from joblib import load
from datetime import datetime
import os
from os import path
from shutil import copyfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import botorch.settings as botorch_settings

from botorch.models import FixedNoiseGP
from gpytorch.kernels import RFFKernel, ScaleKernel

from quantum_bo_env import QuantumFuselageEnv


# =========================
#  Utils: h-GP (sklearn GPR) posterior-only update
# =========================

def freeze_sklearn_gpr(model):
    """Disable hyperparameter optimization. Then fit() becomes posterior-only update."""
    if hasattr(model, "named_steps"):  # Pipeline
        gpr = None
        for _, step in model.named_steps.items():
            if step.__class__.__name__ == "GaussianProcessRegressor":
                gpr = step
        if gpr is None:
            raise RuntimeError("Pipeline has no GaussianProcessRegressor step.")
        gpr.optimizer = None
    else:
        # GaussianProcessRegressor
        model.optimizer = None
    return model


def gpr_predict_mean_std(model, X: torch.Tensor):
    """sklearn GPR / Pipeline predict mean & std on torch tensor X [N,d]."""
    X_np = X.detach().cpu().numpy()
    mu_np, std_np = model.predict(X_np, return_std=True)
    mu = torch.from_numpy(np.asarray(mu_np).reshape(-1)).to(X.device, dtype=torch.float64)
    sig = torch.from_numpy(np.asarray(std_np).reshape(-1)).to(X.device, dtype=torch.float64).clamp_min(1e-12)
    return mu, sig


def observe_h_from_gp(tsai_wu_gp, x_new_torch):
    """
    Placeholder observation of h(x).
    Right now you do NOT have true Tsai-Wu returned by env, so we use GP mean as pseudo-observation.
    Replace this with real Tsai-Wu evaluation when available.
    """
    x_np = x_new_torch.detach().cpu().numpy()
    mu_np = tsai_wu_gp.predict(x_np)  # mean as pseudo-observation
    return torch.tensor(mu_np, dtype=torch.float64, device=x_new_torch.device).view(-1, 1)


def append_data_np(Xh, yh, x_new_torch, h_new_torch):
    x_new = x_new_torch.detach().cpu().numpy()
    h_new = h_new_torch.detach().cpu().numpy().reshape(-1)
    Xh = np.vstack([Xh, x_new])
    yh = np.concatenate([yh, h_new])
    return Xh, yh


# =========================
#  Safe set
# =========================

def build_safe_set(search_space: torch.Tensor, h_gp, xi: float = 1.0, beta_safe: float = 1.645, topk_fallback: int = 20):
    """
    S = {x | mu_h(x) + beta_safe * sigma_h(x) < xi}
    if empty -> fallback topK closest-to-safe points
    """
    search_space = search_space * 1000

    mu_h, sig_h = gpr_predict_mean_std(h_gp, search_space)
    upper_h = mu_h + float(beta_safe) * sig_h
    safe_mask = upper_h < xi

    if safe_mask.any():
        safe_space = search_space[safe_mask]/1000
    else:
        violation = torch.relu(upper_h - xi)
        K = min(int(topk_fallback), search_space.shape[0])
        idx_topk = torch.topk(-violation, k=K).indices
        safe_space = search_space[idx_topk]/1000

    return safe_space, safe_mask, upper_h


# =========================
#  f-GP (your RFF/Gram UCB)
# =========================

def initialize_f_model(actions, response, device, M_target):
    # 统一 dtype（用 X 的 dtype）
    dtype = actions.dtype

    actions = actions.to(device=device, dtype=dtype)
    response = response.to(device=device, dtype=dtype)

    yvar = torch.full_like(response, 1e-8, device=device, dtype=dtype)

    kernel_kwargs = {"nu": 2.5, "ard_num_dims": actions.shape[-1]}
    base_kernel = RFFKernel(**kernel_kwargs, num_samples=M_target)
    covar_module = ScaleKernel(base_kernel).to(device=device, dtype=dtype)

    model = FixedNoiseGP(actions, response, yvar, covar_module=covar_module).to(device)

    with torch.no_grad():
        # lengthscale 也要同 dtype
        ls_init = torch.tensor(
            [[0.1] + [0.6931] * 16 + [0.1]],
            dtype=dtype, device=device
        )
        model.covar_module.base_kernel.lengthscale.copy_(ls_init)
        model.covar_module.outputscale.copy_(torch.tensor(1.0, dtype=dtype, device=device))

    model.eval()
    return model


def W_GP_UCB(model, x_D, response, V_t, Unweighted_Phi, eps_list, beta, lam):
    device = V_t.device
    dtype  = V_t.dtype

    eps_list = torch.tensor(eps_list, device=device, dtype=dtype)
    W = torch.diag(1 / torch.pow(eps_list, 2)).to(device=device, dtype=dtype)

    Unweighted_Phi = Unweighted_Phi.to(device=device, dtype=dtype)
    response = response.to(device=device, dtype=dtype)   # <-- 关键：强制 Double

    V_t_inv = torch.linalg.inv(V_t).to(device=device, dtype=dtype)
    Y_weighted = W @ response.reshape(-1, 1)
    nu_t = V_t_inv @ Unweighted_Phi.T @ Y_weighted

    Phi = model.covar_module.base_kernel.get_features(x_D, 18, normalize=True).to(device=device, dtype=dtype)
    mean = (model.covar_module.outputscale * (Phi @ nu_t)).reshape(-1)

    var = model.covar_module.outputscale * torch.diagonal(lam * Phi @ V_t_inv @ Phi.T)
    sigma = var.clamp_min(1e-12).sqrt()

    score = mean + beta.sqrt() * sigma
    idx = torch.argmax(score)
    x_next = x_D[idx].reshape(1, -1)
    return x_next, mean[idx], var[idx]


# =========================
#  Search space
# =========================

def sample_sobol_on_grid(n=1, device=None, seed=123):
    device = device or torch.device("cpu")
    D = 18
    var_dims = [0, 17]
    V = len(var_dims)

    sobol = torch.quasirandom.SobolEngine(dimension=V, scramble=True, seed=seed)
    u = sobol.draw(n).to(device)

    x = 2.0 * u - 1.0

    values = torch.linspace(-0.5, 0.5, 21, device=device, dtype=torch.float64)  # <-- 加 dtype
    step = values[1] - values[0]
    idx = torch.round((x - values[0]) / step).clamp(0, values.numel()-1).long()

    out = torch.zeros(n, D, device=device, dtype=torch.float64)                  # float64
    out[:, var_dims] = values[idx]                                               # now matches
    return out


def save_data(actions, response, true_response, num_of_queries, eps, file_path):
    torch.save({
        'actions': actions,
        'response': response,
        'queries': num_of_queries,
        'uncertainty': eps,
        'true_response': true_response
    }, file_path)


# =========================
#  Main
# =========================

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    print("Quantum Discrete SAFE GP-UCB (posterior-only h-GP)")

    obs_noise = 0.1 ** 2
    M_target = 400
    xi = 1.0

    # ---- paths ----
    __file__ = 'FuselageActuators'
    folder = path.join(__file__, 'AnsysFiles', "Test")
    file = 'SolutionInputDP52.inp'
    original_input_filename = path.join(folder, file)

    log_dir = "Experiments"
    env_name = "Quantum_Discrete_failure_averse"
    run_dir = path.join(log_dir, env_name, "exp_set_1")
    os.makedirs(run_dir, exist_ok=True)

    input_filename = path.join(run_dir, file)
    copyfile(original_input_filename, input_filename)

    # ---- load h GP ----
    tsai_wu = load('surrogate_tsaiwu.joblib')
    tsai_wu = freeze_sklearn_gpr(tsai_wu)  # posterior-only updates

    # ---- experiment ----
    trails = np.arange(1)
    for tri in trails:
        seed = int(tri)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_default_dtype(torch.float64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        botorch_settings.debug._set_state(True)

        env = QuantumFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise)
        _ = env.reset(input_filename, seed=seed)

        search_space = sample_sobol_on_grid(n=441, seed=seed, device=device)

        lam = 1.0
        eta = 1.0
        eps_list = [1.0]

        # ---- initial safe set + initial x ----
        safe_space0, safe_mask0, _ = build_safe_set(search_space, tsai_wu, xi=xi, beta_safe=0.5, topk_fallback=20)
        x0 = safe_space0[torch.randint(0, safe_space0.shape[0], (1,)).item()].view(1, -1)

        # ---- evaluate f ----
        y0, true_y0, q0 = env.step_surrogate(x0.to(device), eps=eta * 1.0)
        actions = x0.to(device)
        response = y0.view(1, 1).to(device)
        true_response = torch.tensor(true_y0).view(1, 1).to(device)
        num_of_queries = q0.to(device)
        num_of_queries_num = float(q0.item())

        # ---- initial h observation + initialize h posterior-only dataset ----
        h0 = observe_h_from_gp(tsai_wu, actions)    # replace with true Tsai-Wu if available
        Xh = actions.detach().cpu().numpy()
        yh = h0.detach().cpu().numpy().reshape(-1)
        tsai_wu.fit(Xh * 1000, yh)  # posterior-only fit with optimizer=None

        # ---- init f-model (your RFF GP) ----
        f_model = initialize_f_model(actions=actions, response=response, device=device, M_target=M_target)

        # ---- init gram stats ----
        Unweighted_Phi = torch.zeros((1, 2 * M_target), device=device, dtype=torch.float64)
        Phi0 = f_model.covar_module.base_kernel.get_features(actions, 18, normalize=True)
        Unweighted_Phi[0, :] = Phi0

        Weighted_Phi0 = Phi0 * (1 / eps_list[0])
        V_t = f_model.covar_module.outputscale * (Weighted_Phi0.T @ Weighted_Phi0) + lam * torch.eye(2 * M_target, device=device, dtype=torch.float64)

        # ---- loop ----
        iterations = 1
        n_iter = 5000
        ts = torch.arange(1, n_iter, device=device, dtype=torch.float64)
        beta_t = (1 + 1 * torch.sqrt(torch.log(ts) ** 2)) ** 2

        while num_of_queries_num < n_iter:
            _ = env.reset(input_filename, seed=seed)
            iterations += 1
            torch.cuda.empty_cache()

            # 1) safe set from current h posterior
            safe_space, safe_mask, upper_h = build_safe_set(
                search_space=search_space,
                h_gp=tsai_wu,
                xi=xi,
                beta_safe=1,     # ~95% one-sided
                topk_fallback=20
            )
            
            # 2) choose x in safe_space using f UCB
            x_next, post_mean, post_var = W_GP_UCB(
                model=f_model,
                x_D=safe_space,
                response=response,
                V_t=V_t,
                Unweighted_Phi=Unweighted_Phi,
                eps_list=eps_list,
                beta=beta_t[len(response) - 1],
                lam=lam
            )

            # 3) evaluate f
            eps = (torch.sqrt(post_var) / np.sqrt(lam)).item()
            y_next, true_y_next, q_next = env.step_surrogate(x_next, eps * eta, device)

            # update f data
            eps_list.append(float(eps))
            num_of_queries_num += float(q_next.item())
            actions = torch.cat([actions, x_next.to(device)], dim=0)
            response = torch.cat([response, y_next.view(1, 1).to(device)], dim=0)
            true_response = torch.cat([true_response, true_y_next.view(1, 1).to(device)], dim=0)
            num_of_queries = torch.cat([num_of_queries, q_next.to(device)], dim=0)

            # gram update
            Phi_next = f_model.covar_module.base_kernel.get_features(x_next, 18, normalize=True).to(device)
            weighted_next = Phi_next * (1 / eps_list[-1])
            V_t = V_t + (weighted_next.T @ weighted_next)
            Unweighted_Phi = torch.cat([Unweighted_Phi, Phi_next], dim=0)

            # 4) observe h and posterior-only update h GP
            h_next = observe_h_from_gp(tsai_wu, x_next)  # TODO: replace with true Tsai-Wu observation
            Xh, yh = append_data_np(Xh, yh, x_next, h_next)
            tsai_wu.fit(Xh * 1000, yh)  # posterior-only update (optimizer=None)

            print(f"Iter {iterations:4d} | y={y_next.item():.3f} | true={true_y_next.item():.3f} | "
                  f"mean={post_mean.item():.3f} | eps={eps_list[-1]:.3f} | safe#={int(safe_mask.sum())}")

            save_data(actions, response, true_response, num_of_queries, eps_list,
                      path.join(run_dir, f"{obs_noise}_quan_training_data_{tri}_.pth"))