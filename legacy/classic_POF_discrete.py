"""
Classic Constrained Bayesian Optimization — PoF (Probability of Feasibility)
===============================================================================
Classical counterpart of quantum_cbo_POF.py.
Uses ClassicFuselageEnv (Chebyshev Monte-Carlo sampling) instead of
QuantumFuselageEnv. Constraint is evaluated inline via surrogate_tsaiwu.joblib.
"""
from joblib import dump, load
from datetime import datetime
import os
import sys
from os import path
import gpytorch
from shutil import copyfile
import torch
from bo_env_constraints import ClassicFuselageEnv
import botorch.settings as botorch_settings
from botorch.models import FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import RFFKernel, ScaleKernel
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from utils import sample_actions
import math
from torch.distributions import Normal

# ──────────────────────────────── helpers ────────────────────────────────

@torch.no_grad()
def compute_pof_from_posterior(mu_c: torch.Tensor, sig_c: torch.Tensor, eps: float = 1e-12):
    z = mu_c / sig_c.clamp_min(eps)
    return Normal(0.0, 1.0).cdf(z)


def save_data(actions, response, true_response, num_of_queries, eps, file_path):
    torch.save({
        'actions': actions,
        'response': response,
        'queries': num_of_queries,
        'uncertainty': eps,
        'true_response': true_response
    }, file_path)


def initialize_c_model(actions, response, device):
    yvar = torch.full_like(response, 1e-6, device=device)
    model = FixedNoiseGP(actions, response, yvar).to(device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def initialize_model(actions, response, device, M_target, state_dict=None):
    yvar = torch.full_like(response, 1e-6, device=device)
    kernel_kwargs = {"nu": 2.5, "ard_num_dims": actions.shape[-1]}
    base_kernel = RFFKernel(**kernel_kwargs, num_samples=M_target)
    covar_module = ScaleKernel(base_kernel).to(device)
    model = FixedNoiseGP(actions, response, yvar, covar_module=covar_module).to(device)
    model.eval()
    return model, model.covar_module


def fit_rff_fixednoise_gp(model: FixedNoiseGP):
    model.train(); model.likelihood.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval(); model.likelihood.eval()
    return model


def sample_sobol_on_grid(n=1, device=None, seed=123):
    device = device or torch.device("cpu")
    D = 18
    var_dims = [0, 17]
    V = len(var_dims)
    sobol = torch.quasirandom.SobolEngine(dimension=V, scramble=True, seed=seed)
    u = sobol.draw(n).to(device)
    x = 2.0 * u - 1.0
    values = torch.linspace(-1, 1, 41, device=device)
    step = values[1] - values[0]
    idx = torch.round((x - values[0]) / step).clamp(0, values.numel()-1).long()
    out = torch.zeros(n, D, device=device)
    out[:, var_dims] = values[idx]
    return out


@torch.no_grad()
def compute_lcb_c_botorch(c_model, X, beta_c: float):
    post = c_model.posterior(X)
    mu = post.mean.view(-1)
    var = post.variance.view(-1).clamp_min(1e-12)
    sigma = var.sqrt()
    rad = math.sqrt(beta_c)
    lcb = mu - rad * sigma
    ucb = mu + rad * sigma
    return mu, sigma, lcb, ucb


def W_GP_UCB_scores(model, x_D, response, V_t, Unweighted_Phi, eps_list, beta, lam):
    eps_list = torch.tensor(eps_list, device=V_t.device, dtype=V_t.dtype)
    W = torch.diag(1.0 / (eps_list**2)).to(V_t)
    Unweighted_Phi = Unweighted_Phi.to(V_t)
    response = response.to(V_t)
    V_t_inv = torch.linalg.inv(V_t)
    Y_weighted = (W @ response.reshape(-1, 1))
    nu_t = V_t_inv @ Unweighted_Phi.T @ Y_weighted
    features = model.covar_module.base_kernel.get_features(x_D, 18, normalize=True)
    mean = (model.covar_module.outputscale / model.covar_module.outputscale * (features @ nu_t)).reshape(-1)
    var = model.covar_module.outputscale / model.covar_module.outputscale * torch.diagonal(lam * features @ V_t_inv @ features.T)
    sigma = var.clamp_min(1e-12).sqrt()
    ucb = mean + beta.sqrt() * sigma
    return ucb, mean, sigma, var


def eval_tsai_wu(tsai_wu_model, action):
    """Evaluate Tsai-Wu constraint: c_val = 1 - FI. Feasible if c_val >= 0."""
    if isinstance(action, torch.Tensor):
        action_np = action.detach().cpu().numpy().reshape(1, -1)
    else:
        action_np = np.array(action).reshape(1, -1)
    FI = tsai_wu_model.predict(action_np, return_std=False)
    return float(1 - FI)


# ──────────────────────────────── main ────────────────────────────────

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    print("Classic Discrete PoF, 2 actuators")

    noise_std = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    obs_noise = noise_std ** 2
    M_target = 1024
    print('noise_level: ', obs_noise)

    __file__ = 'FuselageActuators'
    folder = path.join(__file__, 'AnsysFiles', "Test")
    file = 'SolutionInputDP52.inp'
    filepath = path.join(folder, file)
    original_input_filename = filepath
    print("Initial shape from", file.split('.')[0])

    log_dir = "Experiments_constraints"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env_name = "Classic_Discrete_cUCB_POF"
    log_dir = log_dir + '/' + env_name + '/' + 'exp_set_1' + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    input_filename = log_dir + file
    copyfile(original_input_filename, input_filename)

    # Load Tsai-Wu constraint model
    tsai_wu_model = load('surrogate_tsaiwu.joblib')

    trails = np.arange(5)
    for tri in trails:
        seed = int(tri)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_default_dtype(torch.float64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        botorch_settings.debug._set_state(True)
        env = ClassicFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise)
        _ = env.reset(input_filename, seed=seed)

        search_space = sample_sobol_on_grid(n=1681, seed=seed, device=device)
        N = search_space.shape[0]

        lam = 1
        eta = 1

        # ── Initialize constraint GP with 10 random points ──
        train_actions = sample_actions(N=10, d=18, low=-1, high=1, active_idx=(0, 17), seed=10086+tri)
        train_response = []
        for i in range(train_actions.shape[0]):
            env.reset(input_filename, seed=seed)
            c_val = eval_tsai_wu(tsai_wu_model, train_actions[i])
            train_response.append(c_val)
        train_actions_tensor = torch.tensor(train_actions)
        train_response_tensor = torch.tensor(train_response).reshape(-1, 1)

        c_model, c_model_mll = initialize_c_model(actions=train_actions_tensor, response=train_response_tensor, device=device)
        fit_gpytorch_mll(c_model_mll)
        c_model.eval()

        c_X = train_actions_tensor.clone().to(device)
        c_Y = train_response_tensor.clone().to(device)

        # ── Initialize objective GP with 50 random points ──
        initial_num_points = 50
        random_index = torch.randint(0, search_space.shape[0], (initial_num_points,))
        eps_list = [1] * initial_num_points
        actions = search_space[random_index].to(device)

        f_X = actions.clone().to(device)

        init_train_response = []
        for i in range(f_X.shape[0]):
            env.reset(input_filename, seed=seed)
            # Classic env returns 3 values (no c_val)
            response, true_response, num_of_queries = env.step_surrogate(action=f_X[i, :], device=device, eps=1)
            init_train_response.append(response)
        f_Y = torch.tensor(init_train_response).reshape(-1, 1).to(dtype=torch.float64, device=device)

        Weighted_Phi = torch.zeros((f_X.shape[0], 2 * M_target)).to(actions)
        Unweighted_Phi = torch.zeros((f_X.shape[0], 2 * M_target)).to(actions)

        model_ei, kernel = initialize_model(actions=f_X, response=f_Y, M_target=M_target, device=device)
        model_ei = fit_rff_fixednoise_gp(model_ei)

        N = search_space.shape[0]

        for i in range(f_X.shape[0]):
            x = f_X[i, :].reshape(1, -1)
            features = model_ei.covar_module.base_kernel.get_features(x, 18, normalize=True)
            Unweighted_Phi[i, :] = features
            Weighted_Phi[i, :] = features

        V_t = model_ei.covar_module.outputscale / model_ei.covar_module.outputscale * torch.matmul(Weighted_Phi.T, Weighted_Phi) + lam * torch.eye(2 * M_target).to(actions)

        iterations = 1
        n_iter = 30001
        ts = torch.arange(1, n_iter)
        beta_t = (1 + 1 * np.sqrt(np.log(ts) ** 2)) ** 2

        _ = env.reset(input_filename, seed=seed)
        num_of_queries_num = num_of_queries.item()

        while num_of_queries_num < n_iter:
            model_ei.eval()
            iterations += 1
            torch.cuda.empty_cache()

            # ===== 1) Objective UCB =====
            ucb_f, mean_f, sig_f, var_f = W_GP_UCB_scores(
                model=model_ei, x_D=search_space, response=f_Y,
                V_t=V_t, Unweighted_Phi=Unweighted_Phi,
                eps_list=eps_list, beta=beta_t[len(response)-1], lam=lam
            )

            # ===== 2) PoF from constraint GP =====
            beta_c = 3.0
            mu_c, sig_c, lcb_c, ucb_c = compute_lcb_c_botorch(c_model, search_space, beta_c)
            pof = compute_pof_from_posterior(mu_c, sig_c)

            # ===== 3) Acquisition = UCB * PoF =====
            pof_min = 1e-6
            acq = ucb_f * pof.clamp_min(pof_min)

            idx = torch.argmax(acq)
            new_actions = search_space[idx].reshape(1, -1)

            # ===== 4) eps from posterior variance =====
            var = var_f[idx].item()
            eps = (math.sqrt(var) / math.sqrt(lam))

            # Classic env returns 3 values (no c_val)
            new_response, true_responses, num_oracle_queries = env.step_surrogate(new_actions, device=device, eps=eta * eps)

            # Evaluate constraint inline
            c_val = eval_tsai_wu(tsai_wu_model, new_actions)

            num_of_queries_num += num_oracle_queries.item()

            # Update constraint GP
            c_X = torch.cat([c_X, new_actions.to(c_X)], dim=0)
            c_Y = torch.cat([c_Y, torch.tensor([[float(c_val)]], device=device, dtype=torch.float64)], dim=0)
            c_model, c_mll = initialize_c_model(actions=c_X, response=c_Y, device=device)
            fit_gpytorch_mll(c_mll)
            c_model.eval()

            eps_list.append(eps)

            actions = torch.cat([actions, new_actions.to(torch.float64).to(device)])
            response = torch.cat([response, new_response.to(torch.float64).to(device)])
            true_response = torch.cat([true_response, true_responses.to(true_response)])
            f_Y = torch.cat([f_Y, new_response.to(torch.float64).to(device)])

            x_max_feature = model_ei.covar_module.base_kernel.get_features(new_actions, 18, normalize=True)
            weighted_feature = x_max_feature * (1 / eps_list[-1])
            V_t = V_t + torch.matmul(weighted_feature.T, weighted_feature)
            Unweighted_Phi = torch.cat([Unweighted_Phi, x_max_feature])

            _ = env.reset(input_filename, seed=seed)
            num_of_queries = torch.cat([num_of_queries, num_oracle_queries.to(device)])

            print('Stage {0} y: {1} f(x): {2} eps: {3} PoF>0.9: {4}/{5}'.format(
                iterations, round(new_response[-1].item(), 3),
                round(true_responses[-1].item(), 3),
                round(eps_list[-1], 3),
                int((pof > 0.9).sum().item()), N))

            save_data(actions, response, true_response, num_of_queries, eps_list,
                      log_dir + str(obs_noise) + 'quan_training_data_' + str(tri) + '_.pth')
            _ = env.reset(input_filename, seed=seed)
