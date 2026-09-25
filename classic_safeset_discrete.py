"""
Classic Constrained Bayesian Optimization — Safe-Set cUCB
============================================================
Classical counterpart of quantum_cbo_discrete.py.
Uses ClassicFuselageEnv (Chebyshev Monte-Carlo sampling) instead of
QuantumFuselageEnv. Constraint is evaluated inline via surrogate_tsaiwu.joblib.
"""
import argparse
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

NOISE_FREE_VARIANCE = 1e-6

# ──────────────────────────────── helpers ────────────────────────────────

def minmax_norm(a: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12):
    v = a[mask]
    if v.numel() == 0:
        return a * 0.0
    lo = v.min()
    hi = v.max()
    return (a - lo) / (hi - lo + eps)

def lambda_by_stage(stage, lam0=1, t0=20, p=1.0):
    return float(lam0 * (t0 / (t0 + max(1, stage)))**p)


def stage_epsilon(var_value, lam, eps_max):
    """epsilon_t = min(epsilon_max, sqrt(var(x) / lambda))."""
    return min(float(eps_max), math.sqrt(max(float(var_value), 1e-12) / float(lam)))


def save_data(
    actions,
    response,
    true_response,
    num_of_queries,
    eps,
    file_path,
    active_records=None,
    metadata=None,
):
    payload = {
        'actions': actions,
        'response': response,
        'queries': num_of_queries,
        'uncertainty': eps,
        'true_response': true_response,
    }
    if active_records is not None:
        payload['active_records'] = active_records
    if metadata is not None:
        payload['metadata'] = metadata
    torch.save(payload, file_path)


def initialize_c_model(actions, response, device, ls_init=None, outputscale=None):
    yvar = torch.full_like(response, NOISE_FREE_VARIANCE, device=device)
    model = FixedNoiseGP(actions, response, yvar).to(device)
    if ls_init is not None or outputscale is not None:
        with torch.no_grad():
            if ls_init is not None:
                model.covar_module.base_kernel.lengthscale.copy_(ls_init.to(device))
            if outputscale is not None:
                model.covar_module.outputscale.copy_(outputscale.to(device))
        model.eval()
        return model, None
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def initialize_model(actions, response, device, M_target, state_dict=None,
                     obs_noise=NOISE_FREE_VARIANCE, ls_init=None, outputscale=None):
    noise_tensor = torch.as_tensor(obs_noise, dtype=response.dtype, device=device)
    if noise_tensor.ndim == 0:
        yvar = torch.full_like(response, float(noise_tensor.item()), device=device)
    else:
        yvar = noise_tensor.reshape(-1, 1) if noise_tensor.ndim == 1 else noise_tensor
        yvar = yvar.expand_as(response).clone()
    kernel_kwargs = {"nu": 2.5, "ard_num_dims": actions.shape[-1]}
    base_kernel = RFFKernel(**kernel_kwargs, num_samples=M_target)
    covar_module = ScaleKernel(base_kernel).to(device)
    model = FixedNoiseGP(actions, response, yvar, covar_module=covar_module).to(device)
    with torch.no_grad():
        if ls_init is not None:
            model.covar_module.base_kernel.lengthscale.copy_(ls_init.to(device))
        model.covar_module.outputscale.copy_(
            torch.tensor(1.0, dtype=torch.double, device=device)
        )
    model.eval()
    return model, model.covar_module


def fit_rff_fixednoise_gp(model: FixedNoiseGP):
    model.train(); model.likelihood.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval(); model.likelihood.eval()
    return model


def sample_sobol_on_grid(grid_size=21, device=None, seed=None):
    device = device or torch.device("cpu")

    D = 18

    var_dims = [0, 17]

    values = torch.linspace(-1, 1, grid_size, device=device, dtype=torch.float64)

    pairs = torch.cartesian_prod(values, values)

    X = torch.zeros(pairs.shape[0], D, device=device, dtype=torch.float64)

    X[:, var_dims] = pairs

    return X


def _normalize_scores(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    return (values - lo) / (hi - lo + eps)


def select_safe_initial_points(search_space, safety_margin, num_points, active_dims=(0, 17)):
    """Choose conservative safe warmup points from the discrete grid."""
    safety_margin = torch.as_tensor(
        safety_margin, dtype=search_space.dtype, device=search_space.device
    ).view(-1)
    safe_idx = torch.where(safety_margin >= 0.0)[0]
    if safe_idx.numel() == 0:
        raise RuntimeError("No safe grid points available for initialization.")

    target = min(int(num_points), int(safe_idx.numel()))
    if target <= 0:
        return torch.empty(0, dtype=torch.long, device=search_space.device)

    active_coords = search_space[:, list(active_dims)]
    preferred_pairs = torch.tensor(
        [
            [0.0, 0.0],
            [-0.2, 0.0],
            [0.2, 0.0],
            [0.0, -0.2],
            [0.0, 0.2],
            [-0.2, -0.2],
            [-0.2, 0.2],
            [0.2, -0.2],
            [0.2, 0.2],
            [-0.4, 0.0],
            [0.4, 0.0],
            [0.0, -0.4],
            [0.0, 0.4],
        ],
        dtype=search_space.dtype,
        device=search_space.device,
    )

    selected = []
    for pair in preferred_pairs:
        matches = safe_idx[
            torch.isclose(active_coords[safe_idx, 0], pair[0], atol=1e-9)
            & torch.isclose(active_coords[safe_idx, 1], pair[1], atol=1e-9)
        ]
        if matches.numel() == 0:
            continue
        best_match = matches[torch.argmax(safety_margin[matches])]
        best_idx = int(best_match.item())
        if best_idx not in selected:
            selected.append(best_idx)
        if len(selected) >= target:
            break

    while len(selected) < target:
        remaining = [idx for idx in safe_idx.tolist() if idx not in selected]
        if not remaining:
            break
        remaining_idx = torch.tensor(
            remaining, dtype=torch.long, device=search_space.device
        )

        margin_score = _normalize_scores(safety_margin[remaining_idx])
        if selected:
            selected_idx = torch.tensor(
                selected, dtype=torch.long, device=search_space.device
            )
            min_dist = torch.cdist(
                active_coords[remaining_idx], active_coords[selected_idx]
            ).min(dim=1).values
            dist_score = _normalize_scores(min_dist)
        else:
            dist_score = torch.ones_like(margin_score)

        center_dist = active_coords[remaining_idx].norm(dim=1)
        center_score = 1.0 - _normalize_scores(center_dist)
        total_score = 0.60 * margin_score + 0.30 * dist_score + 0.10 * center_score
        best_idx = int(remaining_idx[torch.argmax(total_score)].item())
        selected.append(best_idx)

    return torch.tensor(selected, dtype=torch.long, device=search_space.device)


def format_safe_initial_points(search_space, safety_margin, indices, active_dims=(0, 17)):
    lines = []
    for rank, idx in enumerate(indices.tolist(), start=1):
        a0 = float(search_space[idx, active_dims[0]].item())
        a1 = float(search_space[idx, active_dims[1]].item())
        margin = float(safety_margin[idx].item())
        lines.append(
            f"  {rank:02d}. idx={idx:03d}, dim{active_dims[0]}={a0:+.2f}, "
            f"dim{active_dims[1]}={a1:+.2f}, 1-FI={margin:+.4f}"
        )
    return lines


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


def W_GP_UCB_scores(model, x_D, response, V_t, Unweighted_Phi, eps_list, beta, lam, chunk_size=1024):
    eps_list = torch.as_tensor(eps_list, device=V_t.device, dtype=V_t.dtype).view(-1, 1)
    Unweighted_Phi = Unweighted_Phi.to(device=V_t.device, dtype=V_t.dtype)
    response = response.to(device=V_t.device, dtype=V_t.dtype).reshape(-1, 1)
    x_D = x_D.to(device=V_t.device, dtype=V_t.dtype)
    beta = torch.as_tensor(beta, device=V_t.device, dtype=V_t.dtype)

    Y_weighted = response / (eps_list ** 2)

    L = torch.linalg.cholesky(V_t)
    rhs = Unweighted_Phi.T @ Y_weighted
    nu_t = torch.cholesky_solve(rhs, L)

    mean_chunks = []
    var_chunks = []
    for start in range(0, x_D.shape[0], chunk_size):
        x_chunk = x_D[start : start + chunk_size]
        features = model.covar_module.base_kernel.get_features(x_chunk, x_chunk.shape[-1], normalize=True)
        mean_chunks.append((features @ nu_t).reshape(-1))
        proj = torch.cholesky_solve(features.T, L).T
        chunk_var = (lam * (proj * features).sum(dim=1)).clamp_min(1e-12)
        var_chunks.append(chunk_var)

    mean = torch.cat(mean_chunks, dim=0)
    var = torch.cat(var_chunks, dim=0)
    sigma = var.sqrt()

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


def _as_float(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def evaluate_initial_points(env, actions, device, obs_precision: float = 0.04):
    """Initial objective evaluations using fixed observation precision."""
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
    else:
        actions_np = np.asarray(actions)

    actions_np = actions_np.reshape(-1, 18).astype(np.float64)
    response_list = []
    true_response_list = []
    query_list = []
    variance_list = []

    for action_np in actions_np:
        env.forces = np.zeros(18, dtype=np.float32)
        y_obs, y_true, n_q = env.step_surrogate(
            torch.tensor(action_np, dtype=torch.float64, device=device).reshape(1, -1),
            device=device,
            eps=obs_precision,
            method='chebyshev',
        )
        response_list.append(y_obs.to(torch.float64).reshape(1, 1))
        true_response_list.append(y_true.to(torch.float64).reshape(1, 1))
        query_list.append(n_q.reshape(1, 1))
        variance_list.append(
            torch.full((1, 1), obs_precision ** 2, dtype=torch.float64, device=device)
        )

    response_tensor = torch.cat(response_list, dim=0)
    true_response_tensor = torch.cat(true_response_list, dim=0)
    query_tensor = torch.cat(query_list, dim=0)
    variance_tensor = torch.cat(variance_list, dim=0)
    return response_tensor, true_response_tensor, query_tensor, variance_tensor


# ──────────────────────────────── main ────────────────────────────────

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    print("Classic Discrete Safe-Set cUCB, 2 actuators")

    parser = argparse.ArgumentParser(
        description="Classic Discrete Safe-Set cUCB.")
    parser.add_argument("--query_budget", type=int, default=20000,
                        help="Oracle query budget.")
    parser.add_argument("--eps_max", type=float, default=0.04,
                        help="Maximum per-stage epsilon before eta scaling.")
    parser.add_argument("--obs_noise", type=float, default=0.2**2,
                        help="Observation noise variance.")
    parser.add_argument("--min_shots", type=int, default=20,
                        help="Minimum shots for Monte Carlo Estimation.")
    parser.add_argument("--M_features", type=int, default=400,
                        help="Number of RFF features.")
    parser.add_argument("--B", type=float, default=3.0,
                        help="Exploration coefficient B in beta_t.")
    parser.add_argument("--obj_ls", type=float, default=0.2,
                        help="Objective lengthscale for active dimensions.")
    parser.add_argument("--lam0", type=float, default=1.0,
                        help="Initial lambda weight for boundary expansion.")
    parser.add_argument("--t0", type=float, default=10.0,
                        help="Decay timescale for lambda schedule.")
    parser.add_argument("--lam_p", type=float, default=2.0,
                        help="Decay power for lambda schedule.")
    parser.add_argument("--init_num_points", type=int, default=5,
                        help="Number of safe warmup points for both objective and constraint models.")
    args = parser.parse_args()

    query_budget = args.query_budget
    Big_B = args.B
    init_num_points = args.init_num_points
    obs_noise = args.obs_noise
    min_shots = args.min_shots
    M_target = args.M_features
    print('noise_level: ', obs_noise)
    print('eps_max: ', args.eps_max)

    __file__ = 'FuselageActuators'
    folder = path.join(__file__, 'AnsysFiles', "Test")
    file = 'SolutionInputDP52.inp'
    filepath = path.join(folder, file)
    original_input_filename = filepath
    print("Initial shape from", file.split('.')[0])

    log_dir = "Experiments_constraints"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env_name = "Classic_Discrete_cUCB"
    log_dir = log_dir + '/' + env_name + '/' + 'exp_set_1' + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    input_filename = log_dir + file
    copyfile(original_input_filename, input_filename)

    # Load Tsai-Wu constraint model (same as quantum env uses)
    tsai_wu_model = load('surrogate_tsaiwu.joblib')

    trails = np.arange(5)
    for tri in trails:
        seed = int(tri)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_default_dtype(torch.float64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        botorch_settings.debug._set_state(True)
        env = ClassicFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise, min_shots=min_shots)
        _ = env.reset(input_filename, seed=seed)

        search_space = sample_sobol_on_grid(grid_size=21, seed=seed, device=device)
        N = search_space.shape[0]

        lam = 1
        eta = 1

        # Compute safe set on the full grid
        grid_np = search_space.detach().cpu().numpy().reshape(-1, 18)
        FI_all = tsai_wu_model.predict(grid_np, return_std=False)
        safety_margin = torch.tensor(1.0 - FI_all, dtype=torch.float64, device=device)
        safe_idx = torch.where(safety_margin >= 0.0)[0]
        print(f"[Init] {len(safe_idx)}/{N} grid points are safe")

        # ── Initialize constraint GP with points from safe set ──
        init_pool_size = init_num_points

        # Seed-dependent random init from safe grid points (different per trial)
        safe_margins = safety_margin[safe_idx]
        weights = safe_margins.clamp_min(0.0).cpu().numpy().astype(float)
        weights /= weights.sum() + 1e-12
        rng = np.random.RandomState(seed)
        chosen_safe = rng.choice(safe_idx.cpu().numpy(), size=min(init_pool_size, len(safe_idx)),
                                 replace=False, p=weights)
        init_idx = torch.tensor(chosen_safe, dtype=torch.long, device=device)

        print(f"[Init] Selected {init_idx.numel()} safe warmup point(s) (seed={seed}):")
        for line in format_safe_initial_points(search_space, safety_margin, init_idx):
            print(line)

        c_chosen = init_idx[:init_num_points]
        train_actions_tensor = search_space[c_chosen].to(device)
        train_actions_np = train_actions_tensor.detach().cpu().numpy().reshape(-1, 18)
        train_response = []
        for i in range(train_actions_np.shape[0]):
            env.reset(input_filename, seed=seed)
            c_val = eval_tsai_wu(tsai_wu_model, train_actions_np[i])
            train_response.append(c_val)
        train_response_tensor = torch.tensor(
            train_response, dtype=torch.float64, device=device).reshape(-1, 1)

        # Manual constraint GP hyperparameters (validated by smoke_test_constraint_ls.py)
        c_ls = torch.full((1, 18), 0.6931, dtype=torch.float64, device=device)
        c_ls[0, 0]  = 0.50
        c_ls[0, 17] = 0.50
        c_os = torch.tensor(1.0, dtype=torch.float64, device=device)

        c_model, _ = initialize_c_model(
            actions=train_actions_tensor, response=train_response_tensor,
            device=device, ls_init=c_ls, outputscale=c_os)
        c_model.eval()

        c_X = train_actions_tensor.clone().to(device)
        c_Y = train_response_tensor.clone().to(device)

        # ── Noise-free warmup: sample from safe set ──
        chosen = init_idx[:init_num_points]
        actions = search_space[chosen].to(device)
        f_X = actions.clone().to(device)

        env.reset(input_filename, seed=seed)
        f_Y, true_response, init_queries, init_obs_var = evaluate_initial_points(
            env, f_X, device=device, obs_precision=0.04
        )
        response = f_Y.clone()

        # ── Manual lengthscale (validated by smoke_test_ls.py) ──
        # Combo A: active=0.20, inactive=0.6931 (now configurable via args.obj_ls)
        lengthscale = torch.full((1, 18), 0.6931, dtype=torch.float64, device=device)
        lengthscale[0, 0]  = args.obj_ls   # active dim
        lengthscale[0, 17] = args.obj_ls   # active dim
        outputscale = torch.tensor(1.0, dtype=torch.float64, device=device)
        print(f"[Manual LS] active={lengthscale[0,0].item():.2f}, "
              f"inactive={lengthscale[0,1].item():.4f}, "
              f"outputscale={outputscale.item():.2f}")

        # Initialise objective GP kernel with manual hyperparameters.
        init_X = f_X[:init_num_points]
        init_Y = f_Y[:init_num_points]
        init_yvar = init_obs_var[:init_num_points].clone()
        model_ei, kernel = initialize_model(
            actions=init_X, response=init_Y, M_target=M_target, device=device,
            obs_noise=init_yvar, ls_init=lengthscale, outputscale=outputscale)
        model_ei.eval()

        # Seed objective GP with first 2 warmup observations for initial posterior structure
        n_seed = init_num_points
        f_X = f_X[:n_seed].clone()
        f_Y = f_Y[:n_seed].clone()
        actions = f_X.clone()
        response = f_Y.clone()
        true_response = true_response[:n_seed].clone()
        eps_list = [args.eps_max] * n_seed
        num_of_queries = init_queries[:n_seed].clone()
        Weighted_Phi = torch.zeros((n_seed, 2 * M_target), device=device, dtype=torch.float64)
        Unweighted_Phi = torch.zeros((n_seed, 2 * M_target), device=device, dtype=torch.float64)

        N = search_space.shape[0]
        beta_c = 3.0
        mu_c, sig_c, lcb_c, ucb_c = compute_lcb_c_botorch(c_model, search_space, beta_c)
        safe_mask = lcb_c >= 0.0
        print("verified safe:", int(safe_mask.sum().item()), "/", N)

        for i in range(f_X.shape[0]):
            x = f_X[i, :].reshape(1, -1)
            features = model_ei.covar_module.base_kernel.get_features(x, 18, normalize=True)
            Unweighted_Phi[i, :] = features
            Weighted_Phi[i, :] = features * (1.0 / args.eps_max)

        V_t = 1 * torch.matmul(Weighted_Phi.T, Weighted_Phi) + lam * torch.eye(2 * M_target).to(actions)

        iterations = 1
        n_iter = query_budget + 10  # used only for beta_t scheduling
        ts = torch.arange(1, n_iter + 1, device=device, dtype=torch.float64)
        beta_t = (1.0 + Big_B * torch.log(ts).abs()).pow(2)

        _ = env.reset(input_filename, seed=seed)
        num_of_queries_num = 0

        # Separate lists for ACTIVE (post-init) queried points only.
        # Saved to disk; init points are excluded.
        active_actions_list   = []
        active_response_list  = []
        active_true_resp_list = []
        active_queries_list   = []
        active_eps_list_only  = []
        active_records = []
        selected_counts = {}
        print(f"Initial total queries: {num_of_queries_num}")
        while num_of_queries_num < query_budget:
            model_ei.eval()
            iterations += 1
            torch.cuda.empty_cache()

            # ===== 1) Objective UCB =====
            t_acq = max(1, len(response) + 1)
            beta = beta_t[t_acq - 1]
            ucb_f, mean_f, sig_f, var_f = W_GP_UCB_scores(
                model=model_ei, x_D=search_space, response=f_Y,
                V_t=V_t, Unweighted_Phi=Unweighted_Phi,
                eps_list=eps_list, beta=beta, lam=lam
            )

            # ===== 2) Constraint safe set =====
            beta_c = 3.0
            mu_c, sig_c, lcb_c, ucb_c = compute_lcb_c_botorch(c_model, search_space, beta_c)
            safe_mask = lcb_c >= 0.0
            safe_mask_sum_before = int(safe_mask.sum().item())

            if safe_mask.sum().item() == 0:
                idx = torch.argmax(lcb_c)
                new_actions = search_space[idx].reshape(1, -1)
            else:
                # ===== 3) Boundary expansion =====
                eps_sigma = 1e-9
                a_bnd = -torch.abs(mu_c / (sig_c + eps_sigma))

                # ===== 4) Normalize + blend =====
                ucb_n = minmax_norm(ucb_f, mask=safe_mask)
                bnd_n = minmax_norm(a_bnd, mask=safe_mask)
                lam_t = lambda_by_stage(int(iterations), lam0=args.lam0, t0=args.t0, p=args.lam_p)
                score = (1.0 - lam_t) * ucb_n + lam_t * bnd_n
                score[~safe_mask] = -1e18
                idx = torch.argmax(score)
                new_actions = search_space[idx].reshape(1, -1)

            # ===== 5) Evaluate new point =====
            selected_idx = int(idx.item())
            mu_c_idx = float(mu_c[selected_idx].item())
            sig_c_idx = float(sig_c[selected_idx].item())
            lcb_c_idx = float(lcb_c[selected_idx].item())
            ucb_c_idx = float(ucb_c[selected_idx].item())
            gp_safe = bool(safe_mask[selected_idx].item())

            var = var_f[idx].item()
            eps = stage_epsilon(var, lam, args.eps_max)

            # Classic env returns 3 values (no c_val)
            new_response, true_responses, num_oracle_queries = env.step_surrogate(new_actions, device=device, eps=eta * eps, method='chebyshev')

            # Evaluate constraint inline
            c_val = eval_tsai_wu(tsai_wu_model, new_actions)
            actual_safe = bool(float(c_val) >= 0.0)
            fi_scalar = float(1.0 - c_val)

            num_of_queries_num += int(num_oracle_queries.item())

            # Update constraint GP
            c_X = torch.cat([c_X, new_actions.to(c_X)], dim=0)
            c_Y = torch.cat([c_Y, torch.tensor([[float(c_val)]], device=device, dtype=torch.float64)], dim=0)
            c_model, _ = initialize_c_model(
                actions=c_X,
                response=c_Y,
                device=device,
                ls_init=c_ls,
                outputscale=c_os,
            )
            _, _, lcb_c_next, _ = compute_lcb_c_botorch(c_model, search_space, beta_c)
            safe_mask_next = lcb_c_next >= 0.0
            safe_mask_sum_after = int(safe_mask_next.sum().item())
            safe_set_expanded = bool(safe_mask_sum_after > safe_mask_sum_before)

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

            # Accumulate active-only records
            active_actions_list.append(new_actions.to(torch.float64).to(device))
            active_response_list.append(new_response.to(torch.float64).to(device))
            active_true_resp_list.append(true_responses.to(torch.float64).to(device))
            active_queries_list.append(num_oracle_queries.to(device))
            active_eps_list_only.append(eps)

            selection_count_for_idx = selected_counts.get(selected_idx, 0) + 1
            selected_counts[selected_idx] = selection_count_for_idx
            active_records.append(
                {
                    "trial": int(tri),
                    "seed": int(seed),
                    "iteration": int(iterations),
                    "active_step": int(len(active_records) + 1),
                    "selected_index": selected_idx,
                    "selected_idx": selected_idx,
                    "c_val": float(c_val),
                    "FI": fi_scalar,
                    "safe": actual_safe,
                    "unsafe": bool(not actual_safe),
                    "actual_safe": actual_safe,
                    "actual_unsafe": bool(not actual_safe),
                    "gp_safe": gp_safe,
                    "gp_unsafe": bool(not gp_safe),
                    "gp_safe_but_actual_unsafe": bool(gp_safe and (not actual_safe)),
                    "mu_c": mu_c_idx,
                    "sig_c": sig_c_idx,
                    "lcb_c": lcb_c_idx,
                    "ucb_c": ucb_c_idx,
                    "mu_c_idx": mu_c_idx,
                    "sig_c_idx": sig_c_idx,
                    "lcb_c_idx": lcb_c_idx,
                    "ucb_c_idx": ucb_c_idx,
                    "safe_mask_sum": safe_mask_sum_before,
                    "safe_mask_sum_before": safe_mask_sum_before,
                    "safe_mask_sum_after": safe_mask_sum_after,
                    "safe_set_expanded": safe_set_expanded,
                    "num_oracle_queries": int(num_oracle_queries.item()),
                    "eps": float(eps),
                    "selection_count_for_idx": int(selection_count_for_idx),
                    "repeated_sampling_count": int(selection_count_for_idx - 1),
                    "total_budget_after_step": int(num_of_queries_num),
                    "budget_overspent": bool(num_of_queries_num > query_budget),
                    "objective_observation": float(new_response.reshape(-1)[0].item()),
                    "objective_true": float(true_responses.reshape(-1)[0].item()),
                }
            )

            num_safe = safe_mask_sum_before
            safe_rate = num_safe / N
            print('Stage {0} y: {1} f(x): {2} eps: {3} safeset rate: {4:.3f} ({5}/{6})'.format(
                iterations, round(new_response[-1].item(), 3),
                round(true_responses[-1].item(), 3),
                round(eps_list[-1], 3),
                safe_rate, num_safe, N))

            # Save ONLY the actively queried points (no init mixing)
            def _cat(lst):
                return torch.cat(lst, dim=0) if lst else torch.empty(0)
            save_data(_cat(active_actions_list), _cat(active_response_list),
                      _cat(active_true_resp_list), _cat(active_queries_list),
                      active_eps_list_only,
                      log_dir + str(obs_noise) + 'training_data_' + str(tri) + '_.pth',
                      active_records=active_records,
                      metadata={
                          "query_budget": int(query_budget),
                          "trial": int(tri),
                          "seed": int(seed),
                          "num_active_steps": int(len(active_records)),
                          "final_total_budget": int(num_of_queries_num),
                      })
            _ = env.reset(input_filename, seed=seed)

            # ---- End-of-trial save (always runs) ----
        def _cat(lst):
            return torch.cat(lst, dim=0) if lst else torch.empty(0)
        save_data(_cat(active_actions_list), _cat(active_response_list),
                  _cat(active_true_resp_list), _cat(active_queries_list),
                  active_eps_list_only,
                  log_dir + str(obs_noise) + 'training_data_' + str(tri) + '_.pth',
                  active_records=active_records,
                  metadata={
                      "query_budget": int(query_budget),
                      "trial": int(tri),
                      "seed": int(seed),
                      "num_active_steps": int(len(active_records)),
                      "final_total_budget": int(num_of_queries_num),
                  })
        print(f"[Trial {tri}] Saved  ({int(num_of_queries_num)} oracle queries,  {len(active_actions_list)} active steps)")
