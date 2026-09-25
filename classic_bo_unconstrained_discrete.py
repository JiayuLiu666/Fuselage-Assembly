"""
Classic Unconstrained Bayesian Optimization — GP-UCB (Discrete)
============================================================
Unconstrained baseline aligned to classic_safeset_discrete.py so the two can
be compared apples-to-apples. Every initial setting is identical:

  - Same environment (bo_env_constraints.ClassicFuselageEnv, Chebyshev MC)
  - Same 21x21 discrete grid over dims 0 and 17 (sample_sobol_on_grid)
  - Same safe warmup-point selection (init_num_points), so both start alike
  - Same objective GP: initialize_model, RFF kernel, manual obj_ls lengthscale
  - Same beta_t schedule, V_t / Phi init weighted by eps_max, stage_epsilon
  - Same save_data schema / active-record layout

The ONLY difference vs. the safe-set script is the acquisition:
  - NO constraint GP, NO safe-set mask, NO boundary-expansion blend.
  - Selection is pure GP-UCB argmax over the FULL grid.
  (Tsai-Wu is still evaluated per query, but ONLY to report the violation rate.)
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


def sample_sobol_on_grid(grid_size=21, device=None, seed=None):
    device = device or torch.device("cpu")

    D = 18

    var_dims = [0, 17]

    values = torch.linspace(-1, 1, grid_size, device=device, dtype=torch.float64)

    pairs = torch.cartesian_prod(values, values)

    X = torch.zeros(pairs.shape[0], D, device=device, dtype=torch.float64)

    X[:, var_dims] = pairs

    return X


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
    print("Classic Discrete Unconstrained GP-UCB, 2 actuators")

    parser = argparse.ArgumentParser(
        description="Classic Discrete Unconstrained GP-UCB (aligned to safe-set baseline).")
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
    parser.add_argument("--init_num_points", type=int, default=5,
                        help="Number of safe warmup points for the objective model.")
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

    env_name = "Classic_Discrete_Unconstrained"
    log_dir = log_dir + '/' + env_name + '/' + 'exp_set_1' + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    input_filename = log_dir + file
    copyfile(original_input_filename, input_filename)

    # Load Tsai-Wu constraint model — used ONLY for warmup-point selection and
    # per-query violation-rate reporting (never for acquisition).
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

        # Compute safe set on the full grid (for warmup selection / reporting only)
        grid_np = search_space.detach().cpu().numpy().reshape(-1, 18)
        FI_all = tsai_wu_model.predict(grid_np, return_std=False)
        safety_margin = torch.tensor(1.0 - FI_all, dtype=torch.float64, device=device)
        safe_idx = torch.where(safety_margin >= 0.0)[0]
        print(f"[Init] {len(safe_idx)}/{N} grid points are safe")

        # ── Warmup point selection: SAME safe-set draw as the safe-set script ──
        init_pool_size = init_num_points

        safe_margins = safety_margin[safe_idx]
        weights = safe_margins.clamp_min(0.0).cpu().numpy().astype(float)
        weights /= weights.sum() + 1e-12
        rng = np.random.RandomState(seed)
        chosen_safe = rng.choice(safe_idx.cpu().numpy(), size=min(init_pool_size, len(safe_idx)),
                                 replace=False, p=weights)
        init_idx = torch.tensor(chosen_safe, dtype=torch.long, device=device)

        print(f"[Init] Selected {init_idx.numel()} warmup point(s) (seed={seed}):")
        for line in format_safe_initial_points(search_space, safety_margin, init_idx):
            print(line)

        # ── Noise-free warmup: evaluate objective at warmup points ──
        chosen = init_idx[:init_num_points]
        actions = search_space[chosen].to(device)
        f_X = actions.clone().to(device)

        env.reset(input_filename, seed=seed)
        f_Y, true_response, init_queries, init_obs_var = evaluate_initial_points(
            env, f_X, device=device, obs_precision=0.04
        )
        response = f_Y.clone()

        # ── Manual lengthscale (validated by smoke_test_ls.py) ──
        # Combo A: active=obj_ls, inactive=0.6931
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

        # Seed objective GP with warmup observations for initial posterior structure
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
        active_actions_list   = []
        active_response_list  = []
        active_true_resp_list = []
        active_queries_list   = []
        active_eps_list_only  = []
        active_records = []
        selected_counts = {}
        safe_queries_count = 0
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

            # ===== 2) Pure GP-UCB pick over the FULL grid (no safe-set filtering) =====
            idx = torch.argmax(ucb_f)
            new_actions = search_space[idx].reshape(1, -1)

            # ===== 3) Evaluate new point =====
            selected_idx = int(idx.item())
            var = var_f[idx].item()
            eps = stage_epsilon(var, lam, args.eps_max)

            # Classic env returns 3 values (no c_val)
            new_response, true_responses, num_oracle_queries = env.step_surrogate(new_actions, device=device, eps=eta * eps, method='chebyshev')

            # Evaluate constraint inline — reporting ONLY (never influences selection)
            c_val = eval_tsai_wu(tsai_wu_model, new_actions)
            actual_safe = bool(float(c_val) >= 0.0)
            fi_scalar = float(1.0 - c_val)

            num_of_queries_num += int(num_oracle_queries.item())
            if actual_safe:
                safe_queries_count += 1

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
            # Record schema mirrors the safe-set script. Constraint-GP-derived
            # fields carry the true Tsai-Wu margin as a proxy (no constraint GP
            # exists here); gp_safe is True since the acquisition never filters.
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
                    "gp_safe": True,
                    "gp_unsafe": False,
                    "gp_safe_but_actual_unsafe": bool(not actual_safe),
                    "mu_c": float(c_val),
                    "sig_c": 0.0,
                    "lcb_c": float(c_val),
                    "ucb_c": float(c_val),
                    "mu_c_idx": float(c_val),
                    "sig_c_idx": 0.0,
                    "lcb_c_idx": float(c_val),
                    "ucb_c_idx": float(c_val),
                    "safe_mask_sum": int(N),
                    "safe_mask_sum_before": int(N),
                    "safe_mask_sum_after": int(N),
                    "safe_set_expanded": False,
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

            safety_rate = safe_queries_count / max(1, len(active_records))
            violation_rate = 1.0 - safety_rate
            print('Stage {0} y: {1} f(x): {2} eps: {3} safety rate: {4:.3f} violation rate: {5:.3f}'.format(
                iterations, round(new_response[-1].item(), 3),
                round(true_responses[-1].item(), 3),
                round(eps_list[-1], 3),
                safety_rate, violation_rate))

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
