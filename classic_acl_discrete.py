import argparse
import copy
import math
import os
from os import path
from shutil import copyfile
import warnings

import numpy as np
import torch
from bo_env_constraints import ClassicFuselageEnv
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from joblib import load


try:
    from botorch.models import FixedNoiseGP as _NoiseAwareGP

    def _build_gp(train_X, train_Y, yvar, covar_module=None, mean_module=None):
        return _NoiseAwareGP(
            train_X,
            train_Y,
            yvar,
            covar_module=covar_module,
            mean_module=mean_module,
        )

except ImportError:
    from botorch.models import SingleTaskGP as _NoiseAwareGP

    def _build_gp(train_X, train_Y, yvar, covar_module=None, mean_module=None):
        return _NoiseAwareGP(
            train_X,
            train_Y,
            train_Yvar=yvar,
            covar_module=covar_module,
            mean_module=mean_module,
        )


warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

NOISE_FREE_VARIANCE = 1e-6


def fit_gp_model(train_X, train_Y, noise_var=1e-4):
    X_rounded = torch.round(train_X * 1e5) / 1e5
    unique_X, inverse = torch.unique(X_rounded, dim=0, return_inverse=True)
    unique_Y = torch.zeros(
        unique_X.shape[0], 1, dtype=train_Y.dtype, device=train_Y.device
    )
    counts = torch.zeros(unique_X.shape[0], dtype=train_Y.dtype, device=train_Y.device)

    # Build per-point noise tensor from input
    if isinstance(noise_var, torch.Tensor):
        nv = noise_var.view(-1).to(train_Y.device)
    else:
        nv = torch.full((train_X.shape[0],), max(float(noise_var), 1e-6),
                        dtype=train_Y.dtype, device=train_Y.device)
    unique_nv_sum = torch.zeros(unique_X.shape[0], dtype=train_Y.dtype, device=train_Y.device)

    for i in range(train_X.shape[0]):
        unique_Y[inverse[i]] += train_Y[i]
        counts[inverse[i]] += 1
        unique_nv_sum[inverse[i]] += nv[i]
    unique_Y = unique_Y / counts.unsqueeze(1)
    unique_nv = (unique_nv_sum / counts).clamp_min(1e-6)

    yvar = unique_nv.unsqueeze(1)
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]))
    model = _build_gp(unique_X, unique_Y, yvar, covar_module=covar_module)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except Exception as err:
        warnings.warn(f"GP fitting failed; using default hyperparameters. Error: {err}")
    model.eval()
    model.likelihood.eval()
    return model


def update_gp_model_fixed_params(base_model, train_X, train_Y, noise_var=1e-4):
    X_rounded = torch.round(train_X * 1e5) / 1e5
    unique_X, inverse = torch.unique(X_rounded, dim=0, return_inverse=True)
    unique_Y = torch.zeros(
        unique_X.shape[0], 1, dtype=train_Y.dtype, device=train_Y.device
    )
    counts = torch.zeros(unique_X.shape[0], dtype=train_Y.dtype, device=train_Y.device)

    if isinstance(noise_var, torch.Tensor):
        nv = noise_var.view(-1).to(train_Y.device)
    else:
        nv = torch.full((train_X.shape[0],), max(float(noise_var), 1e-6),
                        dtype=train_Y.dtype, device=train_Y.device)
    unique_nv_sum = torch.zeros(unique_X.shape[0], dtype=train_Y.dtype, device=train_Y.device)

    for i in range(train_X.shape[0]):
        unique_Y[inverse[i]] += train_Y[i]
        counts[inverse[i]] += 1
        unique_nv_sum[inverse[i]] += nv[i]
    unique_Y = unique_Y / counts.unsqueeze(1)
    unique_nv = (unique_nv_sum / counts).clamp_min(1e-6)

    yvar = unique_nv.unsqueeze(1)
    model = _build_gp(
        unique_X,
        unique_Y,
        yvar,
        covar_module=copy.deepcopy(base_model.covar_module),
        mean_module=copy.deepcopy(base_model.mean_module),
    )
    model.eval()
    model.likelihood.eval()
    return model


def fit_or_update_gp_model(base_model, train_X, train_Y, noise_var=1e-4):
    if base_model is None:
        return fit_gp_model(train_X, train_Y, noise_var=noise_var)
    return update_gp_model_fixed_params(base_model, train_X, train_Y, noise_var=noise_var)


@torch.no_grad()
def get_gp_predictions(model, X):
    posterior = model.posterior(X)
    mean = posterior.mean.view(-1)
    sigma = posterior.variance.view(-1).clamp_min(1e-12).sqrt()
    return mean, sigma


def initialize_c_model(actions, response, device, ls_init=None, outputscale=None):
    yvar = torch.full_like(response, NOISE_FREE_VARIANCE, device=device)
    model = _NoiseAwareGP(actions, response, yvar).to(device)
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


@torch.no_grad()
def compute_lcb_c(c_model, X, beta_c: float):
    post = c_model.posterior(X)
    mu = post.mean.view(-1)
    sigma = post.variance.view(-1).clamp_min(1e-12).sqrt()
    rad = math.sqrt(beta_c)
    return mu, sigma, mu - rad * sigma, mu + rad * sigma


def evaluate_initial_points(env, actions, device, obs_precision: float = 0.04):
    """Initial objective evaluations using fixed observation precision (matches classic_safeset_discrete.py)."""
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
    else:
        actions_np = np.asarray(actions)

    actions_np = actions_np.reshape(-1, 18).astype(np.float64)
    response_list, true_response_list, query_list, variance_list = [], [], [], []

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

    return (
        torch.cat(response_list, dim=0),
        torch.cat(true_response_list, dim=0),
        torch.cat(query_list, dim=0),
        torch.cat(variance_list, dim=0),
    )


def norm_01(values):
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    if hi > lo:
        return (values - lo) / (hi - lo + 1e-12)
    return torch.zeros_like(values)

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


def build_discrete_search_space(
    device,
    grid_min=-1.0,
    grid_max=1.0,
    grid_points=21,
    dim=18,
    active_dims=(0, 17),
):
    values = torch.linspace(
        grid_min,
        grid_max,
        grid_points,
        device=device,
        dtype=torch.float64,
    )
    mesh = torch.cartesian_prod(values, values)
    search_space = torch.zeros(mesh.shape[0], dim, device=device, dtype=torch.float64)
    search_space[:, active_dims[0]] = mesh[:, 0]
    search_space[:, active_dims[1]] = mesh[:, 1]
    return search_space


def eval_tsai_wu(tsai_wu_model, action):
    if isinstance(action, torch.Tensor):
        action_np = action.detach().cpu().numpy().reshape(1, -1)
    else:
        action_np = np.asarray(action).reshape(1, -1)
    failure_index = tsai_wu_model.predict(action_np, return_std=False)
    return float(1.0 - failure_index)


def best_feasible_value(train_Y, train_C):
    feasible_mask = train_C.view(-1) >= 0.0
    if not feasible_mask.any():
        return np.nan
    return float(train_Y[feasible_mask].max().item())


def save_data(
    actions,
    response,
    true_response,
    num_of_queries,
    uncertainty,
    constraint_margin,
    queried_idx,
    best_feasible_hist,
    batch_history,
    file_path,
):
    torch.save(
        {
            "actions": actions,
            "response": response,
            "true_response": true_response,
            "queries": num_of_queries,
            "uncertainty": uncertainty,
            "constraint_margin": constraint_margin,
            "queried_idx": queried_idx,
            "best_feasible_hist": best_feasible_hist,
            "batch_history": batch_history,
        },
        file_path,
    )


def run_trial(args, tri, log_dir, input_filename):
    seed = int(args.seed_offset + tri)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ClassicFuselageEnv(ip=args.ip, obs_noise=args.noise_level**2, min_shots=args.min_shots, eps_max=args.eps_max)
    tsai_wu_model = load(args.constraint_model)


    env.reset(input_filename)


    search_space = build_discrete_search_space(
        device=device,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        grid_points=args.grid_points,
    )
    n_candidates = int(search_space.shape[0])

    # Compute safety margin on full grid (same as classic_safeset_discrete.py)
    grid_np = search_space.detach().cpu().numpy().reshape(-1, 18)
    FI_all = tsai_wu_model.predict(grid_np, return_std=False)
    safety_margin = torch.tensor(1.0 - FI_all, dtype=torch.float64, device=device)
    safe_idx_grid = torch.where(safety_margin >= 0.0)[0]
    print(f"[Init] {len(safe_idx_grid)}/{n_candidates} grid points are safe")

    init_pool_size = int(args.initial_num_points)

    # Seed-dependent random init from safe grid points (different per trial)
    safe_mask = safety_margin >= 0.0
    safe_indices = torch.where(safe_mask)[0]
    safe_margins = safety_margin[safe_indices]
    # Weight by safety margin so safer points are more likely
    weights = safe_margins.clamp_min(0.0).cpu().numpy().astype(float)
    weights /= weights.sum() + 1e-12
    rng = np.random.RandomState(seed)
    chosen = rng.choice(safe_indices.cpu().numpy(), size=min(init_pool_size, len(safe_indices)),
                        replace=False, p=weights)
    init_idx_tensor = torch.tensor(chosen, dtype=torch.long, device=device)

    print(f"[Init] Selected {init_idx_tensor.numel()} safe warmup point(s) (seed={seed}):")
    for line in format_safe_initial_points(search_space, safety_margin, init_idx_tensor):
        print(line)

    init_idx = init_idx_tensor.tolist()
    queried_idx = init_idx.copy()

    train_X_list = []
    train_Y_list = []
    true_Y_list = []
    train_C_list = []
    query_list = []
    eps_list = []
    batch_history = []
    train_noise_list = []

    init_actions = search_space[init_idx].to(device)  # [n_init, 18]
    env.reset(input_filename, seed=seed)
    init_response, init_true_response, init_queries, init_obs_var = evaluate_initial_points(
        env, init_actions, device=device, obs_precision=args.eps_max
    )

    for i in range(len(init_idx)):
        x = init_actions[i:i+1]
        c_margin = eval_tsai_wu(tsai_wu_model, x)

        train_X_list.append(x)
        train_Y_list.append(init_response[i:i+1])
        true_Y_list.append(init_true_response[i:i+1])
        train_C_list.append(
            torch.tensor([[c_margin]], dtype=torch.float64, device=device)
        )
        query_list.append(init_queries[i:i+1])
        eps_list.append(args.eps_max)
        train_noise_list.append(float(init_obs_var[i].item()))

    train_X = torch.cat(train_X_list, dim=0)
    train_Y = torch.cat(train_Y_list, dim=0)
    true_Y = torch.cat(true_Y_list, dim=0)
    train_C = torch.cat(train_C_list, dim=0)
    num_of_queries = torch.cat(query_list, dim=0)
    total_queries = 0.0

    # Separate lists for ACTIVE (post-init) queried points only.
    active_X_list      = []
    active_Y_list      = []
    active_true_Y_list = []
    active_C_list      = []
    active_q_list      = []
    active_eps_list    = []

    def _cat(lst):
        return torch.cat(lst, dim=0) if lst else torch.empty(0)

    active_best_feasible_hist = []

    c_ls = torch.full((1, 18), 0.6931, dtype=torch.float64, device=device)
    c_ls[0, 0]  = 0.50
    c_ls[0, 17] = 0.50
    c_os = torch.tensor(1.0, dtype=torch.float64, device=device)
    c_model, _ = initialize_c_model(
        actions=train_X, response=train_C,
        device=device, ls_init=c_ls, outputscale=c_os,
    )
    c_X = train_X.clone()
    c_Y = train_C.clone()
    print(f"[Constraint GP] Fixed LS: active={c_ls[0,0].item():.2f}, inactive={c_ls[0,1].item():.4f}")

    # Create initial objective GP with fixed hyperparameters (no MLL fitting)
    obj_ls = torch.full((1, 18), 0.6931, dtype=torch.float64, device=device)
    obj_ls[0, 0]  = args.obj_ls   # active dim
    obj_ls[0, 17] = args.obj_ls   # active dim
    obj_covar = ScaleKernel(RBFKernel(ard_num_dims=18)).to(device=device, dtype=torch.float64)
    with torch.no_grad():
        obj_covar.base_kernel.lengthscale = obj_ls
        obj_covar.outputscale.fill_(1.0)
    noise_tensor_init = torch.tensor(train_noise_list, dtype=torch.float64, device=device)
    yvar_init = noise_tensor_init.unsqueeze(1).clamp_min(1e-6)
    f_model = _build_gp(train_X, train_Y, yvar_init,
                         covar_module=copy.deepcopy(obj_covar)).to(device)
    f_model.eval()
    f_model.likelihood.eval()
    f_model_m = None
    print(f"[Objective GP] Fixed LS: active={obj_ls[0,0].item():.2f}, inactive={obj_ls[0,1].item():.4f}")

    stage = 0
    print(f"Initial total queries: {total_queries}")
    while total_queries < args.query_budget:
        alpha_t = args.alpha_0 * (args.alpha_decay**stage)
        uncertainty_weight = max(0.0, 1.0 - alpha_t - args.beta_weight)

        # 1. Update Constraint GP & Compute Safe Mask
        c_model, _ = initialize_c_model(c_X, c_Y, device=device, ls_init=c_ls, outputscale=c_os)
        mu_c, sig_c, lcb_c, _ = compute_lcb_c(c_model, search_space, beta_c=3.0)
        predicted_safe = lcb_c >= 0.0
        safe_score = sig_c          # constraint uncertainty: peaks near the decision boundary
        p_feasible = mu_c           # used for fallback selection

        # 2. Train Objective GP (with Pseudo-Labeling Trick)
        feasible_mask = train_C.view(-1) >= 0.0
        X_m = train_X[feasible_mask]
        Y_m = train_Y[feasible_mask]
        X_u = train_X[~feasible_mask]

        # Build per-point noise tensor
        noise_tensor = torch.tensor(train_noise_list, dtype=torch.float64, device=device)
        noise_m = noise_tensor[feasible_mask]

        if len(X_m) == 0:
            f_model = fit_or_update_gp_model(f_model, train_X, train_Y, noise_var=noise_tensor)
            f_model_m = None
        else:
            f_model_m = fit_or_update_gp_model(f_model_m, X_m, Y_m, noise_var=noise_m)
            if len(X_u) > 0:
                # Predict pseudo-labels for failed states
                mu_u, _ = get_gp_predictions(f_model_m, X_u)
                X_aug = torch.cat([X_m, X_u], dim=0)
                Y_aug = torch.cat([Y_m, mu_u.view(-1, 1)], dim=0)
                # Pseudo-labels are uncertain: use conservative (eps_max / 1.96)²
                noise_u = torch.full((len(X_u),), (args.eps_max / 1.96)**2,
                                     dtype=torch.float64, device=device)
                noise_aug = torch.cat([noise_m, noise_u], dim=0)
                f_model = fit_or_update_gp_model(f_model, X_aug, Y_aug, noise_var=noise_aug)
            else:
                f_model = f_model_m

        # 3. Part 1: Exploitation (Select First Point)
        mu_f, sigma_f = get_gp_predictions(f_model, search_space)
        acquisition_raw = mu_f + math.sqrt(args.beta_f) * sigma_f  # Upper Confidence Bound

        if args.allow_repeated_queries:
            available_mask = torch.ones(n_candidates, dtype=torch.bool, device=device)
        else:
            available_mask = torch.ones(n_candidates, dtype=torch.bool, device=device)
            available_mask[queried_idx] = False

        if not available_mask.any():
            break

        feasible_objective_mask = predicted_safe & available_mask

        # Fallback if no safe point is available
        if feasible_objective_mask.sum() == 0:
            fallback_scores = p_feasible.clone()
            fallback_scores[~available_mask] = -float("inf")
            idx_star = int(torch.argmax(fallback_scores).item())
        else:
            masked_acq = acquisition_raw.clone()
            masked_acq[~feasible_objective_mask] = -float("inf")
            idx_star = int(torch.argmax(masked_acq).item())

        batch_indices = [idx_star]

        # 4. Part 2: Active Exploration (Select Remaining Points in Batch)
        unlabeled_mask = available_mask.clone()
        if args.allow_repeated_queries:
            unlabeled_mask[batch_indices] = False
        else:
            unlabeled_mask[queried_idx + batch_indices] = False
        
        U_s_indices = torch.where(unlabeled_mask)[0].tolist()

        for k in range(max(0, args.batch_size - 1)):
            if len(U_s_indices) == 0: 
                break
            
            U_s = search_space[U_s_indices]
            
            # 1. Uncertainty Component
            S_u_bar = norm_01(safe_score[U_s_indices])
            
            # 2. Representativeness Component
            if len(U_s_indices) == 1:
                S_r_bar = torch.ones_like(S_u_bar)
            else:
                dist_matrix = torch.cdist(U_s, U_s)
                S_r_raw = dist_matrix.sum(dim=1) / max(1, len(U_s) - 1)
                S_r_bar = 1.0 - norm_01(S_r_raw) 
                
            # 3. Diversity Component
            selected_X = search_space[queried_idx + batch_indices]
            S_d_raw, _ = torch.cdist(U_s, selected_X).min(dim=1)
            S_d_bar = norm_01(S_d_raw)
            
            # Combine Q score and pick max
            Q = (uncertainty_weight * S_u_bar 
                 + alpha_t * S_r_bar 
                 + args.beta_weight * S_d_bar)
            
            local_idx = int(torch.argmax(Q).item())
            batch_indices.append(U_s_indices[local_idx])
            U_s_indices.pop(local_idx)

        batch_history.append(batch_indices.copy())

        # 5. Execution: Query Environment for entire batch
        for next_idx in batch_indices:
            x = search_space[next_idx].view(1, -1)
            eps = args.eps_max

            env.reset(input_filename, seed=seed)
            y_obs, y_true, q = env.step_surrogate(
                action=x,
                device=device,
                eps=args.eta * eps,
                method=args.method,
            )
            c_margin = eval_tsai_wu(tsai_wu_model, x)
            c_X = torch.cat([c_X, x], dim=0)
            c_Y = torch.cat([c_Y, torch.tensor([[c_margin]], dtype=torch.float64, device=device)], dim=0)
            train_X = torch.cat([train_X, x], dim=0)
            train_Y = torch.cat([train_Y, y_obs.to(dtype=torch.float64)], dim=0)
            true_Y = torch.cat([true_Y, y_true.to(dtype=torch.float64)], dim=0)
            train_C = torch.cat(
                [
                    train_C,
                    torch.tensor([[c_margin]], dtype=torch.float64, device=device),
                ],
                dim=0,
            )
            num_of_queries = torch.cat([num_of_queries, q.to(dtype=torch.float64)], dim=0)
            queried_idx.append(next_idx)
            eps_list.append(eps)
            train_noise_list.append((args.eta * eps / 1.96)**2)
            total_queries += float(q.item())

            # Accumulate active-only records (no init points mixed in)
            active_X_list.append(x)
            active_Y_list.append(y_obs.to(dtype=torch.float64))
            active_true_Y_list.append(y_true.to(dtype=torch.float64))
            active_C_list.append(torch.tensor([[c_margin]], dtype=torch.float64, device=device))
            active_q_list.append(q.to(dtype=torch.float64))
            active_eps_list.append(eps)

            best_val = best_feasible_value(
                _cat(active_Y_list), _cat(active_C_list)
            )
            active_best_feasible_hist.append(best_val)

            # Compute f(x*): best true MAE among feasible points
            feasible_mask_all = train_C.view(-1) >= 0.0
            if feasible_mask_all.any():
                best_true_mae = float(true_Y[feasible_mask_all].min().item())
            else:
                best_true_mae = float('nan')

            print(
                "Stage {stage:03d} | idx={idx:04d} | x={x} | y={y:.4f} | MAE={mae:.4f} | "
                "c_margin={c_margin:.4f} | p_safe={p_safe:.3f} | eps={eps:.4f} | f(x*)={best_true:.4f}".format(
                    stage=stage + 1,
                    idx=next_idx,
                    x=np.round(x[0, [0, 17]].detach().cpu().numpy(), 4),
                    y=float(y_obs.item()),
                    mae=float(y_true.item()),
                    c_margin=float(c_margin),
                    p_safe=float(p_feasible[next_idx].item()),
                    eps=eps,
                    best_true=best_true_mae,
                )
            )

            if total_queries >= args.query_budget:
                break

        save_data(
            actions=_cat(active_X_list),
            response=_cat(active_Y_list),
            true_response=_cat(active_true_Y_list),
            num_of_queries=_cat(active_q_list),
            uncertainty=active_eps_list,
            constraint_margin=_cat(active_C_list),
            queried_idx=queried_idx[init_pool_size:],   # active indices only
            best_feasible_hist=active_best_feasible_hist,
            batch_history=batch_history,
            file_path=path.join(
                log_dir,
                f"{args.noise_level ** 2}acl_training_data_{tri}_.pth",
            ),
        )

        stage += 1

    return {
        "seed": seed,
        "n_init": init_pool_size,
        "actions": _cat(active_X_list),
        "response": _cat(active_Y_list),
        "true_response": _cat(active_true_Y_list),
        "constraint_margin": _cat(active_C_list),
        "queries": _cat(active_q_list),
        "uncertainty": active_eps_list,
        "best_feasible_hist": active_best_feasible_hist,
        "batch_history": batch_history,
        "queried_idx": queried_idx[init_pool_size:],
    }


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare-safe-methods style ACL transferred to the discrete fuselage case."
    )
    # ── Aligned with classic_safeset_discrete.py defaults ──
    parser.add_argument("--noise_level", type=float, default=0.2,
                        help="Noise std-dev (squared internally → obs_noise).")
    parser.add_argument("--query_budget", type=int, default=20000,
                        help="Oracle query budget.")
    parser.add_argument("--eps_max", type=float, default=0.04,
                        help="Maximum per-stage epsilon before eta scaling.")
    parser.add_argument("--min_shots", type=int, default=20,
                        help="Minimum shots for Monte Carlo Estimation.")
    parser.add_argument(
        "--initial_num_points",
        "--n_constraint_init",
        dest="initial_num_points",
        type=int,
        default=5,
        help="Number of safe warmup points used for both objective and constraint initialization.",
    )
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Number of points per ACL batch.")
    parser.add_argument("--grid_min", type=float, default=-1.0)
    parser.add_argument("--grid_max", type=float, default=1.0)
    parser.add_argument("--grid_points", type=int, default=21,
                        help="Grid resolution per active dim (21 → 441 points).")
    parser.add_argument("--beta_f", type=float, default=2.0)
    parser.add_argument("--obj_ls", type=float, default=0.20,
                        help="Objective lengthscale for active dimensions.")
    parser.add_argument("--alpha_0", type=float, default=0.7)
    parser.add_argument("--beta_weight", type=float, default=0.2)
    parser.add_argument("--alpha_decay", type=float, default=0.8)
    parser.add_argument("--feasible_threshold", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--method", type=str, default="non_monte_carlo")
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--ip", type=str, default="129.161.91.97")
    parser.add_argument("--constraint_model", type=str, default=os.path.join(SCRIPT_DIR, "surrogate_tsaiwu.joblib"))
    parser.add_argument(
        "--input_file",
        type=str,
        default=os.path.join(SCRIPT_DIR, "FuselageActuators", "AnsysFiles", "Test", "SolutionInputDP52.inp"),
    )
    parser.add_argument("--log_root", type=str, default=os.path.join(SCRIPT_DIR, "Experiments_constraints"))
    parser.add_argument("--env_name", type=str, default="Classic_ACL_Discrete")
    parser.set_defaults(allow_repeated_queries=True)
    parser.add_argument("--allow_repeated_queries", dest="allow_repeated_queries", action="store_true")
    parser.add_argument("--no_allow_repeated_queries", dest="allow_repeated_queries", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Classic Discrete ACL, 2 actuators")
    print("noise_level:", args.noise_level**2)
    print("eps_max:", args.eps_max)
    print("min_shots:", args.min_shots)
    print("batch_size:", args.batch_size)
    print("grid_points:", args.grid_points)
    print("beta_f:", args.beta_f)
    print("alpha_0:", args.alpha_0)
    print("beta_weight:", args.beta_weight)
    print("alpha_decay:", args.alpha_decay)
    print("Initial shape from", path.basename(args.input_file).split(".")[0])

    log_dir = path.join(args.log_root, args.env_name, "exp_set_1")
    os.makedirs(log_dir, exist_ok=True)

    input_filename = path.join(log_dir, path.basename(args.input_file))
    copyfile(args.input_file, input_filename)

    for tri in range(args.n_trials):
        run_trial(args, tri, log_dir, input_filename)
