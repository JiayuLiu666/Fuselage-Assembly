"""
Quantum Constrained Bayesian Optimization — Multi-GPU (Continuous)
===================================================================
Parallelism model (matches classic_safeset_continuous.py):
  - ProcessPoolExecutor launches one subprocess per inner trial.
  - Each trial selects its own GPU via  tri % device_count  (no shared state).
  - Single-GPU W-GP-UCB scoring inside each trial avoids cross-trial contention.

Algorithm:
  - Same LHS warmup + vectorised MAE, with noise-free warmup observations
  - Constraint GP: fit_c_model once on warmup+init, fantasy-model O(1) updates
  - initialize_f_model on actions[:data_count] (warmup + init_action)
  - Full Phi/V_t over all warmup+init data
  - Multi-start gradient-descent refinement with constraint penalty
  - lambda_by_stage(lam0=1, t0=400, p=2.0)
  - save_data(actions, true_response, response, queries, eps, error_init, path)
  - Saves actions[warmup_n:data_count] (no warmup data on disk)
"""
import argparse
import concurrent.futures
import math
import os
from os import path
import random
from shutil import copyfile
import warnings

import botorch.settings as botorch_settings
import numpy as np
import torch
from scipy.stats import qmc

from botorch.models import FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import RFFKernel, ScaleKernel

def minmax_norm(a: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12):
    v = a[mask]
    if v.numel() == 0:
        return a * 0.0
    lo = v.min()
    hi = v.max()
    return (a - lo) / (hi - lo + eps)

def lambda_by_stage(stage, lam0=0.8, t0=10, p=2.0):
    return float(lam0 * (t0 / (t0 + max(1, stage))) ** p)

def save_data(actions, true_response, response, num_of_queries, eps, error_init, file_path):
    torch.save({
        'actions': actions,
        'true_response': true_response,
        'response': response,
        'queries': num_of_queries,
        'uncertainty': eps,
        'error_init': error_init,
    }, file_path)

def effective_noise_variance(obs_noise, eps_max):
    chebyshev_samples = (obs_noise / eps_max**2) * (1/0.05)
    effective_variance = obs_noise / chebyshev_samples
    return effective_variance

def scaled_effective_var(obs_noise, eps_max, eta, error_init):
    effective_var = effective_noise_variance(
        obs_noise=obs_noise,
        eps_max=eps_max * eta,
    )
    return float(effective_var) / (float(error_init) ** 2)

def initialize_c_model(actions, response, device):
    yvar = torch.full_like(response, 1e-6, device=device)
    model = FixedNoiseGP(actions, response, yvar).to(device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll

def fit_c_model(actions, response, device):
    model, mll = initialize_c_model(actions=actions, response=response, device=device)
    fit_gpytorch_mll(mll)
    model.eval()
    model.likelihood.eval()
    return model
    
def initialize_f_model(actions, response, device, M_target, ls_init=None, obs_noise=1e-6):
    noise_tensor = torch.as_tensor(obs_noise, dtype=response.dtype, device=device)
    if noise_tensor.ndim == 0:
        yvar = torch.full_like(response, float(noise_tensor.item()), device=device)
    else:
        yvar = noise_tensor.reshape(-1, 1) if noise_tensor.ndim == 1 else noise_tensor
        yvar = yvar.expand_as(response).clone()
    
    input_dim = actions.shape[-1]
    ls_init = ls_init.to(device) if ls_init is not None else None
    
    base_kernel = RFFKernel(
        num_samples=M_target,
        num_dims=input_dim,
        ard_num_dims=input_dim,
    )
    covar_module = ScaleKernel(base_kernel).to(device)
    model = FixedNoiseGP(
        actions, response, yvar,
        covar_module=covar_module).to(device)
    with torch.no_grad():
        if ls_init is not None:
            model.covar_module.base_kernel.lengthscale.copy_(ls_init)
        # Keep objective outputscale fixed; exploration scaling is handled by beta_t.
        model.covar_module.outputscale.copy_(torch.tensor(1.0, dtype=torch.double, device=device))
    model.eval()
    return model, model.covar_module

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

def sample_sobol_in_active_subspace(
    n,
    active_idx=(0, 1, 2, 3, 14, 15, 16, 17),
    d=18,
    low=-0.5,
    high=0.5,
    seed=0,
    device=None,
    dtype=torch.float64,
):
    device = device or torch.device("cpu")
    active_idx = tuple(active_idx)
    seed = int(seed)
    sobol = torch.quasirandom.SobolEngine(
        dimension=len(active_idx), scramble=True, seed=seed
    )
    u = sobol.draw(n).to(device=device, dtype=dtype)
    active_vals = low + (high - low) * u
    out = torch.zeros((n, d), device=device, dtype=dtype)
    out[:, list(active_idx)] = active_vals
    return out

def sample_safe_sobol_in_active_subspace(
    n, tsai_wu_model, active_idx=(0, 1, 2, 3, 14, 15, 16, 17), d=18,
    low=-0.5, high=0.5, seed=0, device=None, dtype=torch.float64,
    pool_size=8192, max_attempts=8,
):
    device = device or torch.device("cpu")
    safe_batches = []

    for attempt in range(int(max_attempts)):
        candidates = sample_sobol_in_active_subspace(
            n=max(int(n), int(pool_size)),
            active_idx=active_idx, d=d, low=low, high=high,
            seed=int(seed) + 1009 * attempt, device=device, dtype=dtype,
        )
        candidates_np = candidates.detach().cpu().numpy().reshape(-1, d)
        try:
            failure_indices = tsai_wu_model.predict(candidates_np, return_std=False)
        except TypeError:
            failure_indices = tsai_wu_model.predict(candidates_np)

        safe_mask_np = (1.0 - np.asarray(failure_indices).reshape(-1)) >= 0.0
        safe_idx = np.flatnonzero(safe_mask_np)
        if safe_idx.size > 0:
            safe_batches.append(candidates[torch.as_tensor(safe_idx, device=device)])
            num_safe = sum(batch.shape[0] for batch in safe_batches)
            if num_safe >= int(n):
                return torch.cat(safe_batches, dim=0)[:int(n)]

    raise RuntimeError(
        f"Could not find {n} safe Sobol init action(s) after "
        f"{int(max_attempts) * int(pool_size)} candidates."
    )

def candidate_pool_size_by_stage(stage, base_size=1024, growth=256, max_size=4096, every=100):
    growth_steps = max(0, (int(stage) - 1) // max(1, every))
    return int(min(max_size, base_size + growth_steps * growth))
from utils import build_train_gp_with_rff, sample_actions
from quantum_bo_env_constraint import QuantumFuselageEnv

warnings.filterwarnings("ignore")

WARMUP_PRECISION = 0.04

ACTUATOR_INDICES_BY_COUNT = {
    2: (0, 17),
    4: (0, 1, 17, 16),
    6: (0, 1, 2, 17, 16, 15),
    8: (0, 1, 2, 3, 14, 15, 16, 17),
}


def active_indices_for_count(actuator_count):
    try:
        return ACTUATOR_INDICES_BY_COUNT[int(actuator_count)]
    except KeyError as exc:
        valid_counts = ", ".join(str(k) for k in sorted(ACTUATOR_INDICES_BY_COUNT))
        raise ValueError(
            f"Unsupported actuator count {actuator_count}. "
            f"Valid counts are: {valid_counts}."
        ) from exc


def default_results_root(actuator_count, force_scale=1000.0):
    if int(actuator_count) == 8:
        base = 'Experiments_constraint_continuous'
    else:
        base = f'Experiments_constraint_continuous_actuators_{int(actuator_count)}'
    if int(force_scale) != 1000:
        base += f'_force_{int(force_scale)}'
    return base

# ─────────────────────────── helpers ─────────────────────────────────────────

# def save_checkpoint(ckpt_path, **kwargs):
#     """Save a full checkpoint so the trial can be resumed after kill/restart."""
#     tmp_path = ckpt_path + ".tmp"
#     torch.save(kwargs, tmp_path)
#     os.replace(tmp_path, ckpt_path)   # atomic on POSIX
#     print(f"  [checkpoint] saved → {ckpt_path}")


def load_checkpoint(ckpt_path, device):
    """Load a checkpoint, moving tensors to *device*."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  [checkpoint] loaded ← {ckpt_path}")
    return ckpt


def sample_lhs_in_active_subspace(
    n,
    active_idx=(0, 1, 2, 3, 14, 15, 16, 17),
    d=18,
    low=-0.5,
    high=0.5,
    seed=0,
    device=None,
    dtype=torch.float64,
):
    device = device or torch.device("cpu")
    sampler = qmc.LatinHypercube(d=len(active_idx), optimization="random-cd", seed=int(seed))
    lhs_unit = sampler.random(n=int(n))
    active_vals = low + (high - low) * lhs_unit

    out = np.zeros((int(n), d), dtype=np.float64)
    out[:, list(active_idx)] = active_vals
    return torch.tensor(out, device=device, dtype=dtype)


def compute_lcb_c_botorch_grad(c_model, X, beta_c: float):
    """With-gradient version for use inside optimisation loop."""
    post = c_model.posterior(X)
    mu = post.mean.view(-1)
    var = post.variance.view(-1).clamp_min(1e-12)
    sigma = var.sqrt()
    rad = math.sqrt(beta_c)
    lcb = mu - rad * sigma
    ucb = mu + rad * sigma
    return mu, sigma, lcb, ucb


def add_constraint_fantasy(c_model, new_actions, c_val, device):
    """Append one constraint observation without refitting hyperparameters."""
    c_model.eval()
    if hasattr(c_model, "likelihood"):
        c_model.likelihood.eval()

    with torch.no_grad():
        # ExactGP fantasy updates require prediction caches from a prior eval call.
        _ = c_model.posterior(new_actions)
        c_fantasy_noise = torch.full(
            c_val.view(-1).shape, 1e-6, device=device, dtype=torch.float64)
        c_model = c_model.get_fantasy_model(
            new_actions, c_val.view(-1), noise=c_fantasy_noise)
        c_model.eval()
        if hasattr(c_model, "likelihood"):
            c_model.likelihood.eval()
    return c_model


def objective_noise_variance(response, n_warm_up, eps_list=None, eta=1, effective_var=None):
    """Per-point GP noise: adaptive yvar for warmup+init, (η·ε)² for active BO."""
    if eps_list is None:
        eps_list = [WARMUP_PRECISION] * response.shape[0]
    eps_t = torch.as_tensor(eps_list, dtype=response.dtype, device=response.device)
    yvar = ((eta * eps_t) ** 2).view(-1, 1)
    yvar[:n_warm_up] = effective_var
    return yvar


def stage_epsilon(var_value, lam, eps_max):
    """epsilon_t = min(epsilon_max, sqrt(var(x) / lambda))."""
    return min(float(eps_max), math.sqrt(max(float(var_value), 1e-12) / float(lam)))

# ─────────────────────────── scoring ─────────────────────────────────────────

@torch.no_grad()
def W_GP_UCB_scores(model, x_D, response, V_t, Unweighted_Phi,
                    eps_list, beta, lam, chunk_size=4096):
    """
    Memory-safe W-GP-UCB scoring.

    Avoids constructing dense W = diag(1 / eps^2).
    Uses Cholesky solve instead of explicit inverse.
    """

    device = V_t.device
    dtype  = V_t.dtype

    # eps: [n]
    eps_tensor = torch.as_tensor(eps_list, device=device, dtype=dtype).view(-1)

    # 防止 eps 太小导致权重爆炸
    eps_tensor = eps_tensor.clamp_min(1e-12)

    # weights: [n]
    weights = 1.0 / eps_tensor.square()

    # Phi: [n, feature_dim]
    Phi = Unweighted_Phi.to(device=device, dtype=dtype)

    # y: [n, 1]
    y = response.reshape(-1, 1).to(device=device, dtype=dtype)

    # Phi^T W y, but without explicitly forming W
    # [feature_dim, n] @ [n, 1] -> [feature_dim, 1]
    PhiT_W_y = Phi.T @ (weights[:, None] * y)

    # Cholesky factorization of V_t
    # V_t = Phi^T W Phi + lambda I
    L = torch.linalg.cholesky(V_t)

    # nu_t = V_t^{-1} Phi^T W y
    nu_t = torch.cholesky_solve(PhiT_W_y, L)

    beta_value = float(beta.item()) if isinstance(beta, torch.Tensor) else float(beta)
    beta_sqrt  = math.sqrt(max(beta_value, 0.0))

    mean_chunks, var_chunks = [], []

    for start in range(0, x_D.shape[0], chunk_size):
        x_chunk = x_D[start: start + chunk_size]

        # features: [chunk, feature_dim]
        features = model.covar_module.base_kernel.get_features(
            x_chunk, x_chunk.shape[-1], normalize=True
        ).to(device=device, dtype=dtype)

        # mean = phi(x)^T nu_t
        mean_chunk = (features @ nu_t).reshape(-1)

        # var = lambda * phi(x)^T V_t^{-1} phi(x)
        # solve V_t^{-1} features^T by Cholesky
        solved = torch.cholesky_solve(features.T, L).T

        var_chunk = (
            lam * torch.sum(solved * features, dim=1)
        ).clamp_min(1e-12)

        mean_chunks.append(mean_chunk)
        var_chunks.append(var_chunk)

    mean  = torch.cat(mean_chunks)
    var   = torch.cat(var_chunks)
    sigma = var.sqrt()
    ucb   = mean + beta_sqrt * sigma

    return ucb, mean, sigma, var, L, nu_t, beta_sqrt

# ─────────────────────────── trial ───────────────────────────────────────────

def run_trial(tri, seed, args, input_filename, target_filename,
              log_dir, active_idx, obs_noise, eta, lam, Big_B, min_shots):
    # ── Assign GPU (matches classic_safeset_continuous.py) ────────────────
    gpu_id = tri % torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    ckpt_path = path.join(
        log_dir,
        f"{obs_noise}quantum_safeset_ckpt_{tri}_.pth")

    try:
        print(f"[Trial {tri}] -> {device}")
        torch.cuda.empty_cache()
        torch.manual_seed(tri)
        np.random.seed(12345 + seed * 100 + tri)

        M_target = args.M_target
        n_iter   = args.max_iteration
        beta_c   = 3.0

        env = QuantumFuselageEnv(ip="129.161.91.97", obs_noise=obs_noise, min_shots=min_shots, force_scale=args.force_scale)

        # ── LHS warmup (same seed as before) ──────────────────────────────
        warmup_actions = sample_lhs_in_active_subspace(
            n=args.warmup_points, active_idx=active_idx,
            d=18, low=-0.5, high=0.5, seed=10086 + tri,
            device=torch.device("cpu"), dtype=torch.float64,
        )

        env.reset(input_filename, seed=None, target_npy=target_filename)
        warmup_np     = warmup_actions.numpy()
        forces_batch  = (warmup_np * args.force_scale).astype(np.float64)
        u_batch       = forces_batch @ env.surrogate.T
        p_init_flat   = env.initPos[:, 0:2].flatten()
        p_target_flat = env.targetPos[:, 0:2].flatten()
        dev_batch     = (p_init_flat[None, :] + u_batch) - p_target_flat[None, :]
        n_dev         = dev_batch.shape[1]
        mae_batch     = np.abs(dev_batch).sum(axis=1) / n_dev
                
        effective_var = effective_noise_variance(
            obs_noise=obs_noise,
            eps_max=args.eps_max * eta,
        )
        scaled_var = scaled_effective_var(
            obs_noise=obs_noise,
            eps_max=args.eps_max,
            eta=eta,
            error_init=env.error_init,
        )

        # Warmup observations with Gaussian noise: N(0, effective_var)
        noise = np.random.normal(
            loc=0.0,
            scale=np.sqrt(effective_var),
            size=mae_batch.shape,
        )
        y_obs_batch = (-mae_batch.copy() + noise)/ env.error_init

        response_tensor      = torch.tensor(y_obs_batch, dtype=torch.double).to(device)
        true_response_tensor = torch.tensor(mae_batch,   dtype=torch.double)

        # Constraint warmup: vectorised over same LHS points
        FI_batch          = env.tsai_wu.predict(warmup_np, return_std=False)
        constraint_response = (1 - FI_batch).tolist()


        # Only LS is reused here; outputscale is intentionally fixed to 1.0.
        _, lengthscale, _ = build_train_gp_with_rff(
            actions=warmup_actions, response=response_tensor,
            known_noise=scaled_var, max_retries=4,
        )
        env.reset(input_filename, seed=seed, target_npy=target_filename)

        # ── Pre-allocated tensors ─────────────────────────────────────────
        c_X = warmup_actions.to(device=device, dtype=torch.double)
        c_Y = torch.tensor(constraint_response, dtype=torch.double,
                           device=device).reshape(-1, 1)

        total_capacity = n_iter + args.warmup_points + 10
        actions       = torch.zeros((total_capacity, 18), dtype=torch.float64, device=device)
        response      = torch.zeros((total_capacity, 1),  dtype=torch.float64, device=device)
        true_response = torch.zeros((total_capacity, 1),  dtype=torch.float64, device=device)
        stages        = torch.zeros((total_capacity, 1),  dtype=torch.float64, device=device)

        warmup_X      = warmup_actions.to(device=device, dtype=torch.float64)
        warmup_Y      = response_tensor.view(-1, 1).to(device, dtype=torch.float64)
        warmup_true_Y = true_response_tensor.view(-1, 1).to(device, dtype=torch.float64)
        warmup_n      = warmup_X.shape[0]

        actions[:warmup_n]       = warmup_X
        response[:warmup_n]      = warmup_Y
        true_response[:warmup_n] = warmup_true_Y
        stages[:warmup_n]        = 0.0

        # ── Constraint & Objective models built from warmup data ──────────
        c_model = fit_c_model(c_X, c_Y, device=device)
        print(f"[Trial {tri}] Constraint GP fitted and frozen on warmup prior to init.")

        eps_list = [args.eps_max] * warmup_n
        objective_yvar = objective_noise_variance(
            response[:warmup_n], warmup_n, eps_list[:warmup_n], eta, effective_var=scaled_var)
        f_model, _ = initialize_f_model(
            actions[:warmup_n], response[:warmup_n],
            device, M_target,
            ls_init=lengthscale,
            obs_noise=objective_yvar,
        )
        ts     = torch.arange(1, max(n_iter, 2), device=device, dtype=torch.float64)
        beta_t = (1 + Big_B * torch.sqrt(torch.log(ts) ** 2)) ** 2
        f_model.eval()

        # ── Random initial point (noisy) ─────────────────────────────────
        env.reset(input_filename, seed=None, target_npy=target_filename)
        init_action = sample_safe_sobol_in_active_subspace(
            n=1, tsai_wu_model=env.tsai_wu, active_idx=active_idx,
            d=18, low=-0.5, high=0.5, seed=20086 + tri,
            device=device, dtype=torch.float64,
            pool_size=max(8192, args.candidate_pool_base),
        )
        init_response, init_true_response, num_of_queries, init_c_val, init_total_gp_variance = env.step_surrogate(
            init_action, eps=args.eps_max * eta, device=device)
        if init_c_val.item() < 0.0:
            raise RuntimeError(
                f"Initial action is outside the safe set: margin={init_c_val.item():.6g}"
            )

        num_of_queries_num = num_of_queries.item()
        data_count = warmup_n
        actions[data_count]       = init_action.squeeze(0)
        response[data_count]      = init_response.squeeze(0)
        true_response[data_count] = init_true_response.squeeze(0)
        stages[data_count]        = num_of_queries.squeeze(0)
        eps_list.append(args.eps_max)  # Placeholder; will be updated in the loop
        data_count += 1

        # Constraint GP: update with fantasy model for the zero-action point
        c_X = torch.cat([c_X, init_action.to(c_X)], dim=0)
        c_Y = torch.cat([c_Y, init_c_val.to(c_Y)], dim=0)
        c_model = add_constraint_fantasy(c_model, init_action, init_c_val, device)

        feature_dim = f_model.covar_module.base_kernel.get_features(
            actions[:1], actions.shape[-1], normalize=True
        ).shape[-1]
        Unweighted_Phi = torch.zeros((total_capacity, feature_dim),
                                     device=device, dtype=actions.dtype)
        Weighted_Phi   = torch.zeros((total_capacity, feature_dim),
                                     device=device, dtype=actions.dtype)

        warm_features = f_model.covar_module.base_kernel.get_features(
            actions[:data_count], actions.shape[-1], normalize=True)
        Unweighted_Phi[:data_count] = warm_features
        Weighted_Phi[:data_count]   = warm_features * (1.0 / args.eps_max)

        V_t = (torch.matmul(Weighted_Phi[:data_count].T, Weighted_Phi[:data_count])
               + lam * torch.eye(feature_dim, device=device, dtype=actions.dtype))

        error_init, _ = env.reset(input_filename, seed=None, target_npy=target_filename)

        stage_count         = 1
        safe_queries_count  = 0
        total_queries_count = 0

        # ── Resume from checkpoint if one exists ──────────────────────────
        # if path.isfile(ckpt_path):
        #     ckpt = load_checkpoint(ckpt_path, device)
        #     data_count          = ckpt['data_count']
        #     stage_count         = ckpt['stage_count']
        #     num_of_queries_num  = ckpt['num_of_queries_num']
        #     safe_queries_count  = ckpt['safe_queries_count']
        #     total_queries_count = ckpt['total_queries_count']
        #     eps_list            = ckpt['eps_list']
        #     actions[:data_count]       = ckpt['actions'].to(device)
        #     response[:data_count]      = ckpt['response'].to(device)
        #     true_response[:data_count] = ckpt['true_response'].to(device)
        #     stages[:data_count]        = ckpt['stages'].to(device)
        #     V_t                        = ckpt['V_t'].to(device)
        #     Unweighted_Phi[:data_count] = ckpt['Unweighted_Phi'].to(device)
        #     Weighted_Phi[:data_count]   = ckpt['Weighted_Phi'].to(device)
        #     c_X                        = ckpt['c_X'].to(device)
        #     c_Y                        = ckpt['c_Y'].to(device)
        #     # Rebuild objective GP from saved data
        #     objective_yvar = objective_noise_variance(
        #         response[:data_count], warmup_n, eps_list[:data_count], eta)
        #     f_model, _ = initialize_f_model(
        #         actions[:data_count], response[:data_count],
        #         device, M_target,
        #         ls_init=lengthscale,
        #         obs_noise=objective_yvar,
        #     )
        #     f_model.eval()
        #     # Rebuild constraint GP from saved constraint data
        #     c_model = fit_c_model(c_X, c_Y, device=device)
        #     print(f"[Trial {tri}] Resumed from checkpoint at stage {stage_count}, "
        #           f"queries {num_of_queries_num}/{n_iter}")

        # ── BO loop ───────────────────────────────────────────────────────
        while num_of_queries_num < n_iter:
            pool_size = candidate_pool_size_by_stage(
                stage_count,
                base_size=args.candidate_pool_base,
                growth=args.candidate_pool_growth,
                max_size=args.candidate_pool_max,
                every=args.candidate_pool_growth_interval,
            )
            candidate_seed    = 10086 + tri * 100000 + stage_count
            candidate_actions = sample_sobol_in_active_subspace(
                n=pool_size, active_idx=active_idx, d=18,
                low=-0.5, high=0.5, seed=candidate_seed,
                device=device, dtype=actions.dtype,
            )

            beta_idx = min(stage_count - 1, beta_t.numel() - 1)
            ucb_f, _, _, var_f, chol_V_t, nu_t, beta_sqrt = W_GP_UCB_scores(
                model=f_model, x_D=candidate_actions,
                response=response[:data_count], V_t=V_t,
                Unweighted_Phi=Unweighted_Phi[:data_count],
                eps_list=eps_list[:data_count], beta=beta_t[beta_idx],
                lam=lam, chunk_size=args.score_chunk_size,
            )

            mu_c, sig_c, lcb_c, _ = compute_lcb_c_botorch(c_model, candidate_actions, beta_c)
            safe_mask = lcb_c >= 0.0

            if safe_mask.sum().item() == 0:
                idx             = torch.argmax(lcb_c)
                selection_score = lcb_c
            else:
                a_bnd           = -torch.abs(lcb_c)
                ucb_n           = minmax_norm(ucb_f, mask=safe_mask)
                bnd_n           = minmax_norm(a_bnd, mask=safe_mask)
                lam_t           = lambda_by_stage(int(stage_count), lam0=1, t0=100, p=2.0)
                selection_score = (1.0 - lam_t) * ucb_n + lam_t * bnd_n
                selection_score[~safe_mask] = -1e18
                idx             = torch.argmax(selection_score)

            # ── Multi-start local refinement from top candidates ──────────
            if safe_mask.sum().item() > 0:
                
                safe_indices = torch.where(safe_mask)[0]
                top_k = min(10, safe_indices.numel())
                safe_scores = selection_score[safe_indices]
                top_local_indices = torch.topk(safe_scores, top_k).indices
                top_indices = safe_indices[top_local_indices]
                top_candidates = candidate_actions[top_indices].clone().detach()

                ucb_min, ucb_max = ucb_f[safe_mask].min().item(), ucb_f[safe_mask].max().item()
                bnd_min, bnd_max = a_bnd[safe_mask].min().item(), a_bnd[safe_mask].max().item()
                if ucb_max == ucb_min:
                    ucb_max += 1e-9
                if bnd_max == bnd_min:
                    bnd_max += 1e-9

                top_candidates.requires_grad_(True)
                optimizer = torch.optim.Adam([top_candidates], lr=0.01)

                for _ in range(50):
                    optimizer.zero_grad()
                    features  = f_model.covar_module.base_kernel.get_features(
                        top_candidates, top_candidates.shape[-1], normalize=True)
                    mean_part = (features @ nu_t).reshape(-1)
                    proj      = torch.cholesky_solve(features.T, chol_V_t).T
                    var_part  = (lam * torch.sum(proj * features, dim=1)).clamp_min(1e-12)
                    ucb_opt   = mean_part + beta_sqrt * var_part.sqrt()

                    mu_c_opt, sig_c_opt, lcb_c_opt, _ = compute_lcb_c_botorch_grad(
                        c_model, top_candidates, beta_c)
                    a_bnd_opt = -torch.abs(lcb_c_opt)

                    ucb_n_opt   = (ucb_opt   - ucb_min) / (ucb_max - ucb_min)
                    bnd_n_opt   = (a_bnd_opt - bnd_min) / (bnd_max - bnd_min)
                    score_opt   = (1.0 - lam_t) * ucb_n_opt + lam_t * bnd_n_opt
                    penalty = 1000.0 * torch.relu(-lcb_c_opt).pow(2)

                    (-(score_opt - penalty).sum()).backward()
                    optimizer.step()

                    with torch.no_grad():
                        top_candidates.clamp_(-0.5, 0.5)
                        inactive = [i for i in range(18) if i not in active_idx]
                        top_candidates[:, inactive] = 0.0

                with torch.no_grad():
                    _, _, final_lcb_c, _ = compute_lcb_c_botorch_grad(
                        c_model, top_candidates, beta_c)
                    final_safe = final_lcb_c >= 0.0

                    if final_safe.sum() > 0:

                        features = f_model.covar_module.base_kernel.get_features(

                            top_candidates, top_candidates.shape[-1], normalize=True

                        )

                        final_var = (
                            lam * torch.sum(
                                torch.cholesky_solve(features.T, chol_V_t).T * features,
                                dim=1
                            )
                        ).clamp_min(1e-12)

                        final_ucb = (features @ nu_t).reshape(-1) + beta_sqrt * final_var.sqrt()

                        mu_c_final, sig_c_final, final_lcb_c, _ = compute_lcb_c_botorch_grad(

                            c_model, top_candidates, beta_c

                        )

                        final_safe = final_lcb_c >= 0.0

                        final_a_bnd = -torch.abs(final_lcb_c)

                        final_ucb_n = (final_ucb - ucb_min) / (ucb_max - ucb_min)

                        final_bnd_n = (final_a_bnd - bnd_min) / (bnd_max - bnd_min)

                        final_score = (1.0 - lam_t) * final_ucb_n + lam_t * final_bnd_n

                        final_score[~final_safe] = -1e18

                        best_idx = torch.argmax(final_score)

                        new_actions = top_candidates[best_idx].reshape(1, -1).detach()

                        eps = stage_epsilon(final_var[best_idx].item(), lam, args.eps_max)
                    else:
                        new_actions = candidate_actions[idx].reshape(1, -1).detach()
                        eps = stage_epsilon(var_f[idx].item(), lam, args.eps_max)
            else:
                new_actions = candidate_actions[idx].reshape(1, -1).detach()
                eps = stage_epsilon(var_f[idx].item(), lam, args.eps_max)

            # ── Evaluate ──────────────────────────────────────────────────
            env.reset(input_filename, seed=None, target_npy=target_filename)
            new_response, true_responses, num_oracle_queries, c_val, CI_estimation = env.step_surrogate(
                new_actions, eps=eps * eta, device=device)

            total_queries_count += 1
            if c_val.item() >= 0:
                safe_queries_count += 1

            num_of_queries_num += num_oracle_queries.item()
            eps_list.append(eps)

            actions[data_count]       = new_actions.squeeze(0)
            response[data_count]      = new_response.squeeze(0)
            true_response[data_count] = true_responses.squeeze(0)
            stages[data_count]        = num_oracle_queries.squeeze(0)

            x_feat        = f_model.covar_module.base_kernel.get_features(
                new_actions, new_actions.shape[-1], normalize=True)
            V_t           = V_t + torch.matmul(
                (x_feat * (1.0 / eps_list[-1])).T,
                x_feat * (1.0 / eps_list[-1]))
            Unweighted_Phi[data_count] = x_feat.squeeze(0)
            data_count += 1

            # Fantasy update appends one constraint observation without refitting hyperparameters.
            c_X = torch.cat([c_X, new_actions.to(c_X)], dim=0)
            c_Y = torch.cat([c_Y, c_val.to(c_Y)], dim=0)
            c_model = add_constraint_fantasy(c_model, new_actions, c_val, device)

            # Logging every 100 stages
            if stage_count % 100 == 0:
                running_min    = true_response[warmup_n:data_count].min().item()
                safety_rate = safe_queries_count / max(1, total_queries_count)
                violation_rate = 1.0 - safety_rate
                print(
                    f"[Trial {tri}] Stage {stage_count} | "
                    f"y: {new_response[-1].item():.3f} "
                    f"f(x): {true_responses[-1].item():.3f} | "
                    f"eps: {eps:.4f} | safe: {int(safe_mask.sum())}/{pool_size} | "
                    f"running_min: {running_min:.4f} | safety rate: {safety_rate:.3f}"
                )

            # Periodic save (no warmup data)
            if stage_count % 100 == 0 or num_of_queries_num >= n_iter:
                save_data(
                    actions[warmup_n:data_count],
                    true_response[warmup_n:data_count],
                    response[warmup_n:data_count],
                    stages[warmup_n:data_count],
                    eps_list[warmup_n:data_count],
                    error_init,
                    path.join(log_dir,
                              str(obs_noise) + 'quan_training_data_' + str(tri) + '_.pth'),
                )
                # Full checkpoint for resume
                # save_checkpoint(
                #     ckpt_path,
                #     data_count=data_count,
                #     stage_count=stage_count,
                #     num_of_queries_num=num_of_queries_num,
                #     safe_queries_count=safe_queries_count,
                #     total_queries_count=total_queries_count,
                #     eps_list=eps_list,
                #     actions=actions[:data_count].cpu(),
                #     response=response[:data_count].cpu(),
                #     true_response=true_response[:data_count].cpu(),
                #     stages=stages[:data_count].cpu(),
                #     V_t=V_t.cpu(),
                #     Unweighted_Phi=Unweighted_Phi[:data_count].cpu(),
                #     Weighted_Phi=Weighted_Phi[:data_count].cpu(),
                #     c_X=c_X.cpu(),
                #     c_Y=c_Y.cpu(),
                #     error_init=error_init,
                # )

            error_init, _ = env.reset(input_filename, seed=None, target_npy=target_filename)
            stage_count += 1

        # Final summary
        final_min = true_response[warmup_n:data_count].min().item()
        print("=" * 60)
        print(f"Trial {tri} | stages: {stage_count} | final min f(x): {final_min:.6f}")
        print("=" * 60)
        return True

    except Exception as e:
        import traceback
        print(f"Error in Trial {tri}: {e}")
        traceback.print_exc()
        return False


# ─────────────────────────── main ────────────────────────────────────────────

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    torch.set_default_dtype(torch.float64)
    botorch_settings.debug._set_state(True)

    parser = argparse.ArgumentParser(
        description="Multi-GPU Quantum Continuous Constraint BO (ProcessPoolExecutor).")

    # ── Model ──────────────────────────────────────────────────────────────
    parser.add_argument("--M_target",    type=int,   default=256,
                        help="Number of RFF features.")
    parser.add_argument("--obs_noise",   type=float, default=0.1**2,
                        help="Observation noise variance (0.01 → σ=0.1, 0.04 → σ=0.2).")
    parser.add_argument("--min_shots",   type=int,   default=20,
                        help="Minimum shots for Monte Carlo Estimation.")
    parser.add_argument("--eta",         type=float, default=1.0,
                        help="Precision trade-off η.")
    parser.add_argument("--eps_max",     type=float, default=0.04,
                        help="Maximum per-stage epsilon before eta scaling.")
    parser.add_argument("--lam",         type=float, default=1.0,
                        help="Regularisation λ for V_t.")
    parser.add_argument("--B",           type=float, default=1.0,
                        help="Exploration coefficient B in β_t = (1 + B√log(t)²)².")

    # ── Budget ─────────────────────────────────────────────────────────────
    parser.add_argument("--max_iteration", type=int, default=50000,
                        help="Oracle query budget.")

    # ── Warmup ─────────────────────────────────────────────────────────────
    parser.add_argument("--warmup_points", type=int, default=200,
                        help="LHS warm-start points.")

    # ── Actuator subspace / output location ───────────────────────────────
    parser.add_argument("--actuator_count", type=int, default=6,
                        choices=sorted(ACTUATOR_INDICES_BY_COUNT),
                        help=("Number of active actuators. "
                              "6 maps to indices (0, 1, 2, 17, 16, 15)."))
    parser.add_argument("--force_scale", type=float, default=1000.0,
                        help=("Action-to-force scale in lb: action in (-1,1) maps to "
                              "(-force_scale, +force_scale). Default 1000 (-1000..1000 lb). "
                              "Use e.g. 200 or 500 for narrower force ranges."))
    parser.add_argument("--results_root", type=str, default=None,
                        help=("Root folder for saved results. Defaults to "
                              "Experiments_constraint_continuous_actuators_<count>, "
                              "except 8 uses Experiments_constraint_continuous. "
                              "A non-1000 --force_scale appends _force_<scale>."))

    # ── Candidate pool ─────────────────────────────────────────────────────
    parser.add_argument("--candidate_pool_base",     type=int, default=2048)
    parser.add_argument("--candidate_pool_growth",   type=int, default=2048)
    parser.add_argument("--candidate_pool_max",      type=int, default=2**16)
    parser.add_argument("--candidate_pool_growth_interval", type=int, default=10)
    parser.add_argument("--score_chunk_size", type=int, default=2048,
                        help="Candidates scored per chunk on GPU.")

    # ── Experiment grid ────────────────────────────────────────────────────
    parser.add_argument("--outer_seeds",  type=int, default=10,
                        help="Number of outer initial-shape seeds.")
    parser.add_argument("--inner_trials", type=int, default=5,
                        help="BO trials per outer seed (run in parallel, one per GPU).")

    args = parser.parse_args()
    if args.lam <= 0:
        raise ValueError("--lam must be positive.")
    if args.eps_max <= 0:
        raise ValueError("--eps_max must be positive.")

    active_idx = active_indices_for_count(args.actuator_count)
    results_root = args.results_root or default_results_root(args.actuator_count, args.force_scale)
    obs_noise  = args.obs_noise
    min_shots  = args.min_shots
    eta        = args.eta
    lam        = args.lam
    Big_B      = args.B
    print("Multi-GPU Quantum Continuous Constraint BO (ProcessPoolExecutor)")
    print("noise_level  :", obs_noise)
    print("min_shots    :", min_shots)
    print("eta          :", eta, "| eps_max:", args.eps_max, "| lam:", lam, "| B:", Big_B)
    print("warmup_points:", args.warmup_points)
    print("max_iteration:", args.max_iteration)
    print("candidate_pool_schedule:", {
        "base":     args.candidate_pool_base,
        "growth":   args.candidate_pool_growth,
        "max":      args.candidate_pool_max,
        "interval": args.candidate_pool_growth_interval,
    })
    print("actuator_count:", args.actuator_count, "| active_idx:", active_idx)
    print("results_root :", results_root)
    print(f"inner_trials: {args.inner_trials}  (one per GPU via tri % device_count)")

    Initial_shape_file = ['DP52', 'DP50', 'DP49', 'DP45', 'DP45',
                          'DP48', 'DP43', 'DP53', 'DP60', 'DP54']
    Target_shape_file  = ['DP53', 'DP53', 'DP57', 'DP53', 'DP58',
                          'DP53', 'DP53', 'DP55', 'DP44', 'DP53']

    for seed in range(args.outer_seeds):
        random.seed(seed)
        ansys_folder = path.join('FuselageActuators', 'AnsysFiles', 'Test')
        shape_folder = path.join('FuselageActuators', 'Shapes', 'Test')

        init_name = 'SolutionInput' + Initial_shape_file[seed]
        file      = init_name + '.inp'
        original_input_filename = path.join(ansys_folder, file)
        print("Initial shape:", init_name)

        env_name = "Quantum_EXP_{}_{}_{}_{}_multi_gpu_noise_{}".format(
            args.actuator_count, eta, lam, Big_B, obs_noise)
        log_dir = path.join(results_root, env_name, 'exp_set_' + str(seed))
        os.makedirs(log_dir, exist_ok=True)

        input_filename = path.join(log_dir, file)
        copyfile(original_input_filename, input_filename)

        target_name = 'SolutionInput' + Target_shape_file[seed]
        target_npy  = target_name + '.npy'
        original_target_filename = path.join(shape_folder, target_npy)
        target_filename = path.join(log_dir, target_npy)
        copyfile(original_target_filename, target_filename)

        # One subprocess per trial; each picks its own GPU
        active_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        safe_max_workers = min(args.inner_trials, active_gpu_count * 2)

        with concurrent.futures.ProcessPoolExecutor(max_workers=safe_max_workers) as executor:
            futures = [
                executor.submit(
                    run_trial, tri, seed, args,
                    input_filename, target_filename, log_dir,
                    active_idx, obs_noise, eta, lam, Big_B, min_shots
                )
                for tri in range(args.inner_trials)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
