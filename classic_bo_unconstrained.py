"""
Classic Unconstrained Bayesian Optimization — GP-UCB (Continuous)
========================================================================
Unconstrained baseline aligned to classic_safeset_continuous.py so the two
can be compared apples-to-apples. Every initial setting is identical:

  - Same environment (bo_env_constraints.ClassicFuselageEnv) and objective
  - Same LHS warmup + vectorised MAE with the SAME (effective-variance) noise
  - Same GP setup: initialize_f_model on warmup+init, RFF kernel, scaled_var noise
  - Same initial point (safe-Sobol init, identical seed) so both start alike
  - Same beta_t, V_t / Phi initialisation weighted by eps_max
  - Same candidate-pool schedule, multi-start gradient refinement, stage_epsilon
  - Same save_data schema and results-root layout

The ONLY difference vs. the safe-set script is the acquisition:
  - NO constraint GP, NO safe-set mask, NO constraint penalty.
  - Selection is pure GP-UCB argmax over the FULL candidate pool.
  (Tsai-Wu is still evaluated per query, but ONLY to report the violation rate.)
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
from joblib import load

from botorch.models import FixedNoiseGP
from gpytorch.kernels import RFFKernel, ScaleKernel

from bo_env_constraints import ClassicFuselageEnv
from utils import build_train_gp_with_rff

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
        base = 'Experiments_unconstraint_continuous'
    else:
        base = f'Experiments_unconstraint_continuous_actuators_{int(actuator_count)}'
    if int(force_scale) != 1000:
        base += f'_force_{int(force_scale)}'
    return base


# ─────────────────────────── helpers ─────────────────────────────────────────

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


def save_data(actions, true_response, response, num_of_queries, eps, error_init, file_path):
    """Signature matches classic_safeset_continuous.py save_data exactly."""
    torch.save({
        'actions':       actions,
        'true_response': true_response,
        'response':      response,
        'queries':       num_of_queries,
        'uncertainty':   eps,
        'error_init':    error_init,
    }, file_path)


def sample_lhs_in_active_subspace(
    n, active_idx=(0, 1, 2, 3, 14, 15, 16, 17), d=18,
    low=-0.5, high=0.5, seed=0, device=None, dtype=torch.float64,
):
    device = device or torch.device("cpu")
    sampler = qmc.LatinHypercube(d=len(active_idx), optimization="random-cd", seed=int(seed))
    lhs_unit = sampler.random(n=int(n))
    active_vals = low + (high - low) * lhs_unit
    out = np.zeros((int(n), d), dtype=np.float64)
    out[:, list(active_idx)] = active_vals
    return torch.tensor(out, device=device, dtype=dtype)


def sample_sobol_in_active_subspace(
    n, active_idx=(0, 1, 2, 3, 14, 15, 16, 17), d=18,
    low=-0.5, high=0.5, seed=0, device=None, dtype=torch.float64,
):
    device = device or torch.device("cpu")
    sobol = torch.quasirandom.SobolEngine(dimension=len(active_idx), scramble=True, seed=int(seed))
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
    """Identical to the safe-set script: gives both methods the SAME initial point."""
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


# ─────────────────────────── models ──────────────────────────────────────────

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
        model.covar_module.outputscale.copy_(torch.tensor(1.0, dtype=torch.double, device=device))
    model.eval()
    return model, model.covar_module


def eval_tsai_wu(tsai_wu_model, action):
    if isinstance(action, torch.Tensor):
        action_np = action.detach().cpu().numpy().reshape(1, -1)
    else:
        action_np = np.array(action).reshape(1, -1)
    return float(1 - tsai_wu_model.predict(action_np, return_std=False))


def objective_noise_variance(response, n_warm_up, eps_list, eta, effective_var=None):
    """Per-point GP noise: noisy points for warmup+init, (η·ε)² for active BO."""
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
    Memory-safe W-GP-UCB scoring (identical to classic_safeset_continuous.py).

    Avoids constructing dense W = diag(1 / eps^2).
    Uses Cholesky solve instead of explicit inverse.
    """
    device = V_t.device
    dtype  = V_t.dtype

    eps_tensor = torch.as_tensor(eps_list, device=device, dtype=dtype).view(-1)
    eps_tensor = eps_tensor.clamp_min(1e-12)
    weights    = 1.0 / eps_tensor.square()

    Phi = Unweighted_Phi.to(device=device, dtype=dtype)
    y   = response.reshape(-1, 1).to(device=device, dtype=dtype)

    PhiT_W_y = Phi.T @ (weights[:, None] * y)

    L    = torch.linalg.cholesky(V_t)
    nu_t = torch.cholesky_solve(PhiT_W_y, L)

    beta_value = float(beta.item()) if isinstance(beta, torch.Tensor) else float(beta)
    beta_sqrt  = math.sqrt(max(beta_value, 0.0))

    mean_chunks, var_chunks = [], []
    for start in range(0, x_D.shape[0], chunk_size):
        x_chunk  = x_D[start: start + chunk_size]
        features = model.covar_module.base_kernel.get_features(
            x_chunk, x_chunk.shape[-1], normalize=True).to(device=device, dtype=dtype)
        mean_chunks.append((features @ nu_t).reshape(-1))
        solved = torch.cholesky_solve(features.T, L).T
        var_chunks.append((lam * torch.sum(solved * features, dim=1)).clamp_min(1e-12))

    mean  = torch.cat(mean_chunks)
    var   = torch.cat(var_chunks)
    sigma = var.sqrt()
    ucb   = mean + beta_sqrt * sigma
    return ucb, mean, sigma, var, L, nu_t, beta_sqrt


# ─────────────────────────── trial ───────────────────────────────────────────

def run_trial(tri, seed, args, input_filename, target_filename,
              log_dir, SCRIPT_DIR, active_idx, obs_noise, eta, lam, Big_B, min_shots):
    # ── Assign GPU (matches classic_safeset_continuous.py) ───────────────
    gpu_id = tri % torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    try:
        print(f"[Trial {tri}] -> {device}")
        torch.cuda.empty_cache()
        torch.manual_seed(tri)
        np.random.seed(12345 + seed * 100 + tri)

        tsai_wu_model = load(os.path.join(SCRIPT_DIR, "surrogate_tsaiwu.joblib"))
        M_target = args.M_features
        n_iter   = args.max_iteration

        env = ClassicFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise,
                                 min_shots=min_shots, force_scale=args.force_scale)

        # ── LHS warmup (same seed as safe-set) ───────────────────────────
        warmup_actions = sample_lhs_in_active_subspace(
            n=args.warmup_points, active_idx=active_idx,
            d=18, low=-0.5, high=0.5, seed=10086 + tri,
            device=torch.device("cpu"), dtype=torch.float64,
        )

        env.reset(input_filename, seed=seed, target_npy=target_filename)
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
        y_obs_batch = (-mae_batch.copy() + noise) / env.error_init

        response_tensor      = torch.tensor(y_obs_batch, dtype=torch.double).to(device)
        true_response_tensor = torch.tensor(mae_batch,   dtype=torch.double)

        # GP hyperparameter (lengthscale) fitting on warmup data
        _, lengthscale, _ = build_train_gp_with_rff(
            actions=warmup_actions, response=response_tensor,
            known_noise=scaled_var, max_retries=4,
        )
        env.reset(input_filename, seed=seed, target_npy=target_filename)

        # ── Pre-allocated tensors ─────────────────────────────────────────
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

        # ── Objective GP on warmup data ONLY ─────────────────────────────
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

        # ── Initial point: SAME safe-Sobol point as the safe-set script ──
        env.reset(input_filename, seed=seed, target_npy=target_filename)
        init_action = sample_safe_sobol_in_active_subspace(
            n=1, tsai_wu_model=tsai_wu_model, active_idx=active_idx,
            d=18, low=-0.5, high=0.5, seed=20086 + tri,
            device=device, dtype=torch.float64,
            pool_size=max(8192, args.candidate_pool_base),
        )
        init_response, init_true_response, num_of_queries = env.step_surrogate(
            init_action, eps=args.eps_max * eta, device=device, method='chebyshev')

        num_of_queries_num = num_of_queries.item()
        data_count = warmup_n
        actions[data_count]       = init_action.squeeze(0)
        response[data_count]      = init_response.squeeze(0)
        true_response[data_count] = init_true_response.squeeze(0)
        stages[data_count]        = num_of_queries.squeeze(0)
        eps_list.append(args.eps_max)
        data_count += 1

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

        error_init, _ = env.reset(input_filename, seed=seed, target_npy=target_filename)

        stage_count         = 1
        safe_queries_count  = 0
        total_queries_count = 0

        # ── BO loop (unconstrained: pure GP-UCB over the full pool) ───────
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
            ucb_f, mean_f, sig_f, var_f, chol_V_t, nu_t, beta_sqrt = W_GP_UCB_scores(
                model=f_model, x_D=candidate_actions,
                response=response[:data_count], V_t=V_t,
                Unweighted_Phi=Unweighted_Phi[:data_count],
                eps_list=eps_list[:data_count], beta=beta_t[beta_idx],
                lam=lam, chunk_size=args.score_chunk_size,
            )

            # Discrete pick over the full pool (no safe-set filtering)
            idx = torch.argmax(ucb_f)

            # ── Multi-start gradient-descent refinement (pure UCB) ─────────
            top_k = min(10, candidate_actions.shape[0])
            top_indices = torch.topk(ucb_f, top_k).indices
            top_candidates = candidate_actions[top_indices].clone().detach()

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

                (-ucb_opt.sum()).backward()
                optimizer.step()

                with torch.no_grad():
                    top_candidates.clamp_(-0.5, 0.5)
                    inactive = [i for i in range(18) if i not in active_idx]
                    top_candidates[:, inactive] = 0.0

            with torch.no_grad():
                features = f_model.covar_module.base_kernel.get_features(
                    top_candidates, 18, normalize=True)
                final_var = (
                    lam * torch.sum(
                        torch.cholesky_solve(features.T, chol_V_t).T * features, dim=1)
                ).clamp_min(1e-12)
                final_ucb = (features @ nu_t).reshape(-1) + beta_sqrt * final_var.sqrt()

                best_idx = torch.argmax(final_ucb)
                new_actions = top_candidates[best_idx].reshape(1, -1).detach()
                eps = stage_epsilon(final_var[best_idx].item(), lam, args.eps_max)

            # ── Evaluate ──────────────────────────────────────────────────
            env.reset(input_filename, seed=seed, target_npy=target_filename)
            new_response, true_responses, num_oracle_queries = env.step_surrogate(
                new_actions, eps=eps * eta, device=device, method='chebyshev')

            # Tsai-Wu evaluated ONLY to report the violation rate of this baseline.
            c_val_float = eval_tsai_wu(tsai_wu_model, new_actions)
            total_queries_count += 1
            if c_val_float >= 0:
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

            # Logging every 100 stages
            if stage_count % 100 == 0:
                running_min    = true_response[warmup_n:data_count].min().item()
                safety_rate    = safe_queries_count / max(1, total_queries_count)
                violation_rate = 1.0 - safety_rate
                print(
                    f"[Trial {tri}] Stage {stage_count} | "
                    f"y: {new_response[-1].item():.3f} "
                    f"f(x): {true_responses[-1].item():.3f} | "
                    f"eps: {eps:.4f} | pool: {pool_size} | "
                    f"running_min: {running_min:.4f} | safety rate: {safety_rate:.3f} | violation rate: {violation_rate:.3f}"
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
                              str(obs_noise) + 'classic_unconstrained_training_data_' + str(tri) + '_.pth'),
                )

            error_init, _ = env.reset(input_filename, seed=seed, target_npy=target_filename)
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
        description="Classic Continuous Unconstrained GP-UCB (aligned to safe-set baseline).")

    # ── Model ──────────────────────────────────────────────────────────────
    parser.add_argument("--M_features",  type=int,   default=256,
                        help="Number of RFF features.")
    parser.add_argument("--obs_noise",   type=float, default=0.2**2,
                        help="Observation noise variance (0.01 → σ=0.1, 0.04 → σ=0.2).")
    parser.add_argument("--min_shots", type=int, default=20,
                        help="Minimum shots for Monte Carlo Estimation, 0.1=20, 0.2=80")
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
    parser.add_argument("--actuator_count", type=int, default=8,
                        choices=sorted(ACTUATOR_INDICES_BY_COUNT),
                        help=("Number of active actuators. "
                              "6 maps to indices (0, 1, 2, 17, 16, 15)."))
    parser.add_argument("--force_scale", type=float, default=1000.0,
                        help=("Action-to-force scale in lb: action in (-1,1) maps to "
                              "(-force_scale, +force_scale). Default 1000 (-1000..1000 lb). "
                              "Use e.g. 200 or 500 for narrower force ranges."))
    parser.add_argument("--results_root", type=str, default=None,
                        help=("Root folder for saved results. Defaults to "
                              "Experiments_unconstraint_continuous_actuators_<count>, "
                              "except 8 uses Experiments_unconstraint_continuous. "
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
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    print("Classic Continuous Unconstrained GP-UCB")
    print("noise_level  :", obs_noise)
    print("eta          :", eta, "| eps_max:", args.eps_max,
          "| lam:", lam, "| B:", Big_B, "| Min_shots:", min_shots)
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

        env_name = "Classic_Unconstrained_{}_{}_{}_{}_noise_{}".format(
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
                    input_filename, target_filename, log_dir, SCRIPT_DIR,
                    active_idx, obs_noise, eta, lam, Big_B, min_shots
                )
                for tri in range(args.inner_trials)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
