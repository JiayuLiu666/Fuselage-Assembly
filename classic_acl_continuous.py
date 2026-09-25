import argparse
import math
import os
from os import path
from shutil import copyfile
import warnings
import random
import concurrent.futures

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import qmc
from joblib import load

import botorch.settings as botorch_settings
from botorch.models import FixedNoiseGP
from gpytorch.kernels import RFFKernel, ScaleKernel
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import RBFKernel

from bo_env_constraints import ClassicFuselageEnv
from utils import build_train_gp_with_rff, sample_actions

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
        base = "Experiments_constraint_continuous"
    else:
        base = f"Experiments_constraint_continuous_actuators_{int(actuator_count)}"
    if int(force_scale) != 1000:
        base += f"_force_{int(force_scale)}"
    return base


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

def sample_sobol_in_active_subspace(
    n, active_idx=(0, 1, 2, 3, 14, 15, 16, 17), d=18,
    low=-0.5, high=0.5, seed=0, device=None, dtype=torch.float64,
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


def rff_features(model, X):
    k = model.covar_module.base_kernel
    # 兼容你本地如果有 get_features
    if hasattr(k, "get_features"):
        return k.get_features(X, X.shape[-1], normalize=True)

    # 兼容官方 RFFKernel
    if not hasattr(k, "randn_weights"):
        k._init_weights(X.size(-1), k.num_samples)
    Xw = X.matmul(k.randn_weights / k.lengthscale.transpose(-1, -2))
    Z = torch.cat([torch.cos(Xw), torch.sin(Xw)], dim=-1)
    return Z / math.sqrt(k.num_samples)


def build_V_t_nu_t(model, X, Y, eps_list, lam):
    Z = rff_features(model, X)                                  # [n, m]
    eps = torch.as_tensor(eps_list, device=X.device, dtype=X.dtype).view(-1).clamp_min(1e-6)
    w = 1.0 / eps.square()                                      # [n]

    Z_weighted = Z * w.sqrt().unsqueeze(-1)                     # [n, m]
    V = Z_weighted.T @ Z_weighted + lam * torch.eye(
        Z.size(-1), device=X.device, dtype=X.dtype
    )
    L = torch.linalg.cholesky(V)

    rhs = Z.T @ (w.unsqueeze(-1) * Y)                           # [m, 1]
    nu = torch.cholesky_solve(rhs, L)                           # [m, 1]
    return L, nu


def get_gp_predictions_rff(model, X, L, nu, lam, chunk_size=1024):
    mean_chunks, var_chunks = [], []
    for start in range(0, X.size(0), chunk_size):
        Z = rff_features(model, X[start:start + chunk_size])
        mean_chunks.append((Z @ nu).reshape(-1))

        solved = torch.cholesky_solve(Z.T, L).T
        var_chunks.append((lam * torch.sum(solved * Z, dim=1)).clamp_min(1e-12))

    mean = torch.cat(mean_chunks, dim=0)
    std = torch.cat(var_chunks, dim=0).sqrt()
    return mean, std

def objective_noise_variance(response, n_warm_up, eps_list, eta, effective_var=None):
    """Per-point GP noise: noisy points for warmup+init, (η·ε)² for active BO."""
    eps_t = torch.as_tensor(eps_list, dtype=response.dtype, device=response.device)
    yvar = ((eta * eps_t) ** 2).view(-1, 1)
    yvar[:n_warm_up] = effective_var
    return yvar

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

class GPClassificationModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=inducing_points.shape[-1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def choose_inducing_points(train_X, m=128):
    if train_X.size(0) <= m:
        return train_X.clone()
    idx = torch.randperm(train_X.size(0), device=train_X.device)[:m]
    return train_X[idx].clone()


def fit_constraint_model(train_X, train_C, num_inducing=128, batch_size=512, training_iterations=30, lr=0.01):
    inducing_points = choose_inducing_points(train_X, m=num_inducing)
    model = GPClassificationModel(inducing_points).to(train_X.device, train_X.dtype)
    likelihood = BernoulliLikelihood().to(train_X.device, train_X.dtype)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}, {'params': likelihood.parameters()}],
        lr=lr,
    )
    mll = VariationalELBO(likelihood, model, num_data=train_C.numel())

    loader = DataLoader(
        TensorDataset(train_X, train_C.view(-1).float()),
        batch_size=min(batch_size, train_X.size(0)),
        shuffle=True,
    )

    for _ in range(training_iterations):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()

    return model, likelihood


def update_constraint_model_fixed_params(model, likelihood, train_X, train_C, batch_size=512, training_iterations=10, lr=0.01):
    """Fine-tune the existing model in-place on new data (fixed inducing set)."""
    model.train()
    likelihood.train()

    # Only optimize variational parameters (keep kernel/mean/inducing frozen)
    for param in model.covar_module.parameters():
        param.requires_grad = False
    for param in model.mean_module.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        [{'params': model.variational_strategy._variational_distribution.parameters()}],
        lr=lr,
    )
    mll = VariationalELBO(likelihood, model, num_data=train_C.numel())

    loader = DataLoader(
        TensorDataset(train_X, train_C.view(-1).float()),
        batch_size=min(batch_size, train_X.size(0)),
        shuffle=True,
    )

    for _ in range(training_iterations):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()

    # Re-enable gradients for next full fit if needed
    for param in model.covar_module.parameters():
        param.requires_grad = True
    for param in model.mean_module.parameters():
        param.requires_grad = True

    return model, likelihood


@torch.no_grad()
def get_constraint_probabilities(model, likelihood, test_X, chunk_size=1024):
    model.eval()
    likelihood.eval()
    means = []
    for start in range(0, test_X.shape[0], chunk_size):
        x_chunk = test_X[start : start + chunk_size]
        latent_dist = model(x_chunk)
        pred_dist = likelihood(latent_dist)
        means.append(pred_dist.mean.view(-1))
    return torch.cat(means, dim=0)

def norm_01(values):
    if values.numel() == 0:
        return values
    lo = values.min()
    hi = values.max()
    if hi > lo:
        return (values - lo) / (hi - lo + 1e-12)
    return torch.zeros_like(values)

def eval_tsai_wu(tsai_wu_model, action):
    if isinstance(action, torch.Tensor):
        action_np = action.detach().cpu().numpy().reshape(1, -1)
    else:
        action_np = np.asarray(action).reshape(1, -1)
    failure_index = tsai_wu_model.predict(action_np, return_std=False)
    return float(1.0 - failure_index)

def save_data(actions, response, true_response, num_of_queries, uncertainty, constraint_margin, error_init, file_path):
    torch.save(
        {
            "actions": actions,
            "response": response,
            "true_response": true_response,
            "queries": num_of_queries,
            "uncertainty": uncertainty,
            "constraint_margin": constraint_margin,
            "error_init": error_init
        },
        file_path,
    )


def run_trial(tri, seed, args, input_filename, target_filename, log_dir, SCRIPT_DIR, active_idx, obs_noise, eta, lam, Big_B, min_shots):
    try:
        torch.set_default_dtype(torch.float64)
        gpu_id = tri % torch.cuda.device_count() if torch.cuda.is_available() else 0
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        print(f"Trial: {tri} strictly bound to computing {device}")
        
        torch.cuda.empty_cache()
        torch.manual_seed(10086 + tri)
        np.random.seed(12345 + seed * 100 + tri)

        tsai_wu_model = load(os.path.join(SCRIPT_DIR, "surrogate_tsaiwu.joblib"))

        n_iters = args.max_iteration
        M_target = args.M_features
        env = ClassicFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise, min_shots=min_shots, force_scale=args.force_scale)
        safe_queries_count = 0
        total_queries_count = 0

        file_2, error_init = env.reset(input_filename, seed=seed, target_npy=target_filename)
        print(f"[Trial {tri}]: target shape is {file_2}")

        warmup_actions = sample_lhs_in_active_subspace(
            n=args.warmup_points,
            active_idx=active_idx,
            d=18,
            low=-0.5,
            high=0.5,
            seed=10086 + tri,
            device=torch.device("cpu"),
            dtype=torch.float64,
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
        y_obs_batch = (-mae_batch.copy() + noise)/ env.error_init
        
        response_tensor = torch.tensor(y_obs_batch, dtype=torch.float64).reshape(-1, 1).to(device)

        _, lengthscale, _ = build_train_gp_with_rff(
            actions=warmup_actions, response=response_tensor,
            known_noise=scaled_var,
            max_retries=4)

        # ---- Initialize BO ----
        env.reset(input_filename, seed=seed, target_npy=target_filename)

        total_capacity = n_iters + args.warmup_points + 10
        actions = torch.zeros((total_capacity, 18), dtype=torch.float64, device=device)
        response = torch.zeros((total_capacity, 1), dtype=torch.float64, device=device)
        true_response = torch.zeros((total_capacity, 1), dtype=torch.float64, device=device)
        num_of_queries = torch.zeros((total_capacity, 1), dtype=torch.float64, device=device)

        warmup_X = warmup_actions.to(device=device, dtype=torch.float64)
        warmup_Y = response_tensor.view(-1, 1).to(device=device, dtype=torch.float64)
        warmup_true_Y = torch.tensor(mae_batch, dtype=torch.float64).view(-1, 1).to(device=device, dtype=torch.float64)
        warmup_n = warmup_X.shape[0]

        actions[:warmup_n] = warmup_X
        response[:warmup_n] = warmup_Y
        true_response[:warmup_n] = warmup_true_Y

        env.reset(input_filename, seed=seed, target_npy=target_filename)
        init_action = sample_safe_sobol_in_active_subspace(
            n=1, tsai_wu_model=tsai_wu_model, active_idx=active_idx,
            d=18, low=-0.5, high=0.5, seed=20086 + tri,
            device=device, dtype=torch.float64,
            pool_size=max(8192, args.candidate_pool_base),
        )
        init_response, init_true_response, num_of_oracle_queries = env.step_surrogate(
            action=init_action, device=device, eps=args.eps_max * eta, method='chebyshev')
        init_c_margin_check = eval_tsai_wu(tsai_wu_model, init_action)
        if init_c_margin_check < 0.0:
            raise RuntimeError(
                f"Initial action is outside the safe set: margin={init_c_margin_check:.6g}"
            )

        data_count = warmup_n
        actions[data_count] = init_action.squeeze(0)
        response[data_count] = init_response.squeeze(0)
        true_response[data_count] = init_true_response.squeeze(0)
        num_of_queries[data_count] = num_of_oracle_queries.squeeze(0)
        data_count += 1

        f_model, _ = initialize_f_model(
            actions[:warmup_n], response[:warmup_n],
            device, M_target,
            ls_init=lengthscale,
            obs_noise=scaled_var,
        )

        train_X = [warmup_X[i].view(1, -1) for i in range(warmup_n)]
        train_Y = [warmup_Y[i].view(1, -1) for i in range(warmup_n)]
        train_true_Y = [warmup_true_Y[i].view(1, -1) for i in range(warmup_n)]
        train_eps = [WARMUP_PRECISION] * warmup_n
        train_q = [torch.tensor([[0.0]], dtype=torch.float64, device=device)] * warmup_n
        train_C = []
        for i in range(warmup_n):
            m = eval_tsai_wu(tsai_wu_model, warmup_X[i])
            train_C.append(torch.tensor([[m]], dtype=torch.float64, device=device))

        init_c_margin = eval_tsai_wu(tsai_wu_model, init_action)
        train_X.append(init_action)
        train_Y.append(init_response.to(dtype=torch.float64))
        train_true_Y.append(init_true_response.to(dtype=torch.float64))
        train_C.append(torch.tensor([[init_c_margin]], dtype=torch.float64, device=device))
        train_eps.append(WARMUP_PRECISION)
        train_q.append(num_of_oracle_queries.to(torch.float64))

        train_X_t = torch.cat(train_X, dim=0)
        train_Y_t = torch.cat(train_Y, dim=0)
        train_C_t = torch.cat(train_C, dim=0)
        
        c_model, c_likelihood = fit_constraint_model(train_X_t, (train_C_t >= 0.0).to(train_X_t.dtype))

        total_queries_num = num_of_oracle_queries.item()
        iterations = 1
        stage = 0

        while total_queries_num < n_iters:
            train_X_t = torch.cat(train_X, dim=0)
            train_Y_t = torch.cat(train_Y, dim=0)
            train_C_t = torch.cat(train_C, dim=0)
            
            alpha_t = args.alpha_0 * (args.alpha_decay**stage)
            uncertainty_weight = max(0.0, 1.0 - alpha_t - args.beta_weight)

            c_model, c_likelihood = update_constraint_model_fixed_params(
                c_model, c_likelihood, train_X_t, (train_C_t >= 0.0).to(train_X_t.dtype)
            )

            pool_size = candidate_pool_size_by_stage(
                iterations,
                base_size=args.candidate_pool_base,
                growth=args.candidate_pool_growth,
                max_size=args.candidate_pool_max,
                every=args.candidate_pool_growth_interval,
            )
            candidate_seed = 10086 + tri * 100000 + iterations
            candidate_actions = sample_sobol_in_active_subspace(
                n=pool_size, active_idx=active_idx, d=18,
                low=-0.5, high=0.5, seed=candidate_seed,
                device=device, dtype=torch.float64,
            )

            p_feasible = get_constraint_probabilities(c_model, c_likelihood, candidate_actions)
            predicted_safe = p_feasible >= args.feasible_threshold
            safe_score = 1.0 - torch.abs(2.0 * p_feasible - 1.0)

            feasible_mask = train_C_t.view(-1) >= 0.0
            X_m = train_X_t[feasible_mask]
            Y_m = train_Y_t[feasible_mask]
            eps_list_m = [train_eps[i] for i in range(len(train_eps)) if feasible_mask[i]]

            if len(X_m) == 0:
                eps_aug = train_eps
                L_t, nu_t = build_V_t_nu_t(f_model, train_X_t, train_Y_t, eps_aug, lam)
            else:
                L_t_m, nu_t_m = build_V_t_nu_t(f_model, X_m, Y_m, eps_list_m, lam)
                X_u = train_X_t[~feasible_mask]
                if len(X_u) > 0:
                    mu_u, _ = get_gp_predictions_rff(f_model, X_u, L_t_m, nu_t_m, lam)
                    X_aug = torch.cat([X_m, X_u], dim=0)
                    Y_aug = torch.cat([Y_m, mu_u.view(-1, 1)], dim=0)
                    eps_u = [1.0] * len(X_u)
                    eps_aug = eps_list_m + eps_u
                    L_t, nu_t = build_V_t_nu_t(f_model, X_aug, Y_aug, eps_aug, lam)
                else:
                    L_t, nu_t = L_t_m, nu_t_m

            mu_f, sigma_f = get_gp_predictions_rff(f_model, candidate_actions, L_t, nu_t, lam)
            
            beta_t = (1 + Big_B * math.sqrt(math.log(iterations + 1)**2))**2
            acquisition_raw = mu_f + math.sqrt(beta_t) * sigma_f

            if predicted_safe.sum() == 0:
                idx_star = int(torch.argmax(p_feasible).item())
                x_star = candidate_actions[idx_star].view(1, -1).detach()
                eps_star = args.eps_max
            else:
                masked_acq = acquisition_raw.clone()
                masked_acq[~predicted_safe] = -float("inf")
                
                top_k = min(5, predicted_safe.sum().item())
                top_indices = torch.topk(masked_acq, top_k).indices
                top_candidates = candidate_actions[top_indices].clone().detach()
                
                L_t_d = L_t.detach()
                nu_t_d = nu_t.detach()
                
                top_candidates.requires_grad_(True)
                optimizer = torch.optim.Adam([top_candidates], lr=0.01)

                for _ in range(50):
                    optimizer.zero_grad()
                    features = rff_features(f_model, top_candidates)
                    mean_part = (features @ nu_t_d).reshape(-1)
                    proj = torch.cholesky_solve(features.T, L_t_d).T
                    var_part = (lam * torch.sum(proj * features, dim=1)).clamp_min(1e-12)
                    ucb_opt = mean_part + math.sqrt(beta_t) * var_part.sqrt()
                    loss = -ucb_opt.sum()
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        top_candidates.clamp_(-0.5, 0.5)
                        inactive_idx = [i for i in range(18) if i not in active_idx]
                        top_candidates[:, inactive_idx] = 0.0

                with torch.no_grad():
                    p_feas_opt = get_constraint_probabilities(c_model, c_likelihood, top_candidates)
                    valid_mask = p_feas_opt >= args.feasible_threshold
                    
                    if valid_mask.sum() > 0:
                        features = rff_features(f_model, top_candidates)
                        solved_f = torch.cholesky_solve(features.T, L_t_d).T
                        final_var = (lam * torch.sum(solved_f * features, dim=1)).clamp_min(1e-12)
                        final_ucb = (features @ nu_t_d).reshape(-1) + math.sqrt(beta_t) * final_var.sqrt()
                        final_ucb[~valid_mask] = -float("inf")
                        best_opt_idx = torch.argmax(final_ucb)
                        x_star = top_candidates[best_opt_idx].view(1, -1).detach()
                        eps_star = args.eps_max
                        idx_star = None
                    else:
                        idx_star = int(torch.argmax(masked_acq).item())
                        x_star = candidate_actions[idx_star].view(1, -1).detach()
                        eps_star = args.eps_max

            batch_xs = [x_star]
            batch_eps = [eps_star]

            if args.batch_size > 1:
                pool_n = candidate_actions.size(0)
                sub_size = min(2000, pool_n)
                sub_indices = torch.randperm(pool_n, device=device)[:sub_size]

                # x_star 已经被 optimization loop 选过，不再进入 ACL 候选池
                if idx_star is not None:
                    sub_indices = sub_indices[sub_indices != idx_star]

                # ACL 当前候选池
                U_s = candidate_actions[sub_indices].clone()

                # Su 在一个 batch 内固定
                Su_bar = norm_01(safe_score[sub_indices]).clone()

                # 论文 Eq.(9): X_{0,t} 是 labeled dataset，即 train_X_t
                # Sd(x_i) = min_j ||x_i - X_{m-1,t}||
                # X_selected = labeled set + 当前 batch 已选点
                X_selected = torch.cat([train_X_t, x_star], dim=0)

                for _ in range(args.batch_size - 1):
                    if U_s.size(0) == 0:
                        break

                    # Sr：每选一个点后基于剩余 U_s 重新计算
                    if U_s.size(0) > 1:
                        dist_uu = torch.cdist(U_s, U_s)
                        Sr_raw = dist_uu.sum(dim=1) / (U_s.size(0) - 1)
                        Sr_bar = 1.0 - norm_01(Sr_raw)
                    else:
                        Sr_bar = torch.ones(1, device=device, dtype=U_s.dtype)

                    # Sd：论文定义，相对于 labeled set + 当前 batch 已选点
                    Sd_raw = torch.cdist(U_s, X_selected).min(dim=1).values
                    Sd_bar = norm_01(Sd_raw)

                    Q = (
                        uncertainty_weight * Su_bar
                        + alpha_t * Sr_bar
                        + args.beta_weight * Sd_bar
                    )

                    local_idx = int(torch.argmax(Q).item())
                    x_k = U_s[local_idx:local_idx + 1].detach()
                    eps_k = args.eps_max

                    batch_xs.append(x_k)
                    batch_eps.append(eps_k)

                    # 加入 selected set
                    X_selected = torch.cat([X_selected, x_k], dim=0)

                    # 从候选池删掉刚选中的点
                    keep = torch.ones(U_s.size(0), dtype=torch.bool, device=device)
                    keep[local_idx] = False
                    U_s = U_s[keep]
                    Su_bar = Su_bar[keep]


            
            def evaluate_candidate(idx, action, eps_value):
                # Using the identical device dedicated explicitly for this trial to avoid cross-gpu leakage.
                local_device = device
                local_env = ClassicFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise, min_shots=min_shots, force_scale=args.force_scale)
                local_env.reset(input_filename, seed=seed, target_npy=target_filename)
                
                y_obs, y_true, q = local_env.step_surrogate(
                    action=action, device=local_device, eps=eta * eps_value, method='non_monte_carlo')
                
                c_margin = eval_tsai_wu(tsai_wu_model, action)
                
                return {
                    "action": action,
                    "y_obs": y_obs.to(dtype=torch.float64, device=device),
                    "y_true": y_true.to(dtype=torch.float64, device=device),
                    "c_margin": c_margin,
                    "eps": eps_value,
                    "q": q.to(dtype=torch.float64, device=device)
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_xs)) as executor:
                futures = [executor.submit(evaluate_candidate, i, x, eps) for i, (x, eps) in enumerate(zip(batch_xs, batch_eps))]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            for res in results:
                train_X.append(res["action"])
                train_Y.append(res["y_obs"])
                train_true_Y.append(res["y_true"])
                train_C.append(torch.tensor([[res["c_margin"]]], dtype=torch.float64, device=device))
                train_eps.append(res["eps"])
                train_q.append(res["q"])
                
                total_queries_num += float(res["q"].item())
                total_queries_count += 1
                if res["c_margin"] >= 0:
                    safe_queries_count += 1

            stage += 1
            iterations += 1

            if iterations % 10 == 0:
                running_min = torch.cat(train_true_Y, dim=0).min().item()
                violation_rate = safe_queries_count / max(1, total_queries_count)
                print(f'[Trial {tri}] Iteration {iterations}, y: {res["y_obs"].item():.3f} f(x): {res["y_true"].item():.3f} pool: {pool_size} | running_min: {running_min:.4f} | safe rate: {violation_rate:.3f}')

            if iterations % 20 == 0 or total_queries_num >= n_iters:
                save_data(
                    actions=torch.cat(train_X, dim=0), 
                    response=torch.cat(train_Y, dim=0),
                    true_response=torch.cat(train_true_Y, dim=0),
                    num_of_queries=torch.cat(train_q, dim=0), 
                    uncertainty=train_eps,
                    constraint_margin=torch.cat(train_C, dim=0),
                    error_init=error_init,
                    file_path=path.join(log_dir, str(obs_noise) + 'training_data_' + str(tri) + '_.pth'))

        final_min = torch.cat(train_true_Y, dim=0).min().item()
        print("="*60)
        print("Trial {0} finished | total queries: {1} | final minimum f(x): {2:.6f}".format(
            tri, total_queries_num, final_min
        ))
        print("="*60)
        return True

    except Exception as e:
        import traceback
        print(f"Error in Trial {tri}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    torch.set_default_dtype(torch.float64)
    botorch_settings.debug._set_state(True)

    parser = argparse.ArgumentParser(description='Classic Continuous ACL (fast RFF).')
    parser.add_argument('--M_features', type=int, default=256, help='Number of features used in RFF.')
    parser.add_argument('--max_iteration', type=int, default=50000, help='Maximum iterations.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for exploration logic')
    parser.add_argument('--obs_noise', type=float, default=0.1**2, help='noise level of observation')
    parser.add_argument("--min_shots", type=int, default=20, help="Minimum shots for Monte Carlo Estimation.")
    parser.add_argument('--eta', type=float, default=1.0, help='precision trade-off parameters')
    parser.add_argument('--lam', type=float, default=1.0, help='lambda hyperparameters')
    parser.add_argument('--eps_max', type=float, default=0.04, help='Maximum per-stage epsilon before eta scaling.')
    parser.add_argument('--B', type=float, default=1.0, help='exploit hyperparameters')
    parser.add_argument('--alpha_0', type=float, default=0.7)
    parser.add_argument('--beta_weight', type=float, default=0.2)
    parser.add_argument('--alpha_decay', type=float, default=0.8)
    parser.add_argument('--feasible_threshold', type=float, default=0.5)
    parser.add_argument('--warmup_points', type=int, default=200, help='Number of LHS warm-start points across the active dimensions.')
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
    parser.add_argument('--candidate_pool_base', type=int, default=2048, help='Initial Sobol candidate count.')
    parser.add_argument('--candidate_pool_growth', type=int, default=2048, help='Additional candidates per growth step.')
    parser.add_argument('--candidate_pool_max', type=int, default=2**16, help='Maximum Sobol candidate count.')
    parser.add_argument('--candidate_pool_growth_interval', type=int, default=10, help='Grow pool every N stages.')
    parser.add_argument('--outer_seeds', type=int, default=10, help='Number of outer initial-shape seeds.')
    parser.add_argument('--inner_trials', type=int, default=5, help='Number of BO trials per outer seed.')
    args = parser.parse_args()

    active_idx = active_indices_for_count(args.actuator_count)
    results_root = args.results_root or default_results_root(args.actuator_count, args.force_scale)
    obs_noise = args.obs_noise
    min_shots = args.min_shots
    eta = args.eta
    lam = args.lam
    Big_B = args.B

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    print('Classic Continuous ACL (fast RFF)')
    print('noise_level:', obs_noise)
    print("Min shots:", min_shots)
    print('batch_size:', args.batch_size)
    print('trade off parameters:', eta)
    print('lambda:', lam)
    print('Big B:', Big_B)
    print('eps_max:', args.eps_max)
    print('warmup_points:', args.warmup_points)
    print("actuator_count:", args.actuator_count, "| active_idx:", active_idx)
    print("results_root :", results_root)

    Initial_shape_file = ['DP52', 'DP50', 'DP49', 'DP45', 'DP45', 'DP48', 'DP43', 'DP53', 'DP60', 'DP54']
    Target_shape_file = ['DP53', 'DP53', 'DP57', 'DP53', 'DP58', 'DP53', 'DP53', 'DP55', 'DP44', 'DP53']

    for seed in range(args.outer_seeds):
        random.seed(seed)
        ansys_folder = path.join('FuselageActuators', 'AnsysFiles', "Test")
        shape_folder = path.join('FuselageActuators', 'Shapes', 'Test')

        init_name = 'SolutionInput' + Initial_shape_file[seed]
        file = init_name + '.inp'
        filepath = path.join(ansys_folder, file)
        original_input_filename = filepath
        print("Initial shape from", init_name)

        log_dir = results_root
        os.makedirs(log_dir, exist_ok=True)

        env_name = "Classic_Continuous_ACL_{}_{}_{}_{}_noise_{}".format(
            args.actuator_count, eta, lam, Big_B, args.obs_noise)
        log_dir = path.join(log_dir, env_name, 'exp_set_' + str(seed))
        os.makedirs(log_dir, exist_ok=True)

        input_filename = path.join(log_dir, file)
        copyfile(original_input_filename, input_filename)

        target_name = 'SolutionInput' + Target_shape_file[seed]
        target_npy = target_name + '.npy'
        original_target_filename = path.join(shape_folder, target_npy)
        target_filename = path.join(log_dir, target_npy)
        copyfile(original_target_filename, target_filename)

        # Limit max_workers to 2 trials per visible GPU to prevent OOM
        active_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        safe_max_workers = min(args.inner_trials, active_gpu_count)

        with concurrent.futures.ProcessPoolExecutor(max_workers=safe_max_workers) as executor:
            futures = []
            for tri in range(args.inner_trials):
                futures.append(executor.submit(
                    run_trial, tri, seed, args, input_filename, target_filename, log_dir, SCRIPT_DIR, active_idx, obs_noise, eta, lam, Big_B, min_shots
                ))
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
