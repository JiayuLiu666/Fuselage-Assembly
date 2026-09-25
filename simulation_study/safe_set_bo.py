import math
import warnings
import copy

import numpy as np
import torch
from types import SimpleNamespace

try:
    from .experiment_env import (
        build_paper_sim2_environment,
        compute_global_safe_optimum,
        sample_initial_indices,
    )
except ImportError:
    from experiment_env import (
        build_paper_sim2_environment,
        compute_global_safe_optimum,
        sample_initial_indices,
    )

try:
    from botorch.fit import fit_gpytorch_mll
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

    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.kernels import MaternKernel, ScaleKernel

    BOTORCH_AVAILABLE = True
except ModuleNotFoundError:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    BOTORCH_AVAILABLE = False

    class _SklearnGPWrapper:
        def __init__(self, train_X, train_Y, noise_var=1e-6):
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(train_X.shape[-1]),
                length_scale_bounds=(1e-3, 1e3),
                nu=2.5,
            )
            self._gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=max(float(noise_var), 1e-8),
                normalize_y=False,
                n_restarts_optimizer=0,
            )
            self._gp.fit(
                train_X.detach().cpu().numpy(),
                train_Y.view(-1).detach().cpu().numpy(),
            )

        def eval(self):
            return self

        @property
        def likelihood(self):
            return SimpleNamespace(eval=lambda: None)

        def posterior(self, X):
            mean, std = self._gp.predict(X.detach().cpu().numpy(), return_std=True)
            mean_t = torch.as_tensor(mean, dtype=X.dtype, device=X.device).view(-1, 1)
            var_t = torch.as_tensor(std**2, dtype=X.dtype, device=X.device).view(-1, 1)
            return SimpleNamespace(mean=mean_t, variance=var_t)

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)


# =========================================================
# 2a. RFF objective surrogate  (from synth_bo.py / unconstrained_bo.py)
# =========================================================

def _rff_sample_features(d: int, M: int, v_kernel: float = 1.0,
                          lengthscale: float = 1.0, seed: int = 0) -> dict:
    """Pre-sample random Fourier features  s ~ N(0, I/l²),  b ~ U[0,2π]."""
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((M, d)) / lengthscale   # [M, d]
    b = rng.uniform(0.0, 2 * np.pi, M)               # [M]
    return {"s": s, "b": b, "v_kernel": v_kernel}


def _rff_phi(x_np: np.ndarray, rf: dict, M: int) -> np.ndarray:
    """Feature vector φ(x) [M] — mirrors bayesian_optimization_bo.py lines 95-100."""
    s, b, v = rf["s"], rf["b"], rf["v_kernel"]
    x = np.asarray(x_np).reshape(1, -1)
    phi = np.sqrt(2.0 / M) * np.cos(np.squeeze(x @ s.T) + b)
    norm = float(np.sqrt(np.inner(phi, phi)))
    if norm > 1e-12:
        phi /= norm
    return np.sqrt(v) * phi   # [M]


def _rff_ucb_all(domain_np: np.ndarray, rf: dict, nu_t: np.ndarray,
                  Sigma_t_inv: np.ndarray, beta: float, M: int,
                  lam: float = 1.0, mode: str = "max") -> tuple:
    """
    W-GP-UCB scores over the full domain grid.
    Returns (scores [N], variances [N]).
    """
    N = domain_np.shape[0]
    scores = np.empty(N)
    vars_  = np.empty(N)
    for i in range(N):
        phi  = _rff_phi(domain_np[i], rf, M).reshape(-1, 1)
        mean = float(phi.T @ nu_t)
        var  = float(lam * (phi.T @ Sigma_t_inv @ phi))
        if mode == "max":
            scores[i] = mean + beta * math.sqrt(max(var, 0.0))
        else:
            scores[i] = -mean + beta * math.sqrt(max(var, 0.0))
        vars_[i] = var
    return scores, vars_


# =========================================================
# 2. Utilities
# =========================================================
def minmax_norm(a: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12):
    v = a[mask]
    if v.numel() == 0:
        return torch.zeros_like(a)
    lo = v.min()
    hi = v.max()
    return (a - lo) / (hi - lo + eps)


def lambda_by_stage(stage, lam0=0.8, t0=10, p=2.0):
    return float(lam0 * (t0 / (t0 + max(1, stage))) ** p)


def incumbent_simple_regret(best_safe_value, global_safe_opt, mode):
    if np.isnan(best_safe_value) or np.isnan(global_safe_opt):
        return np.nan
    if mode == "max":
        return float(global_safe_opt - best_safe_value)
    if mode == "min":
        return float(best_safe_value - global_safe_opt)
    raise ValueError("mode must be 'max' or 'min'")


def initialize_gp(train_X, train_Y, noise_var=1e-6, covar_module=None, mean_module=None):
    """
    train_X: [N,d]
    train_Y: [N,1]
    """
    noise_var = max(noise_var, 1e-6)  # numerical stability floor for duplicate points
    yvar = torch.full_like(train_Y, noise_var)
    if not BOTORCH_AVAILABLE:
        return _SklearnGPWrapper(train_X, train_Y, noise_var=noise_var), None
    if covar_module is None:
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]))
    model = _build_gp(train_X, train_Y, yvar, covar_module=covar_module, mean_module=mean_module)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def _deduplicate(train_X, train_Y):
    """Average observations at duplicate X locations."""
    # Round to avoid floating point issues on grid
    X_rounded = torch.round(train_X * 1e8) / 1e8
    unique_X, inverse = torch.unique(X_rounded, dim=0, return_inverse=True)
    unique_Y = torch.zeros(unique_X.shape[0], 1, dtype=train_Y.dtype, device=train_Y.device)
    counts = torch.zeros(unique_X.shape[0], dtype=train_Y.dtype, device=train_Y.device)
    for i in range(train_X.shape[0]):
        unique_Y[inverse[i]] += train_Y[i]
        counts[inverse[i]] += 1
    unique_Y = unique_Y / counts.unsqueeze(1)
    return unique_X, unique_Y


def fit_gp_model(train_X, train_Y, noise_var=1e-6, lengthscale=0.5, outputscale=1.0):
    dedup_X, dedup_Y = _deduplicate(train_X, train_Y)
    model, mll = initialize_gp(dedup_X, dedup_Y, noise_var=noise_var)
    if BOTORCH_AVAILABLE:
        # Instead of fitting ML hyperparameters, we use fixed ones:
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale, dtype=train_X.dtype, device=train_X.device)
            model.covar_module.outputscale = torch.tensor(outputscale, dtype=train_X.dtype, device=train_X.device)
    model.eval()
    model.likelihood.eval()
    return model


def update_gp_model_fixed_params(base_model, train_X, train_Y, noise_var=1e-6):
    """
    Rebuild a GP on the expanded dataset while keeping the previously learned
    hyperparameters fixed. Only the posterior changes with the new data.
    """
    if not BOTORCH_AVAILABLE:
        return fit_gp_model(train_X, train_Y, noise_var=noise_var)
    dedup_X, dedup_Y = _deduplicate(train_X, train_Y)
    model, _ = initialize_gp(
        dedup_X,
        dedup_Y,
        noise_var=noise_var,
        covar_module=copy.deepcopy(base_model.covar_module),
        mean_module=copy.deepcopy(base_model.mean_module),
    )
    model.eval()
    model.likelihood.eval()
    return model


@torch.no_grad()
def posterior_mean_var(model, X):
    post = model.posterior(X)
    mu = post.mean.view(-1)
    var = post.variance.view(-1).clamp_min(1e-12)
    sigma = var.sqrt()
    return mu, var, sigma


@torch.no_grad()
def compute_lcb_ucb(model, X, beta=3.0):
    mu, var, sigma = posterior_mean_var(model, X)
    rad = math.sqrt(beta) * sigma
    lcb = mu - rad
    ucb = mu + rad
    return mu, sigma, lcb, ucb



def sample_initial_points(X_grid_np, n_init=10, seed=0):
    """
    在整个网格上随机抽初始点，不要求安全
    """
    return np.asarray(sample_initial_indices(len(X_grid_np), n_init=n_init, seed=seed), dtype=int)


def resolve_n_init(total_points, n_init=10, init_percentage=None):
    """
    Resolve the number of initial points from either an absolute count or
    a percentage of the full grid.

    init_percentage supports either:
    - 0 < p <= 1: fraction form, e.g. 0.05 means 5%
    - 1 < p <= 100: percentage form, e.g. 5 means 5%
    """
    if init_percentage is None:
        resolved_n_init = int(n_init)
    else:
        p = float(init_percentage)
        if p <= 0:
            raise ValueError("init_percentage must be positive.")
        if p <= 1.0:
            frac = p
        elif p <= 100.0:
            frac = p / 100.0
        else:
            raise ValueError("init_percentage must be in (0, 1] or (1, 100].")
        resolved_n_init = int(np.ceil(total_points * frac))

    resolved_n_init = max(1, min(int(resolved_n_init), int(total_points)))
    return resolved_n_init
# =========================================================
# 3. Safe BO simulated study
# =========================================================
def run_safe_bo_simulation(
    mode="max",              # "max" or "min"
    xi=0,
    grid_size=50,
    x1_bounds=(-1.0, 1.0),
    x2_bounds=(-1.0, 1.0),
    n_init=10,
    n_init_c=None,
    init_percentage=None,
    n_iter=80,
    obj_noise_std=0.02,
    con_noise_std=1e-6,
    beta_f=1.0,
    beta_c=3.0,
    lam0=0.8,
    lam_t0=10,
    lam_p=1.0,
    seed=0,
    allow_repeated_queries=True,
    init_idx=None,
    init_idx_c=None,
    # RFF surrogate settings for objective
    M_rff: int = 256,
    lam_rff: float = 1.0,
    lengthscale_rff: float = 1.0,
    v_kernel_rff: float = 1.0,
    # Constraint GP hyperparameters (fixed, not learned)
    con_lengthscale: float = 0.5,
    con_outputscale: float = 1.0,
):
    """
    Classical Safe BO with Chebyshev MC query.

    Structure mirrors quantum_safe_bo.py exactly — the ONLY difference
    is the query step (Chebyshev MC estimation vs quantum IAE).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Grid / ground truth
    # ------------------------------------------------------------------
    env = build_paper_sim2_environment(
        grid_size=grid_size, xi=xi, dtype=torch.float64,
        constraint_representation="margin",
        x1_bounds=x1_bounds, x2_bounds=x2_bounds,
    )
    xx             = env["xx"]
    yy             = env["yy"]
    zz             = env["zz"]
    X_grid         = env["X_grid"]
    safe_true_mask = env["safe_true_mask"]
    has_true_safe_points = bool(safe_true_mask.any())

    print(f"[Safe BO] Grid size: {len(xx)}")
    print(f"[Safe BO] Failure Ratio: {np.count_nonzero(zz < xi)/len(xx):.3f}")
    print(f"[Safe BO] Safe points: {safe_true_mask.sum()} / {len(xx)}")

    global_safe_opt = compute_global_safe_optimum(yy, safe_true_mask, mode)
    N = len(xx)

    # ------------------------------------------------------------------
    # Initial design
    # ------------------------------------------------------------------
    n_init = resolve_n_init(N, n_init=n_init, init_percentage=init_percentage)
    if n_init_c is None:
        n_init_c = n_init
    elif n_init_c != n_init:
        raise ValueError("n_init_c must match n_init so c_model and the objective surrogate share the same initial set.")

    if init_idx is None:
        init_idx = sample_initial_points(xx, n_init=n_init, seed=seed)
    else:
        init_idx = np.asarray(init_idx, dtype=int)
        if len(init_idx) != n_init:
            raise ValueError("len(init_idx) must match the resolved n_init.")

    if init_idx_c is None:
        init_idx_c = init_idx.copy()
    else:
        init_idx_c = np.asarray(init_idx_c, dtype=int)
        if len(init_idx_c) != n_init or not np.array_equal(init_idx_c, init_idx):
            raise ValueError("init_idx_c must match init_idx so c_model and the objective surrogate use the same initial points.")

    # Objective observations (n_init pts)
    train_X = X_grid[init_idx].clone()
    obj_obs, obj_con_obs = [], []
    for idx in init_idx:
        obj_obs.append(yy[idx] + np.random.randn() * obj_noise_std)
        obj_con_obs.append((zz[idx] - xi) + np.random.randn() * con_noise_std)

    train_Y = torch.tensor(obj_obs, dtype=torch.float64).view(-1, 1)

    # Constraint observations use the same shared init points with no duplicate
    # constraint measurements at initialization.
    c_train_X = X_grid[init_idx_c].clone()
    c_train_C = torch.tensor(obj_con_obs, dtype=torch.float64).view(-1, 1)

    # ------------------------------------------------------------------
    # W-GP-UCB RFF setup  (aligned with quantum_safe_bo.py)
    # ------------------------------------------------------------------
    xx_np = np.asarray(xx)
    d     = xx_np.shape[1]
    rf    = _rff_sample_features(d=d, M=M_rff, v_kernel=v_kernel_rff,
                                  lengthscale=lengthscale_rff, seed=seed)

    beta_t = np.sqrt(3) * beta_f
    eps_list = np.ones(n_init)
    Y_obs_np = np.array(obj_obs)

    eps_weights = 1.0 / eps_list

    Phi_obj = np.stack([
        _rff_phi(xx_np[idx], rf, M_rff) * eps_weights[i]
        for i, idx in enumerate(init_idx)
    ])

    Y_weighted  = (Y_obs_np * eps_weights).reshape(-1, 1)
    Sigma_t     = Phi_obj.T @ Phi_obj + lam_rff * np.eye(M_rff)
    Sigma_t_inv = np.linalg.inv(Sigma_t)
    nu_t        = Sigma_t_inv @ Phi_obj.T @ Y_weighted

    # Constraint GP
    c_model = fit_gp_model(c_train_X, c_train_C, noise_var=con_noise_std**2,
                           lengthscale=con_lengthscale, outputscale=con_outputscale)
    init_mu_c, _, init_lcb_c, _ = compute_lcb_ucb(c_model, X_grid, beta=beta_c)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    queried_idx                  = list(init_idx)
    queried_c_obs                = list(obj_con_obs)
    best_safe_value_hist         = []
    recommended_idx_hist         = []
    safe_set_size_hist           = []
    queried_instant_regret_hist  = []
    queried_cumu_regret_hist     = []

    track_queries          = []
    cumulative_oracle_hist = []

    total_oracle_queries = 0
    for _ in range(n_init):
        track_queries.append(0)

    obs_safe_mask = torch.tensor(obj_con_obs, dtype=torch.float64) >= 0.0
    if obs_safe_mask.any():
        yy_init = torch.tensor([yy[i] for i in init_idx], dtype=torch.float64)
        bsv0 = yy_init[obs_safe_mask].max().item() if mode == "max" else yy_init[obs_safe_mask].min().item()
    else:
        bsv0 = np.nan
    best_safe_value_hist.append(bsv0)

    t = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while total_oracle_queries < n_iter:
        t += 1

        # Objective W-GP-UCB scores
        ucb_f_np, var_np = _rff_ucb_all(
            xx_np, rf, nu_t, Sigma_t_inv, beta_t, M_rff, lam=lam_rff, mode=mode
        )
        ucb_f = torch.as_tensor(ucb_f_np, dtype=torch.float64)

        # Constraint GP
        mu_c, sig_c, lcb_c, ucb_c = compute_lcb_ucb(c_model, X_grid, beta=beta_c)

        available_mask = torch.ones(N, dtype=torch.bool)
        if not allow_repeated_queries:
            available_mask[queried_idx] = False

        verified_safe = (lcb_c >= 0.0) & available_mask
        safe_set_size_hist.append(int(verified_safe.sum().item()))

        if not available_mask.any():
            break

        if verified_safe.sum() == 0:
            fallback = lcb_c.clone()
            fallback[~available_mask] = -1e18
            next_idx = int(torch.argmax(fallback).item())
        else:
            obj_score = ucb_f
            a_bnd     = -torch.abs(mu_c / (sig_c + 1e-9))
            lam_t     = lambda_by_stage(t, lam0=lam0, t0=lam_t0, p=lam_p)

            obj_n = minmax_norm(obj_score, verified_safe)
            bnd_n = minmax_norm(a_bnd, verified_safe)
            score = (1.0 - lam_t) * obj_n + lam_t * bnd_n
            score[~verified_safe] = -1e18

            next_idx = int(torch.argmax(score).item())

        # --- Compute eps (same as quantum_safe_bo.py) ---
        phi_star = _rff_phi(xx_np[next_idx], rf, M_rff).reshape(-1, 1)
        var_star = float(lam_rff * (phi_star.T @ Sigma_t_inv @ phi_star))
        eps_star = math.sqrt(max(var_star, 1e-12)) / math.sqrt(lam_rff)
        eps_star = max(eps_star, 1e-6)

        # ---------------------------------------------------------------
        # QUERY STEP — Classical Chebyshev MC Estimation
        # (THE ONLY DIFFERENCE FROM quantum_safe_bo.py)
        # ---------------------------------------------------------------
        delta = 0.05
        var_true = obj_noise_std ** 2
        n_required = math.ceil((var_true / (eps_star**2 + 1e-12)) * (1.0 / delta))
        n_required = min(n_required, 4999)
        n_required = max(n_required, 10)

        samples = [float(np.random.normal(yy[next_idx], obj_noise_std))
                   for _ in range(n_required)]
        y_new = sum(samples) / n_required
        n_oracle = n_required

        c_new = float(np.random.normal(zz[next_idx] - xi, con_noise_std))

        # --- Strict budget enforcement ---
        remaining_budget = n_iter - total_oracle_queries
        if n_oracle > remaining_budget:
            n_oracle = remaining_budget

        # --- Record oracle usage ---
        total_oracle_queries += n_oracle
        track_queries.append(n_oracle)
        cumulative_oracle_hist.append(total_oracle_queries)

        # --- Update tensors ---
        train_X = torch.cat([train_X, X_grid[next_idx].view(1, -1)], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor([[y_new]], dtype=torch.float64)], dim=0)
        c_train_X = torch.cat([c_train_X, X_grid[next_idx].view(1, -1)], dim=0)
        c_train_C = torch.cat([c_train_C, torch.tensor([[c_new]], dtype=torch.float64)], dim=0)
        queried_idx.append(next_idx)
        queried_c_obs.append(c_new)

        # --- Update eps_list ---
        eps_new = obj_noise_std / math.sqrt(n_required)
        eps_list = np.append(eps_list, eps_new)

        # --- Rebuild weighted Phi and Sigma_t ---
        eps_weights = 1.0 / eps_list

        Phi_obj = np.stack([
            _rff_phi(xx_np[queried_idx[i]], rf, M_rff) * eps_weights[i]
            for i in range(len(queried_idx))
        ])

        Y_obs_np = np.append(Y_obs_np, y_new)

        Sigma_t     = Phi_obj.T @ Phi_obj + lam_rff * np.eye(M_rff)
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        Y_weighted  = (Y_obs_np * eps_weights).reshape(-1, 1)
        nu_t        = Sigma_t_inv @ Phi_obj.T @ Y_weighted

        # Constraint GP update
        c_model = update_gp_model_fixed_params(c_model, c_train_X, c_train_C, noise_var=con_noise_std**2)

        # --- Regret ---
        f_query = yy[next_idx]
        r_t = abs(global_safe_opt - f_query) if mode == "max" else abs(f_query - global_safe_opt)
        queried_instant_regret_hist.append(r_t)
        queried_cumu_regret_hist.append(np.sum(queried_instant_regret_hist))

        # --- Recommendation ---
        rff_mean_np, _ = _rff_ucb_all(xx_np, rf, nu_t, Sigma_t_inv, 0.0, M_rff, lam_rff, mode="max")
        rff_mean = torch.as_tensor(rff_mean_np, dtype=torch.float64)
        mu_c_now, _, lcb_c_now, _ = compute_lcb_ucb(c_model, X_grid, beta=beta_c)
        safe_now = lcb_c_now >= 0.0
        if safe_now.any():
            rec_idx = (torch.argmax if mode == "max" else torch.argmin)(
                torch.where(safe_now, rff_mean,
                            torch.tensor(-1e18 if mode == "max" else 1e18, dtype=torch.float64))
            ).item()
        else:
            rec_idx = torch.argmax(lcb_c_now).item()
        recommended_idx_hist.append(rec_idx)

        # --- Best safe observed ---
        obs_safe_mask = torch.tensor(queried_c_obs, dtype=torch.float64) >= 0.0
        if obs_safe_mask.any():
            yy_queried = torch.tensor([yy[i] for i in queried_idx], dtype=torch.float64)
            bsv = yy_queried[obs_safe_mask].max().item() if mode == "max" else yy_queried[obs_safe_mask].min().item()
        else:
            bsv = np.nan
        best_safe_value_hist.append(bsv)

        print(
            f"Iter {t:04d} | oracle={total_oracle_queries}/{n_iter} | "
            f"n_q={n_oracle} | eps={eps_star:.4f} | "
            f"x={xx[next_idx]} | y_obs={y_new:.4f} | f_true={yy[next_idx]:.4f} | h_true={zz[next_idx]:.4f} | "
            f"safe={int(verified_safe.sum())}/{N}"
        )

    # ------------------------------------------------------------------
    # Final recommendation
    # ------------------------------------------------------------------
    if recommended_idx_hist:
        final_rec_idx = int(recommended_idx_hist[-1])
    else:
        rff_final, _ = _rff_ucb_all(xx_np, rf, nu_t, Sigma_t_inv, 0.0, M_rff, lam_rff)
        final_rec_idx = int(np.argmax(rff_final))

    final_rec_x    = xx[final_rec_idx]
    final_rec_f    = yy[final_rec_idx]
    final_rec_h    = zz[final_rec_idx]
    final_rec_safe = final_rec_h >= xi
    simple_regret  = (
        (global_safe_opt - final_rec_f) if mode == "max" else (final_rec_f - global_safe_opt)
    ) if final_rec_safe else np.nan

    # ------------------------------------------------------------------
    # Oracle-expanded arrays  (same as quantum_safe_bo.py)
    # ------------------------------------------------------------------
    bo_track = track_queries[n_init:]

    def _expand(arr):
        out = []
        for val, reps in zip(arr, bo_track):
            out.extend([val] * int(reps))
        return np.array(out)

    f_values_per_step  = yy[queried_idx[n_init:]]
    f_values_expanded  = _expand(f_values_per_step)
    regret_expanded    = _expand(queried_instant_regret_hist)
    cumu_regret_expanded = np.nancumsum(regret_expanded)
    best_safe_expanded = _expand(best_safe_value_hist[1:])

    with torch.no_grad():
        mu_c_final, _, lcb_c_final, _ = compute_lcb_ucb(c_model, X_grid, beta=beta_c)

    return {
        "environment"                : env["environment"],
        "xx"                         : xx,
        "yy"                         : yy,
        "zz"                         : zz,
        "xi"                         : xi,
        "safe_true_mask"             : safe_true_mask,
        "has_true_safe_points"       : has_true_safe_points,
        "init_idx"                   : np.asarray(init_idx, dtype=int),
        "train_X"                    : train_X.detach().cpu().numpy(),
        "train_Y"                    : train_Y.detach().cpu().numpy(),
        "c_train_X"                  : c_train_X.detach().cpu().numpy(),
        "c_train_C"                  : c_train_C.detach().cpu().numpy(),
        "queried_idx"                : queried_idx,
        "best_safe_value_hist"       : np.array(best_safe_value_hist),
        "recommended_idx_hist"       : np.array(recommended_idx_hist),
        "safe_set_size_hist"         : np.array(safe_set_size_hist),
        "global_safe_opt"            : global_safe_opt,
        "final_rec_idx"              : final_rec_idx,
        "final_rec_x"                : final_rec_x,
        "final_rec_f"                : final_rec_f,
        "final_rec_h"                : final_rec_h,
        "final_rec_safe"             : final_rec_safe,
        "simple_regret"              : simple_regret,
        "mode"                       : mode,
        "n_init"                     : n_init,
        "init_percentage"            : init_percentage,
        "allow_repeated_queries"     : allow_repeated_queries,
        "queried_instant_regret_hist": np.array(queried_instant_regret_hist),
        "queried_cumu_regret_hist"   : np.array(queried_cumu_regret_hist),
        "f_values_expanded"          : f_values_expanded,
        "regret_expanded"            : regret_expanded,
        "cumu_regret_expanded"       : cumu_regret_expanded,
        "best_safe_expanded"         : best_safe_expanded,
        "track_queries"              : np.array(track_queries),
        "cumulative_oracle_hist"     : np.array(cumulative_oracle_hist),
        "total_oracle_queries"       : total_oracle_queries,
        "init_mu_c"  : init_mu_c.detach().cpu().numpy(),
        "init_lcb_c" : init_lcb_c.detach().cpu().numpy(),
        "final_mu_c" : mu_c_final.detach().cpu().numpy(),
        "final_lcb_c": lcb_c_final.detach().cpu().numpy(),
    }
