import torch
import numpy as np
import warnings
import copy
import matplotlib.pyplot as plt
import math

from types import SimpleNamespace
import gpytorch

try:
    from .experiment_env import (
        build_paper_sim2_environment,
        compute_global_safe_optimum,
    )
except ImportError:
    from experiment_env import (
        build_paper_sim2_environment,
        compute_global_safe_optimum,
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
    from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel

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

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import RBFKernel, ScaleKernel

def update_constraint_model_fixed_params(
    base_model,
    base_likelihood,
    train_X,
    train_C,
    training_iterations=20,
    lr=0.1,
):
    """
    Rebuild classifier on new train_X, copy kernel/mean hyperparameters from base_model,
    freeze them, and only update variational posterior.
    """
    new_model = GPClassificationModel(train_X).to(train_X.device)
    new_likelihood = BernoulliLikelihood().to(train_X.device)

    # copy learned kernel / mean params
    new_model.covar_module.load_state_dict(base_model.covar_module.state_dict())
    new_model.mean_module.load_state_dict(base_model.mean_module.state_dict())
    new_likelihood.load_state_dict(base_likelihood.state_dict())

    # freeze mean/kernel hyperparameters
    for param in new_model.covar_module.parameters():
        param.requires_grad = False
    for param in new_model.mean_module.parameters():
        param.requires_grad = False

    new_model.train()
    new_likelihood.train()

    optimizer = torch.optim.Adam(
        [{'params': new_model.variational_strategy.parameters()}],
        lr=lr
    )

    mll = VariationalELBO(new_likelihood, new_model, train_C.numel())
    targets = train_C.view(-1).to(torch.float)

    for _ in range(training_iterations):
        optimizer.zero_grad()
        output = new_model(train_X)
        loss = -mll(output, targets)
        loss.backward()
        optimizer.step()

    return new_model, new_likelihood

# 1. Define the GP Classifier Structure
class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# 2. Define the Training Function
def fit_constraint_model(train_X, train_C, training_iterations=80, lr=0.1):
    """
    Train GP classifier on binary labels train_C in {0,1}.

    Kernel hyperparameters (lengthscale, outputscale) are fixed to
    constant values (ls=0.5, sigma^2=1.0) — same convention as Safe BO
    and Quantum Safe BO.  Only variational distribution parameters are
    optimised via the ELBO.
    """
    model = GPClassificationModel(train_X).to(train_X.device)
    likelihood = BernoulliLikelihood().to(train_X.device)

    # --- Fix kernel hyperparameters (no gradient) ---
    with torch.no_grad():
        model.covar_module.base_kernel.lengthscale = torch.tensor(
            0.5, dtype=train_X.dtype, device=train_X.device
        )
        model.covar_module.outputscale = torch.tensor(
            1.0, dtype=train_X.dtype, device=train_X.device
        )
    for param in model.covar_module.parameters():
        param.requires_grad = False
    for param in model.mean_module.parameters():
        param.requires_grad = False

    model.train()
    likelihood.train()

    # Only optimise the variational parameters
    optimizer = torch.optim.Adam(
        [{'params': model.variational_strategy.parameters()}], lr=lr
    )
    mll = VariationalELBO(likelihood, model, train_C.numel())

    targets = train_C.view(-1).to(torch.float)

    for _ in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, targets)
        loss.backward()
        optimizer.step()

    return model, likelihood

# 3. Define the Prediction Function
@torch.no_grad()
def get_constraint_probabilities(model, likelihood, test_X):
    model.eval()
    likelihood.eval()

    latent_dist = model(test_X)
    pred_dist = likelihood(latent_dist)
    return pred_dist.mean.view(-1)   # p(c=1|x)

def resolve_n_init(total_points, n_init=10, init_percentage=None):
    """
    Resolve the number of initial points from either an absolute count or
    a percentage of the full grid.

    init_percentage supports either:
    - 0 < p <= 1: fraction form, e.g. 0.05 means 5%
    - 1 < p <= 100: percentage form, e.g. 5 means 5%
    """
    if init_percentage is None:
        if n_init is None:
            raise ValueError("Provide either n_init or init_percentage.")
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
# 2. Gaussian Process Utilities
# =========================================================
def fit_gp_model(train_X, train_Y, noise_var=1e-4, covar_module=None):
    """Standard GP setup. Accepts an optional covar_module to override the default Matern52."""
    # Deduplicate points to prevent Cholesky matrix decomposition errors
    noise_var = max(float(noise_var), 1e-4)
    X_rounded = torch.round(train_X * 1e5) / 1e5
    unique_X, inverse = torch.unique(X_rounded, dim=0, return_inverse=True)
    unique_Y = torch.zeros(unique_X.shape[0], 1, dtype=train_Y.dtype, device=train_Y.device)
    counts = torch.zeros(unique_X.shape[0], dtype=train_Y.dtype, device=train_Y.device)
    for i in range(train_X.shape[0]):
        unique_Y[inverse[i]] += train_Y[i]
        counts[inverse[i]] += 1
    unique_Y = unique_Y / counts.unsqueeze(1)

    yvar = torch.full_like(unique_Y, noise_var)
    if BOTORCH_AVAILABLE:
        if covar_module is None:
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=train_X.shape[-1]))
        model = _build_gp(unique_X, unique_Y, yvar, covar_module=covar_module)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception as err:
            warnings.warn(f"GP fitting failed; using default hyperparameters. Error: {err}")
    else:
        model = _SklearnGPWrapper(unique_X, unique_Y, noise_var=noise_var)
    model.eval()
    model.likelihood.eval()
    return model


def update_gp_model_fixed_params(base_model, train_X, train_Y, noise_var=1e-4):
    """Rebuild GP on new data while keeping learned hyperparameters fixed."""
    if not BOTORCH_AVAILABLE:
        return fit_gp_model(train_X, train_Y, noise_var=noise_var)
    noise_var = max(float(noise_var), 1e-4)
    X_rounded = torch.round(train_X * 1e5) / 1e5
    unique_X, inverse = torch.unique(X_rounded, dim=0, return_inverse=True)
    unique_Y = torch.zeros(unique_X.shape[0], 1, dtype=train_Y.dtype, device=train_Y.device)
    counts = torch.zeros(unique_X.shape[0], dtype=train_Y.dtype, device=train_Y.device)
    for i in range(train_X.shape[0]):
        unique_Y[inverse[i]] += train_Y[i]
        counts[inverse[i]] += 1
    unique_Y = unique_Y / counts.unsqueeze(1)

    yvar = torch.full_like(unique_Y, noise_var)
    model = _build_gp(
        unique_X,
        unique_Y,
        yvar,
        covar_module=copy.deepcopy(base_model.covar_module),
        mean_module=copy.deepcopy(base_model.mean_module) if hasattr(base_model, 'mean_module') else None,
    )
    model.eval()
    model.likelihood.eval()
    return model



@torch.no_grad()
def get_gp_predictions(model, X):
    post = model.posterior(X)
    return post.mean.view(-1), post.variance.view(-1).clamp_min(1e-12).sqrt()

def norm_01(t):
    """Eq (11): Normalizes a vector to [0, 1]"""
    if t.max() > t.min():
        return (t - t.min()) / (t.max() - t.min() + 1e-12)
    return torch.zeros_like(t)

# =========================================================
# 3. Main BO-ACL Framework (Algorithms 1, 2, & 3)
# =========================================================
def _best_feasible_value(yy, queried_idx, C_true, mode):
    feasible_indices = [idx for idx in queried_idx if C_true[idx].item() == 1.0]
    if len(feasible_indices) == 0:
        return np.nan
    feasible_yy = [yy[idx] for idx in feasible_indices]
    if mode == "min":
        return min(feasible_yy)
    else:
        return max(feasible_yy)


def _run_bo_acl_paper_simulation2(
    grid_res=40,
    grid_size=None,           # alias for grid_res (shared interface)
    n_init=None,
    init_percentage=None,
    n_batches=15,
    batch_size=2,
    n_iter=None,              # alias: sets n_batches=n_iter, batch_size=1
    seed=42,
    allow_repeated_queries=True,
    x1_bounds=(-1.0, 1.0),
    x2_bounds=(-1.0, 1.0),
    # Shared-interface parameters
    mode="min",               # 'min' or 'max'
    xi=0.25,                   # constraint threshold h(x) >= xi is feasible
    obj_noise_std=0.3,        # std of Gaussian noise on objective observations
    con_noise_std=1e-6,       # std of Gaussian noise on constraint observations
    beta_f=1.0,               # objective exploration weight (unused internally, kept for API compat)
    beta_w=0.2,               # boundary exploration weight (maps to alpha_0)
    init_idx=None,            # pre-computed shared initial indices
    lengthscale=1.0,          # GP lengthscale (unused internally, kept for API compat)
):
    """
    Paper Simulation 2 on the discrete [-1, 1]^2 grid.

    By default this follows the paper-style setup of using 10% of the grid
    as the initial labeled dataset.

    Extended to accept the shared interface used by compare_safe_methods.py
    (mode, xi, grid_size, n_iter, noise stds, shared init_idx) while keeping
    the BO-ACL algorithm logic identical.
    """
    # ------------------------------------------------------------------
    # Resolve aliases
    # ------------------------------------------------------------------
    if grid_size is not None:
        grid_res = grid_size
    if n_iter is not None:
        n_batches = n_iter // batch_size

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = build_paper_sim2_environment(
        grid_size=grid_res,
        xi=xi,
        dtype=torch.float64,
        constraint_representation="binary",
        x1_bounds=x1_bounds,
        x2_bounds=x2_bounds,
    )
    xx_np  = np.asarray(env["xx"])   # [N, 2] numpy  (for return dict)
    yy     = env["yy"]               # noise-free objective [N] numpy
    zz     = env["zz"]               # constraint values    [N] numpy
    X_grid = env["X_grid"]
    Y_true = env["Y_true"]
    Z_true = env["Z_true"]
    C_true = env["C_true"]
    
    # ------------------------------------------------------------------
    # Initial dataset
    # ------------------------------------------------------------------
    n_init = resolve_n_init(len(X_grid), n_init=n_init)
    if init_idx is not None:
        # Use the externally supplied (shared) initial indices
        init_idx = np.asarray(init_idx, dtype=int).tolist()
        n_init   = len(init_idx)
    else:
        init_idx = np.random.choice(len(X_grid), size=n_init, replace=False).tolist()

    train_X = X_grid[init_idx].clone()
    # Noisy observations (obj_noise_std=0 reproduces the original noise-free behaviour)
    noise_Y = torch.randn(n_init, 1, dtype=torch.float64) * obj_noise_std
    train_Y = Y_true[init_idx].clone() + noise_Y
    train_C = C_true[init_idx].clone()   # binary {0,1}, no noise needed
    queried_idx = init_idx.copy()

    c_model, c_likelihood = fit_constraint_model(train_X, train_C, training_iterations=80)
    init_p_feasible = get_constraint_probabilities(c_model, c_likelihood, X_grid)
    initial_feasible_mask = (train_C.view(-1) == 1.0)
    if initial_feasible_mask.sum() == 0:
        f_model = fit_gp_model(train_X, train_Y)
        f_model_m = None
    else:
        X_m0 = train_X[initial_feasible_mask]
        Y_m0 = train_Y[initial_feasible_mask]
        f_model_m = fit_gp_model(X_m0, Y_m0)
        f_model = f_model_m
    
    # Hyperparameters from paper (Simulation 2)
    alpha_0, beta_div, eps = 0.6, 0.3, 0.7
    gamma_ucb = np.sqrt(2) * beta_f  # Controls LCB/UCB exploration

    # Ground-truth safe optimum (noise-free)
    safe_mask_true = (zz >= xi)
    if safe_mask_true.any():
        yy_safe = yy[safe_mask_true]
        global_feasible_opt = float(yy_safe.min() if mode == "min" else yy_safe.max())
    else:
        global_feasible_opt = float(Y_true[C_true.view(-1) == 1.0].min().item())

    best_feasible_value_hist   = []
    simple_regret_hist         = []
    batch_history              = []
    safe_set_size_hist         = []   # |Omega_f| per oracle call  (new)
    queried_instant_regret_hist = []  # |f(x_t) - opt|             (new)
    queried_cumu_regret_hist   = []   # cumulative sum of above     (new)

    print("Starting BO-ACL...")
    for t in range(n_batches):
        alpha_t = alpha_0 * (eps ** t)
        
        # -------------------------------------------------------------
        # ALGORITHM 1 PART 1: Constraint Model & Feasible Region
        # -------------------------------------------------------------
        # Train GP Classifier proxy on {0, 1} labels
        c_model, c_likelihood = update_constraint_model_fixed_params(
            c_model,
            c_likelihood,
            train_X,
            train_C,
            training_iterations=20,
        )
        p_feasible = get_constraint_probabilities(c_model, c_likelihood, X_grid)
        
        # Eq (7): Prediction Uncertainty S_u(x) 
        # S_u = 1 - |p(feasible) - p(infeasible)| = 1 - |p - (1-p)| = 1 - |2p - 1|
        S_u_raw = 1.0 - torch.abs(2.0 * p_feasible - 1.0)
        
        # Predicted feasible region \Omega_f
        omega_f_mask = p_feasible >= 0.5
        
        # -------------------------------------------------------------
        # ALGORITHM 2: Objective Optimization loop with Pseudo-Labeling
        # -------------------------------------------------------------
        feasible_mask = (train_C.view(-1) == 1.0)

        if feasible_mask.sum() == 0:
            # no feasible samples yet: fit/update objective model on all observed data
            if f_model is None:
                f_model = fit_gp_model(train_X, train_Y)
            else:
                f_model = update_gp_model_fixed_params(f_model, train_X, train_Y)
        else:
            X_m, Y_m = train_X[feasible_mask], train_Y[feasible_mask]
            X_u = train_X[~feasible_mask]

            # 1. objective surrogate trained only on feasible points
            if f_model_m is None:
                f_model_m = fit_gp_model(X_m, Y_m)
            else:
                f_model_m = update_gp_model_fixed_params(f_model_m, X_m, Y_m)

            # 2. pseudo-label infeasible points using feasible-only model
            if len(X_u) > 0:
                mu_u, _ = get_gp_predictions(f_model_m, X_u)
                X_aug = torch.cat([X_m, X_u], dim=0)
                Y_aug = torch.cat([Y_m, mu_u.view(-1, 1)], dim=0)

                # 3. final surrogate trained on feasible + pseudo-labeled infeasible
                if f_model is None:
                    f_model = fit_gp_model(X_aug, Y_aug)
                else:
                    f_model = update_gp_model_fixed_params(f_model, X_aug, Y_aug)
            else:
                f_model = f_model_m
                
        # 4. Formulate the valid search space (\Omega_f excluding already queried points)
        objective_mask = torch.ones(len(X_grid), dtype=torch.bool)
        if not allow_repeated_queries:
            objective_mask[queried_idx] = False
            
        # The true Safe Set: \Omega_f AND unqueried
        feasible_objective_mask = omega_f_mask & objective_mask

        if not objective_mask.any():
            break # Entire grid has been queried
            
        # 5. Acquisition strictly within \Omega_f  (LCB for min, UCB for max)
        if feasible_objective_mask.sum() == 0:
            # Fallback: explore unqueried point with highest probability of feasibility
            fallback_scores = p_feasible.clone()
            fallback_scores[~objective_mask] = -float("inf")
            idx_star = int(torch.argmax(fallback_scores).item())
        else:
            # Get objective predictions for the grid
            mu_f, sigma_f = get_gp_predictions(f_model, X_grid)
            
            if mode == "min":
                acq = mu_f - gamma_ucb * sigma_f          # LCB
                acq[~feasible_objective_mask] = float("inf")
                idx_star = int(torch.argmin(acq).item())
            else:
                acq = mu_f + gamma_ucb * sigma_f          # UCB
                acq[~feasible_objective_mask] = -float("inf")
                idx_star = int(torch.argmax(acq).item())
            
        batch_indices = [idx_star]
        
        # -------------------------------------------------------------
        # ALGORITHM 1 PART 2: Dynamic Multi-Criteria Sampling
        # -------------------------------------------------------------
        unlabeled_mask = torch.ones(len(X_grid), dtype=torch.bool)
        if allow_repeated_queries:
            unlabeled_mask[batch_indices] = False
        else:
            unlabeled_mask[queried_idx + batch_indices] = False
        U_s_indices = torch.where(unlabeled_mask)[0].tolist()
        
        for k in range(batch_size - 1):
            if len(U_s_indices) == 0: break
            
            U_s = X_grid[U_s_indices]
            
            # 1. Uncertainty Component
            S_u_bar = norm_01(S_u_raw[U_s_indices])
            
            # 2. Representativeness Component (Eq 8 & Eq 12)
            dist_matrix = torch.cdist(U_s, U_s)
            S_r_raw = dist_matrix.sum(dim=1) / max(1, len(U_s) - 1)
            # Inverse normalization: smaller avg distance = higher representativeness
            S_r_bar = 1.0 - norm_01(S_r_raw) 
            
            # 3. Diversity Component (Eq 9)
            selected_X = X_grid[queried_idx + batch_indices]
            S_d_raw, _ = torch.cdist(U_s, selected_X).min(dim=1)
            S_d_bar = norm_01(S_d_raw)
            
            # 4. Aggregated function Q(x_i) (Eq 13)
            Q = (1.0 - alpha_t - beta_div) * S_u_bar + alpha_t * S_r_bar + beta_div * S_d_bar
            
            # Select max Q, add to batch, remove from unlabeled pool
            local_idx = int(torch.argmax(Q).item())
            batch_indices.append(U_s_indices[local_idx])
            U_s_indices.pop(local_idx)

        batch_history.append(batch_indices.copy())
            
        # -------------------------------------------------------------
        # Execute Experiments & Update Dataset (Algorithm 3)
        # -------------------------------------------------------------
        for idx in batch_indices:
            noise_y = float(np.random.randn() * obj_noise_std)
            y_obs   = Y_true[idx].item() + noise_y
            train_X = torch.cat([train_X, X_grid[idx].view(1, -1)])
            train_Y = torch.cat([train_Y, torch.tensor([[y_obs]], dtype=torch.float64)])
            train_C = torch.cat([train_C, C_true[idx].view(1, -1)])
            queried_idx.append(idx)

            # per-oracle-call histories
            safe_set_size_hist.append(int(omega_f_mask.sum().item()))

            f_query = yy[idx]          # noise-free ground truth
            r_t = abs(f_query - global_feasible_opt)
            queried_instant_regret_hist.append(r_t)
            queried_cumu_regret_hist.append(float(np.sum(queried_instant_regret_hist)))

            best_val = _best_feasible_value(yy, queried_idx, C_true, mode)
            best_feasible_value_hist.append(best_val)
            if np.isnan(best_val):
                simple_regret_hist.append(np.nan)
            else:
                simple_regret_hist.append(abs(best_val - global_feasible_opt))
            
        # Logging progress — instantaneous regret at the current objective query

        print(f"Iter {t+1:02d} | Batch Selected: {batch_indices} | "
              f"x*={X_grid[idx_star].detach().cpu().numpy()} | "
              f"f_obs={Y_true[idx_star].item():.4f}")

    # Final learned feasibility map after all queried data are assimilated.
    c_model, c_likelihood = update_constraint_model_fixed_params(
        c_model,
        c_likelihood,
        train_X,
        train_C,
        training_iterations=30,
    )

    final_p_feasible = get_constraint_probabilities(c_model, c_likelihood, X_grid)

    # Final simple regret (noise-free post-init queries in the true safe set)
    _q    = np.asarray(queried_idx, dtype=int)
    _post = _q[n_init:]
    _safe_post = _post[zz[_post] >= xi] if len(_post) > 0 else np.array([], dtype=int)
    if len(_safe_post) > 0:
        best_nf = float(np.min(yy[_safe_post])) if mode == "min" else float(np.max(yy[_safe_post]))
        simple_regret_final = abs(best_nf - global_feasible_opt)
    else:
        simple_regret_final = np.nan

    return {
        # --- original keys (unchanged) ---
        "environment": env["environment"],
        "grid_res": grid_res,
        "X_grid": X_grid,
        "Y_true": Y_true,
        "Z_true": Z_true,
        "C_true": C_true,
        "init_idx": init_idx,
        "queried_idx": queried_idx,
        "batch_history": batch_history,
        "train_X": train_X,
        "train_Y": train_Y,
        "train_C": train_C,
        "best_feasible_value_hist": np.asarray(best_feasible_value_hist, dtype=float),
        "simple_regret_hist": np.asarray(simple_regret_hist, dtype=float),
        "global_feasible_opt": float(global_feasible_opt),
        "final_p_feasible": final_p_feasible,
        "seed": seed,
        "n_init": n_init,
        "init_percentage": init_percentage,
        "allow_repeated_queries": allow_repeated_queries,
        "n_batches": n_batches,
        "batch_size": batch_size,
        # --- shared-interface keys (for compare_safe_methods.py) ---
        "xx": xx_np,
        "yy": yy,
        "zz": zz,
        "xi": xi,
        "mode": mode,
        "global_safe_opt": float(global_feasible_opt),
        "simple_regret": simple_regret_final,
        "safe_set_size_hist": np.asarray(safe_set_size_hist, dtype=int),
        "queried_instant_regret_hist": np.asarray(queried_instant_regret_hist, dtype=float),
        "queried_cumu_regret_hist": np.asarray(queried_cumu_regret_hist, dtype=float),
        # Learned constraint boundary (for plotting) — p(feasible) over the grid
        "init_p_feasible_np": init_p_feasible.detach().cpu().numpy(),
        "final_p_feasible_np": final_p_feasible.detach().cpu().numpy(),
    }



def run_bo_acl_simulation2(**kwargs):
    return _run_bo_acl_paper_simulation2(**kwargs)


def plot_bo_acl_simulation2_results(result):
    grid_res = result["grid_res"]
    x_axis = np.linspace(-1.0, 1.0, grid_res)
    X1, X2 = np.meshgrid(x_axis, x_axis, indexing="ij")

    y_true = result["Y_true"].view(grid_res, grid_res).cpu().numpy()
    z_true = result["Z_true"].view(grid_res, grid_res).cpu().numpy()
    c_true = result["C_true"].view(grid_res, grid_res).cpu().numpy()
    p_feasible = result["final_p_feasible"].view(grid_res, grid_res).cpu().numpy()

    queried_idx = np.asarray(result["queried_idx"], dtype=int)
    init_idx = np.asarray(result["init_idx"], dtype=int)
    model_idx = queried_idx[len(init_idx):]

    queried_points = result["X_grid"][queried_idx].cpu().numpy()
    init_points = result["X_grid"][init_idx].cpu().numpy()
    model_points = result["X_grid"][model_idx].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    cs = ax.contourf(X1, X2, y_true, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, shrink=0.85, label="Objective f(x)")
    ax.contour(X1, X2, z_true, levels=[0.0], colors="white", linewidths=2.0, linestyles="--")
    ax.scatter(init_points[:, 0], init_points[:, 1], s=40, c="white", edgecolors="black", label="Initial data")
    if len(model_points) > 0:
        ax.scatter(model_points[:, 0], model_points[:, 1], s=32, c=np.arange(len(model_points)), cmap="plasma", label="BO-ACL queries")
    ax.set_title("Simulation 2 Objective and Queried Points")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.contourf(X1, X2, c_true, levels=[-0.1, 0.5, 1.1], cmap="Greens", alpha=0.9)
    ax.contour(X1, X2, z_true, levels=[0.0], colors="black", linewidths=2.0, linestyles="--")
    ax.scatter(queried_points[:, 0], queried_points[:, 1], s=26, c="tab:red")
    ax.set_title("True Feasible Region")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    ax = axes[1, 0]
    hm = ax.contourf(X1, X2, p_feasible, levels=np.linspace(0.0, 1.0, 21), cmap="Blues")
    fig.colorbar(hm, ax=ax, shrink=0.85, label="Predicted feasibility")
    ax.contour(X1, X2, p_feasible, levels=[0.5], colors="cyan", linewidths=2.0)
    ax.contour(X1, X2, z_true, levels=[0.0], colors="red", linewidths=1.8, linestyles="--")
    ax.scatter(queried_points[:, 0], queried_points[:, 1], s=26, c="black")
    ax.set_title("Learned Feasibility Map")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    ax = axes[1, 1]
    plot_convergence_rate_like_paper(result, ax=ax)

    fig.suptitle("BO-ACL on Paper Simulation 2", fontsize=15)
    fig.tight_layout()
    return fig


def plot_convergence_rate_like_paper(result, ax=None):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
        created_fig = True
    else:
        fig = ax.figure

    batch_size = result["batch_size"]
    regret_hist = np.asarray(result["simple_regret_hist"], dtype=float)
    batch_end_regret = regret_hist[batch_size - 1 :: batch_size]
    x_eval = np.arange(batch_size, batch_size * len(batch_end_regret) + 1, batch_size)

    plot_regret = batch_end_regret

    ax.step(
        x_eval,
        plot_regret,
        where="post",
        color="tab:blue",
        linewidth=2.5,
        label="BO-ACL Framework",
    )
    ax.plot(
        x_eval,
        plot_regret,
        linestyle="None",
        marker="s",
        markersize=4.5,
        color="tab:blue",
    )
    ax.set_xlabel("Total Number of Experiments (Evaluations)", fontsize=13, fontfamily="serif")
    ax.set_ylabel(r"Convergence Rate $r_t$", fontsize=13, fontfamily="serif")
    ax.set_xlim(0, x_eval[-1] + batch_size)
    ax.grid(True, which="major", linestyle="--", color="#b8b8b8", alpha=0.7, linewidth=0.9)
    ax.grid(True, which="minor", linestyle=":", color="#d0d0d0", alpha=0.6, linewidth=0.7)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="0.75", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("serif")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    if created_fig:
        fig.tight_layout()
        return fig, ax
    return ax

if __name__ == "__main__":
    SEED = 42
    INIT_PERCENTAGE = 10
    N_BATCHES = 20
    BATCH_SIZE = 4

    print("\n--- Running BO-ACL on Simulation 2 ---")
    result = run_bo_acl_simulation2(
        init_percentage=INIT_PERCENTAGE,
        n_batches=N_BATCHES,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    final_best = result["best_feasible_value_hist"][-1]
    final_regret = result["simple_regret_hist"][-1]
    print("\n--- Summary ---")
    print(f"Initial percentage: {INIT_PERCENTAGE}%")
    print(f"Initial points: {result['n_init']}")
    print(f"Batches: {N_BATCHES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total post-initial evaluations: {N_BATCHES * BATCH_SIZE}")
    print(f"Global feasible optimum: {result['global_feasible_opt']:.4f}")
    print(f"Best feasible value found: {final_best:.4f}")
    print(f"Final simple regret: {final_regret:.4f}")

    plot_bo_acl_simulation2_results(result)
    fig_conv, ax_conv = plt.subplots(figsize=(8, 6))
    plot_convergence_rate_like_paper(result, ax=ax_conv)
    fig_conv.tight_layout()
    plt.show()
