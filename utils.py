import torch
from typing import Optional, Tuple
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RFFKernel
import numpy as np

torch.set_default_dtype(torch.double)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_train_gp_with_rff(
    actions: torch.Tensor,                  # [N, 18]
    response: torch.Tensor,                 # [N] or [N,1]
    test_actions: Optional[torch.Tensor] = None,
    test_response: Optional[torch.Tensor] = None,
    M_target: int = 1024,                   # number of RFF features
    known_noise: Optional[float] = 1e-6,    # set None to learn noise (SingleTaskGP)
    nu: float = 2.5,                        # e.g., for Matérn(ν) approximation
    r2_threshold: float = 0.7,              # retrain until R^2 > threshold
    max_retries: int = 1,                   # number of additional re-trains allowed
    adam_steps: int = 50,                  # warmup steps per attempt
    adam_lr: float = 1e-4,                  # Adam learning rate
    seed: Optional[int] = None,             # optional seed for reproducibility
    verbose: bool = True,                   # print training logs
) -> Tuple[torch.nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Trains a GP with an RFF kernel using the training data only.

    The `test_actions` and `test_response` arguments are kept only for backward
    compatibility and are ignored. If the training-set R^2 <= r2_threshold,
    automatically retries (re-initializing the model) up to `max_retries`.
    Returns (model, lengthscale, outputscale).
    """

    # ---------- (0) Prepare tensors ----------
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    X = torch.as_tensor(actions, dtype=torch.double, device=device)
    Y = torch.as_tensor(response, dtype=torch.double, device=device)
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)
    assert X.ndim == 2 and X.shape[1] == 18, "X must be [N,18]"
    assert Y.ndim == 2 and Y.shape[1] == 1, "Y must be [N,1]"

    # ---------- (helper) Build a fresh model ----------
    def make_model() -> torch.nn.Module:
        input_dim = X.shape[-1]
        base_kernel = RFFKernel(
            num_samples=M_target,
            num_dims=input_dim,
            ard_num_dims=input_dim,
        )
        covar_module = ScaleKernel(base_kernel).to(device)
        with torch.no_grad():
            covar_module.outputscale.copy_(torch.tensor(1.0, dtype=X.dtype, device=device))

        if known_noise is not None:
            yvar = torch.full_like(Y, float(known_noise))
            mdl = FixedNoiseGP(X, Y, yvar, covar_module=covar_module).to(device)
        else:
            mdl = SingleTaskGP(X, Y, covar_module=covar_module).to(device)
        return mdl

    # ---------- (helper) One training attempt ----------
    def train_once(restart_idx: int = 0) -> Tuple[torch.nn.Module, float]:
        model = make_model()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.train(); model.likelihood.train()

        # Adam warmup
        opt = torch.optim.Adam(model.parameters(), lr=adam_lr)
        with torch.autograd.set_detect_anomaly(False), \
             torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False), \
             torch.no_grad() if adam_steps <= 0 else torch.enable_grad():
            for it in range(adam_steps):
                opt.zero_grad()
                out = model(X)
                loss = -mll(out, Y)
                if loss.dim() > 0:
                    loss = loss.sum()
                if not torch.isfinite(loss).all():
                    raise RuntimeError("Non-finite loss encountered.")
                loss.backward()
                opt.step()
                if verbose and (it + 1) % max(100, adam_steps // 3 or 1) == 0:
                    print(f"[Warmup {restart_idx}] iter {it+1}/{adam_steps} loss={loss.item():.4f}")

        # LBFGS polish (best‑effort)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            if verbose:
                print(f"[Warn] LBFGS fit failed on attempt {restart_idx}: {e}")

        model.eval(); model.likelihood.eval()

        with torch.no_grad():
            y_true = Y.squeeze(-1)
            y_pred = model.posterior(X).mean.squeeze(-1)
            mse = torch.mean((y_true - y_pred) ** 2).item()
            ss_res = torch.sum((y_true - y_pred) ** 2)
            ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
            if ss_tot.abs().item() < 1e-12:
                r2_val = torch.tensor(1.0 if ss_res.abs().item() < 1e-12 else 0.0, device=ss_res.device)
            else:
                r2_val = 1 - ss_res / ss_tot

        if verbose:
            print(f"[Attempt {restart_idx}] Fit R^2 = {float(r2_val):.4f} MSE = {mse:.4f}")
        return model, float(r2_val)

    # ---------- (helper) Read hyperparameters robustly ----------
    def read_hparams(model: torch.nn.Module) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # outputscale: always on ScaleKernel
        outputscale = model.covar_module.outputscale.detach().cpu()
        lengthscale = model.covar_module.base_kernel.lengthscale.detach().cpu()

        return lengthscale, outputscale

    # ---------- (1) Try training + conditional retries ----------
    best_model, best_r2 = None, float("-inf")
    attempts = 1 + max(0, int(max_retries))
    for k in range(attempts):
        # Different seeds across retries (if user passed seed)
        if seed is not None:
            torch.manual_seed(seed + k + 1)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + k + 1)

        try:
            model_k, r2_k = train_once(restart_idx=k)
        except Exception as e:
            if verbose:
                print(f"[Warn] RFF GP training failed on attempt {k}: {e}")
            continue

        if r2_k > best_r2:
            best_model, best_r2 = model_k, r2_k
        if r2_k >= r2_threshold:
            break

    # ---------- (2) Extract hyperparameters ----------
    if best_model is None:
        raise RuntimeError(
            f"RFF GP training failed on all {attempts} attempt(s); "
            "no model was available for hyperparameter extraction."
        )
    lengthscale, outputscale = read_hparams(best_model)

    # ---------- (3) Final logs ----------
    if verbose:
        ls_str = "None" if lengthscale is None else f"{tuple(lengthscale.shape)} tensor"
        print(f"[Done] Best R^2={best_r2:.4f} | lengthscale: {ls_str}")

    return best_model, lengthscale, outputscale

def sample_actions_variable_active(
    N=10,
    d=18,
    low=-0.5,
    high=0.5,
    k_range=(1, 18),          # number of active dims sampled uniformly from [k_min, k_max]
    active_idx_pool=None,     # allowed indices to activate; default = all 0..d-1
    seed=None,
    return_active_idx=False,
):
    """
    Sample actions in R^d where each sample can have a DIFFERENT active_idx set.

    For each sample i:
      1) draw k_i ~ UniformInt[k_min, k_max]
      2) choose k_i indices from active_idx_pool without replacement
      3) fill those dims with Uniform(low, high), others are 0

    Returns:
        actions: [N, d] numpy array
        (optional) active_idx_list: list of tuples, each is the active indices for that sample
    """
    rng = np.random.default_rng(seed)

    if active_idx_pool is None:
        active_idx_pool = np.arange(d)
    else:
        active_idx_pool = np.array(active_idx_pool, dtype=int)

    k_min, k_max = k_range
    k_min = max(1, int(k_min))
    k_max = min(len(active_idx_pool), int(k_max))
    if k_min > k_max:
        raise ValueError(f"Invalid k_range={k_range} for pool size={len(active_idx_pool)}")

    actions = np.zeros((N, d), dtype=float)
    active_idx_list = []

    for i in range(N):
        k = rng.integers(k_min, k_max + 1)  # inclusive
        idx = rng.choice(active_idx_pool, size=k, replace=False)
        actions[i, idx] = rng.uniform(low=low, high=high, size=k)
        active_idx_list.append(tuple(sorted(idx.tolist())))

    if return_active_idx:
        return actions, active_idx_list
    actions = torch.from_numpy(actions).float()
    return actions

def sample_actions(N=10, d=18, low=-0.5, high=0.5, active_idx=(0,1,2,3,14,15,16,17), seed=None):
    """
    Sample actions in R^d where only indices in active_idx are nonzero.
    
    Args:
        N: number of samples
        d: dimension (default 18)
        low, high: sampling range for active dims
        active_idx: tuple/list of indices that are allowed nonzero
        seed: optional random seed
    Returns:
        actions: [N, d] numpy array
    """
    rng = np.random.default_rng(seed)
    actions = np.zeros((N, d))
    actions[:, active_idx] = rng.uniform(low=low, high=high, size=(N, len(active_idx)))
    return actions

import torch
import numpy as np
import warnings
from scipy.stats import qmc
from sklearn.cluster import KMeans

def sample_kmeans_safe_subspace(
    n,
    tsai_wu_model,
    pool_size=10000,
    active_idx=(0, 1, 2, 3, 14, 15, 16, 17),
    d=18,
    low=-0.5,
    high=0.5,
    seed=0,
    device=None,
    dtype=torch.float64,
):
    """
    Initializes BO by generating a massive pool of LHS candidates, filtering
    out the physical failures, and clustering the surviving safe points.
    """
    device = device or torch.device("cpu")
    np.random.seed(int(seed))

    # 1. Generate a massive candidate pool in the active subspace
    sampler = qmc.LatinHypercube(d=len(active_idx), optimization="random-cd", seed=int(seed))
    lhs_unit = sampler.random(n=int(pool_size))
    active_vals = low + (high - low) * lhs_unit

    # Pad the pool to the full 18D space so the Tsai-Wu model can read it
    pool_18d = np.zeros((int(pool_size), d), dtype=np.float64)
    pool_18d[:, list(active_idx)] = active_vals

    # 2. Evaluate the constraint model in a single fast batch
    # Assuming tsai_wu_model.predict() handles 2D arrays (pool_size, 18)
    try:
        failure_indices = tsai_wu_model.predict(pool_18d, return_std=False)
    except TypeError:
        # Fallback if return_std is not accepted in batch
        failure_indices = tsai_wu_model.predict(pool_18d)
        
    # Your logic: c_label = 1.0 if (1.0 - failure_index) >= 0.0 else 0.0
    margins = 1.0 - failure_indices

    # 3. Filter for strictly safe points
    safe_mask = margins >= 0.0
    safe_active_vals = active_vals[safe_mask]
    
    # 4. Fallback Edge Case: What if the safe zone is too small?
    if len(safe_active_vals) < n:
        warnings.warn(
            f"Only found {len(safe_active_vals)} safe points out of {pool_size}. "
            f"Falling back to taking the top {n} points with the highest margins."
        )
        # Sort by margin descending (safest/least unsafe points first)
        best_indices = np.argsort(margins)[::-1][:n]
        best_active_vals = active_vals[best_indices]
        
        out = np.zeros((int(n), d), dtype=np.float64)
        out[:, list(active_idx)] = best_active_vals
        return torch.tensor(out, device=device, dtype=dtype)

    # 5. Run K-Means clustering on the safe subspace
    # n_init="auto" suppresses sklearn warnings.
    kmeans = KMeans(n_clusters=int(n), random_state=int(seed), n_init="auto")
    kmeans.fit(safe_active_vals)
    
    # The centroids act as our perfectly spaced, safe initialization points
    centroids_active = kmeans.cluster_centers_

    # 6. Reconstruct the full 18D output with the inactive dimensions as 0
    out = np.zeros((int(n), d), dtype=np.float64)
    out[:, list(active_idx)] = centroids_active

    return torch.tensor(out, device=device, dtype=dtype)



# # Load your model once
# from joblib import load
# tsai_wu_model = load(args.constraint_model)

# # Generate 30 perfectly spaced safe points
# init_tensor = sample_kmeans_safe_subspace(
#     n=30, 
#     tsai_wu_model=tsai_wu_model,
#     pool_size=10000,           # Increase this if you want an even tighter fit
#     active_idx=(0, 1, 2, 3, 14, 15, 16, 17)
# )
