from datetime import datetime
import os
from os import path
from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel, LinearKernel, RBFKernel
from ansys.mapdl.core import launch_mapdl, launcher, Mapdl
from ansys.mapdl import reader as mapdl_reader
from shutil import copyfile
import torch
from botorch.models.transforms.input import Normalize
from bo_env_constraints import ClassicFuselageEnv

from utils import build_train_gp_with_rff, sample_actions
from botorch.models.transforms import Normalize, Standardize
from botorch.generation.gen import gen_candidates_scipy, gen_candidates_torch, TGenCandidates
import botorch
import numpy as np
import warnings
import argparse
import random
import joblib
warnings.filterwarnings("ignore")

def nnz_count(F, tol=1e-2):
    return int(np.sum(np.abs(F) > tol))

def algorithm2_search_lambda(
    U, B, psi,
    M,                      # target nnz in F
    rho=1.0,
    F_lower=None,
    F_upper=None,
    e3=1e-4,
    e4=1e-2,
    max_admm_iter=1000,
    max_bs_iter=40,
    zero_tol=1e-2,
    lam_min=0.0,
    lam_max=None,
    expand_factor=2.0,
    expand_steps=40,
    verbose=False,
):
    """
    Algorithm 2: binary search λ so that ||F||_0 ≈ M, using Algorithm 1 (ADMM) as subroutine.
    Returns: lam_best, F_best, (z_best,u_best), info
    """

    U = np.asarray(U, float)
    B = np.asarray(B, float)
    psi = np.zeros(U.shape[0]) if psi is None else np.asarray(psi, float).reshape(-1)

    # ---- init λ_max ----
    if lam_max is None:
        # paper init: λ_max = ||2 U^T B ψ||_∞
        lam_max = np.max(np.abs(2.0 * (U.T @ (B @ psi))))
        if lam_max == 0.0:
            lam_max = 1e-6  # if psi=0, paper init degenerates

    # ADMM warm-start state
    warm = {"F": None, "z": None, "u": None}

    def solve(lam):
        F, z, u, hist = admm_estimate_F(
            U=U, B=B, psi=psi,
            rho=rho, lam=lam,
            F_lower=F_lower, F_upper=F_upper,
            e3=e3, e4=e4,
            max_iter=max_admm_iter,
            F0=warm["F"], z0=warm["z"], u0=warm["u"],
            verbose=False
        )
        F = np.where(np.abs(F) > zero_tol, F, 0.0)
        # update warm-start for next call
        warm["F"], warm["z"], warm["u"] = F, z, u
        return F, z, u, hist

    # ---- bracket so that nnz(lam_min) >= M and nnz(lam_max) <= M ----
    F0, z0, u0, _ = solve(lam_min)
    nnz0 = nnz_count(F0, tol=zero_tol)
    if verbose:
        print(f"[init] lam_min={lam_min:.3e}, nnz={nnz0}")

    Fh, zh, uh, _ = solve(lam_max)
    nnzh = nnz_count(Fh, tol=zero_tol)
    if verbose:
        print(f"[init] lam_max={lam_max:.3e}, nnz={nnzh}")

    # expand lam_max until sparse enough
    steps = 0
    while nnzh > M and steps < expand_steps:
        lam_max *= expand_factor
        Fh, zh, uh, _ = solve(lam_max)
        nnzh = nnz_count(Fh, tol=zero_tol)
        steps += 1
        if verbose:
            print(f"[expand] lam_max={lam_max:.3e}, nnz={nnzh}")

    if nnzh > M:
        return None, Fh, (zh, uh), {
            "status": "failed_to_bracket",
            "message": "Could not find λ_max making nnz(F) <= M. M may be infeasible.",
            "lam_max": lam_max,
            "nnz_at_lam_max": nnzh
        }

    if nnz0 < M:
        return None, F0, (z0, u0), {
            "status": "infeasible_target",
            "message": "At λ=0 already nnz(F) < M; cannot reach M by increasing λ.",
            "nnz_at_lam0": nnz0
        }

    # ---- binary search ----
    best = {"lam": lam_min, "F": F0, "z": z0, "u": u0, "nnz": nnz0, "diff": abs(nnz0 - M)}

    for it in range(max_bs_iter):
        lam = 0.5 * (lam_min + lam_max)
        F, z, u, _ = solve(lam)
        nnz = nnz_count(F, tol=zero_tol)
        diff = abs(nnz - M)

        if verbose:
            print(f"[bs {it:02d}] lam={lam:.6e}, nnz={nnz}")

        if diff < best["diff"]:
            best = {"lam": lam, "F": F, "z": z, "u": u, "nnz": nnz, "diff": diff}

        if nnz == M:
            return lam, F, (z, u), {"status": "success_exact", "iters": it + 1, "nnz": nnz}

        # Algorithm 2 decision:
        if nnz < M:
            lam_max = lam      # too sparse -> λ too big
        else:
            lam_min = lam      # too dense  -> λ too small

    return best["lam"], best["F"], (best["z"], best["u"]), {
        "status": "success_best_effort",
        "iters": max_bs_iter,
        "nnz": best["nnz"],
        "diff": best["diff"]
    }
    


def soft_threshold(v, kappa):
    """
    S_{kappa}(v) = sign(v) * max(|v| - kappa, 0)
    """
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

def project_box(v, lower, upper):
    """
    Π_C(v) for box C = {F: lower <= F <= upper}
    """
    return np.minimum(np.maximum(v, lower), upper)

def admm_estimate_F(
    U, B,
    psi=None,                 # psi vector; set None or zeros for psi=0
    rho=1.0,
    lam=1e-2,                 # lambda (λ)
    F_lower=None,             # F_L (vector or scalar)
    F_upper=None,             # F_Q (vector or scalar)
    e3=1e-4,
    e4=1e-3,
    max_iter=1000,
    LN = 1e3,
    z0=None, u0=None, F0=None,
    verbose=False
):
    """
    Solves the ADMM iterations for:
        minimize_F  χ_C(F) + (psi + U F)^T B (psi + U F) + λ ||z||_1
        s.t.         F - z = 0
    using the update equations in your screenshots:
        F^{k+1} = Π_C( (2 U^T B U + ρ I)^{-1} (ρ z^k - ρ u^k - 2 U^T B ψ) )
        z^{k+1} = S_{λ/ρ}(F^{k+1} + u^k)
        u^{k+1} = u^k + F^{k+1} - z^{k+1}

    Returns: F, z, u, history dict
    """

    U = np.asarray(U, dtype=float)
    B = np.asarray(B, dtype=float)

    # Dimensions: U is (n, m), F is (m,)
    n, m = U.shape
    I = np.eye(m)

    if psi is None:
        psi = np.zeros(n, dtype=float)
    else:
        psi = np.asarray(psi, dtype=float)
        assert psi.shape == (n,), f"psi must be shape ({n},), got {psi.shape}"

    # Bounds
    if F_lower is None:
        F_lower = -np.inf * np.ones(m)
    elif np.isscalar(F_lower):
        F_lower = float(F_lower) * np.ones(m)
    else:
        F_lower = np.asarray(F_lower, dtype=float).reshape(-1)
        assert F_lower.shape == (m,)

    if F_upper is None:
        F_upper = np.inf * np.ones(m)
    elif np.isscalar(F_upper):
        F_upper = float(F_upper) * np.ones(m)
    else:
        F_upper = np.asarray(F_upper, dtype=float).reshape(-1)
        assert F_upper.shape == (m,)

    # Initialize
    if z0 is None: z = np.zeros(m)
    else: z = np.asarray(z0, dtype=float).reshape(m)

    if u0 is None: u = np.zeros(m)
    else: u = np.asarray(u0, dtype=float).reshape(m)

    if F0 is None: F = np.zeros(m)
    else: F = np.asarray(F0, dtype=float).reshape(m)

    # Precompute constant pieces
    # A = (2 U^T B U + rho I)
    A = 2.0 * LN * (U.T @ (B @ U)) + rho * I
    c = -2.0 * LN * (U.T @ (B @ psi))

    # For speed/stability: solve linear systems with np.linalg.solve
    # If A is ill-conditioned, consider scipy.linalg.cho_factor/cho_solve if SPD.

    history = {
        "r_norm": [],
        "s_norm": [],
        "e1": [],
        "e2": [],
    }
    lam_scaled = lam / LN
    kappa = lam_scaled / rho

    for k in range(max_iter):
        z_prev = z.copy()

        # ---- (4) F-update ----
        # v = A^{-1} (rho z^k - rho u^k + c)
        rhs = rho * z - rho * u + c
        v = np.linalg.solve(A, rhs)
        F = project_box(v, F_lower, F_upper)

        # ---- (5) z-update ----
        z = soft_threshold(F + u, kappa)

        # ---- (6) u-update ----
        u = u + (F - z)

        # ---- (7)(8) residuals ----
        r = F - z
        s = rho * (z - z_prev)

        r_norm = np.linalg.norm(r, 2)
        s_norm = np.linalg.norm(s, 2)

        # ---- (9)(10) tolerances ----
        # e1 = sqrt(m)*e3 + e4 * max(||z||2, ||F||2)
        # e2 = sqrt(m)*e3 + e4 * ||rho*u||2
        e1 = np.sqrt(m) * e3 + e4 * max(np.linalg.norm(z, 2), np.linalg.norm(F, 2))
        e2 = np.sqrt(m) * e3 + e4 * np.linalg.norm(rho * u, 2)

        history["r_norm"].append(r_norm)
        history["s_norm"].append(s_norm)
        history["e1"].append(e1)
        history["e2"].append(e2)

        if verbose and (k % 10 == 0 or k == max_iter - 1):
            print(f"iter {k:4d} | r={r_norm:.3e} (<= {e1:.3e}) | s={s_norm:.3e} (<= {e2:.3e})")

        # ---- (11) stopping ----
        if (r_norm <= e1) and (s_norm <= e2):
            break

    return F, z, u, history



if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    botorch.settings.debug(True)
    parser = argparse.ArgumentParser(description='Example of using argparse to adjust parameters.')
    parser.add_argument('--M_features', type=int, default=1024, help='Number of features used in RFF.')
    parser.add_argument('--max_iteration', type=int, default=10000, help='Maximum iterations.')
    parser.add_argument('--obs_noise', type=float, default=0.2**2, help='noise level of observation') #0.05 and 0.1
    parser.add_argument('--eta', type=float, default=1, help='precision trade-off parameters')
    parser.add_argument('--lam', type=float, default=0.5, help='lambda hyperparameters')   
    parser.add_argument('--B', type=float, default=1, help='exploit hyperparameters')
    args = parser.parse_args()
    seed = 0
    random.seed(seed)
    __file__ = 'FuselageActuators'
    folder = path.join(__file__, 'AnsysFiles', "Test") 
    file = random.choice(os.listdir(folder))
        
    filepath = path.join(folder, file)
    original_input_filename = filepath
    print("Initial shape from", file.split('.')[0])
    ###

    log_dir = "Experiments"
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    env_name = "Classic_EXP_ADMM"
    log_dir = log_dir + '/' + env_name + '/' + 'exp_set_' + str(0) + '/'
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    input_filename = log_dir + file
    copyfile(original_input_filename, input_filename)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    obs_noise = 0.1 ** 2
    env = ClassicFuselageEnv(ip='129.161.91.97', obs_noise=obs_noise)  
    psi = env.reset(input_filename, seed=seed) #354
    
    #### create new log file for each run
    U = joblib.load('surrogate_likeDu_v22.joblib').coef_ # 354,18
    
    print(U)
    ####
    B = np.identity(354)
    m = U.shape[1]
    F_lower = -200 * np.ones(m)   # example
    F_upper =  200 * np.ones(m)   # example
    Ln = 1e3
    M_target = 10
    lam_star, F_star, (z_star, u_star), info = algorithm2_search_lambda(
        U=U, B=B, psi=psi,    # psi can be None/zeros if you want
        M=M_target,
        rho=1.0,
        F_lower=F_lower, F_upper=F_upper,   # set your bounds
        zero_tol=1e-2,
        verbose=True
    )

    print(info)
    print("lambda* =", lam_star)
    print("nnz(F*) =", np.sum(np.abs(F_star) > 1e-2))
    print(",".join(map(str, np.array(F_star).ravel())))