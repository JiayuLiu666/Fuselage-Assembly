"""
unconstrained_bo.py
===================
Unconstrained Bayesian Optimisation on the simulated 2-D grid benchmark.

Logic follows Quantum_Bayesian_Optimization/synth/synth_bo.py exactly:

  synth_bo.py pattern
  -------------------
  1.  Load precomputed random features (s, b, obs_noise, v_kernel).
  2.  Define synth_func(param) -> (obs, f_true):
          obs      = np.random.normal(f[ind], sqrt(obs_noise))   # noisy
          f_true   = f[ind]                                       # noiseless
  3.  Set beta_t = sqrt(2) * ones(T)   (constant, NOT a schedule).
  4.  Run BO.maximize():
        a. Build Phi [n x M] from RFF features (cos projection, normalised, v_kernel-scaled).
        b. Sigma_t = Phi.T @ Phi + lam * I
        c. nu_t   = Sigma_t_inv @ Phi.T @ Y   (unweighted ridge)
        d. next_x = argmax UCB over domain
        e. Rank-1 update: Sigma_t += phi @ phi.T

  Adapted from 1-D synthetic domain → our 2-D simulated grid.
  Return dict is interface-compatible with compare_safe_methods.py.
"""

import math
import warnings

import numpy as np

warnings.filterwarnings("ignore")

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
    from .safe_set_bo import resolve_n_init, incumbent_simple_regret
except ImportError:
    from safe_set_bo import resolve_n_init, incumbent_simple_regret


# ---------------------------------------------------------------------------
# RFF helpers  (from helper_funcs_bo.py / bayesian_optimization_bo.py)
# ---------------------------------------------------------------------------

def _sample_random_features(d: int, M: int, obs_noise: float,
                              v_kernel: float = 1.0,
                              lengthscale: float = 1.0,
                              seed: int = 0) -> dict:
    """
    Pre-sample random features  s ~ N(0, I/l²),  b ~ U[0, 2π].
    Mirrors the format of random_features pkl used in synth_bo.py.
    """
    rng = np.random.default_rng(seed)
    ls = np.asarray(lengthscale, dtype=float)  # scalar or [d] array
    s = rng.standard_normal((M, d)) / ls  # broadcasts correctly for both scalar and array
    b = rng.uniform(0.0, 2 * np.pi, M)             # [M]
    return {"s": s, "b": b, "obs_noise": obs_noise, "v_kernel": v_kernel}


def _rff_phi(x: np.ndarray, rf: dict, M: int) -> np.ndarray:
    """
    φ(x) for a single point x [d].
    Mirrors bayesian_optimization_bo.py lines 95-100:
        features = sqrt(2/M) * cos(x @ s.T + b)
        features /= ||features||
        features *= sqrt(v_kernel)
    Returns [M] vector.
    """
    s = rf["s"]       # [M, d]
    b = rf["b"]       # [M]
    v = rf["v_kernel"]
    x = np.asarray(x).reshape(1, -1)
    phi = np.sqrt(2.0 / M) * np.cos(np.squeeze(x @ s.T) + b)  # [M]
    norm = np.sqrt(np.inner(phi, phi))
    if norm > 1e-12:
        phi /= norm
    return np.sqrt(v) * phi  # [M]


def _ucb_all(domain: np.ndarray, rf: dict, nu_t: np.ndarray,
              Sigma_t_inv: np.ndarray, beta: float, M: int, mode: str = "max") -> np.ndarray:
    """
    UCB score for every point in domain [N, d].
    Mirrors UtilityFunction._ucb() from helper_funcs_bo.py:
        mean = phi.T @ nu_t
        var  = lam * phi.T @ Sigma_t_inv @ phi   (lam=1 absorbed into Sigma_t_inv)
        ucb  = mean + sqrt(beta) * sqrt(var)
    """
    N = domain.shape[0]
    scores = np.empty(N)
    for i in range(N):
        phi = _rff_phi(domain[i], rf, M).reshape(-1, 1)          # [M, 1]
        mean = float(phi.T @ nu_t)
        var  = float(phi.T @ Sigma_t_inv @ phi)                   # lam=1
        if mode == "max":
            scores[i] = mean + math.sqrt(beta) * math.sqrt(max(var, 0.0))
        else:
            scores[i] = -mean + math.sqrt(beta) * math.sqrt(max(var, 0.0))
    return scores


# ---------------------------------------------------------------------------
# synth_func equivalent  (mirrors synth_bo.py lines 27-36)
# ---------------------------------------------------------------------------

def _query(idx: int, yy: np.ndarray, obs_noise_var: float,
           xi: float, zz: np.ndarray, con_noise_var: float):
    """
    Query the environment at grid index idx.
    Returns (y_obs, y_true, c_obs) — mirrors synth_func returning (obs, f_true).
        y_obs  = Normal(f_true, sqrt(obs_noise))   [noisy, as in synth_bo.py]
        y_true = yy[idx]                           [noiseless ground truth]
        c_obs  = Normal(h(x) - xi, sqrt(con_noise)) [constraint margin, logged only]
    """
    y_obs  = float(np.random.normal(yy[idx], math.sqrt(obs_noise_var)))
    y_true = float(yy[idx])
    c_obs  = float(np.random.normal(zz[idx] - xi, math.sqrt(con_noise_var)))
    return y_obs, y_true, c_obs


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_unconstrained_bo_simulation(
    mode: str = "max",
    xi: float = 0.0,               # constraint threshold — NOT used for acquisition
    grid_size: int = 50,
    x1_bounds=(-1.0, 1.0),
    x2_bounds=(-1.0, 1.0),
    n_init: int = 1,
    init_percentage=None,
    n_iter: int = 80,
    obj_noise_std: float = 1e-6,
    con_noise_std: float = 1e-6,   # kept for interface compat; not used in acq
    # algorithm settings
    M_target: int = 256,           # number of RFF features
    lam: float = 1.0,              # ridge λ
    beta_f: float = 2.0,           # sqrt(2) constant in synth_bo.py; exposed here
    lengthscale: float = 1.0,
    v_kernel: float = 1.0,
    # shared
    seed: int = 0,
    allow_repeated_queries: bool = True,
    init_idx=None,
    **kwargs,
) -> dict:
    """
    Classical GP-UCB with RFF, following synth_bo.py / bayesian_optimization_bo.py.

    Key differences from a standard GP UCB:
      * beta_t is CONSTANT = sqrt(2) * beta_f   (not a log schedule)
      * Y observations are unweighted (no eps-scaling on nu_t)
      * Rank-1 gram update (O(M²) instead of O(n M²) full rebuild)
      * Noisy observation via Normal(f_true, sqrt(obs_noise)) — like synth_bo.py
    """
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Environment  (shared 2-D grid)
    # ------------------------------------------------------------------
    env = build_paper_sim2_environment(
        grid_size=grid_size,
        xi=xi,
        dtype=None,
        constraint_representation="margin",
        x1_bounds=x1_bounds,
        x2_bounds=x2_bounds,
    )
    xx  = np.asarray(env["xx"])              # [N, 2]
    yy  = np.asarray(env["yy"])              # [N]
    zz  = np.asarray(env["zz"])              # [N]
    safe_true_mask = np.asarray(env["safe_true_mask"])   # [N] bool

    print(f"[Unconstrained BO] Grid size: {len(xx)}")
    print(f"[Unconstrained BO] Failure Region Ratio: {np.count_nonzero(zz < xi) / len(xx):.3f}")
    print(f"[Unconstrained BO] Safe points: {safe_true_mask.sum()} / {len(xx)}")

    has_true_safe_points = bool(safe_true_mask.any())
    global_safe_opt = compute_global_safe_optimum(yy, safe_true_mask, mode)
    N = len(xx)

    # ------------------------------------------------------------------
    # Initial design
    # ------------------------------------------------------------------
    n_init = resolve_n_init(N, n_init=n_init, init_percentage=init_percentage)
    if init_idx is None:
        init_idx = sample_initial_indices(N, n_init=n_init, seed=seed)
    else:
        init_idx = list(np.asarray(init_idx, dtype=int))
        if len(init_idx) != n_init:
            raise ValueError("len(init_idx) must equal resolved n_init.")

    # ------------------------------------------------------------------
    # Pre-sample random features  (done once, like synth_bo.py lines 24-25)
    # ------------------------------------------------------------------
    d = xx.shape[1]   # 2
    rf = _sample_random_features(
        d=d, M=M_target,
        obs_noise=obj_noise_std ** 2,
        v_kernel=v_kernel,
        lengthscale=lengthscale,
        seed=seed,
    )

    # beta_t = sqrt(2) constant  (synth_bo.py line 39: beta_t = sqrt(2)*ones)
    # We allow scaling via beta_f so the user can tune it.
    beta_const = math.sqrt(2.0) * beta_f

    # ------------------------------------------------------------------
    # Initialise  (mirrors BO.init())
    # ------------------------------------------------------------------
    Y_obs  = []    # noisy observations      (used for nu_t)
    Y_true = []    # noiseless ground truth   (for regret)
    C_obs  = []    # constraint margin obs    (logged, not used in acq)

    for idx in init_idx:
        y_o, y_t, c_o = _query(idx, yy, obj_noise_std**2, xi, zz, con_noise_std**2)
        Y_obs.append(y_o)
        Y_true.append(y_t)
        C_obs.append(c_o)

    Y_obs  = np.array(Y_obs)
    Y_true = np.array(Y_true)
    C_obs  = np.array(C_obs)

    # Build initial Phi [n_init, M]
    Phi = np.stack([_rff_phi(xx[idx], rf, M_target) for idx in init_idx])  # [n_init, M]

    # Sigma_t = Phi.T @ Phi + lam * I   (bayesian_optimization_bo.py line 107)
    Sigma_t     = Phi.T @ Phi + lam * np.eye(M_target)
    Sigma_t_inv = np.linalg.inv(Sigma_t)
    nu_t        = Sigma_t_inv @ Phi.T @ Y_obs.reshape(-1, 1)   # [M, 1]  unweighted

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    queried_idx                  = list(init_idx)
    best_safe_value_hist         = []
    recommended_idx_hist         = []
    safe_set_size_hist           = []
    queried_instant_regret_hist  = []
    queried_cumu_regret_hist     = []

    init_safe = C_obs >= 0.0
    if init_safe.any():
        # Use noise-free Y_true for performance metrics (not noisy Y_obs)
        bsv0 = float(Y_true[init_safe].max() if mode == "max" else Y_true[init_safe].min())
    else:
        bsv0 = float("nan")
    best_safe_value_hist.append(bsv0)

    # ------------------------------------------------------------------
    # Main BO loop  (mirrors bayesian_optimization_bo.py maximize())
    # ------------------------------------------------------------------
    for t in range(1, n_iter + 1):

        # Score all grid points with constant beta (synth_bo.py line 39)
        avail = np.ones(N, dtype=bool)
        if not allow_repeated_queries:
            avail[queried_idx] = False

        scores = _ucb_all(xx, rf, nu_t, Sigma_t_inv, beta_const, M_target, mode=mode)
        obj_scores = scores
        obj_scores[~avail] = -1e18
        safe_set_size_hist.append(N)

        next_idx = int(np.argmax(obj_scores))

        # --- Query  (mirrors synth_func: obs = Normal(f_true, sqrt(obs_noise))) ---
        y_new, y_true_new, c_new = _query(
            next_idx, yy, obj_noise_std**2, xi, zz, con_noise_std**2
        )

        # Append data
        Y_obs  = np.append(Y_obs,  y_new)
        Y_true = np.append(Y_true, y_true_new)
        C_obs  = np.append(C_obs,  c_new)
        queried_idx.append(next_idx)

        # --- Rank-1 gram update (bayesian_optimization_bo.py lines 143-148) ---
        phi_new     = _rff_phi(xx[next_idx], rf, M_target).reshape(-1, 1)  # [M, 1]
        Sigma_t     = Sigma_t + phi_new @ phi_new.T
        Phi         = np.vstack([Phi, phi_new.T])
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        nu_t        = Sigma_t_inv @ Phi.T @ Y_obs.reshape(-1, 1)   # unweighted

        # --- Regret (noise-free yy, all queries count) ---
        f_q      = yy[next_idx]
        h_q      = zz[next_idx]
        is_safe  = h_q >= xi
        # Compute regret for ALL queries against the safe optimum.
        # Unsafe queries are NOT skipped (that would give a free 0 via nansum).
        # |f(x) - f*| for every step reflects the cost of spending budget in the wrong region.
        if mode == "max":
            r_t = float(global_safe_opt - f_q)   # positive when f_q < f*
        else:
            r_t = float(f_q - global_safe_opt)   # positive when f_q > f* (higher than minimum)
        r_t = max(r_t, 0.0)                       # clamp: if query beats safe opt (unlikely for unsafe), don't go negative
        queried_instant_regret_hist.append(r_t)
        queried_cumu_regret_hist.append(float(np.sum(queried_instant_regret_hist)))

        # --- Best safe observed + recommendation ---
        # Use noise-free Y_true for performance metrics
        obs_safe = C_obs >= 0.0
        if obs_safe.any():
            safe_Y_true = Y_true[obs_safe]   # noise-free ground truth
            bsv = float(safe_Y_true.max() if mode == "max" else safe_Y_true.min())
            qs  = np.array(queried_idx)[obs_safe]
            rec_idx = int(qs[int(safe_Y_true.argmax() if mode == "max" else safe_Y_true.argmin())])
        else:
            bsv, rec_idx = float("nan"), next_idx
        best_safe_value_hist.append(bsv)
        recommended_idx_hist.append(rec_idx)

        current_sr = incumbent_simple_regret(bsv, global_safe_opt, mode)
        if t % 20 == 0:
            print(
                f"Iter {t:03d} | next={next_idx} | x={xx[next_idx]} | "
                f"y_obs={y_new:.4f} | f_true={y_true_new:.4f} | "
                f"h_true={h_q:.4f} | safe={is_safe}"
            )

    # ------------------------------------------------------------------
    # Final recommendation
    # ------------------------------------------------------------------
    if recommended_idx_hist:
        final_rec_idx = int(recommended_idx_hist[-1])
    else:
        final_rec_idx = int(np.argmax(Y_obs))

    final_rec_x    = xx[final_rec_idx]
    final_rec_f    = yy[final_rec_idx]
    final_rec_h    = zz[final_rec_idx]
    final_rec_safe = final_rec_h >= xi

    if mode == "max":
        simple_regret = np.abs(global_safe_opt - final_rec_f) if final_rec_safe else float("nan")
    else:
        simple_regret = np.abs(final_rec_f - global_safe_opt) if final_rec_safe else float("nan")

    return {
        "environment": env["environment"],
        "xx": xx,
        "yy": yy,
        "zz": zz,
        "xi": xi,
        "safe_true_mask": safe_true_mask,
        "has_true_safe_points": has_true_safe_points,
        "init_idx": np.asarray(init_idx, dtype=int),
        "train_X": xx[queried_idx],
        "train_Y": Y_obs.reshape(-1, 1),
        "train_C": C_obs.reshape(-1, 1),
        "queried_idx": queried_idx,
        "best_safe_value_hist": np.array(best_safe_value_hist),
        "recommended_idx_hist": np.array(recommended_idx_hist),
        "safe_set_size_hist": np.array(safe_set_size_hist),
        "global_safe_opt": global_safe_opt,
        "final_rec_idx": final_rec_idx,
        "final_rec_x": final_rec_x,
        "final_rec_f": final_rec_f,
        "final_rec_h": final_rec_h,
        "final_rec_safe": final_rec_safe,
        "simple_regret": simple_regret,
        "mode": mode,
        "n_init": n_init,
        "init_percentage": init_percentage,
        "allow_repeated_queries": allow_repeated_queries,
        "queried_instant_regret_hist": np.array(queried_instant_regret_hist),
        "queried_cumu_regret_hist": np.array(queried_cumu_regret_hist),
    }


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    res = run_unconstrained_bo_simulation(
        mode="max",
        xi=1.0,
        grid_size=25,
        n_init=1,
        n_iter=50,
        obj_noise_std=0.3,       # matches synth_bo.py: obs_noise = 0.3**2
        M_target=256,
        seed=0,
    )
    print("\n=== Unconstrained BO (synth_bo.py style) Summary ===")
    print(f"global_safe_opt   : {res['global_safe_opt']:.4f}")
    print(f"simple_regret     : {res['simple_regret']}")
    print(f"best_safe_observed: {res['best_safe_value_hist'][-1]}")
    safe_rate = np.mean(res["zz"][res["queried_idx"][res["n_init"]:]] >= res["xi"])
    print(f"safe_query_rate   : {safe_rate:.3f}")
