"""
quantum_safe_bo.py
==================
Quantum-inspired Safe Bayesian Optimisation for the simulated study.

Follows the logic of synth_quantum.py (Quantum_Bayesian_Optimization),
with safe_set_bo.py's constraint handling incorporated.

KEY DIFFERENCES from safe_set_bo.py (classical)
------------------------------------------------
1. STOPPING CONDITION:
     synth_bo.py (classical):   for _ in range(n_iter)
     synth_quantum.py:          while total_oracle_queries < oracle_budget

   One IAE call consumes O(1/eps) Grover oracle queries (many more than 1).
   The budget is shared oracle calls, not iterations.
   safe_set_bo runs the same budget but each classical query = 1 oracle call.

2. WEIGHTED RFF FEATURES (W-GP-UCB):
     safe_set_bo.py:   Phi[i] = phi(x_i)            (unweighted)
                       nu_t   = Sigma_t^{-1} Phi^T Y  (unweighted ridge)

     quantum_safe_bo:  Phi[i] = phi(x_i) / eps_i     (eps-weighted, per synth_quantum.py)
                       nu_t   = Sigma_t^{-1} Phi^T (diag(1/eps^2) Y)

   where eps_i = sqrt(lambda * phi_i^T Sigma_t^{-1} phi_i) / sqrt(lambda)
   = posterior std at x_i at the time it was queried.

3. QUERY FUNCTION:
     safe_set_bo.py:   y = Normal(f_true, obj_noise_std)        [1 oracle call]
     quantum_safe_bo:  y, n_oracle = IAE(f_true, noise, eps)    [O(1/eps) oracle calls]
                       total_oracle_queries += n_oracle

4. DATA RECORDING:
     safe_set_bo.py:   records per-iteration values
     quantum_safe_bo:  records per-step values indexed by CUMULATIVE ORACLE QUERIES
                       so regret curves are plotted vs oracle budget, not iterations

Everything else (safety mask LCB_c >= 0, boundary expansion, Matern constraint GP,
RFF features, rank-1 Sigma_t update) is identical to safe_set_bo.py.
"""

import math
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Qiskit imports  (optional)
# ---------------------------------------------------------------------------

from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit.primitives import Sampler
from qiskit_finance.circuit.library import NormalDistribution



# ---------------------------------------------------------------------------
# Shared infrastructure from safe_set_bo
# ---------------------------------------------------------------------------
try:
    from .safe_set_bo import (
        _rff_sample_features,
        _rff_phi,
        fit_gp_model,
        update_gp_model_fixed_params,
        compute_lcb_ucb,
        minmax_norm,
        lambda_by_stage,
        resolve_n_init,
        sample_initial_points,
    )
    from .experiment_env import (
        build_paper_sim2_environment,
        compute_global_safe_optimum,
    )
except ImportError:
    from safe_set_bo import (
        _rff_sample_features,
        _rff_phi,
        fit_gp_model,
        update_gp_model_fixed_params,
        compute_lcb_ucb,
        minmax_norm,
        lambda_by_stage,
        resolve_n_init,
        sample_initial_points,
    )
    from experiment_env import (
        build_paper_sim2_environment,
        compute_global_safe_optimum,
    )


# ---------------------------------------------------------------------------
# UCB scoring with W-GP-UCB  (eps-weighted, mirrors synth_quantum.py)
# ---------------------------------------------------------------------------
def _rff_ucb_weighted_all(
    domain_np: np.ndarray,
    rf: dict,
    nu_t: np.ndarray,
    Sigma_t_inv: np.ndarray,
    beta: float,
    M: int,
    lam: float = 1.0,
    mode: str = "max",
) -> tuple:
    """
    W-GP-UCB scores over the full domain grid.

    Mirrors bayesian_optimization_quantum.py UtilityFunction._ucb():
        mean = phi^T nu_t
        var  = lam * phi^T Sigma_t^{-1} phi
        ucb  = mean + sqrt(beta) * sqrt(var)

    Returns (scores [N], variances [N]) so caller can compute eps for next query.
    """
    N = domain_np.shape[0]
    scores = np.empty(N)
    vars_  = np.empty(N)
    for i in range(N):
        phi  = _rff_phi(domain_np[i], rf, M).reshape(-1, 1)  # [M,1] unweighted
        mean = float(phi.T @ nu_t)
        var  = float(lam * (phi.T @ Sigma_t_inv @ phi))
        if mode == "max":
            scores[i] = mean + beta * math.sqrt(max(var, 0.0))
        else:
            scores[i] = -mean + beta * math.sqrt(max(var, 0.0))
        vars_[i]  = var
    return scores, vars_


# ---------------------------------------------------------------------------
# Quantum IAE query  (mirrors synth_quantum.py synth_func)
# ---------------------------------------------------------------------------
def quantum_amplitude_query(
    mean: float,
    obs_noise_var: float,
    eps: float,
    num_uncertainty_qubits: int = 6,
    alpha: float = 0.05,
    quantum_noise: bool = False,
    seed: int = 0,
    backend=None,
):
    """
    Iterative Amplitude Estimation query — mirrors synth_quantum.py synth_func().

    Returns (y_estimate, n_oracle_queries).
    Falls back to Normal sample if Qiskit not installed (n_oracle = 1).
    """
    # if not QISKIT_AVAILABLE:
    #     return float(np.random.normal(mean, math.sqrt(max(obs_noise_var, 1e-12)))), 1

    stddev = math.sqrt(max(obs_noise_var, 1e-12))
    low, high = mean - 3 * stddev, mean + 3 * stddev

    uncertainty_model = NormalDistribution(
        num_uncertainty_qubits, mu=mean, sigma=stddev**2, bounds=(low, high)
    )
    linear_payoff = LinearAmplitudeFunction(
        num_uncertainty_qubits, slope=1, offset=0,
        domain=(low, high), image=(low, high), rescaling_factor=1,
    )
    circuit = QuantumCircuit(linear_payoff.num_qubits)
    circuit.append(uncertainty_model, range(num_uncertainty_qubits))
    circuit.append(linear_payoff,     range(linear_payoff.num_qubits))

    # rescale epsilon (synth_quantum.py line 80)
    epsilon_q = float(np.clip(eps / (3 * stddev), 1e-6, 0.5))
    max_shots  = int(np.ceil(
        32 * np.log(2 / alpha * np.log2(np.pi / (4 * epsilon_q)))
    ))

    problem = EstimationProblem(
        state_preparation=circuit, objective_qubits=[0],
        post_processing=linear_payoff.post_processing,
    )

    if backend is not None:
        # Real IBM hardware via Qiskit Runtime — route through the project's
        # custom IAE in circuit_utils.py, which builds Grover circuits by hand
        # and submits via SamplerV2(mode=backend). Bypasses the V1/V2
        # incompatibility between qiskit_algorithms IAE and qiskit-ibm-runtime.
        import os, sys
        _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from circuit_utils import iterative_amplitude_estimation as custom_iae

        # Pass the amplitude-space epsilon directly: the simulated path already converts
        # x-space eps → amplitude-space via eps / (3*stddev) under the
        # LinearAmplitudeFunction(image=(low,high)) encoding. Reusing the
        # same conversion keeps the real-hardware path semantically aligned
        # with the simulated path and avoids the polyfit-degenerate issue
        # for saturating LinearAmplitudeFunction encodings.
        (_, est_proc, _, n_q, _) = custom_iae(
            problem=problem,
            backend=backend,
            shots=int(np.ceil(max_shots)),
            alpha=alpha,
            epsilon=epsilon_q,
        )
        est = float(est_proc)
        n_q = int(n_q) if n_q else int(np.ceil((0.8 / epsilon_q) * np.log((2 / alpha) * np.log2(np.pi / (4 * epsilon_q))))/2)
        return est, n_q

    if quantum_noise:
        try:
            from qiskit.providers.fake_provider import FakeWashington
            from qiskit_aer.noise import NoiseModel
            device = FakeWashington()
            sampler = Sampler(
                backend_options={
                    "method": "density_matrix",
                    "coupling_map": device.configuration().coupling_map,
                    "noise_model": NoiseModel.from_backend(device),
                },
                run_options={"shots": max_shots, "seed_simulator": seed},
                transpile_options={"seed_transpiler": seed},
            )
        except Exception:
            sampler = Sampler(options={
            "shots": int(np.ceil(max_shots)), "seed":seed})
            # sampler = Sampler(run_options={"shots": max_shots, "seed_simulator": seed})
    else:
        sampler = Sampler(options={
            "shots": int(np.ceil(max_shots)), "seed":seed})
        # sampler = Sampler(run_options={"shots": max_shots, "seed_simulator": seed})

    ae     = IterativeAmplitudeEstimation(epsilon_target=epsilon_q, alpha=alpha, sampler=sampler)
    result = ae.estimate(problem)
    est    = float(result.estimation_processed)
    n_q    = int(result.num_oracle_queries)

    if n_q == 0:
        n_q = int(np.ceil((0.8 / epsilon_q) * np.log((2 / alpha) * np.log2(np.pi / (4 * epsilon_q))))/2)
    return est, n_q


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------
def run_quantum_safe_bo_simulation_real(
    mode: str = "max",
    xi: float = 0.0,
    grid_size: int = 50,
    x1_bounds=(-1.0, 1.0),
    x2_bounds=(-1.0, 1.0),
    n_init: int = 1,
    init_percentage=None,
    # STOPPING CONDITION (synth_quantum.py):
    #   oracle_budget = total Grover oracle calls allowed.
    #   For classical (use_quantum_query=False), each step = 1 call → n_iter steps.
    #   For quantum,  each step = O(1/eps) calls → fewer but higher-quality steps.
    oracle_budget: int = 80,
    obj_noise_std: float = 0.09,
    con_noise_std: float = 1e-6,
    beta_f: float = 1.0,
    beta_c: float = 3.0,
    lam0: float = 0.8,
    lam_t0: int = 10,              # lambda_by_stage: half-decay iteration
    lam_p: float = 1.0,            # lambda_by_stage: decay power
    seed: int = 0,
    allow_repeated_queries: bool = True,
    n_init_c=None,             # kept for compatibility; must match n_init if provided
    init_idx=None,
    init_idx_c=None,           # kept for compatibility; must match init_idx if provided
    # RFF settings for objective surrogate
    M_rff: int = 256,
    lam_rff: float = 1.0,
    lengthscale_rff: float = 1.0,
    v_kernel_rff: float = 1.0,
    # Constraint GP hyperparameters (fixed, not learned)
    con_lengthscale: float = 0.5,
    con_outputscale: float = 1.0,
    # Quantum query settings
    use_quantum_query: bool = False,
    backend=None,  # placeholder for potential future backend selection
    quantum_noise: bool = False,
    num_uncertainty_qubits: int = 6,
    **kwargs,
) -> dict:
    """
    Safe BO with quantum IAE query, following synth_quantum.py stopping logic.

    Differences from run_safe_bo_simulation (safe_set_bo.py):
      1. Stopping: while total_oracle_queries < oracle_budget
      2. Weighted RFF: Phi[i] = phi(x_i)/eps_i, Y_w = diag(1/eps^2) Y
      3. Query: IAE returns (y_est, n_oracle); total_oracle_queries += n_oracle
      4. eps_list: computed per step as sqrt(lam * phi^T Sigma^{-1} phi)/sqrt(lam)
      5. Recording: regret and best_safe indexed by cumulative oracle queries
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

    print(f"[Quantum Safe BO] Grid size: {len(xx)}")
    print(f"[Quantum Safe BO] Failure Ratio: {np.count_nonzero(zz < xi)/len(xx):.3f}")
    print(f"[Quantum Safe BO] Safe points: {safe_true_mask.sum()} / {len(xx)}")
    print(f"[Quantum Safe BO] oracle_budget={oracle_budget} | use_quantum_query={use_quantum_query}")

    global_safe_opt = compute_global_safe_optimum(yy, safe_true_mask, mode)
    N = len(xx)

    # ------------------------------------------------------------------
    # Initial design
    # ------------------------------------------------------------------
    n_init   = resolve_n_init(N, n_init=n_init, init_percentage=init_percentage)
    if n_init_c is None:
        n_init_c = n_init
    elif n_init_c != n_init:
        raise ValueError("n_init_c must match n_init so c_model and the objective surrogate share the same initial set.")

    # Objective init indices
    if init_idx is None:
        init_idx = sample_initial_points(xx, n_init=n_init, seed=seed)
    else:
        init_idx = np.asarray(init_idx, dtype=int)
        if len(init_idx) != n_init:
            raise ValueError("len(init_idx) must match the resolved n_init.")

    # Constraint init indices must match the objective init indices so both
    # surrogates are initialized from the same shared design.
    if init_idx_c is None:
        init_idx_c = init_idx.copy()
    else:
        init_idx_c = np.asarray(init_idx_c, dtype=int)
        if len(init_idx_c) != n_init or not np.array_equal(init_idx_c, init_idx):
            raise ValueError("init_idx_c must match init_idx so c_model and the objective surrogate use the same initial points.")

    queried_idx                  = list(init_idx)

    # Objective observations (n_init pts)
    train_X  = X_grid[init_idx].clone()
    obj_obs, obj_con_obs = [], []
    for idx in init_idx:
        obj_obs.append(yy[idx] + np.random.randn() * obj_noise_std)
        obj_con_obs.append((zz[idx] - xi) + np.random.randn() * con_noise_std)

    train_Y = torch.tensor(obj_obs, dtype=torch.float64).view(-1, 1)

    # Constraint observations use the same shared init points with no duplicate
    # constraint measurements at initialization.
    c_train_X = X_grid[init_idx_c].clone()
    c_train_C = torch.tensor(obj_con_obs, dtype=torch.float64).view(-1, 1)
    queried_c_obs                = list(obj_con_obs)  # constraint obs at queried pts


    # ------------------------------------------------------------------
    # W-GP-UCB RFF setup  (synth_quantum.py: eps-weighted features)
    # ------------------------------------------------------------------
    xx_np = np.asarray(xx)
    d     = xx_np.shape[1]
    rf    = _rff_sample_features(d=d, M=M_rff, v_kernel=v_kernel_rff,
                                  lengthscale=lengthscale_rff, seed=seed)

    # beta constant (synth_bo style): beta_t = sqrt(2) * beta_f
    # ts = np.arange(1, oracle_budget + 1)
    # beta_t = 1 + np.sqrt(np.log(ts) ** 2)
    beta_t = np.sqrt(3) * beta_f
    # eps_list: one eps per observed point (init points get eps=1, like synth_quantum.py)
    eps_list = np.ones(n_init)   # synth_quantum.py line 93: eps_list = append(eps_list, 1)
    Y_obs_np = np.array(obj_obs)

    eps_weights = 1.0 / eps_list

    Phi_obj = np.stack([
        _rff_phi(xx_np[idx], rf, M_rff) * eps_weights[i]
        for i, idx in enumerate(queried_idx)
    ])

    Y_weighted = (Y_obs_np * eps_weights).reshape(-1, 1)

    Sigma_t = Phi_obj.T @ Phi_obj + lam_rff * np.eye(M_rff)
    Sigma_t_inv = np.linalg.inv(Sigma_t)
    nu_t = Sigma_t_inv @ Phi_obj.T @ Y_weighted
    # Constraint GP uses the same shared initial set as the objective surrogate.
    c_model = fit_gp_model(c_train_X, c_train_C, noise_var=con_noise_std**2,
                           lengthscale=con_lengthscale, outputscale=con_outputscale)
    init_mu_c, _, init_lcb_c, _ = compute_lcb_ucb(c_model, X_grid, beta=beta_c)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    best_safe_value_hist         = []
    recommended_idx_hist         = []
    safe_set_size_hist           = []
    queried_instant_regret_hist  = []
    queried_cumu_regret_hist     = []

    # ORACLE QUERY TRACKING  (synth_quantum.py: track_queries, total_used_queries)
    track_queries          = []        # oracle calls per BO step
    cumulative_oracle_hist = []        # total oracle calls after each BO step

    # Budget consumed by init (each init point costs 1 oracle call, like synth_quantum.py)
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

    t = 0  # iteration counter (for lambda_by_stage, logging)

    # ------------------------------------------------------------------
    # Main loop — STOPPING: while total_oracle_queries < oracle_budget
    # (synth_quantum.py line 145)
    # ------------------------------------------------------------------
    while total_oracle_queries < oracle_budget:
        t += 1

        # --- Compute eps for the NEXT query point BEFORE scoring ---
        # (synth_quantum.py: compute eps from Sigma_t_inv before querying)
        # We will compute it after selecting next_idx below.

        # Objective W-GP-UCB scores
        ucb_f_np, var_np = _rff_ucb_weighted_all(
            xx_np, rf, nu_t, Sigma_t_inv, beta_t, M_rff, lam=lam_rff, mode=mode
        )
        ucb_f = torch.as_tensor(ucb_f_np, dtype=torch.float64)

        # Constraint GP
        mu_c, sig_c, lcb_c, ucb_c = compute_lcb_ucb(c_model, X_grid, beta=beta_c)

        available_mask = torch.ones(N, dtype=torch.bool)
        if not allow_repeated_queries:
            available_mask[queried_idx] = False

        # Safety mask: LCB_c >= 0  (identical to safe_set_bo.py)
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
            # Boundary expansion: target safe pts nearest the constraint boundary
            a_bnd     = -torch.abs(mu_c / (sig_c + 1e-9))
            lam_t     = lambda_by_stage(t, lam0=lam0, t0=lam_t0, p=lam_p)

            obj_n = minmax_norm(obj_score, verified_safe)
            bnd_n = minmax_norm(a_bnd, verified_safe)
            score = (1.0 - lam_t) * obj_n + lam_t * bnd_n
            score[~verified_safe] = -1e18   # never query outside verified safe set

            next_idx = int(torch.argmax(score).item())


        # --- Compute eps for x* (synth_quantum.py lines 130-141) ---
        phi_star = _rff_phi(xx_np[next_idx], rf, M_rff).reshape(-1, 1)  # unweighted phi
        var_star = float(lam_rff * (phi_star.T @ Sigma_t_inv @ phi_star))
        eps_star = math.sqrt(max(var_star, 1e-12)) / math.sqrt(lam_rff)  # synth_quantum line 141
        
        # ---------------------------------------------------------------
        # Optimization: Prevent IAE from consuming too much budget on a single point 
        # (Mimicking safe_set_bo cap min(n_required, 4999))
        # ---------------------------------------------------------------
        eps_star = max(eps_star, 1e-6)

        # ---------------------------------------------------------------
        # QUERY STEP  (THE ONLY DIFFERENCE FROM safe_set_bo.py)
        # ---------------------------------------------------------------
        y_new, n_oracle = quantum_amplitude_query(
            mean                   = float(yy[next_idx]),
            obs_noise_var          = obj_noise_std ** 2,
            eps                    = eps_star,
            backend                = backend,
            num_uncertainty_qubits = num_uncertainty_qubits,
            quantum_noise          = quantum_noise,
            seed                   = seed + t,
        )


        c_new = float(np.random.normal(zz[next_idx] - xi, con_noise_std))

        # --- Strict budget enforcement ---
        remaining_budget = oracle_budget - total_oracle_queries
        if n_oracle > remaining_budget:
            n_oracle = remaining_budget

        # --- Record oracle usage (synth_quantum.py lines 148-149) ---
        total_oracle_queries += n_oracle
        track_queries.append(n_oracle)
        cumulative_oracle_hist.append(total_oracle_queries)

        # --- Update tensors ---
        train_X = torch.cat([train_X, X_grid[next_idx].view(1, -1)], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor([[y_new]], dtype=torch.float64)], dim=0)
        # Constraint tensors are separate (richer init)
        c_train_X = torch.cat([c_train_X, X_grid[next_idx].view(1, -1)], dim=0)
        c_train_C = torch.cat([c_train_C, torch.tensor([[c_new]], dtype=torch.float64)], dim=0)
        queried_idx.append(next_idx)
        queried_c_obs.append(c_new)

        # --- Update eps_list (synth_quantum.py line 207) ---
        eps_list = np.append(eps_list, eps_star)

        # --- Rebuild weighted Phi and Sigma_t ---
        eps_weights = 1.0 / eps_list
        
        Phi_obj = np.stack([
            _rff_phi(xx_np[queried_idx[i]], rf, M_rff) * eps_weights[i]
            for i in range(len(queried_idx))
        ])

        Y_obs_np = np.append(Y_obs_np, y_new)
        Sigma_t  = Phi_obj.T @ Phi_obj + lam_rff * np.eye(M_rff)
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        Y_weighted  = (Y_obs_np * eps_weights).reshape(-1, 1)
        nu_t        = Sigma_t_inv @ Phi_obj.T @ Y_weighted

        # Constraint GP update
        c_model = update_gp_model_fixed_params(c_model, c_train_X, c_train_C, noise_var=con_noise_std**2)

        # --- Regret (identical to safe_set_bo.py) ---
        f_query       = yy[next_idx]
        # Instant regret for ALL queries (safe or not) against the safe optimum
        r_t = abs(global_safe_opt - f_query) if mode == "max" else abs(f_query - global_safe_opt)
        queried_instant_regret_hist.append(r_t)
        queried_cumu_regret_hist.append(np.sum(queried_instant_regret_hist))

        # --- Recommendation (identical to safe_set_bo.py) ---
        rff_mean_np, _ = _rff_ucb_weighted_all(xx_np, rf, nu_t, Sigma_t_inv, 0.0, M_rff, lam_rff, mode="max")
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

        # --- Best safe observed (use noise-free yy[queried_idx], not noisy train_Y) ---
        obs_safe_mask = torch.tensor(queried_c_obs, dtype=torch.float64) >= 0.0
        if obs_safe_mask.any():
            yy_queried = torch.tensor([yy[i] for i in queried_idx], dtype=torch.float64)
            bsv = yy_queried[obs_safe_mask].max().item() if mode == "max" else yy_queried[obs_safe_mask].min().item()
        else:
            bsv = np.nan
        best_safe_value_hist.append(bsv)

        print(
            f"Iter {t:04d} | oracle={total_oracle_queries}/{oracle_budget} | "
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
        rff_final, _ = _rff_ucb_weighted_all(xx_np, rf, nu_t, Sigma_t_inv, 0.0, M_rff, lam_rff)
        final_rec_idx = int(np.argmax(rff_final))

    final_rec_x    = xx[final_rec_idx]
    final_rec_f    = yy[final_rec_idx]
    final_rec_h    = zz[final_rec_idx]
    final_rec_safe = final_rec_h >= xi
    simple_regret  = (
        (global_safe_opt - final_rec_f) if mode == "max" else (final_rec_f - global_safe_opt)
    ) if final_rec_safe else np.nan

    # ------------------------------------------------------------------
    # Oracle-expanded arrays  (analyze.ipynb pattern)
    # Each BO step produced track_queries[i] oracle calls.
    # Repeat each per-step value that many times so the resulting arrays
    # are indexed by ORACLE CALL NUMBER, matching the classical baseline
    # where every step = 1 oracle call.
    #
    # From analyze.ipynb:
    #   for i in range(len(values)):
    #       values_new += list(np.repeat(values[i], track_queries[i]))
    # ------------------------------------------------------------------
    bo_track = track_queries[n_init:]   # only the loop steps (exclude init)

    def _expand(arr):
        """Repeat arr[i] exactly bo_track[i] times, return flat array."""
        out = []
        for val, reps in zip(arr, bo_track):
            out.extend([val] * int(reps))
        return np.array(out)

    # f_values (true objective at queried point, per oracle call)
    f_values_per_step = yy[queried_idx[n_init:]]          # [n_steps]
    f_values_expanded = _expand(f_values_per_step)         # [total_oracle - n_init]

    # instant regret per oracle call
    regret_expanded   = _expand(queried_instant_regret_hist)

    # cumulative regret per oracle call (recomputed from expanded instants)
    cumu_regret_expanded = np.nancumsum(regret_expanded)
    
    # best safe value per oracle call (pad with nan for unsafe steps)
    best_safe_expanded = _expand(best_safe_value_hist[1:])  # skip init entry

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
        # Per-BO-step arrays (one entry per acquisition step)
        "queried_instant_regret_hist": np.array(queried_instant_regret_hist),
        "queried_cumu_regret_hist"   : np.array(queried_cumu_regret_hist),
        # Oracle-expanded arrays (analyze.ipynb: np.repeat(values[i], track_queries[i]))
        # These are indexed by oracle call number — directly comparable to classical BO
        # where each step = 1 oracle call.
        "f_values_expanded"          : f_values_expanded,       # f_true repeated per oracle call
        "regret_expanded"            : regret_expanded,          # instant regret per oracle call
        "cumu_regret_expanded"       : cumu_regret_expanded,     # cumulative regret per oracle call
        "best_safe_expanded"         : best_safe_expanded,       # best safe obs per oracle call
        # Quantum tracking
        "track_queries"              : np.array(track_queries),       # oracle calls per step (all)
        "cumulative_oracle_hist"     : np.array(cumulative_oracle_hist),
        "total_oracle_queries"       : total_oracle_queries,
        "oracle_budget"              : oracle_budget,
        "eps_list"                   : eps_list,
        # Learned constraint boundary (for plotting)
        "init_mu_c":   init_mu_c.detach().cpu().numpy(),
        "init_lcb_c":  init_lcb_c.detach().cpu().numpy(),
        "final_mu_c":  mu_c_final.detach().cpu().numpy(),
        "final_lcb_c": lcb_c_final.detach().cpu().numpy(),
    }


# Backward-compatible alias for older imports.
run_quantum_safe_bo_simulation = run_quantum_safe_bo_simulation_real
