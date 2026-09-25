# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantum-Classical Hybrid Bayesian Optimization research for aerospace manufacturing. The project compares classical GP-based safe-set Bayesian Optimization (BO) against quantum variants that use Quantum Amplitude Estimation (QAE) for constraint satisfaction checking, applied to fuselage actuator force optimization with Tsai-Wu failure criterion constraints.

The 18-dimensional action space represents forces on fuselage actuators, but only 8 are active (indices 0–3 and 14–17); the rest are zeroed out by `sample_actions` in `utils.py`. However, the discrete grid used in the main experiments (`sample_sobol_on_grid`) is actually a **2D Cartesian product** (21×21 = 441 points) of `linspace(-1, 1, 21)` over **only dims 0 and 17** — despite the function name, it is not Sobol-sampled.

The constraint is defined as `c(x) = 1 - FI(x)` where FI = Tsai-Wu failure index; `c(x) ≥ 0` means structurally safe. `sample_kmeans_safe_subspace` filters candidate warmup points by `1.0 - FI ≥ 0.0` before clustering.

## Running Experiments

All scripts must be run from the project root (surrogate `.joblib` files are loaded relative to CWD). There is no build system.

```bash
# Activate conda environment first
conda activate quantum

# Classical safe-set BO (discrete action space)
python classic_safeset_discrete.py --query_budget 20000 --eps_max 0.04 --obs_noise 0.04 --B 3.0 --obj_ls 0.2

# Quantum safe-set BO (discrete action space)
python quantum_safeset_discrete.py --query_budget 20000 --eps_max 0.04 --obs_noise 0.04 --B 3.0 --obj_ls 0.2

# Hyperparameter sweep — uses classic_safeset_discrete.py as faster proxy for quantum
bash sweep_hyperparams.sh        # saves per-config logs to sweep_results/
python sweep_hyperparams.py      # standalone Python version

# Multi-seed quantum experiments (vary --init_num_points)
bash run_quantum_experiments.sh

# Monitor long-running experiment processes
bash monitor_quantum_runs.sh

# Simulated 2D synthetic benchmark (run from inside simulated_study/)
cd simulated_study
python compare_safe_methods_shared_init.py   # single-seed
python multi_init_cumulative_regret.py       # multi-seed
```

Common CLI parameters (flag names must match exactly — e.g. `--query_budget` not `--budget`):
- `--query_budget`: Number of oracle queries (default: 20000)
- `--eps_max`: Safe-set expansion threshold (default: 0.04)
- `--obs_noise`: Observation noise variance (default: 0.04 = 0.2²)
- `--min_shots`: Minimum Monte Carlo samples per evaluation (default: 20)
- `--B`: UCB exploration coefficient (optimal: 3.0)
- `--obj_ls`: Objective GP lengthscale (optimal: 0.2)
- `--lam0`: Initial lambda for constraint weighting (default: 1.0)
- `--t0`: Lambda decay timescale (optimal: 10.0)
- `--lam_p`: Lambda decay power (default: 2.0)
- `--init_num_points`: Safe warmup points for both GP models (optimal: 5)
- `--M_features`: Number of RFF features for GP kernel (default: 400)

## Architecture

### Core Modules

- **[utils.py](utils.py)** — `build_train_gp_with_rff`: trains a BoTorch `FixedNoiseGP` with `RFFKernel`; auto-retries with varied seeds if training R² < threshold (default 0.7). Note: this utility is not currently called by the main experiment scripts, which build their GPs inline. `sample_actions`: samples from the 8-dim active subspace, returns [N, 18] with inactive dims zeroed. `sample_kmeans_safe_subspace`: clusters safe points via K-Means for diverse initialization.
- **[circuit_utils.py](circuit_utils.py)** — Custom IAE (`iterative_amplitude_estimation`): constructs Qiskit circuits `Q^k A |0⟩`, runs Grover iterations, returns confidence intervals via Clopper-Pearson/Chernoff bounds. `find_next_k` selects the next Grover power to minimize CI width. Oracle cost accumulates as `k × shots` per iteration.

### Two Separate GP Models per Experiment

Each experiment script maintains two independent GPs:
- **Constraint GP** (`c_model`): standard `FixedNoiseGP` with Matérn-2.5 kernel (no RFF), fixed lengthscales (0.6931 for inactive dims, 0.50 for active dims 0 and 17). Updated every iteration; used to compute `lcb_constraint` for safe-set membership.
- **Objective GP** (`model_ei`): `FixedNoiseGP` with `RFFKernel` (M_features random Fourier features, Matérn-2.5). Lengthscale set via `--obj_ls`. Uses W-GP-UCB (weighted by ε) via the Gram matrix `V_t`.

### Environment Classes

| File | Classical/Quantum | Notes |
|------|------------------|-------|
| [bo_env.py](bo_env.py) | Classical | ANSYS/surrogate wrapper, no constraint |
| [bo_env_constraints.py](bo_env_constraints.py) | Classical | Tsai-Wu constraint; `step_surrogate` returns `(obs, true_error, oracle_queries)` as torch tensors; supports `chebyshev`/`clt`/`hoeffding`/`non_monte_carlo` methods |
| [quantum_bo_env.py](quantum_bo_env.py) | Quantum | QAE for constraint checking |
| [quantum_bo_env_constraint.py](quantum_bo_env_constraint.py) | Quantum | Full constraint handling; `step_surrogate` returns `(obs_response, true_response, oracle_queries, c_val, empirical_variance)` where `c_val ≥ 0` is safe |
| [FuselageActuators/FuselageActuators_env_v22.py](FuselageActuators/FuselageActuators_env_v22.py) | Classical | OpenAI Gym env wrapping live ANSYS |

ANSYS calls are commented out in all env classes by default; surrogate inference is used instead.

### Experiment Scripts

**Classical baselines:** `classic_safeset_discrete.py`, `classic_safeset_continuous.py`, `classic_acl_discrete.py`, `classic_acl_continuous.py`, `classic_bo_unconstrained.py`, `classic_turbo_discrete.py`, `classic_POF_discrete.py`, `admmbo_classic.py`

**Quantum variants:** `quantum_safeset_discrete.py`, `quantum_safeset_continuous.py`, `quantum_safeset_discrete_real.py` (IBM real hardware), `quantum_bo_discrete.py`, `quantum_turbo_discrete.py`, `quantum_bo_active.py`

**Unconstrained baselines (shared-init):** `classic_bo_unconstrained.py`, `classic_bo_unconstrained_discrete.py`, `quantum_bo_unconstrained.py`, `quantum_bo.py`. Each is deliberately aligned to its safeset counterpart (same env `bo_env_constraints.ClassicFuselageEnv`, same objective, same initial points) so constrained vs. unconstrained curves are directly comparable. Results land in `Experiments_unconstraint_continuous/` (subdirs `Classic_Unconstrained_*` / `Quantum_Unconstrained_*`).

**Force-range study:** the continuous scripts accept `--actuator_count` (default 8) and `--force_scale` (default 1000.0, in lb). `run_force_range_experiments.sh` runs the 3 continuous methods sequentially at force scales 500 and 200 (1000 is the baseline already in `Experiments_constraint_continuous`), logging to `force_range_logs/`. `compare_force_range_cumulative_regret.py` plots one panel per force range with root mapping 200→`Experiments_constraint_continuous_force_200`, 500→`_force_500`, 1000→`Experiments_constraint_continuous`. Outputs `force_range_cumulative_regret_noise_0p01.{png,csv}`.

**Shape-gap reduction plots:** `extract_shape_gap_force_configs.py` extracts per-actuator force configs; `plot_shape_gap_reduction_panels.py` and `plot_classic_quantum_shape_gap_comparison.py` render the resulting `shape_gap_reduction_extract` CSVs (found under `exp_set_*/` result subdirs).

**Actuator-count scaling study:** `compare_actuator_count_cumulative_regret.py` aggregates cumulative regret across actuator counts 4/6/8 (roots `Experiments_constraint_continuous_actuators_4`, `_6`, and `Experiments_constraint_continuous` for count 8). Regret = `cumsum(abs(F_OPT - true_response))` with rows expanded by the per-step `queries` count; curves are then averaged (± stderr) across trials/exp_sets. Outputs `actuator_count_cumulative_regret_noise_0p01.{png,csv}`. The `--noise` flag is an observation *variance* (default `0.1**2 = 0.01`); the physical σ shown in plots is its square root.

**Analysis tools:**
- `test_run.py` — analysis script for **continuous** experiments; loads `.pth` data from `Experiments_constraint_continuous/`, computes per-seed minimum, cumulative regret, and safe-rate statistics.
- `analyze_discrete.ipynb` — primary analysis notebook for discrete experiments; loads from `Experiments_constraints/`; see [analyze_discrete_README.md](analyze_discrete_README.md) for full metric definitions and key results.
- `print_minimums.py`, `print_mins_script.py` — quick scripts to print best values found per seed.
- `report_violation_rate.py` — recomputes constraint violation rate strictly via `surrogate_tsaiwu.joblib`.
- `plot_constraint_regret.py` — plots running minimum and cumulative regret figures.
- Other notebooks: `analyze.ipynb` (constrained continuous Q/C-Safe BO vs. BO-ACL — MAE, safe-rate, cumulative regret; results summarized in [result_README.md](result_README.md)), `compare_cumulative_regret.ipynb`, `analyze_constraint_continuous_regret.ipynb`

### Simulated Study (`simulated_study/`)

A self-contained 2D synthetic benchmark (Paper Simulation 2) that validates the algorithmic approach on an analytic environment before the full fuselage application. Must be run from inside `simulated_study/`.

**Objective:** `g(x) = x₁² − sin(4x₂²)` on a 25×25 grid over `[−1, 1]²`

**Constraint:** `c(x) = x₂ − x₁²` (safe if ≥ 0)

| Module | Method |
|--------|--------|
| `safe_set_bo.py` | Classical GP safe-set cUCB |
| `quantum_safe_bo.py` | IAE-based quantum safe BO (simulated) |
| `quantum_safe_bo_real.py` | Quantum safe BO on IBM real hardware |
| `unconstrained_bo.py` | Standard UCB-BO (no constraint) |
| `ACL_paper.py` | Active Constraint Learning BO |

Key result: Quantum Safe BO reduces cumulative regret by ~42% vs classical Safe BO while maintaining 100% safety; BO-ACL violates safety on ~31.5% of queries.

**Driver / replot scripts (prefer these to re-running experiments):** The multi-seed benchmark is expensive, so results are cached and figures/tables are regenerated from the cache rather than re-simulated:
- `multi_init_cumulative_regret.py` — canonical multi-seed (seeds 5–9) driver. Resumes from `multi_init_checkpoint.pkl`, skipping already-completed runs; writes `multi_init_regret_mean_std.png` and `multi_init_stats.txt`. Set `INCLUDE_REAL_QUANTUM` to toggle the IBM-hardware method.
- `replot_multi_init_from_pkl.py` — re-aggregates the summary table/figure purely from `multi_init_checkpoint.pkl` (no simulation, no IBM access). Use this to change how metrics are reported.
- `replot_from_pkl.py` / `replot_from_txt.py` — regenerate `comparison_regret.png` (single-seed) from a `results_*.pkl` or its `results_*.txt` sidecar. The `.txt` sidecar embeds the raw cumulative-regret curves, so it is a self-sufficient plotting source.

Result artifacts come in matched pairs — `results_<ts>.pkl` (raw arrays for reloading) and `results_<ts>.txt` (human-readable summary table + embedded regret curves). Regret conventions: quantum methods store `cumu_regret_expanded` (per-oracle-query), classical methods store `queried_cumu_regret_hist`; both are padded/trimmed to `ORACLE_BUDGET` before aggregation. "Simple regret" is computed noise-free as `|global_safe_opt - best ground-truth objective over feasible queried points|`.

**IBM real-hardware access:** `QiskitRuntimeService()` with no args loads the default saved account, which may have no QPUs attached (→ `QiskitBackendNotFoundError` from `least_busy`). Load a QPU-bearing account explicitly, e.g. `QiskitRuntimeService(name="CS102")`. The installed `qiskit-ibm-runtime` only accepts `channel` in `{ibm_cloud, ibm_quantum}`; saved `ibm_quantum_platform` accounts will fail to load.

### Data and Models

- **Surrogate models** are loaded from the project root via relative paths:
  - `surrogate_likeDu_v22.joblib` — linear model (`.coef_` extracts [18,] weights) for objective (shape error)
  - `surrogate_tsaiwu.joblib` — Tsai-Wu failure criterion constraint; full sklearn model
  - A copy of `surrogate_likeDu_v22.joblib` also lives in `FuselageActuators/Surrogates/`
- **Shape files:** `FuselageActuators/Shapes/{Train,Test,Benchmark}/` — `.npy` displacement arrays and `.txt` ANSYS input files
- **Results directories:**
  - `Experiments/` — older unconstrained results (`Classic_TurBO_Discrete/`, `Quantum_Discrete_cUCB/`, etc.)
  - `Experiments_constraints/` — current constrained results (`Classic_Discrete_cUCB/`, `Quantum_Discrete_cUCB/`, `Classic_ACL_Discrete/`, etc.); files named `{noise}training_data_{trial}_.pth` or `{noise}quan_training_data_{trial}_.pth`
  - `Experiments_constraint_continuous/` — continuous-space constrained results (also serves as the 8-actuator root for the scaling study)
  - `Experiments_constraint_continuous_actuators_4/`, `_6/` — same, restricted to 4 and 6 active actuators; subdirs encode `<method>_<count>_<params>_noise_<variance>`
  - `Experiments_constraint_continuous_force_200/`, `_force_500/` — continuous constrained results at 200 lb / 500 lb force scales (1000 lb baseline lives in `Experiments_constraint_continuous/`)
  - `Experiments_unconstraint_continuous/` — unconstrained-baseline results (`Classic_Unconstrained_*`, `Quantum_Unconstrained_*`)

**Saved `.pth` file schema:**
```python
{
  'actions': Tensor[N, 18],
  'response': Tensor[N, 1],        # noisy observation
  'true_response': Tensor[N, 1],   # ground truth
  'queries': Tensor[N],            # oracle queries per step
  'uncertainty': list[float],      # epsilon_t per step
  'active_records': list[dict],    # per-step metadata
  'metadata': {
    'query_budget': int, 'trial': int, 'seed': int,
    'num_active_steps': int, 'final_total_budget': int
  }
}
```

### Key Design Patterns

1. **Safe-set expansion**: Safe set S ⊆ grid grows as GP constraint uncertainty narrows. Each iteration computes `epsilon_t = min(eps_max, sqrt(var(x) / lambda_t))` where `lambda_t` decays via `lam0 * (t0 / (t0 + stage))^lam_p`.
2. **QAE replaces GP constraint bounds**: Instead of GP posterior CI over constraint, a `NormalDistribution` circuit encodes the GP predictive distribution; IAE estimates P(constraint satisfied) with quantum speedup. The CI from IAE substitutes for classical `lcb_constraint`.
3. **Acquisition**: cUCB = `minmax_norm(ucb_obj) - lambda_t * minmax_norm(lcb_constraint)` over the current safe set.
4. **GP training**: The constraint GP uses standard BoTorch `FixedNoiseGP` with Matérn-2.5 and fixed lengthscales. The objective GP uses `RFFKernel` (random Fourier features approximation of Matérn-2.5). The Gram matrix `V_t = Σ (1/ε_i) Φ(xᵢ)Φ(xᵢ)ᵀ + λI` accumulates RFF features for `W_GP_UCB_scores()`.
5. **Multi-seed runs**: `concurrent.futures.ProcessPoolExecutor` or sequential loops with `--seed_offset` in ACL scripts; `run_quantum_experiments.sh` loops sequentially.
6. **Inline GP construction**: The main experiment scripts (`classic_safeset_discrete.py`, `quantum_safeset_discrete.py`) build and update GPs directly via `initialize_model` / `initialize_c_model` defined at the top of each file, rather than importing from `utils.py`.

## Dependencies

```bash
pip install -r requirements.txt
```

Key pinned versions: `qiskit==0.44`, `qiskit_aer==0.12.0`, `qiskit_finance==0.3.4`, `qiskit-algorithms==0.3.0`, `qiskit_ibmq_provider==0.20.2`. BoTorch/PyTorch/scikit-learn are assumed pre-installed in the `quantum` conda environment.

## Optimal Hyperparameters

From sweep results in [hyperparameter_sweep_results.md](hyperparameter_sweep_results.md):

| Parameter | Optimal | Why |
|-----------|---------|-----|
| `init_num_points` | 5 | Eliminates GP cold-start; 1 point → 40% convergence, 5 points → 100% |
| `B` | 3.0 | UCB exploration; B=1.0 stalls safe set at ~45% of grid |
| `t0` | 10.0 | Slow lambda decay gives more time for boundary expansion |
| `obj_ls` | 0.2 | Sharp localized response; slight edge over 0.5 early in search |
| `lam_p` | 2.0 | Quadratic decay toward pure exploitation |

Achieves MAE ~0.0716 (theoretical minimum) with 100% convergence across all seeds and noise levels.
