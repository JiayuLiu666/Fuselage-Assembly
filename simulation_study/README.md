# Simulated Study: Quantum vs. Classical Safe Bayesian Optimization

Benchmarks four safe-BO algorithms on a 2-D synthetic environment (Paper Simulation 2)
with an analytic constraint. The study isolates algorithm behavior from real-hardware
and surrogate-model noise, enabling clean regret comparisons.

---

## Environment

**Objective** (`experiment_env.py`):

```
g(x) = x₁² − sin(4x₂²)
```

**Constraint** (safety condition, `c(x) ≥ 0` is safe):

```
c(x) = x₂ − x₁²   (minus slack ξ)
```

Both functions are evaluated on a discrete **25 × 25 grid** over `[−1, 1]²`
(625 candidate points total). A shared set of `N_INIT` initial points is drawn
once per seed and given identically to all methods.

---

## Methods

| Module | Label | Description |
|--------|-------|-------------|
| `safe_set_bo.py` | **Safe BO** | Classical GP safe-set expansion with Matérn-2.5 kernel and cUCB acquisition; each oracle call = 1 function evaluation |
| `quantum_safe_bo.py` | **Quantum Safe BO** | Same safe-set logic but uses Iterative Amplitude Estimation (IAE) via Qiskit to query the objective; oracle cost = O(1/ε) Grover iterations; ε-weighted RFF (W-GP-UCB) |
| `quantum_safe_bo_real.py` | **Quantum Safe BO (Real)** | Variant of the quantum method adapted for real IBM hardware execution |
| `unconstrained_bo.py` | **Unconstrained BO** | Standard UCB-BO with no constraint model; ignores safety (upper bound on regret) |
| `ACL_paper.py` | **BO-ACL** | Active Constraint Learning BO; jointly learns constraint and objective; uses a binary feasibility probe |

---

## Running the Experiments

All scripts must be run from inside `simulated_study/`:

```bash
cd simulated_study

# Single-seed comparison (seed 0, saves comparison_regret.png and comparison_maps.png)
python compare_safe_methods_shared_init.py

# Multi-seed benchmark (seeds 5–9, saves multi_init_regret_mean_std.png and multi_init_stats.txt)
python multi_init_cumulative_regret.py
```

Key hyperparameters (shared across methods):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `GRID_SIZE` | 25 | Grid resolution (25² = 625 points) |
| `XI` | 0 | Constraint threshold `h(x) ≥ ξ` |
| `N_INIT` | 10 (single-seed) / 5 (multi-seed) | Shared warm-start points |
| `ORACLE_BUDGET` | 500 | Total oracle calls per method |
| `OBJ_NOISE` | 0.3 | Objective observation noise std |
| `BETA_C` | 3.0 | UCB exploration coefficient for constraint GP |
| `LAM0` | 0.8 | Initial safe-set expansion weight |
| `LAM_T0` | 10 | Lambda decay half-life (iterations) |
| `LAM_P` | 1.0 | Lambda decay power |

---

## Results

### Single-Seed Run (Seed 0)

Settings: `GRID=25²`, `N_INIT=10`, `ORACLE_BUDGET=500`, `OBJ_NOISE=0.3`,
`INIT_IDX=[523, 508, 393, 315, 166, 25, 10, 190, 109, 46]`

| Method | Global Opt | Final Best Safe | Simple Regret | Cumul. Regret | Safe Rate | Viol. Rate |
|--------|-----------|----------------|--------------|--------------|-----------|------------|
| **Safe BO** | −0.9787 | −0.9781 | 0.0006 | 44.34 | 1.000 | 0.000 |
| **Quantum Safe BO** | −0.9787 | **−0.9787** | **0.0000** | **8.43** | 1.000 | 0.000 |
| **Quantum Safe BO (Real)** | −0.9787 | **−0.9787** | **0.0000** | 19.45 | 1.000 | 0.000 |
| Unconstrained BO | −0.9787 | −0.7790 | 0.1997 | 19.12 | 0.000 | **1.000** |
| BO-ACL | −0.9787 | −0.8415 | 0.1372 | 363.27 | 0.736 | 0.264 |

**Key observations (single seed):**
- Both quantum variants reach the exact global optimum (zero simple regret) while remaining 100% safe.
- Quantum Safe BO accumulates 5.3× less cumulative regret than classical Safe BO (8.43 vs 44.34).
- Quantum Safe BO (Real) matches the simulated quantum variant on simple regret and stays competitive with the unconstrained baseline on cumulative regret while preserving full safety.
- BO-ACL accumulates 43× more cumulative regret than Quantum Safe BO and still violates safety 26% of the time.

---

### Multi-Seed Benchmark (Seeds 5–9, N=5 runs)

Settings: `GRID=25²`, `N_INIT=5`, `ORACLE_BUDGET=500`, `OBJ_NOISE=0.3`

| Method | Cumul. Regret μ ± σ | Simple Regret μ ± σ | Safe Rate μ ± σ |
|--------|-------------------|-------------------|----------------|
| **Safe BO** | 56.17 ± 31.79 | 0.018 ± 0.025 | **1.000 ± 0.000** |
| **Quantum Safe BO** | 20.21 ± 26.62 | 0.013 ± 0.025 | **1.000 ± 0.000** |
| **Quantum Safe BO (Real)** | **19.85 ± 17.60** | **0.006 ± 0.011** | **1.000 ± 0.000** |
| BO-ACL | 318.18 ± 52.69 | 0.070 ± 0.135 | 0.685 ± 0.069 |

Per-run cumulative regret:

| Method | Seed 5 | Seed 6 | Seed 7 | Seed 8 | Seed 9 |
|--------|--------|--------|--------|--------|--------|
| Safe BO | 26.31 | 20.16 | 84.08 | 101.28 | 49.02 |
| Quantum Safe BO | 3.50 | 3.47 | 6.19 | 72.75 | 15.13 |
| Quantum Safe BO (Real) | 7.55 | 7.93 | 11.56 | 54.25 | 17.96 |
| BO-ACL | 290.95 | 290.21 | 277.20 | 311.27 | 421.29 |

**Key observations (multi-seed):**
- Both quantum variants maintain 100% safety across all seeds.
- Quantum Safe BO reduces cumulative regret by ~64% relative to classical Safe BO on average (20.21 vs 56.17).
- Quantum Safe BO (Real) achieves the lowest mean cumulative regret (19.85), the lowest mean simple regret (0.006 ± 0.011), and the tightest variance (σ = 17.60 vs 26.62 for Quantum Safe BO and 31.79 for Safe BO).
- 4 of 5 seeds for Quantum Safe BO finish under 16 cumulative regret; seed 8 is the lone outlier (72.75) that drives most of the variance.
- BO-ACL accumulates ~16× more cumulative regret than Quantum Safe BO and averages 31.5% constraint violations.

---

## Output Files

| File | Description |
|------|-------------|
| `comparison_regret.png` | Cumulative regret curves (all methods, seed 0) |
| `comparison_maps.png` | Safe-set and query maps on the 2-D grid |
| `acl_query_detail.png` | ACL query trajectory detail |
| `multi_init_regret_mean_std.png` | Mean ± std cumulative-regret curves over 5 seeds |
| `multi_init_stats.txt` | Printed summary statistics table (multi-seed) |
| `results_20260430_023108.txt` | Full per-step regret curves from the latest single-seed run |
| `results_20260430_023108.pkl` | Pickle of the full result dict from the latest single-seed run |
| `comparison_regret_original.png` / `comparison_maps_original.png` / `multi_init_regret_mean_std_origin.png` | Snapshots from the prior run, kept for comparison |
| `multi_init_checkpoint.pkl` | Checkpoint from the multi-seed run |
| `logs/` | Per-method stdout logs (`safe_bo_seed0.log`, `quantum_bo_seed0.log`, etc.) |

---

## Module Overview

```
simulated_study/
├── experiment_env.py            # Grid construction, objective & constraint functions
├── safe_set_bo.py               # Classical safe-set BO (GP + cUCB)
├── quantum_safe_bo.py           # Quantum safe BO (IAE + W-GP-UCB)
├── quantum_safe_bo_real.py      # Quantum safe BO for real IBM hardware
├── unconstrained_bo.py          # Unconstrained UCB-BO baseline
├── ACL_paper.py                 # Active Constraint Learning BO
├── circuit_utils.py             # IAE circuit helpers (Qiskit)
├── compare_safe_methods_shared_init.py   # Single-seed comparison runner
├── compare_safe_methods_lib.py  # Shared runner utilities
├── multi_init_cumulative_regret.py       # Multi-seed benchmark runner
└── tune_hyperparams.py / tune_quantum_hyperparams.py  # Hyperparameter search
```

---

## Relation to the Fuselage Application

This study mirrors the structure of the main fuselage experiments
(`classic_safeset_discrete.py`, `quantum_safeset_discrete.py`) but uses an
analytic 2-D function instead of the ANSYS surrogate. Results here validate
the algorithmic advantage of quantum-enhanced safe exploration before applying
it to the 18-dimensional actuator-force problem.
