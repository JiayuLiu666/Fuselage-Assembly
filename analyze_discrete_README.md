# analyze_discrete.ipynb

Analysis notebook comparing quantum and classical safe-set Bayesian Optimization methods on the discrete fuselage actuator force optimization task.

## What it does

Loads experiment `.pth` files from `Experiments_constraints/`, computes three performance metrics, and produces publication-quality figures.

### Methods compared

| Label | Description |
|---|---|
| Q-cUCB | Quantum Safe-Set UCB (simulated QAE, `Quantum_Discrete_cUCB/`) |
| Q-cUCB Real | Quantum Safe-Set UCB on IBM real hardware (`Quantum_Discrete_cUCB_Real/`) |
| C-cUCB | Classical Safe-Set UCB (`Classic_Discrete_cUCB/`) |
| C-ACL | Classical Active Constraint Learning (`Classic_ACL_Discrete/`) |

Each method is evaluated at two observation noise levels: **σ=0.1** (σ²=0.01) and **σ=0.2** (σ²=0.04), across **5 independent trials** per configuration.

### Metrics

1. **Cumulative regret** — oracle-query-weighted sum of `|f_max - f(x)|` over iterations, where `f_max = 0.07159 in` is the global minimum shape error.
2. **Running minimum MAE** — `min_{t' ≤ t} f(x_{t'})` as a function of cumulative oracle queries; shows convergence speed.
3. **Noise-free minima** — best true-response value found per seed; hit rate against the target (atol = 5×10⁻⁴).
4. **Safe rate** — fraction of queried points satisfying `1 − FI > 0` (Tsai-Wu feasibility), computed from `active_records` or the `surrogate_tsaiwu.joblib` model.
5. **Violation rate** — `1 − safe_rate` re-evaluated strictly via `surrogate_tsaiwu.joblib` at analysis time (no GP predictions used).

## Outputs

| File | Description |
|---|---|
| `discrete_cumulative_regret.png` | Single-panel cumulative regret for all 8 method/noise combinations |
| `discrete_combined.png` | Three-panel figure: (a) cumulative regret, (b) running minimum σ=0.1, (c) running minimum σ=0.2 |

## Key Results

### Convergence to global optimum (f_max = 0.07159 in)

| Method | σ | Hits (5 seeds) | Avg min found |
|---|---|---|---|
| Q-cUCB | 0.1 | 5/5 | 0.071591 |
| Q-cUCB | 0.2 | 5/5 | 0.071591 |
| Q-cUCB Real | 0.1 | 5/5 | 0.071591 |
| Q-cUCB Real | 0.2 | 5/5 | 0.071591 |
| C-cUCB | 0.1 | 5/5 | 0.071591 |
| C-cUCB | 0.2 | 5/5 | 0.071591 |
| C-ACL | 0.1 | 2/5 | 0.082429 |
| C-ACL | 0.2 | 1/5 | 0.079675 |

### Oracle queries to reach the optimum

| Method | σ²=0.01 (avg queries) | σ²=0.04 (avg queries) |
|---|---|---|
| Q-cUCB | **799** | **4,351** |
| Q-cUCB Real | 1,835 | 4,653 |
| C-cUCB | 2,445 | 8,335 |
| C-ACL | 2,873 (unreliable) | 7,315 (unreliable) |

Q-cUCB reaches the global optimum fastest at both noise levels.

### Constraint violation rate (strict surrogate evaluation)

| Method | Avg violation rate | Unsafe queries / total |
|---|---|---|
| Q-cUCB | 0.00% | 0 / 339 |
| Q-cUCB Real | 0.00% | 0 / 354 |
| C-cUCB | 0.00% | 0 / 258 |
| C-ACL | **26.5% ± 5.2%** | 234 / 995 |

Safe-set methods (Q-cUCB, C-cUCB) maintain zero constraint violations across all trials. C-ACL violates the Tsai-Wu constraint on ~21–35% of its queried points.

## Prerequisites

- Conda environment `quantum` activated
- Experiment data present in `Experiments_constraints/`
- `surrogate_tsaiwu.joblib` at the project root (for violation rate re-evaluation)
- Run from the project root so relative paths resolve correctly

## Data layout expected

```
Experiments_constraints/
  Quantum_Discrete_cUCB/exp_set_1/     {noise}quan_training_data_{trial}_.pth
  Quantum_Discrete_cUCB_Real/exp_set_1/ {noise}quan_training_data_{trial}_.pth
  Classic_Discrete_cUCB/exp_set_1/     {noise}training_data_{trial}_.pth
  Classic_ACL_Discrete/exp_set_1/      {noise}acl_training_data_{trial}_.pth
```

Trials are indexed 0–4; noise prefixes are `0.010000000000000002` (σ²=0.01) and `0.04000000000000001` (σ²=0.04).
