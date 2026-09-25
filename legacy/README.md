# legacy/

Superseded, broken, or one-off files kept for reference. Nothing here is maintained, and the main pipeline doesn't import anything from this folder.

| Files | What they are |
|---|---|
| `classic_turbo_discrete.py`, `quantum_turbo_discrete.py`, `quantum_bo.py`, `quantum_bo_active.py` | Early experiments built on the old environments below |
| `classic_POF_discrete.py`, `quantum_cbo_POF.py`, `constrained_classic_bo.py`, `admmbo_classic.py` | Probability-of-feasibility and ADMM baselines. `quantum_cbo_POF.py` and `admmbo_classic.py` no longer match the current env API. |
| `bo_env.py`, `quantum_bo_env.py`, `fuselageENV.py` | Old environments. `fuselageENV.py` is a copy of `FuselageActuators/FuselageActuators_env_v22.py`. |
| `test_run.py`, `plot_constraint_regret.py`, `run_compare_regret.py`, `check_grid_min.py`, `compute_grid_min.py` | Analysis tools that point at result folders or modules that no longer exist |
| `sweep_hyperparams.sh`, `run_quantum_experiments.sh`, `monitor_quantum_runs.sh` | Shell drivers that use removed CLI flags or hardcoded process IDs |
| `optimize_regret.py`, `sweep_regret_target.py`, `check_qae_discretization.py`, `compare_queries.py`, `test_gp.py`, `timeout.py` | One-off exploration and debugging scripts |
| `visualize.ipynb`, `test_qc.ipynb`, `check_data.ipynb`, `test_func.ipynb`, `perfectPos.{csv,npy}` | Old notebooks (pre-1.0 Qiskit APIs, live-ANSYS setup, missing result folders) and the data only they use |
| `surro_classic.joblib` | Unused duplicate of `surrogate_likeDu_v22.joblib` |

To try a script, run it from the repository root so data paths resolve, and put the root on the import path:

```bash
PYTHONPATH=. python legacy/<script>.py
```
