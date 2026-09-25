# Quantum-Enhanced Safe Bayesian Optimization for Fuselage Shape Adjustment

Research code comparing classical safe-set Bayesian optimization (BO) with a quantum variant that estimates the objective by Quantum Amplitude Estimation (QAE). The application is choosing actuator forces that bring an initial fuselage shape close to a target shape. A Tsai-Wu failure-criterion constraint keeps every queried configuration structurally safe. A 2D synthetic benchmark in `simulation_study/` validates the algorithms first.

## Repository contents

| Path | Contents |
|---|---|
| `*_safeset_*.py`, `*_acl_*.py`, `*_bo_unconstrained*.py`, `quantum_bo_discrete.py` | Experiment scripts: classical and quantum safe-set BO, the BO-ACL baseline, and unconstrained baselines |
| `bo_env_constraints.py`, `quantum_bo_env_constraint.py`, `circuit_utils.py`, `utils.py` | Surrogate environments, custom iterative amplitude estimation, and shared GP/sampling helpers |
| `surrogate_likeDu_v22.joblib`, `surrogate_tsaiwu.joblib`, `Surrogate modeling/` | Linear shape surrogate, Tsai-Wu constraint surrogate, and surrogate training data |
| `FuselageActuators/` | ANSYS input decks (`AnsysFiles/`), displacement shapes (`Shapes/`), and a live-ANSYS Gym environment |
| `Experiments_constraints/` | Discrete-grid results (`.pth`) used by `analyze_discrete.ipynb` |
| `simulation_study/` | 2D synthetic benchmark with cached results and figures |
| `*.ipynb`, `compare_*.py`, `plot_*.py`, `report_violation_rate.py` | Analysis notebooks and plotting scripts |
| `sweep_*.py`, `*sweep*.json` | Hyperparameter sweeps and their results |
| `figures/` | Generated plots and the CSVs behind them |
| `legacy/` | Superseded, broken, or one-off scripts, kept for reference (see `legacy/README.md`) |

**Not included:** the continuous-space results (about 8 GB: `Experiments_constraint_continuous*/`, `Experiments_unconstraint_continuous/`) and older result folders. The continuous-run notebooks and compare scripts need those folders at the repository root.

## Setup

The code was developed in a conda env named `quantum` with:

- Python 3.8
- torch 2.0.0, botorch 0.8.5, gpytorch 1.10, scikit-learn 1.2.0
- qiskit 1.2.4, qiskit-aer 0.17.2, qiskit-algorithms 0.3.1, qiskit-finance 0.4.1, qiskit-ibm-runtime 0.34.0
- `ansys-mapdl-core`: imported at module level, but ANSYS itself isn't needed for surrogate runs

`requirements.txt` is out of date.

The code calls `gpytorch.kernels.RFFKernel.get_features`, which stock gpytorch 1.10 lacks. Add this method to `RFFKernel` in `gpytorch/kernels/rff_kernel.py`:

```python
def get_features(self, x, num_dims, normalize=False):
    if not hasattr(self, "randn_weights"):
        self._init_weights(num_dims, self.num_samples)
    return self._featurize(x, normalize=normalize)
```

## Quick start

Run scripts from the repository root, because paths resolve against the working directory. Each run overwrites the result files it writes, so back up the output folder first.

```bash
conda activate quantum
python classic_safeset_discrete.py --query_budget 20000
python quantum_safeset_discrete.py --query_budget 20000
```

Run the synthetic benchmark from inside its folder:

```bash
cd simulation_study
python replot_multi_init_from_pkl.py    # rebuild the figure and table from cached results
python multi_init_cumulative_regret.py  # full multi-seed run (also tries IBM hardware)
```

The real-hardware scripts use `qiskit_ibm_runtime.QiskitRuntimeService()`, which needs a saved IBM Quantum account with QPU access.

## Further reading

- `CLAUDE.md`: algorithm details, script map, results layout, and known issues
- `analyze_discrete_README.md`, `result_README.md`, `simulation_study/README.md`: descriptions of the analyses
