# Quantum-Enhanced Safe Bayesian Optimization for Fuselage Shape Adjustment

**Paper:** [Quantum Safe-Set Bayesian Optimization for Quality Improvement in Fuselage Assembly](https://arxiv.org/abs/2511.22090), Jiayu Liu, Chong Liu, Trevor Rhone and Yinan Wang (arXiv:2511.22090).

Research code comparing classical safe-set Bayesian optimization (BO) with a quantum variant that estimates the objective by Quantum Amplitude Estimation (QAE). The application is choosing actuator forces that bring an initial fuselage shape close to a target shape. A Tsai-Wu failure-criterion constraint keeps every queried configuration structurally safe. A 2D synthetic benchmark in `simulation_study/` validates the algorithms first.

## Repository contents

| Path | Contents |
|---|---|
| `*_safeset_*.py`, `*_acl_*.py`, `*_bo_unconstrained*.py`, `quantum_bo_discrete.py` | Experiment scripts: classical and quantum safe-set BO, the BO-ACL baseline, and unconstrained baselines |
| `bo_env_constraints.py`, `quantum_bo_env_constraint.py`, `circuit_utils.py`, `utils.py` | Surrogate environments, custom iterative amplitude estimation, and shared GP/sampling helpers |
| `surrogate_likeDu_v22.joblib`, `surrogate_tsaiwu.joblib`, `surrogate_modeling/` | Linear shape surrogate, Tsai-Wu constraint surrogate, and surrogate training data |
| `FuselageActuators/` | ANSYS input decks (`AnsysFiles/`), displacement shapes (`Shapes/`), and a live-ANSYS Gym environment |
| `simulation_study/` | 2D synthetic benchmark (its cached results and figures are in the history, see below) |
| `analysis/` | Analysis notebooks and the compare, plot, extract and report scripts (run the scripts from the repository root) |
| `sweeps/` | Hyperparameter sweeps and their JSON results |
| `docs/` | Write-ups of the discrete, continuous and sweep results |
| `scripts/` | Shell driver for the force-range experiments |
| `legacy/` | Superseded, broken, or one-off scripts, kept for reference (see `legacy/README.md`) |

**Not included:** the continuous-space results (about 8 GB: `Experiments_constraint_continuous*/`, `Experiments_unconstraint_continuous/`) and older result folders. The continuous-run notebooks and compare scripts need those folders at the repository root. The discrete-grid results (`Experiments_constraints/`), the generated figures (`figures/`) and the simulation-study result files were removed from the repository on 2026-09-25; they are in the history up to commit `c5a1b8f` (`git checkout c5a1b8f -- Experiments_constraints figures simulation_study`).

## Setup

The code was developed in a conda env named `quantum` with:

- Python 3.8
- torch 2.0.0, botorch 0.8.5, gpytorch 1.10, scikit-learn 1.2.0
- qiskit 1.2.4, qiskit-aer 0.17.2, qiskit-algorithms 0.3.1, qiskit-finance 0.4.1, qiskit-ibm-runtime 0.34.0
- `ansys-mapdl-core`: imported at module level, but ANSYS itself isn't needed for surrogate runs

`requirements.txt` pins these versions.

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
python multi_init_cumulative_regret.py  # full multi-seed run (IBM hardware only with INCLUDE_REAL_QUANTUM = True)
```

The real-hardware scripts use `qiskit_ibm_runtime.QiskitRuntimeService()`, which needs a saved IBM Quantum account with QPU access.

## Further reading

- `CLAUDE.md`: algorithm details, script map, results layout, and known issues
- `docs/analyze_discrete_README.md`, `docs/result_README.md`, `docs/hyperparameter_sweep_results.md`, `simulation_study/README.md`: descriptions of the analyses

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{liu2026quantumsafesetbayesianoptimization,
      title={Quantum Safe-Set Bayesian Optimization for Quality Improvement in Fuselage Assembly},
      author={Jiayu Liu and Chong Liu and Trevor Rhone and Yinan Wang},
      year={2026},
      eprint={2511.22090},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.22090},
}
```
