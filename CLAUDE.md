# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research code comparing classical safe-set Bayesian optimization (BO) with a quantum variant for fuselage actuator-force optimization under a Tsai-Wu structural-safety constraint. The classical and quantum pipelines share the same GP safe-set and acquisition logic. They differ only in how the **objective** is estimated at a queried point: Monte Carlo sampling (sample count from a Chebyshev bound, ~1/ε²) versus Quantum Amplitude Estimation (QAE, ~1/ε oracle calls). The constraint is never estimated by QAE.

- **Action:** 18-dim normalized actuator forces. The env applies `forces += action * force_scale` (lb, default 1000).
- **Objective:** shape error, MAE = mean |initPos[:, :2] + coef_·F − targetPos[:, :2]| over 354 coordinates. `coef_` (354×18) comes from the linear surrogate `surrogate_likeDu_v22.joblib`; the intercept is dropped. The default task is init shape DP52 → target DP53. Lower is better.
- **Constraint:** c(x) = 1 − FI(x) from `surrogate_tsaiwu.joblib` (sklearn Pipeline: StandardScaler + GaussianProcessRegressor). It is evaluated exactly (noise-free) on the *normalized* action, and c ≥ 0 means safe. Because FI ignores `force_scale`, the safe region is identical across force scales.
- **Discrete task:** a 21×21 grid (`linspace(-1, 1, 21)`) over dims 0 and 17, built by `sample_sobol_on_grid`. The function is defined inline in each discrete script and, despite its name, is not Sobol. 294 of 441 points are safe; the safe optimum is MAE 0.07159 at grid index 152.
- **Continuous task:** `--actuator_count` picks the active dims: 2→(0,17), 4→(0,1,16,17), 6→(0,1,2,15,16,17), 8→(0–3,14–17). Actions are clamped to [−0.5, 0.5], so the max force is 0.5·force_scale.

## Layout

- **Top level:** the experiment scripts, the env modules (`bo_env_constraints.py`, `quantum_bo_env_constraint.py`, `circuit_utils.py`, `utils.py`), the two surrogates and `FuselageActuators/`. They stay at the root because every script imports the env modules as top-level modules and resolves data paths against the CWD.
- **`analysis/`:** the compare, plot, extract and report scripts plus the analysis notebooks. Run the scripts from the repo root (`python analysis/<script>.py`); the notebooks chdir to the repo root in their first cell.
- **`sweeps/`:** the two hyperparameter sweeps and their JSON results. Both resolve data against the repo root and write their JSON next to themselves.
- **`docs/`:** the result write-ups (`analyze_discrete_README.md`, `result_README.md`, `hyperparameter_sweep_results.md`).
- **`scripts/`:** `run_force_range_experiments.sh`.
- **Surrogate copies:** `surrogate_modeling/` (training data and notebook) and `FuselageActuators/Surrogates/` hold byte-identical copies of `surrogate_likeDu_v22.joblib`. The scripts load the root copy.
- **`figures/`:** where the compare scripts and `analysis/analyze_discrete.ipynb` write plots and CSVs. The directory was removed from the repo with the results (see **Git scope**), and the compare scripts do not create it, so `mkdir figures` before running them.
- **`legacy/`:** superseded, broken, or one-off code, listed in `legacy/README.md`. Nothing at the top level imports from it. To run a legacy script, use `PYTHONPATH=. python legacy/<script>.py` from the root.
- **Origin:** this folder started as a clean clone of the original working folder `/data/liuj35/quan_fuselage` (no longer configured as a remote). That folder still holds the ~8 GB of continuous results and the `.history/` editor snapshots. The only remote is `origin`, the GitHub repository `JiayuLiu666/Fuselage-Assembly`.

## Gotchas

- **Runs overwrite results.** Output paths are fixed, with no run id, and scripts re-save during the run. Re-running any experiment script overwrites the canonical `.pth` files. `sweep_quantum_safeset_exact.py` rewrites `Experiments_constraints/Quantum_Discrete_cUCB/exp_set_1/` once per config. The discrete `.pth` files are no longer tracked (see **Git scope**), so restore the canonical set from history after a smoke test; continuous scripts accept `--results_root`.
- **Run from the repo root.** Surrogates, `FuselageActuators/{AnsysFiles,Shapes}/Test/`, and result dirs resolve relative to CWD; this includes the `analysis/` scripts. The exception is `simulation_study/`, which must run from inside that directory.
- **Filenames embed `str(obs_noise)`.** The default `0.1**2` produces `0.010000000000000002…`, but `--obs_noise 0.01` produces `0.01…`. To match existing files, omit the flag or pass the exact repr (`0.010000000000000002`, `0.04000000000000001`). `Classic_Discrete_Unconstrained/` was run with `--obs_noise 0.01`, so its σ = 0.1 files carry the short `0.01` prefix and a glob for `0.010000000000000002*` misses them.
- **Git scope.** The discrete results, `figures/` and the `simulation_study/` result files (`results_*.{pkl,txt}`, `multi_init_checkpoint.pkl`, `multi_init_stats.txt` and the PNGs) were removed from the repo on 2026-09-25. The last commit that carries them is `c5a1b8f`; `git checkout c5a1b8f -- Experiments_constraints figures simulation_study` restores all of them into the working tree and index (`git restore --staged .` afterwards if they should stay out of the next commit). Still tracked: the sweep JSONs and `simulation_study/rerun_archives/`. `.gitignore` ignores every `Experiments*` entry at the root except `Experiments_constraints/**/*.pth`, so re-running a discrete script leaves untracked `.pth` files in `git status`. The continuous results live only in the original folder.
  - To analyze them here, point `analysis/compare_actuator_count_cumulative_regret.py` at them with `--root-4/6/8`, or symlink the dirs into the root; `analysis/compare_force_range_cumulative_regret.py` has hardcoded roots.
  - Never `git add -f` them.
- **`scripts/run_force_range_experiments.sh`** cds to the repo root and activates `quantum` itself (conda base from `conda info --base`, falling back to `~/anaconda3`). It runs the three constrained continuous scripts at 500 and 200 lb one after another.
- **`reset()` return order differs.** The classical env returns `(file, error_init)`; the quantum env returns `(error_init, file)`. `classic_safeset_continuous.py` and `classic_bo_unconstrained.py` unpack it backwards, so their saved `error_init` is a filename string.
- **Forces accumulate in the env.** Scripts call `env.reset(...)` after every evaluation; keep that when adding code paths.
- **`classic_acl_discrete.py` samples differently.** Its `--method` defaults to `non_monte_carlo`, which ignores ε_t and draws a fixed n = ⌈obs_noise/(eps_max²·0.05)⌉ (at least `min_shots`) per query: 500 at σ = 0.2 and 125 at σ = 0.1. Only its warmup helper uses `chebyshev`.
- **Continuous defaults differ per script.** `--actuator_count` defaults to 8 in `classic_safeset_continuous.py` and `classic_bo_unconstrained.py` but 6 in the quantum and ACL scripts, and `classic_bo_unconstrained.py` defaults to `obs_noise 0.2**2` where every other continuous script uses `0.1**2`. Pass both flags explicitly for comparable runs.
- **`analysis/analyze.ipynb` uses the legacy 8-actuator dir names** (`Quantum_EXP_10_…`, `Classic_SafeSet_1.0_…`, `Classic_Continuous_ACL_1.0_…`). Results written by the current code carry `_8_` in the name, so substitute those before running it on new runs. `legacy/compare_cumulative_regret.ipynb` is stale (it reads `exp_set_0` dirs that don't exist); use `analysis/analyze_discrete.ipynb`.

## Environment

```bash
conda activate quantum
```

- **Versions (Python 3.8):** torch 2.0.0, botorch 0.8.5, gpytorch 1.10, scikit-learn 1.2.0, qiskit 1.2.4, qiskit-aer 0.17.2, qiskit-algorithms 0.3.1, qiskit-finance 0.4.1, qiskit-ibm-runtime 0.34.0. `requirements.txt` pins these versions (plus numpy 1.23.4, scipy 1.9.3, matplotlib 3.7.5, pandas 1.5.2, ansys-mapdl-core 0.67.0); the gpytorch patch below is still required.
- **Two envs:** `quantum` is the working env. `quantum_fresh` has the same pins but an unpatched gpytorch, so the RFF scripts fail there until the method below is added.
- **Patched gpytorch:** the scripts call `RFFKernel.get_features`, which stock gpytorch 1.10 lacks. It is added in the user-site copy `~/.local/lib/python3.8/site-packages/gpytorch/kernels/rff_kernel.py`, which shadows the conda env's gpytorch. That user site also supplies scikit-learn 1.2.0, numpy 1.23.4, joblib and pandas (the conda env itself has scikit-learn 1.3.0, numpy 1.24.3 and no pandas), so never run with `python -s` or `PYTHONNOUSERSITE=1`. On a new machine, add this method to `RFFKernel`:
  ```python
  def get_features(self, x, num_dims, normalize=False):
      if not hasattr(self, "randn_weights"):
          self._init_weights(num_dims, self.num_samples)
      return self._featurize(x, normalize=normalize)
  ```
- **`ansys.mapdl`:** the env modules import it at the top level, so it must be installed. MAPDL launches are commented out and the envs run surrogate-only.
- **Expected warning:** the joblib surrogates were pickled with scikit-learn 1.1.1, so loading them in a fresh interpreter prints version-mismatch warnings. They are harmless.
- **GPUs:** this machine has four RTX A5000s. The discrete scripts use `cuda:0` when available and the CPU otherwise; the continuous scripts assign trial `i` to GPU `i % device_count`. Use `CUDA_VISIBLE_DEVICES` to choose GPUs.
- **No tests or build:** there is no test suite, linter or build. Check syntax with `python -m py_compile <file>`. Smoke-test a discrete script with a small budget, then remove the results it wrote, or restore the canonical ones from history (all 5 trials take about 10 s classical and 20 s quantum, with warmup plus one to four steps each):
  ```bash
  python classic_safeset_discrete.py --query_budget 300
  rm -rf Experiments_constraints/   # or: git checkout c5a1b8f -- Experiments_constraints/
  ```
  For the continuous scripts pass `--results_root <scratch dir>` instead. `simulation_study/_smoke_compare.py` is the benchmark's smoke test.

## Running Experiments

```bash
# Discrete (21×21 grid). 5 sequential trials, seed = trial index 0–4, one GPU.
python classic_safeset_discrete.py --query_budget 20000   # default obs_noise 0.2**2, lam0 1.0
python quantum_safeset_discrete.py --query_budget 20000   # default obs_noise 0.1**2, lam0 0.5

# Continuous. Outer loop over 10 exp_sets (shape pairs); 5 trials in a ProcessPoolExecutor, GPU = trial % n_gpu.
python classic_safeset_continuous.py --actuator_count 8 --force_scale 1000
python quantum_safeset_continuous.py --actuator_count 8   # --actuator_count defaults to 6 here and in classic_acl_continuous.py

# Scaling studies and plots (compare scripts only read results)
bash scripts/run_force_range_experiments.sh               # 3 continuous methods at 500 and 200 lb
python analysis/compare_force_range_cumulative_regret.py --exp-sets 0
python analysis/compare_actuator_count_cumulative_regret.py   # _expset0 output = --exp-sets 0; _single = --exp-sets 0 --trials 2
```

- **Shared discrete flags:** `--query_budget` (oracle queries, warmup excluded; the last step may overshoot), `--eps_max 0.04`, `--min_shots 20`, `--M_features 400`, `--B 3.0`, `--obj_ls 0.2`, `--t0 10`, `--lam_p 2.0`, `--init_num_points 5`.
- **Continuous flags:** `--max_iteration` is the oracle-query budget (default 50000). Others: `--warmup_points 200`, `--outer_seeds 10`, `--inner_trials 5`, `--B 1.0`, 256 RFF features (`--M_features`, or `--M_target` in the quantum script), and `--candidate_pool_*`.
- **`classic_acl_discrete.py`:** `--noise_level` is σ (default 0.2), not a variance. `--seed_offset` changes seeds but not filenames.

## Architecture

### Safe-set BO loop (`classic_safeset_discrete.py` / `quantum_safeset_discrete.py`)

Both scripts build their GPs inline (`initialize_c_model`, `initialize_f_model`); neither imports GP code from `utils.py`. Each iteration:

1. **Objective model (W-GP-UCB).** φ(x) comes from `RFFKernel.get_features` (2·M features, Matérn-2.5; fixed lengthscale `--obj_ls` on dims 0/17 and 0.6931 elsewhere). The design matrix is `V_t = λI + Σ φ(xᵢ)φ(xᵢ)ᵀ/εᵢ²` with λ = 1 fixed. `W_GP_UCB_scores` returns UCB_f = mean + √β_t·σ, where β_t = (1 + B·|log t|)².
2. **Constraint GP.** A `FixedNoiseGP` with Matérn-2.5 and fixed hyperparameters: lengthscale 0.50 on dims 0/17, 0.6931 elsewhere, outputscale 1, no MLL fit. Each step it absorbs the exact c(x) via `get_fantasy_model` (noise 1e-6). The safe set is S = {μ_c − √3·σ_c ≥ 0} (`beta_c = 3.0` is hardcoded). If S is empty, the loop takes argmax LCB.
3. **Acquisition over S.** score = (1 − λ_t)·minmax(UCB_f) + λ_t·minmax(−|μ_c/σ_c|), where λ_t = lam0·(t0/(t0 + t))^lam_p. This is a boundary-expansion weight that decays toward pure exploitation.
4. **Query precision.** ε_t = min(eps_max, σ_f(x)); `stage_epsilon` divides by the fixed λ = 1, not by λ_t. The env returns an ε_t-accurate estimate and its oracle cost. The loop runs until the cumulative cost reaches `--query_budget`.
5. **Warmup.** `init_num_points` grid points are drawn from the truly safe set (1 − FI ≥ 0), with probability ∝ safety margin (`RandomState(trial)`). The same points seed both GPs. `sample_kmeans_safe_subspace` (utils.py) and `select_safe_initial_points` are unused.

The continuous scripts use the same score over a scrambled-Sobol candidate pool in the active subspace, followed by a local refinement step. They start from 200 LHS warmup points. `utils.build_train_gp_with_rff` (1024 RFF features; callers allow up to 4 refits while train R² ≤ 0.7) runs on those points only to learn the objective lengthscale.

### Objective estimation: classical vs quantum

- **Classical: `bo_env_constraints.ClassicFuselageEnv.step_surrogate(action, method, device, eps)`**
  - Averages n draws of N(−MAE, obs_noise).
  - `method` is required; scripts use `'chebyshev'`: n = max(min(⌈var/(ε²δ)⌉, 29999), min_shots), δ = 0.05. `clt`, `hoeffding` and `non_monte_carlo` also exist.
  - Returns `(obs, true_mae, n)`.
  - The draws use unseeded `random.gauss`, so classical runs aren't bit-reproducible.
  - This env has no Tsai-Wu code; the classical scripts load `surrogate_tsaiwu.joblib` themselves.
- **Quantum: `quantum_bo_env_constraint.QuantumFuselageEnv.step_surrogate(action, eps, device)`**
  - Encodes N(−MAE(x), obs_noise) in a 6-qubit `qiskit_finance` `NormalDistribution` (whose `sigma` argument is a variance), truncated at ±3√obs_noise, followed by an identity `LinearAmplitudeFunction`.
  - Runs IAE at amplitude precision clip(ε/(3σ), 1e-6, 0.5), with α = 0.05.
  - Oracle cost = Σ shots × k over the IAE rounds.
  - Returns `(obs, true_mae, oracle_queries, c_val, empirical_variance)`.
  - **Default path:** `qiskit_algorithms.IterativeAmplitudeEstimation` with the V1 `qiskit.primitives.Sampler(seed=0)`, so the estimate is deterministic for a given (x, ε).
  - **With `backend`:** uses the custom IAE in `circuit_utils.py`. It builds Q^k A|0⟩ circuits by hand, transpiles each round, and submits via `qiskit_ibm_runtime.SamplerV2(mode=backend)`. `find_next_k` picks the next Grover power, and CIs are Clopper-Pearson.
- **Saved values:** `response` is the noisy −MAE/error_init (error_init is the zero-force MAE), and the BO maximizes it. `true_response` is the raw MAE.
- **Legacy envs:** `legacy/bo_env.py` and `legacy/quantum_bo_env.py` are used only by the legacy scripts beside them. The live-ANSYS Gym env is `FuselageActuators/FuselageActuators_env_v22.py`.

### Script map

| Role | Discrete (grid) | Continuous |
|---|---|---|
| Classical safe-set cUCB | `classic_safeset_discrete.py` | `classic_safeset_continuous.py` |
| Quantum safe-set cUCB | `quantum_safeset_discrete.py` (IBM HW: `quantum_safeset_discrete_real.py`) | `quantum_safeset_continuous.py` |
| BO-ACL baseline | `classic_acl_discrete.py` | `classic_acl_continuous.py` |
| Unconstrained, classical | `classic_bo_unconstrained_discrete.py` | `classic_bo_unconstrained.py` |
| Unconstrained, quantum | `quantum_bo_discrete.py` | `quantum_bo_unconstrained.py` |

The unconstrained scripts share seeds, shape lists and the safe init with their safe-set counterparts, so the curves are comparable. The TuRBO, POF, ADMM and `quantum_bo.py` scripts are in `legacy/`; `quantum_bo.py` is not aligned with the other baselines.

### Results layout

- **Discrete (in git history up to `c5a1b8f`; see Git scope):** `Experiments_constraints/<Method>/exp_set_1/{obs_noise}{prefix}training_data_{trial}_.pth`.
  - Methods: `Classic_Discrete_cUCB`, `Quantum_Discrete_cUCB`, `Quantum_Discrete_cUCB_Real`, `Classic_ACL_Discrete`, `Classic_Discrete_Unconstrained`, `Quantum_Discrete_Unconstrained`.
  - Prefixes: none (classical), `quan_` (quantum), `acl_`.
- **Continuous (local only):** `Experiments_constraint_continuous[_actuators_N][_force_F]/<Method>_<count>_<eta>_<lam>_<B>_[multi_gpu_]noise_<var>/exp_set_<k>/`.
  - `exp_set_k` is the k-th (init → target) shape pair: DP52→53, 50→53, 49→57, 45→53, 45→58, 48→53, 43→53, 53→55, 60→44, 54→53.
  - The existing 8-actuator dirs in `Experiments_constraint_continuous/` use legacy names without the count (`Classic_SafeSet_1.0_…`, `Quantum_EXP_10_…`). Re-running current code creates `*_8_*` dirs beside them, and the compare scripts would then pool both.
  - The unconstrained runs mirror this layout under `Experiments_unconstraint_continuous[...]`.
- **Discrete safe-set `.pth` schema:**
  - `actions` [N,18]; `response` and `true_response` [N,1]; `queries` [N,1] int64; `uncertainty` (a list of ε_t).
  - `active_records` has 35 per-step diagnostics, e.g. `c_val`, `FI`, `mu_c`/`sig_c`/`lcb_c`, `safe_mask_sum_*`, `eps`.
  - `metadata` holds `query_budget`, `trial`, `seed`, `num_active_steps`, `final_total_budget`.
  - Warmup rows aren't saved.
- **ACL and continuous `.pth` files:** no `active_records` or `metadata`; they add `error_init` (and `constraint_margin` for ACL). ACL files include the warmup rows, with queries = 0.
- **Cumulative regret convention:** each row is repeated `queries` times, and regret = Σ|f* − true_response|.
  - Discrete: f* = 0.07159.
  - Continuous compare scripts: f* = `--f-opt` (default 0), so their "regret" is cumulative MAE.

### Analysis

Everything below lives in `analysis/`; run the scripts from the repo root.

- `analyze_discrete.ipynb`: the discrete analysis; metrics are defined in `docs/analyze_discrete_README.md`.
- `analyze.ipynb`: the continuous analysis; results in `docs/result_README.md`.
- `report_violation_rate.py`: recomputes violation rates for `Experiments_constraints/` from the Tsai-Wu surrogate.
- `print_minimums.py` reads only `Experiments_constraint_continuous/`; `print_mins_script.py` is the same report over a recursive glob from the CWD.
- `analyze_constraint_continuous_regret.ipynb`: cumulative regret and running minimum for one continuous root (`ROOT` at the top, currently the 4-actuator dir).
- Shape-gap figures: `extract_shape_gap_force_configs.py` reads one continuous `.pth` plus the init/target shapes and writes a `shape_gap_reduction_extract/` dir (CSVs, `.npz`, `summary.json`); `plot_shape_gap_reduction_panels.py --extract-dir` and `plot_classic_quantum_shape_gap_comparison.py --quantum-dir/--classic-dir` plot those extracts. All defaults point into `Experiments_constraint_continuous/`, which exists only in the original folder.
- `draw_shape.ipynb` draws fuselage layers and shapes; `train_GP.ipynb` only sanity-checks the Tsai-Wu surrogate.
- The READMEs' numbers are partly stale because runs were overwritten later; recompute from the data.

### Simulation study (`simulation_study/`)

This is a self-contained 2D benchmark (Paper Simulation 2). Run it from inside the directory.

- **Task:** minimize g(x) = x₁² − sin(4x₂²) subject to x₂ − x₁² ≥ ξ (ξ = 0), on a 25×25 grid over [−1, 1]².
- **Methods:**
  - `safe_set_bo.py`: classical cUCB.
  - `quantum_safe_bo.py`: `qiskit_algorithms` IAE.
  - `quantum_safe_bo_real.py`: IBM hardware; it imports the root `circuit_utils.py` via `sys.path`.
  - `unconstrained_bo.py` and `ACL_paper.py`: baselines.
  - `experiment_env.py`: builds the grid and functions.
- **Configuration:** module-level constants at the top of each driver (`SEED`, `N_INIT`, `ORACLE_BUDGET=500`, `OBJ_NOISE=0.3` std, `BETA_C`, `LAM0`, …); there are no CLI flags. Both drivers use LAM0 0.8, LAM_T0 10 and LAM_P 1.0; `compare_safe_methods_shared_init.py` currently sets SEED 0 and N_INIT 10.
- **Drivers:**
  - Both drivers below have an `INCLUDE_REAL_QUANTUM` constant, currently `False`; setting it to `True` adds the IBM-hardware Q-Safe BO.
  - `compare_safe_methods_shared_init.py`: single seed. With the flag off, its pickle stores `quantum_real: None`, which `replot_from_pkl.py` does not handle.
  - `multi_init_cumulative_regret.py`: seeds 5–9. It resumes from `multi_init_checkpoint.pkl` and skips finished (method, seed) runs, so turning the flag on later runs only the hardware method. The `c5a1b8f` checkpoint also holds the hardware runs; without any checkpoint the script re-runs every enabled method.
  - `_smoke_compare.py`: tiny-grid smoke test of all five methods (grid 10, no plots or pickles); it also tries a 127-qubit IBM backend.
  - `tune_hyperparams.py`: random 80-config sweep of the classical method; it only prints the top 5.
- **Replot instead of re-running:**
  - `replot_multi_init_from_pkl.py` rebuilds the table and figure from the checkpoint.
  - `replot_from_pkl.py` and `replot_from_txt.py` rebuild `comparison_regret.png` from `results_<ts>.{pkl,txt}`; the `.txt` embeds the regret curves.
- **Headline numbers:** `multi_init_stats.txt` (`git show c5a1b8f:simulation_study/multi_init_stats.txt`). The README's tables, seed and path (`simulated_study/`) are older.
- **Unused or one-off:** `compare_safe_methods_lib.py` (imported by nothing) and `cleanup_tmp.py` (deleted an old scratch directory) now sit in `legacy/`; `rerun_archives/` keeps an earlier multi-init run's outputs.
- **Regret conventions:** quantum runs store `cumu_regret_expanded` (per oracle query) and classical runs store `queried_cumu_regret_hist`; both are padded or trimmed to `ORACLE_BUDGET`. Simple regret is noise-free: |global safe optimum − best true objective among feasible queried points|.

### IBM hardware

- **Default account:** every script calls `QiskitRuntimeService()` with no args, which loads the default saved account. That account may have no QPUs, and then `least_busy` raises `QiskitBackendNotFoundError`. Load a QPU-bearing account explicitly, e.g. `QiskitRuntimeService(name="CS102")`.
- **Channels:** qiskit-ibm-runtime 0.34 accepts only `channel` ∈ {`ibm_cloud`, `ibm_quantum`}, so saved `ibm_quantum_platform` accounts fail to load.
- **Silent fallback:** if the backend lookup fails, `quantum_safeset_discrete_real.py` quietly runs the simulator but still writes to `Quantum_Discrete_cUCB_Real/`. Check its log for the selected backend. Its defaults also differ: obs_noise 0.04, min_shots 100, lam0 1.0.

## Hyperparameters

- **Discrete defaults are the tuned values:** B 3.0, obj_ls 0.2, t0 10, lam_p 2.0, init_num_points 5, M 400, eps_max 0.04.
  - lam0 is 0.5 in the quantum script, the winner of `sweeps/sweep_quantum_safeset_exact.py` (see `sweeps/quantum_safeset_exact_sweep_best.json`), and 1.0 in the classical script. The published discrete comparison therefore used different lam0 values. That sweep runs `quantum_safeset_discrete.py` as a subprocess per config (edit its `configs` list) and logs to `sweep_logs/`.
- **Classical proxy sweep:** `sweeps/sweep_hyperparams.py` (no CLI; edit `CONFIGS` and `NOISE_LEVELS` in `main()`) reimplements the loop without the env and writes `sweeps/sweep_results.json`, summarized in `docs/hyperparameter_sweep_results.md`. Findings:
  - With 1 warmup point, only 2–3 of 5 seeds converge; 3 or more points reach 5/5.
  - B = 1 stalls safe-set growth (about 45% of the grid explored, versus about 49% at B = 3).
  - obj_ls 0.2 versus 0.5 made no measurable difference.
  - "Converged" there means best MAE ≤ 0.08.
