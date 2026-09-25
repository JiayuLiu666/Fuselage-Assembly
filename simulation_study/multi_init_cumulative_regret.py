"""
multi_init_cumulative_regret.py
===============================
Runs the 5-method benchmark from compare_safe_methods_shared_init.py over
N_RUNS independent seeds (= different random initial points).

Each seed generates a fresh shared init_idx so all methods start from
identical initial data within that run.

Outputs
-------
multi_init_regret_mean_std.png  — mean ± std cumulative-regret curves
multi_init_stats.txt            — printed + saved summary statistics table
"""

import sys, os, pickle
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib
matplotlib.use('Agg')           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

from experiment_env   import build_paper_sim2_environment, sample_initial_indices, sample_stratified_initial_indices
from safe_set_bo      import run_safe_bo_simulation, resolve_n_init
from quantum_safe_bo  import run_quantum_safe_bo_simulation
from ACL_paper        import run_bo_acl_simulation2
from unconstrained_bo import run_unconstrained_bo_simulation

# ============================================================
# Configuration  (matches compare_safe_methods_shared_init.py)
# ============================================================
N_RUNS        = 5                          # number of independent runs
BASE_SEEDS    = list(range(5, 5 + N_RUNS)) # seeds 5 … 9

GRID_SIZE     = 25
XI            = 0        # constraint threshold: h(x) >= xi is feasible
N_INIT        = 5        # shared initial points per run
ORACLE_BUDGET = 500      # total oracle calls per method
OBJ_NOISE     = 0.3      # matches compare_safe_methods_shared_init.py
CON_NOISE     = 1e-6
MODE          = 'min'
BETA_C        = 3.0

# lambda_by_stage schedule (from compare_safe_methods_shared_init.py)
LAM0   = 0.8
LAM_T0 = 10
LAM_P  = 1.0

# RFF kernel params
M_RFF            = 256
LAM_RFF          = 1.0
LENGTHSCALE_RFF  = np.array([0.2, 0.2])
V_KERNEL_RFF     = 1.0

# ── Methods to run ──
# Set INCLUDE_REAL_QUANTUM = True if you have IBM Quantum access and want
# to include the real-hardware quantum method (slow — requires live backend).
# ── Real quantum hardware (disabled for local testing) ──
INCLUDE_REAL_QUANTUM = False
METHODS = ['C-Safe BO', 'Q-Safe BO', 'Unconstrained BO', 'BO-ACL']
if INCLUDE_REAL_QUANTUM:
    from quantum_safe_bo_real import run_quantum_safe_bo_simulation_real
    METHODS.insert(2, 'Q-Safe BO (Real)')

COLORS = {
    'C-Safe BO'                : '#2196F3',
    'Q-Safe BO'        : '#9C27B0',
    'Q-Safe BO (Real)' : '#E91E63',
    'BO-ACL'                 : '#4CAF50',
    'Unconstrained BO'       : '#FF5722',
}
LINESTYLES = {
    'C-Safe BO'                : '-',
    'Q-Safe BO'        : '--',
    'Q-Safe BO (Real)' : '-.',
    'BO-ACL'                 : '-.',
    'Unconstrained BO'       : ':',
}

# ============================================================
# Checkpointing
# ============================================================
CHECKPOINT_PATH = 'multi_init_checkpoint.pkl'

def _save_checkpoint(all_results):
    """Persist current results to disk so we can resume after a crash."""
    with open(CHECKPOINT_PATH, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'     [checkpoint] saved → {CHECKPOINT_PATH}')


def _load_checkpoint():
    """Load previously saved results, or return None."""
    if os.path.isfile(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'rb') as f:
            data = pickle.load(f)
        print(f'[checkpoint] Loaded existing checkpoint from {CHECKPOINT_PATH}')
        _drop_stale_checkpoint_entries(data)
        return data
    return None


def _has_consistent_constraint_rows(method, res):
    """Safe-set methods should have one constraint row per objective row."""
    if method not in ('Safe BO', 'Q-Safe BO', 'Q-Safe BO (Real)'):
        return True

    train_X = res.get('train_X')
    c_train_X = res.get('c_train_X')
    c_train_C = res.get('c_train_C')
    if train_X is None or c_train_X is None or c_train_C is None:
        return True

    train_rows = int(train_X.shape[0])
    con_x_rows = int(c_train_X.shape[0])
    con_y_rows = int(c_train_C.shape[0])
    return train_rows == con_x_rows == con_y_rows


def _drop_stale_checkpoint_entries(all_results):
    """
    Older Safe BO checkpoints duplicated the initial constraint observations.
    Truncate each affected method at the first stale run so subsequent runs are
    recomputed in order instead of being silently skipped.
    """
    for method, runs in list(all_results.items()):
        if not isinstance(runs, list):
            continue
        for i, res in enumerate(runs):
            if not _has_consistent_constraint_rows(method, res):
                print(
                    f'  [checkpoint] {method} run {i} has stale constraint rows; '
                    f'truncating cached {method} results from this run onward.'
                )
                all_results[method] = runs[:i]
                break


def _run_already_done(all_results, method, run_i):
    """Check whether (method, run_i) was already completed."""
    return len(all_results.get(method, [])) > run_i


# ============================================================
# Helpers
# ============================================================
def _safe_rate(res):
    """Fraction of post-init queries that landed in the true safe set."""
    n  = int(res['n_init'])
    zz = np.asarray(res['zz'])
    xi = float(res['xi'])
    qs = np.asarray(res['queried_idx'], dtype=int)[n:]
    return float(np.mean(zz[qs] >= xi)) if len(qs) > 0 else float('nan')


def _pad_or_trim(arr, length):
    """Return arr exactly `length` long; pad with last value if needed."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= length:
        return arr[:length]
    fval = arr[-1] if len(arr) > 0 else 0.0
    return np.concatenate([arr, np.full(length - len(arr), fval)])


def _get_cumu_regret(res, budget):
    """Cumulative-regret array aligned to `budget` oracle calls."""
    if 'cumu_regret_expanded' in res:          # Quantum BO
        return _pad_or_trim(res['cumu_regret_expanded'], budget)
    return _pad_or_trim(res['queried_cumu_regret_hist'], budget)


def _noise_free_simple_regret(res):
    """Compute simple regret from noise-free ground truth."""
    yy   = np.asarray(res['yy'])
    zz   = np.asarray(res['zz'])
    xi   = float(res['xi'])
    mode = res['mode']
    q    = np.asarray(res['queried_idx'], dtype=int)
    safe_q = q[zz[q] >= xi]
    if len(safe_q) == 0:
        return float('nan')
    best = float(np.min(yy[safe_q])) if mode == 'min' else float(np.max(yy[safe_q]))
    return abs(best - float(res['global_safe_opt']))


# ============================================================
# Main experiment loop
# ============================================================
print(f'Running {N_RUNS} experiments with different initial points')
print(f'  grid={GRID_SIZE}², ξ={XI}, budget={ORACLE_BUDGET}, noise={OBJ_NOISE}')
print(f'  Methods: {", ".join(METHODS)}')
print('=' * 70)

# Try to resume from a previous checkpoint
_ckpt = _load_checkpoint()
if _ckpt is not None:
    all_results = _ckpt
    # Ensure every method key exists (handles added/removed methods)
    for m in METHODS:
        if m not in all_results:
            all_results[m] = []
    done_summary = {m: len(all_results[m]) for m in METHODS}
    print(f'  Resumed state: {done_summary}')
else:
    all_results = {m: [] for m in METHODS}
    print('  Starting fresh (no checkpoint found).')

for run_i, seed in enumerate(BASE_SEEDS):
    print(f'\n{"="*70}')
    print(f'[Run {run_i+1}/{N_RUNS}]  seed={seed}')
    print(f'{"="*70}')

    # Shared initial design for this seed
    env_ref = build_paper_sim2_environment(
        grid_size=GRID_SIZE, xi=XI, dtype=None,
        constraint_representation='margin',
    )
    N_GRID     = len(env_ref['xx'])
    n_init     = resolve_n_init(N_GRID, n_init=N_INIT)
    safe_mask  = env_ref['safe_true_mask']
    init_idx   = np.array(
        sample_stratified_initial_indices(N_GRID, safe_mask, n_init=n_init, min_safe=3, seed=seed),  # 3/5 guaranteed safe
        dtype=int,
    )
    n_safe_init = int(safe_mask[init_idx].sum())
    print(f'  Shared init_idx (n={len(init_idx)}, safe={n_safe_init}): {init_idx}')

    # 1. Safe BO
    if _run_already_done(all_results, 'C-Safe BO', run_i):
        print('  → C-Safe BO ... [SKIP — already checkpointed]')
    else:
        print('  → C-Safe BO ...')
        res = run_safe_bo_simulation(
            mode=MODE, xi=XI, grid_size=GRID_SIZE,
            n_init=N_INIT, n_iter=ORACLE_BUDGET,
            obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
            beta_f=0.2, beta_c=BETA_C, lam0=0.8, lam_t0=10, lam_p=2.0,
            seed=seed, init_idx=init_idx.copy(),
            M_rff=M_RFF, lam_rff=LAM_RFF,
            lengthscale_rff=0.2, v_kernel_rff=V_KERNEL_RFF,
            con_lengthscale=1.0, con_outputscale=1.0,
        )
        all_results['C-Safe BO'].append(res)
        print(f'     cumu_regret={_get_cumu_regret(res, ORACLE_BUDGET)[-1]:.4f}  '
              f'safe_rate={_safe_rate(res):.3f}')
        _save_checkpoint(all_results)

    # 2. Q-Safe BO
    if _run_already_done(all_results, 'Q-Safe BO', run_i):
        print('  → Q-Safe BO ... [SKIP — already checkpointed]')
    else:
        print('  → Q-Safe BO ...')
        res = run_quantum_safe_bo_simulation(
            mode=MODE, xi=XI, grid_size=GRID_SIZE,
            n_init=N_INIT, oracle_budget=ORACLE_BUDGET,
            obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
            beta_f=0.2, beta_c=BETA_C, lam0=0.8, lam_t0=10, lam_p=2.0,
            seed=seed, init_idx=init_idx.copy(),
            M_rff=M_RFF, lam_rff=LAM_RFF,
            lengthscale_rff=0.2, v_kernel_rff=V_KERNEL_RFF,
            con_lengthscale=1.0, con_outputscale=1.0,
            use_quantum_query=True,
        )
        all_results['Q-Safe BO'].append(res)
        print(f'     oracle_used={res["total_oracle_queries"]}  '
              f'cumu_regret={_get_cumu_regret(res, ORACLE_BUDGET)[-1]:.4f}  '
              f'safe_rate={_safe_rate(res):.3f}')
        _save_checkpoint(all_results)

    # 2b. Q-Safe BO (Real)
    if INCLUDE_REAL_QUANTUM:
        if _run_already_done(all_results, 'Q-Safe BO (Real)', run_i):
            print('  → Q-Safe BO (Real) ... [SKIP — already checkpointed]')
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            print('  → Q-Safe BO (Real) ...')
            service_ibm = QiskitRuntimeService()
            try:
                real_backend = service_ibm.least_busy(
                    operational=True, simulator=False, min_num_qubits=127
                )
                print(f'     Selected backend: {real_backend.name}')
            except Exception as e:
                print(f'     Failed to get backend: {e}')
                real_backend = None
    
            res = run_quantum_safe_bo_simulation_real(
                mode=MODE, xi=XI, grid_size=GRID_SIZE,
                n_init=N_INIT, oracle_budget=ORACLE_BUDGET,
                obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
                beta_f=0.2, beta_c=BETA_C, lam0=0.8, lam_t0=10, lam_p=2.0,
                seed=seed, init_idx=init_idx.copy(),
                M_rff=M_RFF, lam_rff=LAM_RFF,
                lengthscale_rff=0.2, v_kernel_rff=V_KERNEL_RFF,
                con_lengthscale=1.0, con_outputscale=1.0,
                use_quantum_query=True,
                backend=real_backend,
            )
            all_results['Q-Safe BO (Real)'].append(res)
            print(f'     oracle_used={res.get("total_oracle_queries", "N/A")}')
            _save_checkpoint(all_results)

    # 4. BO-ACL
    if _run_already_done(all_results, 'BO-ACL', run_i):
        print('  → BO-ACL ... [SKIP — already checkpointed]')
    else:
        print('  → BO-ACL ...')
        res = run_bo_acl_simulation2(
            mode=MODE, xi=XI, grid_size=GRID_SIZE,
            n_init=N_INIT, n_iter=ORACLE_BUDGET,
            batch_size=2,
            obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
            beta_f=1.0, beta_w=0.2,
            seed=seed, init_idx=init_idx.copy(),
            lengthscale=LENGTHSCALE_RFF.copy(),
        )
        all_results['BO-ACL'].append(res)
        print(f'     cumu_regret={_get_cumu_regret(res, ORACLE_BUDGET)[-1]:.4f}  '
              f'safe_rate={_safe_rate(res):.3f}')
        _save_checkpoint(all_results)

    # 5. Unconstrained BO
    if _run_already_done(all_results, 'Unconstrained BO', run_i):
        print('  → Unconstrained BO ... [SKIP — already checkpointed]')
    else:
        print('  → Unconstrained BO ...')
        res = run_unconstrained_bo_simulation(
            mode=MODE, xi=XI, grid_size=GRID_SIZE,
            n_init=N_INIT, n_iter=ORACLE_BUDGET,
            obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
            beta_f=0.5,
            seed=seed, init_idx=init_idx.copy(),
            M_target=256, lengthscale=0.2, v_kernel=1.0,
        )
        all_results['Unconstrained BO'].append(res)
        print(f'     cumu_regret={_get_cumu_regret(res, ORACLE_BUDGET)[-1]:.4f}  '
              f'safe_rate={_safe_rate(res):.3f}')
        _save_checkpoint(all_results)

print('\n' + '=' * 70)
print('All experiments complete. Aggregating metrics...')


# ============================================================
# Aggregate across seeds
# ============================================================
budget = ORACLE_BUDGET
x_axis = np.arange(1, budget + 1)

cumu_regret_mat    = {}   # shape: [N_RUNS, budget]
final_cumu_all     = {}   # shape: [N_RUNS]
simple_regret_all  = {}   # shape: [N_RUNS]
violation_rate_all = {}   # shape: [N_RUNS]  (= 1 - safe rate)

for m in METHODS:
    rows = [_get_cumu_regret(r, budget) for r in all_results[m]]
    cumu_regret_mat[m]   = np.array(rows)
    final_cumu_all[m]    = cumu_regret_mat[m][:, -1]
    simple_regret_all[m] = np.array(
        [_noise_free_simple_regret(r) for r in all_results[m]], dtype=float
    )
    violation_rate_all[m] = np.array(
        [1.0 - _safe_rate(r) for r in all_results[m]], dtype=float
    )


# ============================================================
# Plot: Cumulative Regret vs Oracle Budget  (mean ± std band)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.suptitle(
    f'Cumulative Regret vs Oracle Budget',
    fontsize=13, fontweight='bold',
)

for m in METHODS:
    cr   = cumu_regret_mat[m]
    mean = np.nanmean(cr, axis=0)
    std  = np.nanstd(cr, axis=0)
    ax.plot(x_axis, mean, color=COLORS[m], ls=LINESTYLES[m], lw=2.2,
            label=f'{m}  (final: {mean[-1]:.2f} ± {std[-1]:.2f})')
    ax.fill_between(x_axis, mean - std, mean + std,
                    color=COLORS[m], alpha=0.18, linewidth=0)

ax.set_xlabel('Iterations', fontsize=12)
ax.set_ylabel('Cumulative Regret', fontsize=12)
ax.legend(fontsize=9, loc='upper left', framealpha=0.8)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig('multi_init_regret_mean_std.png', dpi=300, bbox_inches='tight')
plt.close()
print('\nSaved → multi_init_regret_mean_std.png')


# ============================================================
# Summary statistics table
# ============================================================
def _fmt(mean, std):
    return f'{mean:.3f} ± {std:.3f}'

lines = []
lines.append('=' * 100)
lines.append(f'Summary Statistics  — {N_RUNS} runs with different initial points')
lines.append(f'  Grid={GRID_SIZE}²  ξ={XI}  oracle_budget={ORACLE_BUDGET}  obj_noise={OBJ_NOISE}')
lines.append(f'  Seeds: {BASE_SEEDS}')
lines.append('=' * 100)
lines.append(
    f"{'Method':<24} | {'Cumu Regret μ ± σ':>22} "
    f"| {'Simple Regret μ ± σ':>22} | {'Violation Rate μ ± σ':>20}"
)
lines.append('-' * 100)

for m in METHODS:
    cr_m, cr_s   = np.nanmean(final_cumu_all[m]),    np.nanstd(final_cumu_all[m])
    sr_m, sr_s   = np.nanmean(simple_regret_all[m]), np.nanstd(simple_regret_all[m])
    vr_m, vr_s   = np.nanmean(violation_rate_all[m]), np.nanstd(violation_rate_all[m])
    lines.append(
        f'{m:<24} | {_fmt(cr_m, cr_s):>22} '
        f'| {_fmt(sr_m, sr_s):>22} | {_fmt(vr_m, vr_s):>20}'
    )

lines.append('-' * 100)
lines.append('Per-run final cumulative regret:')
for m in METHODS:
    vals = ', '.join(f'{v:.3f}' for v in final_cumu_all[m])
    lines.append(f'  {m:<24}  [{vals}]')
lines.append('=' * 100)

table_str = '\n'.join(lines)
print('\n' + table_str)

with open('multi_init_stats.txt', 'w') as f:
    f.write(table_str + '\n')
print('\nSaved → multi_init_stats.txt')
