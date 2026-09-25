"""
replot_multi_init_from_pkl.py
=============================
Re-aggregate the 5-seed multi-init benchmark from an existing checkpoint
(no simulations are re-executed). Mirrors the aggregation/plotting in
multi_init_cumulative_regret.py.

Usage:
  python replot_multi_init_from_pkl.py [--pkl multi_init_checkpoint.pkl]
"""
import sys, os, pickle, argparse
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--pkl', default='multi_init_checkpoint.pkl')
args = parser.parse_args()

# Settings matching multi_init_cumulative_regret.py
GRID_SIZE     = 25
XI            = 0
ORACLE_BUDGET = 500
OBJ_NOISE     = 0.3
BASE_SEEDS    = [5, 6, 7, 8, 9]
N_RUNS        = len(BASE_SEEDS)

# Method-key alias map: checkpoint may use older names
KEY_ALIASES = {
    'C-Safe BO':        ['C-Safe BO', 'Safe BO'],
    'Q-Safe BO':        ['Q-Safe BO', 'Quantum Safe BO'],
    'Q-Safe BO (Real)': ['Q-Safe BO (Real)', 'Quantum Safe BO (Real)'],
    'Unconstrained BO': ['Unconstrained BO'],
    'BO-ACL':           ['BO-ACL'],
}
METHODS = list(KEY_ALIASES.keys())

COLORS = {
    'C-Safe BO':        '#2196F3',
    'Q-Safe BO':        '#9C27B0',
    'Q-Safe BO (Real)': '#E91E63',
    'Unconstrained BO': '#FF5722',
    'BO-ACL':           '#4CAF50',
}
LINESTYLES = {
    'C-Safe BO':        '-',
    'Q-Safe BO':        '--',
    'Q-Safe BO (Real)': '-.',
    'Unconstrained BO': ':',
    'BO-ACL':           '-.',
}

with open(args.pkl, 'rb') as f:
    raw = pickle.load(f)

# Map runs into canonical method names
all_results = {}
for canonical, aliases in KEY_ALIASES.items():
    for a in aliases:
        if a in raw and len(raw[a]) > 0:
            all_results[canonical] = raw[a]
            break
    else:
        all_results[canonical] = []

print(f'Loaded {args.pkl}')
for m in METHODS:
    print(f'  {m}: {len(all_results[m])} runs')


# ============================================================
# Helpers (copied verbatim from multi_init_cumulative_regret.py)
# ============================================================
def _safe_rate(res):
    n  = int(res['n_init'])
    zz = np.asarray(res['zz'])
    xi = float(res['xi'])
    qs = np.asarray(res['queried_idx'], dtype=int)[n:]
    return float(np.mean(zz[qs] >= xi)) if len(qs) > 0 else float('nan')


def _pad_or_trim(arr, length):
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= length:
        return arr[:length]
    fval = arr[-1] if len(arr) > 0 else 0.0
    return np.concatenate([arr, np.full(length - len(arr), fval)])


def _get_cumu_regret(res, budget):
    # Prefer the expanded array; some methods only have the per-step history
    arr = res.get('cumu_regret_expanded')
    if arr is None or len(arr) == 0:
        arr = res['queried_cumu_regret_hist']
    return _pad_or_trim(arr, budget)


def _noise_free_simple_regret(res):
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
# Aggregate across seeds
# ============================================================
budget = ORACLE_BUDGET
x_axis = np.arange(1, budget + 1)

cumu_regret_mat    = {}
final_cumu_all     = {}
simple_regret_all  = {}
violation_rate_all = {}   # = 1 - safe rate

for m in METHODS:
    runs = all_results[m]
    if len(runs) == 0:
        cumu_regret_mat[m]    = np.zeros((0, budget))
        final_cumu_all[m]     = np.array([])
        simple_regret_all[m]  = np.array([])
        violation_rate_all[m] = np.array([])
        continue
    rows = [_get_cumu_regret(r, budget) for r in runs]
    cumu_regret_mat[m]   = np.array(rows)
    final_cumu_all[m]    = cumu_regret_mat[m][:, -1]
    simple_regret_all[m] = np.array(
        [_noise_free_simple_regret(r) for r in runs], dtype=float
    )
    violation_rate_all[m] = np.array(
        [1.0 - _safe_rate(r) for r in runs], dtype=float
    )


# ============================================================
# Plot: Cumulative Regret vs Oracle Budget  (mean +/- std band)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.suptitle(
    f'Cumulative Regret vs Oracle Budget\n'
    f'({N_RUNS} runs, grid={GRID_SIZE}^2, noise={OBJ_NOISE})',
    fontsize=13, fontweight='bold',
)

for m in METHODS:
    cr = cumu_regret_mat[m]
    if cr.shape[0] == 0:
        continue
    mean = np.nanmean(cr, axis=0)
    std  = np.nanstd(cr, axis=0)
    ax.plot(x_axis, mean, color=COLORS[m], ls=LINESTYLES[m], lw=2.2,
            label=f'{m}  (final: {mean[-1]:.2f} +/- {std[-1]:.2f})')
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
print('\nSaved -> multi_init_regret_mean_std.png')


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
    if len(final_cumu_all[m]) == 0:
        lines.append(f'{m:<24} | {"NO DATA":>22} | {"NO DATA":>22} | {"NO DATA":>20}')
        continue
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
    if len(final_cumu_all[m]) == 0:
        lines.append(f'  {m:<24}  [no data]')
        continue
    vals = ', '.join(f'{v:.3f}' for v in final_cumu_all[m])
    lines.append(f'  {m:<24}  [{vals}]')
lines.append('=' * 100)

table_str = '\n'.join(lines)
print('\n' + table_str)

with open('multi_init_stats.txt', 'w') as f:
    f.write(table_str + '\n')
print('\nSaved -> multi_init_stats.txt')
