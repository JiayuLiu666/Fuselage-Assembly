import sys, os, pickle, argparse
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--pkl', default='results_20260521_111827.pkl')
args = parser.parse_args()

with open(args.pkl, 'rb') as f:
    blob = pickle.load(f)

cfg = blob['config']
res_safe       = blob['safe_bo']
res_qsafe      = blob['quantum_safe']
res_qsafe_real = blob['quantum_real']
res_uncon      = blob['unconstrained']
res_acl        = blob['acl']

SEED          = cfg['SEED']
GRID_SIZE     = cfg['GRID_SIZE']
ORACLE_BUDGET = cfg['ORACLE_BUDGET']
OBJ_NOISE     = cfg['OBJ_NOISE']
INIT_IDX      = np.array(cfg['INIT_IDX'], dtype=int)

print(f"Loaded {args.pkl}  (seed={SEED}, grid={GRID_SIZE}², budget={ORACLE_BUDGET})")

def classical_cumu_regret(res):
    return np.array(res['queried_cumu_regret_hist'])

def quantum_cumu_regret(res):
    arr = np.array(res.get('cumu_regret_expanded', res['queried_cumu_regret_hist']))
    if len(arr) < ORACLE_BUDGET:
        arr = np.concatenate([arr, np.full(ORACLE_BUDGET - len(arr), arr[-1])])
    return arr

def simple_regret(res):
    # Noise-free best safe value: use ground-truth yy at safe queried points
    _yy = np.asarray(res['yy'])
    _zz = np.asarray(res['zz'])
    _xi = float(res['xi'])
    _mode = res['mode']
    _q = np.asarray(res['queried_idx'], dtype=int)
    _safe_queries = _q[_zz[_q] >= _xi]
    if len(_safe_queries) == 0:
        return float('nan')
    _best_nf = float(np.min(_yy[_safe_queries])) if _mode == 'min' else float(np.max(_yy[_safe_queries]))
    return abs(float(res['global_safe_opt']) - _best_nf)

budget = ORACLE_BUDGET
r_safe       = quantum_cumu_regret(res_safe)[:budget]
r_qsafe      = quantum_cumu_regret(res_qsafe)[:budget]
r_qsafe_real = quantum_cumu_regret(res_qsafe_real)[:budget]
r_uncon      = classical_cumu_regret(res_uncon)[:budget]
r_acl        = classical_cumu_regret(res_acl)[:budget]

x_safe       = np.arange(1, len(r_safe) + 1)
x_qsafe      = np.arange(1, len(r_qsafe) + 1)
x_qsafe_real = np.arange(1, len(r_qsafe_real) + 1)
x_uncon      = np.arange(1, len(r_uncon) + 1)
x_acl        = np.arange(1, len(r_acl) + 1)

# ============================================================
# Plot 1: Cumulative Regret vs Iterations
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
fig.suptitle(
    f'Cumulative Regret Comparison  (grid={GRID_SIZE}², noise={OBJ_NOISE})',
    fontsize=13, fontweight='bold'
)

COLORS = {
    'C-Safe BO'        : '#2196F3',
    'Q-Safe BO'        : '#9C27B0',
    'Q-Safe BO (Real)' : '#E91E63',
    'Unconstrained BO'       : '#FF5722',
    'BO-ACL'                 : '#4CAF50',
}
STYLES = {
    'C-Safe BO'        : '-',
    'Q-Safe BO'        : '--',
    'Q-Safe BO (Real)' : '--',
    'Unconstrained BO'       : ':',
    'BO-ACL'                 : '-.',
}

ax.plot(x_safe,       r_safe,       label=f'C-Safe BO        (final: {r_safe[-1]:.2f})',        color=COLORS['C-Safe BO'],        ls=STYLES['C-Safe BO'],        lw=2)
ax.plot(x_qsafe,      r_qsafe,      label=f'Q-Safe BO        (final: {r_qsafe[-1]:.2f})',       color=COLORS['Q-Safe BO'],        ls=STYLES['Q-Safe BO'],        lw=2)
ax.plot(x_qsafe_real, r_qsafe_real, label=f'Q-Safe BO (Real) (final: {r_qsafe_real[-1]:.2f})',  color=COLORS['Q-Safe BO (Real)'], ls=STYLES['Q-Safe BO (Real)'], lw=2)
ax.plot(x_uncon,      r_uncon,      label=f'Unconstrained BO (final: {r_uncon[-1]:.2f})',       color=COLORS['Unconstrained BO'],       ls=STYLES['Unconstrained BO'],       lw=2)
ax.plot(x_acl,        r_acl,        label=f'BO-ACL           (final: {r_acl[-1]:.2f})',         color=COLORS['BO-ACL'],                 ls=STYLES['BO-ACL'],                 lw=2)
ax.set_xlabel('Iterations', fontsize=11)
ax.set_ylabel('Cumulative Regret', fontsize=11)
ax.set_title('Cumulative Regret vs Iterations')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_regret.png', dpi=600, bbox_inches='tight')
print('Saved -> comparison_regret.png')

# ============================================================
# Plot 2: Query Maps + Learned Constraint Boundary
# ============================================================
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
fig.suptitle('Query Maps and Learned Constraint Boundary',
             fontsize=24, fontweight='bold')

RESULTS = [
    ('C-Safe BO',         res_safe,       COLORS['C-Safe BO']),
    ('Q-Safe BO',         res_qsafe,      COLORS['Q-Safe BO']),
    ('Q-Safe BO (Real)',  res_qsafe_real, COLORS['Q-Safe BO (Real)']),
    ('Unconstrained BO',        res_uncon,      COLORS['Unconstrained BO']),
    ('BO-ACL',                  res_acl,        COLORS['BO-ACL']),
]

for col, (label, res, color) in enumerate(RESULTS):
    xx = np.asarray(res['xx'])
    yy_env = np.asarray(res['yy'])
    zz = np.asarray(res['zz'])
    xi_val = float(res['xi'])
    safe_mask = zz >= xi_val

    n = res['n_init']
    all_q = np.array(res['queried_idx'], dtype=int)
    init_q = all_q[:n]
    model_q = all_q[n:]

    gs = GRID_SIZE
    x1_ax = np.linspace(-1, 1, gs)
    x2_ax = np.linspace(-1, 1, gs)
    X1, X2 = np.meshgrid(x1_ax, x2_ax, indexing='ij')
    yy_grid = yy_env.reshape(gs, gs)
    zz_grid = zz.reshape(gs, gs)

    # Row 0: query scatter on objective background
    ax = axes[0, col]
    cs = ax.contourf(X1, X2, yy_grid, levels=20, cmap='viridis', alpha=0.7)
    _cb = fig.colorbar(cs, ax=ax, shrink=0.8, label='f(x)')
    _cb.set_label('f(x)', fontsize=15)
    _cb.ax.tick_params(labelsize=13)
    ax.contour(X1, X2, zz_grid, levels=[xi_val],
               colors='red', linewidths=2.5, linestyles='--')
    ax.contourf(X1, X2, zz_grid, levels=[xi_val, 1e9],
                colors=['white'], alpha=0.15)
    if len(model_q) > 0:
        if label == 'BO-ACL':
            _bh = res['batch_history']
            _obj_q = np.array([b[0] for b in _bh if len(b) > 0], dtype=int)
            _con_q = np.array([b[1] for b in _bh if len(b) > 1], dtype=int)
            if len(_obj_q) > 0:
                ax.scatter(xx[_obj_q, 0], xx[_obj_q, 1],
                           c=np.arange(len(_obj_q)), cmap='plasma',
                           s=45, zorder=4, alpha=0.90, label='Objective queries')
            if len(_con_q) > 0:
                ax.scatter(xx[_con_q, 0], xx[_con_q, 1],
                           c='red', marker='x', s=40, zorder=4,
                           linewidths=1.4, alpha=0.80, label='Constraint queries')
        else:
            ax.scatter(xx[model_q, 0], xx[model_q, 1],
                       c=np.arange(len(model_q)), cmap='plasma',
                       s=30, zorder=4, alpha=0.85, label='Queries')
    ax.set_title(label, fontsize=18, fontweight='bold', color=color)
    ax.set_xlabel('$x_1$', fontsize=16); ax.set_ylabel('$x_2$', fontsize=16)
    ax.tick_params(labelsize=13)
    _leg = [Line2D([0], [0], color='red', lw=2, ls='--', label=f'h(x)={xi_val}')]
    if label == 'BO-ACL':
        _leg += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumorchid',
                   markersize=7, label='Objective'),
            Line2D([0], [0], marker='x', color='red', markersize=7,
                   linewidth=1.4, label='Constraint'),
        ]
    ax.legend(handles=_leg, fontsize=11, loc='upper right')

    _mode_val  = res.get('mode', 'min')
    _safe_idx  = np.where(safe_mask)[0]
    _yy_safe   = yy_env[_safe_idx]
    _local_opt = np.argmin(_yy_safe) if _mode_val == 'min' else np.argmax(_yy_safe)
    _opt_idx   = _safe_idx[_local_opt]
    _opt_x     = xx[_opt_idx]

    # Row 1: INITIAL learned constraint boundary
    ax = axes[1, col]
    if label == 'BO-ACL' and 'init_p_feasible_np' in res:
        p_grid = np.asarray(res['init_p_feasible_np']).reshape(gs, gs)
        hm = ax.contourf(X1, X2, p_grid, levels=np.linspace(0.0, 1.0, 21),
                         cmap='RdYlGn', alpha=0.85)
        _cb = fig.colorbar(hm, ax=ax, shrink=0.8, label='Prob of Feasibility')
        _cb.set_label('Prob of Feasibility', fontsize=15)
        _cb.ax.tick_params(labelsize=13)
        ax.contour(X1, X2, zz_grid, levels=[xi_val], colors='red', linewidths=1.8, linestyles='--')
        ax.scatter([_opt_x[0]], [_opt_x[1]], c='red', s=150, zorder=10, marker='*', edgecolors='black', linewidths=0.6)
        ax.set_title('INITIAL Probability of Feasibility', fontsize=15)
    elif 'init_lcb_c' in res:
        import scipy.stats as stats
        lcb_grid = np.asarray(res['init_lcb_c']).reshape(gs, gs)
        mu_grid  = np.asarray(res['init_mu_c']).reshape(gs, gs)
        sigma_grid = (mu_grid - lcb_grid) / np.sqrt(3.0)
        pof_grid = stats.norm.cdf(mu_grid / (sigma_grid + 1e-12))
        hm = ax.contourf(X1, X2, pof_grid, levels=np.linspace(0.0, 1.0, 21),
                         cmap='RdYlGn', alpha=0.85)
        _cb = fig.colorbar(hm, ax=ax, shrink=0.8, label='Prob of Feasibility')
        _cb.set_label('Prob of Feasibility', fontsize=15)
        _cb.ax.tick_params(labelsize=13)
        ax.contour(X1, X2, zz_grid, levels=[xi_val], colors='red', linewidths=1.8, linestyles='--')
        ax.scatter([_opt_x[0]], [_opt_x[1]], c='red', s=150, zorder=10, marker='*', edgecolors='black', linewidths=0.6)
        ax.set_title('INITIAL Probability of Feasibility', fontsize=15)
    else:
        ax.text(0.5, 0.5, 'No constraint model', ha='center', va='center',
                transform=ax.transAxes, fontsize=15, color='gray')
        ax.set_title('INITIAL Probability of Feasibility', fontsize=15)
    ax.set_xlabel('$x_1$', fontsize=16); ax.set_ylabel('$x_2$', fontsize=16)
    ax.tick_params(labelsize=13)

    # Row 2: FINAL learned constraint boundary
    ax = axes[2, col]
    if label == 'BO-ACL' and 'final_p_feasible_np' in res:
        p_grid = np.asarray(res['final_p_feasible_np']).reshape(gs, gs)
        hm = ax.contourf(X1, X2, p_grid, levels=np.linspace(0.0, 1.0, 21),
                         cmap='RdYlGn', alpha=0.85)
        _cb = fig.colorbar(hm, ax=ax, shrink=0.8, label='Probability of Feasibility')
        _cb.set_label('Probability of Feasibility', fontsize=15)
        _cb.ax.tick_params(labelsize=13)
        ax.contour(X1, X2, zz_grid, levels=[xi_val],
                   colors='red', linewidths=1.8, linestyles='--')
        legend_elems = [
            Line2D([0], [0], color='red', lw=2, ls='--', label=f'True boundary h(x)={xi_val}'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markeredgecolor='black', markersize=10, label='Global safe optimum'),
        ]
        ax.legend(handles=legend_elems, fontsize=11, loc='upper right')
        ax.set_title('Learned Probability of Feasibility', fontsize=15)
    elif 'final_lcb_c' in res:
        import scipy.stats as stats
        lcb_grid = np.asarray(res['final_lcb_c']).reshape(gs, gs)
        mu_grid  = np.asarray(res['final_mu_c']).reshape(gs, gs)
        sigma_grid = (mu_grid - lcb_grid) / np.sqrt(3.0)
        pof_grid = stats.norm.cdf(mu_grid / (sigma_grid + 1e-12))
        hm = ax.contourf(X1, X2, pof_grid, levels=np.linspace(0.0, 1.0, 21),
                         cmap='RdYlGn', alpha=0.85)
        _cb = fig.colorbar(hm, ax=ax, shrink=0.8, label='Probability of Feasibility')
        _cb.set_label('Probability of Feasibility', fontsize=15)
        _cb.ax.tick_params(labelsize=13)
        ax.contour(X1, X2, zz_grid, levels=[xi_val],
                   colors='red', linewidths=1.8, linestyles='--')
        legend_elems = [
            Line2D([0], [0], color='red', lw=2, ls='--', label=f'True boundary: h(x)={xi_val}'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markeredgecolor='black', markersize=10, label='Global safe optimum'),
        ]
        ax.legend(handles=legend_elems, fontsize=11, loc='upper right')
        ax.set_title('Learned Probability of Feasibility', fontsize=15)
    else:
        ax.text(0.5, 0.5, 'No constraint\nmodel', ha='center', va='center',
                transform=ax.transAxes, fontsize=15, color='gray')
        ax.set_title('No constraint model', fontsize=15)

    if label != 'Unconstrained BO':
        ax.scatter([_opt_x[0]], [_opt_x[1]], c='red', s=150, zorder=10,
                   marker='*', edgecolors='black', linewidths=0.6)

    ax.set_xlabel('$x_1$', fontsize=16); ax.set_ylabel('$x_2$', fontsize=16)
    ax.tick_params(labelsize=13)

plt.tight_layout()
plt.savefig('comparison_maps.png', dpi=600, bbox_inches='tight')
print('Saved -> comparison_maps.png')

# ============================================================
# Plot 3: BO-ACL batch[0] vs batch[1]
# ============================================================
_xx   = np.asarray(res_acl['xx'])
_yy   = np.asarray(res_acl['yy'])
_zz   = np.asarray(res_acl['zz'])
_xi   = float(res_acl['xi'])
_n    = res_acl['n_init']
_all_q  = np.array(res_acl['queried_idx'], dtype=int)
_init_q = _all_q[:_n]

_bh     = res_acl['batch_history']
_prim_q = np.array([b[0] for b in _bh if len(b) > 0], dtype=int)
_sec_q  = np.array([b[1] for b in _bh if len(b) > 1], dtype=int)

_gs = GRID_SIZE
_x1 = np.linspace(-1, 1, _gs)
_x2 = np.linspace(-1, 1, _gs)
_X1, _X2 = np.meshgrid(_x1, _x2, indexing='ij')
_yy_grid = _yy.reshape(_gs, _gs)
_zz_grid = _zz.reshape(_gs, _gs)

fig2, (ax_s, ax_u) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('BO-ACL Query Breakdown by Batch Position', fontsize=13, fontweight='bold')

cs2 = ax_s.contourf(_X1, _X2, _yy_grid, levels=20, cmap='viridis', alpha=0.65)
fig2.colorbar(cs2, ax=ax_s, shrink=0.8, label='f(x)')
ax_s.contour(_X1, _X2, _zz_grid, levels=[_xi], colors='red', linewidths=2.5, linestyles='--')
ax_s.contourf(_X1, _X2, _zz_grid, levels=[_xi, 1e9], colors=['white'], alpha=0.15)
ax_s.scatter(_xx[_init_q, 0], _xx[_init_q, 1],
             c='white', edgecolors='black', s=100, zorder=5, marker='*')
if len(_prim_q) > 0:
    ax_s.scatter(_xx[_prim_q, 0], _xx[_prim_q, 1],
                 c=np.arange(len(_prim_q)), cmap='plasma',
                 s=50, zorder=4, alpha=0.90, edgecolors='white', linewidths=0.4)
ax_s.set_title(f'Batch[0]: Objective Queries  (n={len(_prim_q)})', fontsize=11)
ax_s.set_xlabel('$x_1$'); ax_s.set_ylabel('$x_2$')
ax_s.legend(handles=[
    Line2D([0], [0], color='red', lw=2, ls='--', label=f'h(x)={_xi}'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumorchid',
           markersize=8, label='Objective query (circle)'),
], fontsize=8, loc='upper right')

cs3 = ax_u.contourf(_X1, _X2, _yy_grid, levels=20, cmap='viridis', alpha=0.65)
fig2.colorbar(cs3, ax=ax_u, shrink=0.8, label='f(x)')
ax_u.contour(_X1, _X2, _zz_grid, levels=[_xi], colors='red', linewidths=2.5, linestyles='--')
ax_u.contourf(_X1, _X2, _zz_grid, levels=[_xi, 1e9], colors=['white'], alpha=0.15)
ax_u.scatter(_xx[_init_q, 0], _xx[_init_q, 1],
             c='white', edgecolors='black', s=100, zorder=5, marker='*')
if len(_sec_q) > 0:
    ax_u.scatter(_xx[_sec_q, 0], _xx[_sec_q, 1],
                 c='crimson', marker='x', s=50, zorder=4,
                 linewidths=1.6, alpha=0.85)
ax_u.set_title(f'Batch[1]: Constraint Queries  (n={len(_sec_q)})', fontsize=11)
ax_u.set_xlabel('$x_1$'); ax_u.set_ylabel('$x_2$')
ax_u.legend(handles=[
    Line2D([0], [0], color='red', lw=2, ls='--', label=f'h(x)={_xi}'),
    Line2D([0], [0], marker='x', color='crimson', markersize=8,
           linewidth=1.6, label='Constraint query (cross)'),
], fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('acl_query_detail.png', dpi=600, bbox_inches='tight')
print('Saved -> acl_query_detail.png')
