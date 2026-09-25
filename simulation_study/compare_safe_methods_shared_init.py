import sys, os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

from experiment_env import build_paper_sim2_environment, sample_initial_indices
from safe_set_bo      import run_safe_bo_simulation, resolve_n_init
from quantum_safe_bo  import run_quantum_safe_bo_simulation
from unconstrained_bo import run_unconstrained_bo_simulation
from ACL_paper        import run_bo_acl_simulation2

print('All modules loaded OK')

# ============================================================
# Shared experiment settings
# ============================================================
SEED        = 2
GRID_SIZE   = 25
XI          = 0   # constraint threshold h(x) >= xi is feasible
N_INIT      = 5   # shared model init points
BETA_C = 3.0

ORACLE_BUDGET = 500     # total oracle calls (= n_iter for classical methods)
OBJ_NOISE   = 0.3     # matches synth_bo.py obs_noise = 0.3^2
CON_NOISE   = 1e-6
MODE        = 'min'
# ── lambda_by_stage schedule: lam_t = LAM0 * (LAM_T0 / (LAM_T0 + t))^LAM_P ──
LAM0   = 0.8   # initial boundary-expansion weight  (0 = pure objective, 1 = pure boundary)
LAM_T0 = 10    # half-decay iteration               (larger → slower decay)
LAM_P  = 1.0   # decay power                        (larger → faster decay)

# ── Real quantum hardware ──
# Set INCLUDE_REAL_QUANTUM = True if you have IBM Quantum access and want
# to include the real-hardware quantum method (slow — requires live backend).
INCLUDE_REAL_QUANTUM = False
if INCLUDE_REAL_QUANTUM:
    from quantum_safe_bo_real import run_quantum_safe_bo_simulation_real

# ============================================================
# Generate shared initial design ONCE
# All methods receive the same init_idx
# ============================================================
env_ref = build_paper_sim2_environment(
    grid_size=GRID_SIZE, xi=XI, dtype=None,
    constraint_representation='margin',
)
N_GRID = len(env_ref['xx'])

# Objective init (shared)
_n_init = resolve_n_init(N_GRID, n_init=N_INIT)
INIT_IDX = np.array(sample_initial_indices(N_GRID, n_init=_n_init, seed=SEED), dtype=int)

print(f'Grid size : {N_GRID} pts  ({GRID_SIZE}²)')
print(f'oracle_budget : {ORACLE_BUDGET}')
print(f'Shared init_idx : {INIT_IDX}  (n={len(INIT_IDX)})')



# ============================================================

def get_best_queried_point(res):
    if 'best_safe_value_hist' in res:
        best_val = res['best_safe_value_hist'][-1]
    elif 'best_feasible_value_hist' in res:
        best_val = res['best_feasible_value_hist'][-1]
    else:
        return "N/A", np.nan
        
    if np.isnan(best_val):
        return "N/A", np.nan
        
    q_idx = res['queried_idx']
    yy = res['yy']
    for idx in reversed(q_idx):
        if np.isclose(yy[idx], best_val, atol=1e-5):
            return res['xx'][idx], best_val
    return "N/A", np.nan

# 1. Safe BO  (classical, synth_bo.py style)
# ============================================================
res_safe = run_safe_bo_simulation(
    mode=MODE, xi=XI, grid_size=GRID_SIZE,
    n_init=N_INIT, n_iter=ORACLE_BUDGET,
    obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
    beta_f=0.2, beta_c=BETA_C, lam0=0.8, lam_t0=10, lam_p=2.0,
    seed=SEED, init_idx=INIT_IDX.copy(),
    M_rff=256, lam_rff=1.0, lengthscale_rff=0.2, v_kernel_rff=1.0,
    con_lengthscale=1.0, con_outputscale=1.0,
)
print(f"\nSafe BO done | global_opt={res_safe['global_safe_opt']:.4f}")
bx, bv = get_best_queried_point(res_safe)
print(f"  Queried Best Point (x*): {bx}")
print(f"  Queried Best Value f(x*): {bv:.4f}")

# ============================================================
# 2. Q-Safe BO  (IAE query + epsilon-weighted W-GP-UCB)
# ============================================================
res_qsafe = run_quantum_safe_bo_simulation(
    mode=MODE, xi=XI, grid_size=GRID_SIZE,
    n_init=N_INIT, oracle_budget=ORACLE_BUDGET,
    obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
    beta_f=0.2, beta_c=BETA_C, lam0=0.8, lam_t0=10, lam_p=2.0,
    seed=SEED, init_idx=INIT_IDX.copy(),
    M_rff=256, lam_rff=1.0, lengthscale_rff=0.2, v_kernel_rff=1.0,
    con_lengthscale=1.0, con_outputscale=1.0,
    use_quantum_query=True,
)
print(f"\nQ-Safe BO done | total_oracle={res_qsafe['total_oracle_queries']}")
bx_q, bv_q = get_best_queried_point(res_qsafe)
print(f"  Queried Best Point (x*): {bx_q}")
print(f"  Queried Best Value f(x*): {bv_q:.4f}")

# ============================================================
# 2b. Q-Safe BO (Real Hardware)
# ============================================================
res_qsafe_real = None
if INCLUDE_REAL_QUANTUM:
    from qiskit_ibm_runtime import QiskitRuntimeService

    print("\n[Q-Safe BO (Real)] Connecting to IBM Quantum...")
    service_ibm = QiskitRuntimeService()
    real_backend = None
    try:
        real_backend = service_ibm.least_busy(operational=True, simulator=False)
        print(f"[Q-Safe BO (Real)] Selected backend: {real_backend.name}")
    except Exception as e:
        print(f"[Q-Safe BO (Real)] Failed to get backend: {e}")

    res_qsafe_real = run_quantum_safe_bo_simulation_real(
        mode=MODE, xi=XI, grid_size=GRID_SIZE,
        n_init=N_INIT, oracle_budget=ORACLE_BUDGET,
        obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
        beta_f=0.2, beta_c=BETA_C, lam0=0.8, lam_t0=10, lam_p=2.0,
        seed=SEED, init_idx=INIT_IDX.copy(),
        M_rff=256, lam_rff=1.0, lengthscale_rff=0.2, v_kernel_rff=1.0,
        con_lengthscale=1.0, con_outputscale=1.0,
        use_quantum_query=True,
        backend=real_backend,  # real IBM backend, same flow as quantum_safeset_discrete_real.py
    )
    print(f"\nQ-Safe BO (Real) done | total_oracle={res_qsafe_real.get('total_oracle_queries', 'N/A')}")

# ============================================================
# 3. Unconstrained BO  (no safety mask, synth_bo.py style)
# ============================================================
res_uncon = run_unconstrained_bo_simulation(
    mode=MODE, xi=XI, grid_size=GRID_SIZE,
    n_init=N_INIT, n_iter=ORACLE_BUDGET,
    obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
    beta_f=0.5,
    seed=SEED, init_idx=INIT_IDX.copy(),
    M_target=256, lengthscale=0.2, v_kernel=1.0,
)
print(f"\nUnconstrained BO done")
bx_u, bv_u = get_best_queried_point(res_uncon)
print(f"  Queried Best Point (x*): {bx_u}")
print(f"  Queried Best Value f(x*): {bv_u:.4f}")

# ============================================================
# 4. BO-ACL  (probability of feasibility baseline)
# ============================================================
res_acl = run_bo_acl_simulation2(
    mode=MODE, xi=XI, grid_size=GRID_SIZE,
    n_init=N_INIT, n_iter=ORACLE_BUDGET,
    batch_size=2,
    obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
    beta_f=1.0, beta_w=0.2,
    seed=SEED, init_idx=INIT_IDX.copy(),
    lengthscale=0.2,
)
print(f"\nBO-ACL done")
bx_a, bv_a = get_best_queried_point(res_acl)
print(f"  Queried Best Point (x*): {bx_a}")
print(f"  Queried Best Value f(x*): {bv_a:.4f}")

# ============================================================
# Build oracle-budget-aligned regret arrays
# ============================================================
# Classical methods: one value per step = one oracle call
def classical_cumu_regret(res):
    return np.array(res['queried_cumu_regret_hist'])

# Quantum method: post-init BO queries only (LHS init excluded)
# Pad to oracle_budget length so x-axis aligns with other methods
def quantum_cumu_regret(res):
    arr = np.array(res['cumu_regret_expanded'])
    n_init = int(res['n_init'])
    # target = ORACLE_BUDGET - n_init   # same number of post-init steps as other methods expect
    if len(arr) < ORACLE_BUDGET:
        # Repeat last value: budget exhausted, no more queries, regret stays flat
        arr = np.concatenate([arr, np.full(ORACLE_BUDGET - len(arr), arr[-1])])
    return arr

budget = ORACLE_BUDGET

r_safe  = quantum_cumu_regret(res_safe)[:budget]
r_qsafe = quantum_cumu_regret(res_qsafe)[:budget]
if INCLUDE_REAL_QUANTUM:
    r_qsafe_real = quantum_cumu_regret(res_qsafe_real)[:budget]
r_uncon = classical_cumu_regret(res_uncon)[:budget]
r_acl   = classical_cumu_regret(res_acl)[:budget]

x_safe  = np.arange(1, len(r_safe)  + 1)
x_qsafe = np.arange(1, len(r_qsafe) + 1)
if INCLUDE_REAL_QUANTUM:
    x_qsafe_real = np.arange(1, len(r_qsafe_real) + 1)
x_uncon = np.arange(1, len(r_uncon) + 1)
x_acl   = np.arange(1, len(r_acl)   + 1)

print('Cumulative regret at end of budget:')
print(f'  C-Safe BO          : {r_safe[-1]:.4f}  (oracle calls={len(r_safe)})')
print(f'  Q-Safe BO  : {r_qsafe[-1]:.4f}  (oracle calls={len(r_qsafe)})')
if INCLUDE_REAL_QUANTUM:
    print(f'  Q-Safe BO (R): {r_qsafe_real[-1]:.4f}  (oracle calls={len(r_qsafe_real)})')
print(f'  Unconstrained BO : {r_uncon[-1]:.4f}  (steps={len(r_uncon)})')
print(f'  BO-ACL           : {r_acl[-1]:.4f}  (steps={len(r_acl)})')

# ============================================================
# Safe query rate
# ============================================================
def safe_rate(res):
    n = res['n_init']
    import torch; zz=res['zz']; h = zz.numpy() if hasattr(zz,'numpy') else np.asarray(zz)
    qs = np.array(res['queried_idx'][n:], dtype=int)
    return float(np.mean(h[qs] >= res['xi'])) if len(qs) > 0 else float('nan')

print('Safe query rates:')
print(f'  C-Safe BO          : {safe_rate(res_safe):.3f}')
print(f'  Q-Safe BO  : {safe_rate(res_qsafe):.3f}')
if INCLUDE_REAL_QUANTUM:
    print(f'  Q-Safe BO (R): {safe_rate(res_qsafe_real):.3f}')
print(f'  Unconstrained BO : {safe_rate(res_uncon):.3f}')
print(f'  BO-ACL           : {safe_rate(res_acl):.3f}')

# ============================================================
# Plot 1: Cumulative Regret vs Iterations
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
fig.suptitle(
    f'C-Safe BO Comparison  (grid={GRID_SIZE}², noise={OBJ_NOISE})',
    fontsize=13, fontweight='bold'
)

COLORS = {
    'C-Safe BO'         : '#2196F3',
    'Q-Safe BO' : '#9C27B0',
    'Q-Safe BO (Real)' : '#E91E63',
    'Unconstrained BO': '#FF5722',
    'BO-ACL'          : '#4CAF50',
}
STYLES = {
    'C-Safe BO'         : '-',
    'Q-Safe BO' : '--',
    'Q-Safe BO (Real)' : '--',
    'Unconstrained BO': ':',
    'BO-ACL'          : '-.',
}

# --- Cumulative Regret ---
ax.plot(x_safe,  r_safe,  label='C-Safe BO',          color=COLORS['C-Safe BO'],          ls=STYLES['C-Safe BO'],          lw=2)
ax.plot(x_qsafe, r_qsafe, label='Q-Safe BO',  color=COLORS['Q-Safe BO'],  ls=STYLES['Q-Safe BO'],  lw=2)
if INCLUDE_REAL_QUANTUM:
    ax.plot(x_qsafe_real, r_qsafe_real, label='Q-Safe BO (Real)', color=COLORS['Q-Safe BO (Real)'], ls=STYLES['Q-Safe BO (Real)'], lw=2)
ax.plot(x_uncon, r_uncon, label='Unconstrained BO', color=COLORS['Unconstrained BO'], ls=STYLES['Unconstrained BO'], lw=2)
ax.plot(x_acl,   r_acl,   label='BO-ACL',           color=COLORS['BO-ACL'],           ls=STYLES['BO-ACL'],           lw=2)
ax.set_xlabel('Iterations', fontsize=11)
ax.set_ylabel('Cumulative Regret', fontsize=11)
ax.set_title('Cumulative Regret vs Iterations')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_regret.png', dpi=600, bbox_inches='tight')
plt.show()
print('Saved → comparison_regret.png')

# ============================================================
# Plot 2: Safe-Set Growth + Query Map  (with constraint contour)
# ============================================================
import torch
from matplotlib.lines import Line2D
RESULTS = [
    ('C-Safe BO',          res_safe,  COLORS['C-Safe BO']),
    ('Q-Safe BO',  res_qsafe, COLORS['Q-Safe BO']),
    ('Q-Safe BO (Real)',  res_qsafe_real, COLORS['Q-Safe BO (Real)']),
    ('Unconstrained BO', res_uncon, COLORS['Unconstrained BO']),
    ('BO-ACL',           res_acl,   COLORS['BO-ACL']),
]
RESULTS = [r for r in RESULTS if r[1] is not None]  # drops Q-Safe BO (Real) when INCLUDE_REAL_QUANTUM is False

fig, axes = plt.subplots(3, len(RESULTS), figsize=(5 * len(RESULTS), 14))
fig.suptitle('Query Maps and Learned Constraint Boundary',
             fontsize=13, fontweight='bold')

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

    # Reconstruct 2D meshgrid for contour plotting
    gs = GRID_SIZE
    x1_ax = np.linspace(-1, 1, gs)
    x2_ax = np.linspace(-1, 1, gs)
    X1, X2 = np.meshgrid(x1_ax, x2_ax, indexing='ij')
    yy_grid = yy_env.reshape(gs, gs)
    zz_grid = zz.reshape(gs, gs)

    # --- Row 0: query scatter on objective background ---
    ax = axes[0, col]
    cs = ax.contourf(X1, X2, yy_grid, levels=20, cmap='viridis', alpha=0.7)
    fig.colorbar(cs, ax=ax, shrink=0.8, label='f(x)')
    ax.contour(X1, X2, zz_grid, levels=[xi_val],
               colors='red', linewidths=2.5, linestyles='--')
    ax.contourf(X1, X2, zz_grid, levels=[xi_val, 1e9],
                colors=['white'], alpha=0.15)
    if len(model_q) > 0:
        if label == 'BO-ACL':
            # Split by batch role: batch[0]=objective query (○), batch[1]=constraint query (×)
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
    ax.set_title(label, fontsize=10, fontweight='bold', color=color)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    _leg = [Line2D([0], [0], color='red', lw=2, ls='--', label=f'h(x)=ξ={xi_val}')]
    if label == 'BO-ACL':
        _leg += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumorchid',
                   markersize=7, label='Objective (batch[0])'),
            Line2D([0], [0], marker='x', color='red', markersize=7,
                   linewidth=1.4, label='Constraint (batch[1])'),
        ]
    ax.legend(handles=_leg, fontsize=6, loc='upper right')

    # Locate the global safe optimum on the grid (needed for row 1 and 2)
    _mode_val  = res.get('mode', 'min')
    _safe_idx  = np.where(safe_mask)[0]
    _yy_safe   = yy_env[_safe_idx]
    _local_opt = np.argmin(_yy_safe) if _mode_val == 'min' else np.argmax(_yy_safe)
    _opt_idx   = _safe_idx[_local_opt]
    _opt_x     = xx[_opt_idx]   # [x1, x2] coordinates of the global safe optimum

    # --- Row 1: INITIAL learned constraint boundary ---
    ax = axes[1, col]

    if label == 'BO-ACL' and 'init_p_feasible_np' in res:
        p_grid = np.asarray(res['init_p_feasible_np']).reshape(gs, gs)
        hm = ax.contourf(X1, X2, p_grid, levels=np.linspace(0.0, 1.0, 21),
                         cmap='RdYlGn', alpha=0.85)
        fig.colorbar(hm, ax=ax, shrink=0.8, label='Prob of Feasibility')
        ax.contour(X1, X2, zz_grid, levels=[xi_val], colors='red', linewidths=1.8, linestyles='--')
        ax.scatter([_opt_x[0]], [_opt_x[1]], c='red', s=150, zorder=10, marker='*', edgecolors='black', linewidths=0.6)
        ax.set_title('INITIAL Probability of Feasibility', fontsize=9)

    elif 'init_lcb_c' in res:
        import scipy.stats as stats
        lcb_grid = np.asarray(res['init_lcb_c']).reshape(gs, gs)
        mu_grid  = np.asarray(res['init_mu_c']).reshape(gs, gs)
        sigma_grid = (mu_grid - lcb_grid) / np.sqrt(3.0)
        pof_grid = stats.norm.cdf(mu_grid / (sigma_grid + 1e-12))

        hm = ax.contourf(X1, X2, pof_grid, levels=np.linspace(0.0, 1.0, 21),
                         cmap='RdYlGn', alpha=0.85)
        fig.colorbar(hm, ax=ax, shrink=0.8, label='Prob of Feasibility')
        ax.contour(X1, X2, zz_grid, levels=[xi_val], colors='red', linewidths=1.8, linestyles='--')
        ax.scatter([_opt_x[0]], [_opt_x[1]], c='red', s=150, zorder=10, marker='*', edgecolors='black', linewidths=0.6)
        ax.set_title('INITIAL Probability of Feasibility', fontsize=9)

    else:
        ax.text(0.5, 0.5, 'No constraint model', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='gray')
        ax.set_title('INITIAL Probability of Feasibility', fontsize=9)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')

    # --- Row 2: FINAL learned constraint boundary ---
    ax = axes[2, col]

    if label == 'BO-ACL' and 'final_p_feasible_np' in res:
        # Continuous probability of feasibility
        p_grid = np.asarray(res['final_p_feasible_np']).reshape(gs, gs)
        hm = ax.contourf(X1, X2, p_grid, levels=np.linspace(0.0, 1.0, 21),
                         cmap='RdYlGn', alpha=0.85)
        fig.colorbar(hm, ax=ax, shrink=0.8, label='Probability of Feasibility')
        # True boundary for comparison
        ax.contour(X1, X2, zz_grid, levels=[xi_val],
                   colors='red', linewidths=1.8, linestyles='--')
        legend_elems = [
            Line2D([0], [0], color='red', lw=2, ls='--', label='True boundary h(x)=ξ'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markeredgecolor='black', markersize=10, label='Global safe optimum'),
        ]
        ax.legend(handles=legend_elems, fontsize=6, loc='upper right')
        ax.set_title('Learned Probability of Feasibility', fontsize=9)

    elif 'final_lcb_c' in res:
        import scipy.stats as stats
        lcb_grid = np.asarray(res['final_lcb_c']).reshape(gs, gs)
        mu_grid  = np.asarray(res['final_mu_c']).reshape(gs, gs)

        # PROBABILITY OF FEASIBILITY CONTOUR
        # Calculate sigma from LCB: lcb_c = mu_c - sqrt(beta_c) * sigma_c  (beta_c = 3.0 here)
        sigma_grid = (mu_grid - lcb_grid) / np.sqrt(3.0)
        pof_grid = stats.norm.cdf(mu_grid / (sigma_grid + 1e-12))

        hm = ax.contourf(X1, X2, pof_grid, levels=np.linspace(0.0, 1.0, 21),
                         cmap='RdYlGn', alpha=0.85)
        fig.colorbar(hm, ax=ax, shrink=0.8, label='Probability of Feasibility')
        # True boundary
        ax.contour(X1, X2, zz_grid, levels=[xi_val],
                   colors='red', linewidths=1.8, linestyles='--')
        legend_elems = [
            Line2D([0], [0], color='red', lw=2, ls='--', label='True: h(x)=ξ'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markeredgecolor='black', markersize=10, label='Global safe optimum'),
        ]
        ax.legend(handles=legend_elems, fontsize=6, loc='upper right')
        ax.set_title('Learned Probability of Feasibility', fontsize=9)

    else:
        ax.text(0.5, 0.5, 'No constraint\nmodel', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='gray')
        ax.set_title('No constraint model', fontsize=9)

    # Mark global safe optimum (on top of everything) for models with a constraint map.
    if label != 'Unconstrained BO':
        ax.scatter([_opt_x[0]], [_opt_x[1]], c='red', s=150, zorder=10,
                   marker='*', edgecolors='black', linewidths=0.6)

    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')



plt.tight_layout()
plt.savefig('comparison_maps.png', dpi=600, bbox_inches='tight')
plt.show()
print('Saved → comparison_maps.png')


# ============================================================
# Dedicated BO-ACL figure: batch[0] (objective) vs batch[1] (constraint)
# ============================================================
_xx   = np.asarray(res_acl['xx'])
_yy   = np.asarray(res_acl['yy'])
_zz   = np.asarray(res_acl['zz'])
_xi   = float(res_acl['xi'])
_n    = res_acl['n_init']
_all_q  = np.array(res_acl['queried_idx'], dtype=int)
_init_q = _all_q[:_n]

# Extract by batch position
_bh     = res_acl['batch_history']
_prim_q = np.array([b[0] for b in _bh if len(b) > 0], dtype=int)   # objective query (circle)
_sec_q  = np.array([b[1] for b in _bh if len(b) > 1], dtype=int)   # constraint query (cross)

_gs = GRID_SIZE
_x1 = np.linspace(-1, 1, _gs)
_x2 = np.linspace(-1, 1, _gs)
_X1, _X2 = np.meshgrid(_x1, _x2, indexing='ij')
_yy_grid = _yy.reshape(_gs, _gs)
_zz_grid = _zz.reshape(_gs, _gs)

fig2, (ax_s, ax_u) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('BO-ACL Query Breakdown by Batch Position', fontsize=13, fontweight='bold')

# Left panel: batch[0] — objective queries (circles, plasma-colored by iteration)
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
    Line2D([0], [0], color='red', lw=2, ls='--', label=f'h(x)=ξ={_xi}'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumorchid',
           markersize=8, label='Objective query (circle)'),
], fontsize=8, loc='upper right')

# Right panel: batch[1] — constraint queries (red crosses)
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
    Line2D([0], [0], color='red', lw=2, ls='--', label=f'h(x)=ξ={_xi}'),
    Line2D([0], [0], marker='x', color='crimson', markersize=8,
           linewidth=1.6, label='Constraint query (cross)'),
], fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('acl_query_detail.png', dpi=600, bbox_inches='tight')
plt.show()
print('Saved → acl_query_detail.png')

# ============================================================
headers = ['Method', 'Global Opt', 'Final Best Safe', 'Simple Regret', 'Cumu Regret', 'Safe Rate', 'Viol Rate']
rows = []
_table_results = [('Safe BO', res_safe), ('Q-Safe BO', res_qsafe), ('Q-Safe BO (R)', res_qsafe_real),
                  ('Unconstrained BO', res_uncon), ('BO-ACL', res_acl)]
for label, res in _table_results:
    if res is None:  # Q-Safe BO (R) when INCLUDE_REAL_QUANTUM is False
        continue


    sr = safe_rate(res)
    cr_arr = res.get('cumu_regret_expanded', res['queried_cumu_regret_hist'])

    # Noise-free "Final Best Safe": use yy[queried_idx] (ground truth) instead of noisy train_Y
    _yy  = np.asarray(res['yy'])
    _zz  = np.asarray(res['zz'])
    _xi  = float(res['xi'])
    _mode = res['mode']
    _q   = np.asarray(res['queried_idx'], dtype=int)
    _n   = res['n_init']
    _all_queries = _q  # include all queries (init + active) for final best safe
    _safe_queries = _all_queries[_zz[_all_queries] >= _xi]
    if len(_safe_queries) > 0:
        _best_nf = float(np.min(_yy[_safe_queries])) if _mode == 'min' else float(np.max(_yy[_safe_queries]))
    else:
        _best_nf = float('nan')

    # Compute simple_regret uniformly from noise-free best safe value
    # (works for all methods regardless of what keys they store in their result dict)
    if np.isnan(_best_nf):
        _simple_regret = float('nan')
    elif _mode == 'min':
        _simple_regret = abs(_best_nf - float(res['global_safe_opt']))
    else:
        _simple_regret = abs(float(res['global_safe_opt']) - _best_nf)

    rows.append([
        label,
        f"{res['global_safe_opt']:.4f}",
        f"{_best_nf:.4f}",
        f"{_simple_regret:.4f}",
        f"{cr_arr[-1]:.4f}",
        f"{sr:.3f}",
        f"{1.0 - sr:.3f}",
    ])

col_w = max(len(h) for h in headers) + 2
col_w = max(col_w, 18)
sep = '─' * (col_w * len(headers))
print(sep)
print('  '.join(h.ljust(col_w) for h in headers))
print(sep)
for row in rows:
    print('  '.join(v.ljust(col_w) for v in row))
print(sep)

# ============================================================
# Persist results
# ============================================================
import pickle, json
from datetime import datetime

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# 1. Raw result dicts (for reloading arrays / plotting later)
_pkl_path = f'results_{_ts}.pkl'
with open(_pkl_path, 'wb') as _f:
    pickle.dump({
        'config': {
            'SEED': SEED, 'GRID_SIZE': GRID_SIZE, 'XI': XI, 'N_INIT': N_INIT,
            'ORACLE_BUDGET': ORACLE_BUDGET, 'OBJ_NOISE': OBJ_NOISE,
            'CON_NOISE': CON_NOISE, 'BETA_C': BETA_C,
            'LAM0': LAM0, 'LAM_T0': LAM_T0, 'LAM_P': LAM_P,
            'INIT_IDX': INIT_IDX.tolist(),
        },
        'safe_bo':      res_safe,
        'quantum_safe': res_qsafe,
        'quantum_real': res_qsafe_real,
        'unconstrained': res_uncon,
        'acl':          res_acl,
    }, _f)
print(f'Saved raw results → {_pkl_path}')

# 2. Human-readable summary (text)
_txt_path = f'results_{_ts}.txt'
with open(_txt_path, 'w') as _f:
    _f.write(f'compare_safe_methods_shared_init  |  {datetime.now().isoformat()}\n')
    _f.write(f'SEED={SEED}  GRID={GRID_SIZE}²  XI={XI}  N_INIT={N_INIT}  '
             f'ORACLE_BUDGET={ORACLE_BUDGET}  OBJ_NOISE={OBJ_NOISE}  '
             f'BETA_C={BETA_C}  LAM0={LAM0}  LAM_T0={LAM_T0}  LAM_P={LAM_P}\n')
    _f.write(f'INIT_IDX={INIT_IDX.tolist()}\n\n')
    _f.write(sep + '\n')
    _f.write('  '.join(h.ljust(col_w) for h in headers) + '\n')
    _f.write(sep + '\n')
    for row in rows:
        _f.write('  '.join(v.ljust(col_w) for v in row) + '\n')
    _f.write(sep + '\n\n')
    _f.write('Cumulative regret curves:\n')
    _curves = [('C-Safe BO', r_safe), ('Q-Safe BO', r_qsafe)]
    if INCLUDE_REAL_QUANTUM:
        _curves.append(('Quantum Safe (R)', r_qsafe_real))
    _curves += [('Unconstrained BO', r_uncon), ('BO-ACL', r_acl)]
    for label, arr in _curves:
        _f.write(f'  {label}: {arr.tolist()}\n')
print(f'Saved text summary  → {_txt_path}')
