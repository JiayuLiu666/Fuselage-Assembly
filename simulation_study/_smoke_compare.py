"""Smoke test mirroring compare_safe_methods_shared_init.py.

Same imports, same five method calls, same downstream-aggregation pattern
(safe_rate, cumu_regret_expanded, summary table). Differences: tiny grid /
budget / n_init, no plotting, no pickle save. Real-hardware run uses
ibm_rensselaer with the same tiny budget.
"""
import os, sys, time, traceback
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

from experiment_env import build_paper_sim2_environment, sample_initial_indices
from safe_set_bo      import run_safe_bo_simulation, resolve_n_init
from quantum_safe_bo  import run_quantum_safe_bo_simulation
from quantum_safe_bo_real import run_quantum_safe_bo_simulation_real
from unconstrained_bo import run_unconstrained_bo_simulation
from ACL_paper        import run_bo_acl_simulation2
log('imports OK')

# --- Tiny smoke constants (mirroring compare_safe_methods_shared_init.py) ---
SEED          = 0
GRID_SIZE     = 10        # was 25
XI            = 0
N_INIT        = 3         # was 10
BETA_C        = 3.0
ORACLE_BUDGET = 10        # was 500 — tightest cap that still exercises the BO loop
OBJ_NOISE     = 0.3
CON_NOISE     = 1e-6
MODE          = 'min'

env_ref = build_paper_sim2_environment(grid_size=GRID_SIZE, xi=XI, dtype=None, constraint_representation='margin')
N_GRID  = len(env_ref['xx'])
_n_init = resolve_n_init(N_GRID, n_init=N_INIT)
INIT_IDX = np.array(sample_initial_indices(N_GRID, n_init=_n_init, seed=SEED), dtype=int)
log(f'grid={N_GRID}  n_init={len(INIT_IDX)}  budget={ORACLE_BUDGET}')

# 1. Safe BO (classical)
res_safe = run_safe_bo_simulation(
    mode=MODE, xi=XI, grid_size=GRID_SIZE,
    n_init=N_INIT, n_iter=ORACLE_BUDGET,
    obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
    beta_f=0.2, beta_c=BETA_C, lam0=0.8, lam_t0=10, lam_p=2.0,
    seed=SEED, init_idx=INIT_IDX.copy(),
    M_rff=64, lam_rff=1.0, lengthscale_rff=0.2, v_kernel_rff=1.0,
    con_lengthscale=1.0, con_outputscale=1.0,
)
log(f'Safe BO done | global_opt={res_safe["global_safe_opt"]:.4f}')

# 2. Quantum Safe BO (simulator)
res_qsafe = run_quantum_safe_bo_simulation(
    mode=MODE, xi=XI, grid_size=GRID_SIZE,
    n_init=N_INIT, oracle_budget=ORACLE_BUDGET,
    obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
    beta_f=0.2, beta_c=BETA_C, lam0=0.8, lam_t0=10, lam_p=2.0,
    seed=SEED, init_idx=INIT_IDX.copy(),
    M_rff=64, lam_rff=1.0, lengthscale_rff=0.2, v_kernel_rff=1.0,
    con_lengthscale=1.0, con_outputscale=1.0,
    use_quantum_query=True,
)
log(f'Quantum Safe BO (sim) done | total_oracle={res_qsafe["total_oracle_queries"]}')

# 2b. Quantum Safe BO (real hardware) — ibm_rensselaer
log('connecting to IBM Quantum...')
from qiskit_ibm_runtime import QiskitRuntimeService
service_ibm = QiskitRuntimeService()
real_backend = service_ibm.least_busy(operational=True, simulator=False, min_num_qubits=127)
log(f'selected backend: {real_backend.name}')

res_qsafe_real = run_quantum_safe_bo_simulation_real(
    mode=MODE, xi=XI, grid_size=GRID_SIZE,
    n_init=N_INIT, oracle_budget=ORACLE_BUDGET,
    obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
    beta_f=0.2, beta_c=BETA_C, lam0=0.8, lam_t0=10, lam_p=2.0,
    seed=SEED, init_idx=INIT_IDX.copy(),
    M_rff=64, lam_rff=1.0, lengthscale_rff=0.2, v_kernel_rff=1.0,
    con_lengthscale=1.0, con_outputscale=1.0,
    use_quantum_query=True,
    backend=real_backend,
)
log(f'Quantum Safe BO (real) done | total_oracle={res_qsafe_real.get("total_oracle_queries", "N/A")}')

# 3. Unconstrained BO
res_uncon = run_unconstrained_bo_simulation(
    mode=MODE, xi=XI, grid_size=GRID_SIZE,
    n_init=N_INIT, n_iter=ORACLE_BUDGET,
    obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
    beta_f=0.5,
    seed=SEED, init_idx=INIT_IDX.copy(),
    M_target=64, lengthscale=0.2, v_kernel=1.0,
)
log('Unconstrained BO done')

# 4. BO-ACL
res_acl = run_bo_acl_simulation2(
    mode=MODE, xi=XI, grid_size=GRID_SIZE,
    n_init=N_INIT, n_iter=ORACLE_BUDGET,
    batch_size=2,
    obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
    beta_f=1.0, beta_w=0.2,
    seed=SEED, init_idx=INIT_IDX.copy(),
    lengthscale=0.2,
)
log('BO-ACL done')

# --- Downstream aggregation (verify result-dict shapes line up) ---
def safe_rate(res):
    n = res['n_init']
    zz = res['zz']; h = zz.numpy() if hasattr(zz, 'numpy') else np.asarray(zz)
    qs = np.array(res['queried_idx'][n:], dtype=int)
    return float(np.mean(h[qs] >= res['xi'])) if len(qs) > 0 else float('nan')

def cumu_regret_last(res):
    arr = res.get('cumu_regret_expanded', res.get('queried_cumu_regret_hist'))
    return float(arr[-1]) if len(arr) else float('nan')

print()
print('=== SUMMARY ===')
for label, res in [
    ('Safe BO',             res_safe),
    ('Quantum Safe (sim)',  res_qsafe),
    ('Quantum Safe (real)', res_qsafe_real),
    ('Unconstrained BO',    res_uncon),
    ('BO-ACL',              res_acl),
]:
    print(f'{label:20s} | safe_rate={safe_rate(res):.3f} | cumu_regret={cumu_regret_last(res):.4f} | n_queries={len(res["queried_idx"])}')

print('SMOKE OK')
