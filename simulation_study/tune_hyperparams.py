import sys, os
import numpy as np
import time
import itertools
import random

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from safe_set_bo import run_safe_bo_simulation
from experiment_env import build_paper_sim2_environment, sample_initial_indices
from safe_set_bo import resolve_n_init

SEED = 0
GRID_SIZE = 25
XI = 0
ORACLE_BUDGET = 300
OBJ_NOISE = 0.3
CON_NOISE = 1e-6
MODE = 'min'

env_ref = build_paper_sim2_environment(grid_size=GRID_SIZE, xi=XI, dtype=None, constraint_representation='margin')
N_GRID = len(env_ref['xx'])

search_space = {
    'lengthscale_rff': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
    'con_lengthscale': [0.25, 0.5, 1.0],
    'n_init': [5, 10, 15],
    'lam0': [0.5, 0.8, 1.0],
    'lam_t0': [5, 10, 20],
    'lam_p': [0.5, 1.0, 2.0],
    'beta_f': [0.5, 1.0, 2.0]
}

keys = list(search_space.keys())
values = list(search_space.values())
all_combinations = list(itertools.product(*values))

random.seed(42)
selected_combinations = random.sample(all_combinations, min(80, len(all_combinations)))

best_score = float('inf')
best_params = None
results = []

print(f"Testing {len(selected_combinations)} hyperparameter combinations...")

for i, combo in enumerate(selected_combinations):
    params = dict(zip(keys, combo))
    
    _n_init = resolve_n_init(N_GRID, n_init=params['n_init'])
    INIT_IDX = np.array(sample_initial_indices(N_GRID, n_init=_n_init, seed=SEED), dtype=int)
    
    res = run_safe_bo_simulation(
        mode=MODE, xi=XI, grid_size=GRID_SIZE,
        n_init=params['n_init'], n_iter=ORACLE_BUDGET,
        obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
        beta_f=params['beta_f'], beta_c=3.0, 
        lam0=params['lam0'], lam_t0=params['lam_t0'], lam_p=params['lam_p'],
        seed=SEED, init_idx=INIT_IDX.copy(),
        M_rff=256, lam_rff=1.0, lengthscale_rff=params['lengthscale_rff'], v_kernel_rff=1.0,
        con_lengthscale=params['con_lengthscale'], con_outputscale=1.0,
    )
    
    # Calculate simple regret properly using noise-free yy
    _yy  = np.asarray(res['yy'])
    _zz  = np.asarray(res['zz'])
    _q   = np.asarray(res['queried_idx'], dtype=int)
    _safe_queries = _q[_zz[_q] >= XI]
    
    if len(_safe_queries) > 0:
        _best_nf = float(np.min(_yy[_safe_queries]))
        simple_regret = abs(_best_nf - float(res['global_safe_opt']))
    else:
        simple_regret = float('inf')
        
    cumu_regret = res['queried_cumu_regret_hist'][-1] if len(res['queried_cumu_regret_hist']) > 0 else float('inf')
    
    # Safe rate
    safe_rate = float(np.mean(_zz[_q[params['n_init']:]] >= XI)) if len(_q) > params['n_init'] else 0.0
    
    # We heavily penalize simple regret not being 0 and safe rate not being 1.
    score = cumu_regret
    if simple_regret > 1e-4:
        score += 10000.0 * simple_regret
    if safe_rate < 1.0:
        score += 10000.0 * (1.0 - safe_rate)
        
    print(f"[{i+1}/{len(selected_combinations)}] Score: {score:.4f} | SimpleReg: {simple_regret:.4f} | CumuReg: {cumu_regret:.4f} | SafeRate: {safe_rate:.3f} | Params: {params}")
    
    results.append({
        'params': params,
        'simple_regret': simple_regret,
        'cumu_regret': cumu_regret,
        'safe_rate': safe_rate,
        'score': score
    })

results.sort(key=lambda x: x['score'])

print("\n" + "="*50)
print("Top 5 Combinations:")
for i, res in enumerate(results[:5]):
    print(f"{i+1}. Score: {res['score']:.4f} | Simple: {res['simple_regret']:.4f} | CumuReg: {res['cumu_regret']:.4f} | SafeRate: {res['safe_rate']:.3f}")
    print(f"   {res['params']}")
print("="*50)
