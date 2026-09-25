"""
Standalone hyperparameter sweep for SafeSet-UCB GP logic.
Reimplements the core GP optimization loop WITHOUT ansys/env dependencies.
Uses the same surrogate model, constraint model, and GP logic as classic_safeset_discrete.py.
"""
import os
import sys
import json
import math
import copy
import warnings
import numpy as np
import torch
from datetime import datetime
from joblib import load
from itertools import product

from gpytorch.kernels import MaternKernel, RFFKernel, ScaleKernel
from botorch.models import FixedNoiseGP
import botorch.settings as botorch_settings

warnings.filterwarnings("ignore")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)   # surrogates, shapes and decks live at the repository root
NOISE_FREE_VARIANCE = 1e-6

# Helper for RFF feature extraction (avoids monkey-patching issues)
def get_kernel_features(kernel, x, num_dims=None, normalize=True):
    if not hasattr(kernel, "randn_weights"):
        d = x.size(-1) if num_dims is None else num_dims
        kernel._init_weights(d, kernel.num_samples)
    if kernel.randn_weights.dtype != x.dtype:
        kernel.randn_weights = kernel.randn_weights.to(dtype=x.dtype)
    return kernel._featurize(x, normalize=normalize)

# ── Reuse functions from classic_safeset_discrete.py ──

def sample_sobol_on_grid(grid_size=21, seed=None, device=None):
    device = device or torch.device("cpu")
    D = 18
    var_dims = [0, 17]
    values = torch.linspace(-1, 1, grid_size, device=device, dtype=torch.float64)
    pairs = torch.cartesian_prod(values, values)
    X = torch.zeros(pairs.shape[0], D, device=device, dtype=torch.float64)
    X[:, var_dims] = pairs
    return X

def lambda_by_stage(stage, lam0=1, t0=20, p=1.0):
    return float(lam0 * (t0 / (t0 + max(1, stage)))**p)

def stage_epsilon(var_value, lam, eps_max):
    return min(float(eps_max), math.sqrt(max(float(var_value), 1e-12) / float(lam)))

def minmax_norm(a, mask, eps=1e-12):
    v = a[mask]
    if v.numel() == 0:
        return a * 0.0
    lo = v.min()
    hi = v.max()
    return (a - lo) / (hi - lo + eps)

def compute_lcb_c_botorch(c_model, X, beta_c):
    post = c_model.posterior(X)
    mu = post.mean.view(-1)
    var = post.variance.view(-1).clamp_min(1e-12)
    sigma = var.sqrt()
    rad = math.sqrt(beta_c)
    lcb = mu - rad * sigma
    ucb = mu + rad * sigma
    return mu, sigma, lcb, ucb

def initialize_c_model(actions, response, device, ls_init=None, outputscale=None):
    yvar = torch.full_like(response, NOISE_FREE_VARIANCE, device=device)
    model = FixedNoiseGP(actions, response, yvar).to(device)
    if ls_init is not None or outputscale is not None:
        with torch.no_grad():
            if ls_init is not None:
                model.covar_module.base_kernel.lengthscale.copy_(ls_init.to(device))
            if outputscale is not None:
                model.covar_module.outputscale.copy_(outputscale.to(device))
        model.eval()
        return model, None
    mll = None
    model.eval()
    return model, mll

def initialize_model(actions, response, device, M_target, obs_noise=NOISE_FREE_VARIANCE,
                     ls_init=None, outputscale=None):
    noise_tensor = torch.as_tensor(obs_noise, dtype=response.dtype, device=device)
    if noise_tensor.ndim == 0:
        yvar = torch.full_like(response, float(noise_tensor.item()), device=device)
    else:
        yvar = noise_tensor.reshape(-1, 1) if noise_tensor.ndim == 1 else noise_tensor
        yvar = yvar.expand_as(response).clone()
    base_kernel = RFFKernel(nu=2.5, ard_num_dims=actions.shape[-1], num_samples=M_target)
    covar_module = ScaleKernel(base_kernel).to(device)
    model = FixedNoiseGP(actions, response, yvar, covar_module=covar_module).to(device)
    with torch.no_grad():
        if ls_init is not None:
            model.covar_module.base_kernel.lengthscale.copy_(ls_init.to(device))
        model.covar_module.outputscale.copy_(torch.tensor(1.0, dtype=torch.double, device=device))
    model.eval()
    return model, model.covar_module

def W_GP_UCB_scores(model, x_D, response, V_t, Unweighted_Phi, eps_list, beta, lam, chunk_size=1024):
    eps_list_t = torch.as_tensor(eps_list, device=V_t.device, dtype=V_t.dtype).view(-1, 1)
    Unweighted_Phi = Unweighted_Phi.to(device=V_t.device, dtype=V_t.dtype)
    response = response.to(device=V_t.device, dtype=V_t.dtype).reshape(-1, 1)
    x_D = x_D.to(device=V_t.device, dtype=V_t.dtype)
    beta = torch.as_tensor(beta, device=V_t.device, dtype=V_t.dtype)
    Y_weighted = response / (eps_list_t ** 2)
    L = torch.linalg.cholesky(V_t)
    rhs = Unweighted_Phi.T @ Y_weighted
    nu_t = torch.cholesky_solve(rhs, L)
    mean_chunks, var_chunks = [], []
    for start in range(0, x_D.shape[0], chunk_size):
        x_chunk = x_D[start:start+chunk_size]
        features = get_kernel_features(model.covar_module.base_kernel, x_chunk, x_chunk.shape[-1], normalize=True)
        mean_chunks.append((features @ nu_t).reshape(-1))
        proj = torch.cholesky_solve(features.T, L).T
        chunk_var = (lam * (proj * features).sum(dim=1)).clamp_min(1e-12)
        var_chunks.append(chunk_var)
    mean = torch.cat(mean_chunks, dim=0)
    var = torch.cat(var_chunks, dim=0)
    sigma = var.sqrt()
    ucb = mean + beta.sqrt() * sigma
    return ucb, mean, sigma, var


def evaluate_noise_free(surrogate_T, initPos, targetPos, error_init, actions):
    """Vectorized noise-free evaluation using the surrogate model."""
    actions_np = actions.detach().cpu().numpy().reshape(-1, 18).astype(np.float64)
    forces_batch = actions_np * 1000.0
    u_batch = forces_batch @ surrogate_T
    p_init_flat = initPos[:, 0:2].flatten()
    p_target_flat = targetPos[:, 0:2].flatten()
    dev_batch = (p_init_flat[None, :] + u_batch) - p_target_flat[None, :]
    mae_batch = np.abs(dev_batch).sum(axis=1) / dev_batch.shape[1]
    response_batch = -mae_batch / error_init
    response = torch.tensor(response_batch, dtype=torch.float64).reshape(-1, 1)
    true_response = torch.tensor(mae_batch, dtype=torch.float64).reshape(-1, 1)
    return response, true_response


def evaluate_noisy(surrogate_T, initPos, targetPos, error_init, action, obs_noise_var, eps_max):
    """Single noisy evaluation mimicking step_surrogate with non_monte_carlo."""
    response, true_response = evaluate_noise_free(surrogate_T, initPos, targetPos, error_init,
                                                   action)
    noise = torch.randn_like(response) * math.sqrt(obs_noise_var)
    noisy_response = response + noise
    # Query cost: ceil(1 / (4 * eps_max^2))
    n_queries = math.ceil(1.0 / (4.0 * eps_max**2))
    return noisy_response, true_response, n_queries


def load_surrogate_data(input_file):
    """Load the FEM surrogate data (matching ClassicFuselageEnv.reset)."""
    surrogate_model = load(os.path.join(REPO_ROOT, "surrogate_likeDu_v22.joblib"))
    surrogate_T = surrogate_model.coef_.T  # same as env.surrogate.T

    # Load initPos from the .npy matching the input file name
    file_base = os.path.basename(input_file).split(".")[0]
    shapes_dir = os.path.join(REPO_ROOT, "FuselageActuators", "Shapes", "Test")
    initPos = np.load(os.path.join(shapes_dir, file_base + ".npy"))

    # Target: SolutionInputDP53.npy (same default as env)
    targetPos = np.load(os.path.join(shapes_dir, "SolutionInputDP53.npy"))

    # error_init: mean absolute deviation
    dev = initPos[:, 0:2] - targetPos[:, 0:2]
    error_init = float(np.abs(dev).sum() / dev.size)

    return surrogate_T, initPos, targetPos, error_init


def run_single_trial(seed, search_space, tsai_wu_model, surrogate_T, initPos, targetPos,
                     error_init, args, device):
    """Run a single SafeSet-UCB trial and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = search_space.shape[0]
    M_target = args['M_features']
    lam = 1
    eta = 1

    # Compute safe set
    grid_np = search_space.detach().cpu().numpy().reshape(-1, 18)
    FI_all = tsai_wu_model.predict(grid_np, return_std=False)
    safety_margin = torch.tensor(1.0 - FI_all, dtype=torch.float64, device=device)
    safe_idx = torch.where(safety_margin >= 0.0)[0]

    # Seed-dependent random init
    safe_margins = safety_margin[safe_idx]
    weights = safe_margins.clamp_min(0.0).cpu().numpy().astype(float)
    weights /= weights.sum() + 1e-12
    rng = np.random.RandomState(seed)
    init_pool_size = max(args['initial_num_points'], args['n_constraint_init'])
    chosen_safe = rng.choice(safe_idx.cpu().numpy(),
                             size=min(init_pool_size, len(safe_idx)),
                             replace=False, p=weights)
    init_idx = torch.tensor(chosen_safe, dtype=torch.long, device=device)

    # Noise-free init
    chosen = init_idx[:args['initial_num_points']]
    f_X = search_space[chosen].to(device)
    f_Y, true_response = evaluate_noise_free(surrogate_T, initPos, targetPos, error_init, f_X)
    response = f_Y.clone()

    # Constraint GP init
    c_chosen = init_idx[:args['n_constraint_init']]
    c_X = search_space[c_chosen].to(device)
    FI_c = tsai_wu_model.predict(c_X.detach().cpu().numpy().reshape(-1, 18), return_std=False)
    c_Y = torch.tensor(1 - FI_c, dtype=torch.float64, device=device).reshape(-1, 1)

    c_ls = torch.full((1, 18), 0.6931, dtype=torch.float64, device=device)
    c_ls[0, 0] = 0.50
    c_ls[0, 17] = 0.50
    c_os = torch.tensor(1.0, dtype=torch.float64, device=device)
    c_model, _ = initialize_c_model(c_X, c_Y, device, ls_init=c_ls, outputscale=c_os)

    # Objective GP init
    lengthscale = torch.full((1, 18), 0.6931, dtype=torch.float64, device=device)
    lengthscale[0, 0] = args['obj_ls']
    lengthscale[0, 17] = args['obj_ls']

    n_seed = min(2, f_X.shape[0])
    init_X = f_X[:n_seed]
    init_Y = f_Y[:n_seed]
    init_yvar = torch.full_like(init_Y, NOISE_FREE_VARIANCE, device=device)
    model_ei, kernel = initialize_model(
        actions=init_X, response=init_Y, M_target=M_target, device=device,
        obs_noise=init_yvar, ls_init=lengthscale)

    f_X = init_X.clone()
    f_Y = init_Y.clone()
    actions = f_X.clone()
    response = f_Y.clone()
    true_response = true_response[:n_seed].clone()
    eps_list = [args['eps_max']] * n_seed

    Weighted_Phi = torch.zeros((n_seed, 2 * M_target), device=device, dtype=torch.float64)
    Unweighted_Phi = torch.zeros((n_seed, 2 * M_target), device=device, dtype=torch.float64)

    for i in range(n_seed):
        x = f_X[i,:].reshape(1,-1)
        features = get_kernel_features(model_ei.covar_module.base_kernel, x, 18, normalize=True)
        Unweighted_Phi[i, :] = features
        Weighted_Phi[i, :] = features * (1.0 / args['eps_max'])

    V_t = torch.matmul(Weighted_Phi.T, Weighted_Phi) + lam * torch.eye(2 * M_target).to(device).to(torch.float64)

    beta_c = 3.0
    n_iter = args['query_budget'] + 10
    ts = torch.arange(1, n_iter + 1, device=device, dtype=torch.float64)
    beta_t = (1.0 + args['B'] * torch.log(ts).abs()).pow(2)

    total_queries = 0
    iterations = 1
    stage_data = []

    while total_queries < args['query_budget']:
        iterations += 1
        t_acq = max(1, len(response) + 1)
        beta = beta_t[min(t_acq - 1, len(beta_t) - 1)]

        ucb_f, mean_f, sig_f, var_f = W_GP_UCB_scores(
            model=model_ei, x_D=search_space, response=f_Y,
            V_t=V_t, Unweighted_Phi=Unweighted_Phi,
            eps_list=eps_list, beta=beta, lam=lam)

        mu_c, sig_c, lcb_c, ucb_c = compute_lcb_c_botorch(c_model, search_space, beta_c)
        safe_mask = lcb_c >= 0.0
        safe_count = int(safe_mask.sum().item())

        if safe_mask.sum().item() == 0:
            idx = torch.argmax(lcb_c)
        else:
            eps_sigma = 1e-9
            a_bnd = -torch.abs(mu_c / (sig_c + eps_sigma))
            ucb_n = minmax_norm(ucb_f, mask=safe_mask)
            bnd_n = minmax_norm(a_bnd, mask=safe_mask)
            lam_t = lambda_by_stage(iterations, lam0=args['lam0'], t0=args['t0'], p=args['lam_p'])
            score = (1.0 - lam_t) * ucb_n + lam_t * bnd_n
            score[~safe_mask] = -1e18
            idx = torch.argmax(score)

        new_actions = search_space[idx].reshape(1, -1)
        var_val = var_f[idx].item()
        eps = stage_epsilon(var_val, lam, args['eps_max'])

        new_response, true_resp, n_q = evaluate_noisy(
            surrogate_T, initPos, targetPos, error_init,
            new_actions, args['obs_noise'], eps * eta)
        new_response = new_response.to(device)
        true_resp = true_resp.to(device)

        total_queries += n_q
        c_val = float(1.0 - tsai_wu_model.predict(
            new_actions.detach().cpu().numpy().reshape(-1, 18), return_std=False)[0])

        # Update constraint GP
        with torch.no_grad():
            c_fantasy_noise = torch.full((1,), 1e-6, device=device, dtype=torch.float64)
            c_model = c_model.get_fantasy_model(
                new_actions,
                torch.tensor([[c_val]], device=device, dtype=torch.float64).view(-1),
                noise=c_fantasy_noise)
            c_model.eval()

        eps_list.append(eps)
        actions = torch.cat([actions, new_actions.to(torch.float64)])
        response = torch.cat([response, new_response.to(torch.float64)])
        true_response = torch.cat([true_response, true_resp.to(torch.float64)])
        f_Y = torch.cat([f_Y, new_response.to(torch.float64)])

        x_max_feature = get_kernel_features(model_ei.covar_module.base_kernel, new_actions, 18, normalize=True)
        V_t = V_t + torch.matmul((x_max_feature * (1/eps)).T, x_max_feature * (1/eps))
        Unweighted_Phi = torch.cat([Unweighted_Phi, x_max_feature])

        safe_rate = safe_count / N
        fx = float(true_resp.item())
        stage_data.append({
            'stage': iterations,
            'fx': fx,
            'y': float(new_response.item()),
            'eps': eps,
            'safeset_rate': safe_rate,
            'safe_count': safe_count,
        })

    # Best true MAE among feasible points
    FI_queried = tsai_wu_model.predict(actions.detach().cpu().numpy().reshape(-1, 18), return_std=False)
    feasible = (1.0 - FI_queried) >= 0.0
    if feasible.any():
        best_true_mae = float(true_response[feasible].min().item())
    else:
        best_true_mae = float('nan')

    return {
        'seed': seed,
        'total_stages': iterations,
        'total_queries': total_queries,
        'best_true_mae': round(best_true_mae, 4),
        'final_safeset_rate': stage_data[-1]['safeset_rate'] if stage_data else 0,
        'final_safe_count': stage_data[-1]['safe_count'] if stage_data else 0,
        'final_fx': stage_data[-1]['fx'] if stage_data else float('nan'),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    botorch_settings.debug._set_state(True)

    # Load models
    tsai_wu_path = os.path.join(REPO_ROOT, "surrogate_tsaiwu.joblib")
    tsai_wu_model = load(tsai_wu_path)
    print(f"Loaded Tsai-Wu model from {tsai_wu_path}")

    input_file = os.path.join(REPO_ROOT, "FuselageActuators", "AnsysFiles", "Test",
                              "SolutionInputDP52.inp")
    surrogate_T, initPos, targetPos, error_init = load_surrogate_data(input_file)
    print(f"Loaded surrogate data, error_init={error_init:.4f}")

    search_space = sample_sobol_on_grid(grid_size=21, seed=0, device=device)
    print(f"Grid: {search_space.shape[0]} points")

    # ── Sweep grid ──
    NOISE_LEVELS = [0.01, 0.04]
    CONFIGS = [
        # (obj_ls, B, lam0, t0, lam_p, init_pts, c_init, label)
        (0.20, 1.0, 1.0,  5.0, 2.0, 1, 1, "baseline_ls02"),
        (0.50, 1.0, 1.0,  5.0, 2.0, 1, 1, "baseline_ls05"),
        (0.20, 1.0, 1.0,  5.0, 2.0, 5, 5, "init5_ls02"),
        (0.50, 1.0, 1.0,  5.0, 2.0, 5, 5, "init5_ls05"),
        (0.20, 1.0, 1.0, 10.0, 2.0, 5, 5, "t10_ls02"),
        (0.50, 1.0, 1.0, 10.0, 2.0, 5, 5, "t10_ls05"),
        (0.20, 1.0, 1.0, 20.0, 2.0, 5, 5, "t20_ls02"),
        (0.50, 1.0, 1.0, 20.0, 2.0, 5, 5, "t20_ls05"),
        (0.20, 1.0, 1.0, 10.0, 1.0, 5, 5, "t10_p1_ls02"),
        (0.50, 1.0, 1.0, 10.0, 1.0, 5, 5, "t10_p1_ls05"),
        (0.20, 3.0, 1.0, 10.0, 2.0, 5, 5, "B3_ls02"),
        (0.50, 0.5, 1.0, 10.0, 2.0, 5, 5, "B05_ls05"),
        (0.20, 1.0, 1.0, 15.0, 1.5, 5, 5, "t15_p15_ls02"),
        (0.50, 0.8, 1.0, 15.0, 1.5, 5, 5, "t15_p15_ls05"),
        (0.30, 1.0, 1.0, 10.0, 2.0, 5, 5, "t10_ls03"),
        (0.20, 1.0, 1.0, 10.0, 2.0, 3, 3, "t10_init3_ls02"),
    ]
    N_TRIALS = 5
    SEEDS = list(range(N_TRIALS))

    results = []
    total_runs = len(NOISE_LEVELS) * len(CONFIGS)
    run_idx = 0

    print(f"\n{'='*90}")
    print(f"Starting sweep: {total_runs} configs × {N_TRIALS} trials × {len(NOISE_LEVELS)} noise levels")
    print(f"{'='*90}")

    for noise in NOISE_LEVELS:
        for cfg in CONFIGS:
            obj_ls, B, lam0, t0, lam_p, init_pts, c_init, label = cfg
            run_idx += 1

            args = {
                'obs_noise': noise,
                'obj_ls': obj_ls,
                'B': B,
                'lam0': lam0,
                't0': t0,
                'lam_p': lam_p,
                'initial_num_points': init_pts,
                'n_constraint_init': c_init,
                'eps_max': 0.04,
                'M_features': 400,
                'query_budget': 20000,
            }

            print(f"\n[{run_idx}/{total_runs}] noise={noise}, {label} "
                  f"(ls={obj_ls}, B={B}, t0={t0}, p={lam_p}, init={init_pts}, c={c_init})")

            trial_results = []
            for s in SEEDS:
                try:
                    tr = run_single_trial(s, search_space, tsai_wu_model,
                                          surrogate_T, initPos, targetPos, error_init,
                                          args, device)
                    trial_results.append(tr)
                    print(f"  seed={s}: best_mae={tr['best_true_mae']:.4f}, "
                          f"safe={tr['final_safe_count']}/441, stages={tr['total_stages']}")
                except Exception as e:
                    print(f"  seed={s}: ERROR - {e}")

            if trial_results:
                avg_mae = np.mean([t['best_true_mae'] for t in trial_results])
                min_mae = np.min([t['best_true_mae'] for t in trial_results])
                max_mae = np.max([t['best_true_mae'] for t in trial_results])
                converged = sum(1 for t in trial_results if t['best_true_mae'] <= 0.08)
                avg_safe = np.mean([t['final_safeset_rate'] for t in trial_results])

                entry = {
                    'label': label,
                    'obs_noise': noise,
                    'obj_ls': obj_ls,
                    'B': B,
                    'lam0': lam0,
                    't0': t0,
                    'lam_p': lam_p,
                    'initial_num_points': init_pts,
                    'n_constraint_init': c_init,
                    'avg_mae': round(float(avg_mae), 4),
                    'min_mae': round(float(min_mae), 4),
                    'max_mae': round(float(max_mae), 4),
                    'converged': converged,
                    'n_trials': len(trial_results),
                    'avg_safeset_rate': round(float(avg_safe), 3),
                    'trials': trial_results,
                }
                results.append(entry)
                print(f"  → avg_mae={avg_mae:.4f}, conv={converged}/{len(trial_results)}, safe={avg_safe:.3f}")

            # Save intermediate
            with open(os.path.join(SCRIPT_DIR, 'sweep_results.json'), 'w') as f:
                json.dump(results, f, indent=2)

    # Final summary
    print(f"\n{'='*90}")
    print(f"SWEEP COMPLETE — {len(results)} configs")
    print(f"{'='*90}")

    header = f"{'noise':>6} {'label':>20} {'avg_mae':>8} {'min':>6} {'max':>6} {'conv':>6} {'safe%':>6}"
    print(header)
    print('-' * len(header))
    for r in sorted(results, key=lambda x: (x['obs_noise'], x['avg_mae'])):
        print(f"{r['obs_noise']:>6.2f} {r['label']:>20} {r['avg_mae']:>8.4f} {r['min_mae']:>6.4f} "
              f"{r['max_mae']:>6.4f} {r['converged']:>2}/{r['n_trials']} {r['avg_safeset_rate']:>6.3f}")

    with open(os.path.join(SCRIPT_DIR, 'sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to sweep_results.json")


if __name__ == '__main__':
    main()
