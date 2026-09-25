import numpy as np
from joblib import load as jload

def sample_matern52_spectral(d, M, lengthscales, rng):
    ls = np.asarray(lengthscales).reshape(1, d)
    normal = rng.standard_normal((M, d))
    chi2 = rng.chisquare(df=5, size=(M, 1))
    omega = normal / np.sqrt(chi2 / 5.0) / ls
    b = rng.uniform(0, 2 * np.pi, size=M)
    return omega, b

def rff_features(X, omega, b, M):
    proj = X @ omega.T + b[None, :]
    return np.sqrt(2.0 / M) * np.cos(proj)

def load_env():
    surrogate_coef = jload('surrogate_likeDu_v22.joblib').coef_
    initPos = np.load('FuselageActuators/Shapes/Test/SolutionInputDP52.npy')
    targetPos = np.load('FuselageActuators/Shapes/Test/SolutionInputDP53.npy').astype(np.float64)
    p_init = initPos[:, 0:2].flatten()
    p_target = targetPos[:, 0:2].flatten()
    error_init = float(np.sum(np.abs(p_init - p_target)) / len(p_init))
    return surrogate_coef, initPos, targetPos, error_init

def eval_grid(surrogate_coef, initPos, targetPos, error_init, grid):
    forces = grid * 1000.0
    u = forces @ surrogate_coef.T
    p_init = initPos[:, 0:2].flatten()
    p_target = targetPos[:, 0:2].flatten()
    dev = (p_init[None, :] + u) - p_target[None, :]
    mae = np.abs(dev).sum(axis=1) / dev.shape[1]
    y = -mae / error_init
    return y.reshape(-1, 1), mae.reshape(-1, 1)

def eval_constraint(grid):
    r2 = grid[:, 0]**2 + grid[:, 17]**2
    return 0.6 - 0.5 * r2

def gp_predict_constraint(X_train, y_train, X_test, ls_c, alpha=1e-6):
    def rbf_kernel(X1, X2, ls):
        X1s, X2s = X1 / ls, X2 / ls
        sq = np.sum(X1s**2, 1)[:, None] + np.sum(X2s**2, 1)[None, :] - 2 * X1s @ X2s.T
        return np.exp(-0.5 * sq)
    K = rbf_kernel(X_train, X_train, ls_c) + alpha * np.eye(len(X_train))
    K_s = rbf_kernel(X_train, X_test, ls_c)
    K_inv = np.linalg.inv(K)
    mu = K_s.T @ K_inv @ y_train
    var = 1.0 - np.sum((K_s.T @ K_inv) * K_s.T, axis=1)
    sig = np.sqrt(np.maximum(var, 1e-12))
    return mu.ravel(), sig

def run_trial(grid, y_all, mae_all, c_all, seed, B, t0, eps_max, obs_noise_var, n_iters=45):
    rng = np.random.default_rng(seed)
    d, N, M = 18, grid.shape[0], 400
    lam0, lam_p, beta_c = 1.0, 2.0, 3.0
    n_c_init = 5

    obj_ls = np.full(d, 0.6931); obj_ls[0] = 0.20; obj_ls[17] = 0.20
    c_ls = np.full(d, 0.6931); c_ls[0] = 0.50; c_ls[17] = 0.50
    safe_indices = np.where(c_all >= 0)[0]

    omega, b = sample_matern52_spectral(d, M, obj_ls, rng)
    Phi_grid = rff_features(grid, omega, b, M)

    idx0 = rng.choice(safe_indices)
    y_obs = [y_all[idx0, 0] + rng.normal(0, np.sqrt(obs_noise_var))]
    eps_list = [eps_max]
    Phi_mat = Phi_grid[idx0:idx0+1].copy()
    W_diag = np.array([1.0 / eps_max**2])
    lam = 1.0

    c_train_idx = rng.choice(safe_indices, size=min(n_c_init, len(safe_indices)), replace=False)
    c_X, c_y = grid[c_train_idx].copy(), c_all[c_train_idx].copy()

    ts = np.arange(1, n_iters + 10)
    beta_obj = (1 + B * np.sqrt(np.log(ts)**2))**2

    best_mae = mae_all[idx0, 0]
    best_iter = 0
    cum_regret = 0.0

    for it in range(n_iters):
        n_obs = len(y_obs)
        Phi_w = Phi_mat * np.sqrt(W_diag[:, None])
        V_t = Phi_w.T @ Phi_w + lam * np.eye(M)
        V_t_inv = np.linalg.inv(V_t)
        y_arr = np.array(y_obs).reshape(-1, 1)
        nu_t = V_t_inv @ Phi_mat.T @ (W_diag[:, None] * y_arr)
        mean = (Phi_grid @ nu_t).ravel()
        var = np.maximum(lam * np.sum((Phi_grid @ V_t_inv) * Phi_grid, axis=1), 1e-12)
        sigma = np.sqrt(var)
        ucb = mean + np.sqrt(beta_obj[n_obs-1]) * sigma

        mu_c, sig_c = gp_predict_constraint(c_X, c_y, grid, c_ls)
        lcb_c = mu_c - beta_c * sig_c
        safe_mask = lcb_c >= 0

        if not np.any(safe_mask):
            idx = np.argmax(lcb_c)
        else:
            eps_sigma = 1e-9
            a_bnd = -np.abs(mu_c / (sig_c + eps_sigma))
            def minmax_norm(v, mask):
                vals = v[mask]; vmin, vmax = vals.min(), vals.max()
                if vmax - vmin < 1e-12:
                    out = np.zeros_like(v); out[mask] = 0.5; return out
                out = np.zeros_like(v); out[mask] = (v[mask] - vmin) / (vmax - vmin); return out
            ucb_n = minmax_norm(ucb, safe_mask)
            bnd_n = minmax_norm(a_bnd, safe_mask)
            lam_t = float(lam0 * (t0 / (t0 + max(1, it+1)))**lam_p)
            score = (1.0 - lam_t) * ucb_n + lam_t * bnd_n
            score[~safe_mask] = -1e18
            idx = np.argmax(score)

        eps = min(eps_max, np.sqrt(max(var[idx], 1e-12) / lam))
        y_obs.append(y_all[idx, 0] + rng.normal(0, np.sqrt(obs_noise_var)))
        eps_list.append(eps)
        Phi_mat = np.vstack([Phi_mat, Phi_grid[idx]])
        W_diag = np.append(W_diag, 1.0 / eps**2)
        c_X = np.vstack([c_X, grid[idx:idx+1]]); c_y = np.append(c_y, c_all[idx])

        mae_t = mae_all[idx, 0]
        cum_regret += (mae_t - 0.07159)
        
        if mae_t < best_mae:
            best_mae = mae_t
            best_iter = it + 1

    return best_mae, cum_regret

if __name__ == "__main__":
    surrogate_coef, initPos, targetPos, error_init = load_env()
    values = np.linspace(-1, 1, 21)
    g0, g1 = np.meshgrid(values, values)
    grid = np.zeros((441, 18)); grid[:, 0] = g0.ravel(); grid[:, 17] = g1.ravel()
    y_all, mae_all = eval_grid(surrogate_coef, initPos, targetPos, error_init, grid)
    c_all = eval_constraint(grid)
    
    eps_max = 0.04
    obs_noise_var = 0.01

    combos = [
        (0.2, 5.0),
        (0.3, 10.0),
        (0.3, 5.0),
        (0.4, 5.0),
        (0.4, 10.0),
        (0.5, 10.0),
        (0.5, 20.0),
    ]
    
    print(f"Optimizing Hyperparameters for 45 steps (20k budget equivalent)...")
    for B, t0 in combos:
        all_maes = []
        all_regrets = []
        for seed in range(5):
            best_mae, cum_regret = run_trial(grid, y_all, mae_all, c_all, seed, B, t0, eps_max, obs_noise_var, n_iters=45)
            all_maes.append(best_mae)
            all_regrets.append(cum_regret)
            
        print(f"B={B:<4} t0={t0:<5} -> Regret: {np.mean(all_regrets):.2f} ± {np.std(all_regrets):.2f} | Final MAEs: {[round(m, 4) for m in all_maes]}")
