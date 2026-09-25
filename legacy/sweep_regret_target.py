import json
import math
from itertools import product

import numpy as np
from joblib import load as jload


TARGET_MAE = 0.07159
EPS_MAX = 0.04
OBS_NOISE_VAR = 0.01
INIT_POINTS = 5
CONSTRAINT_INIT_POINTS = 5


def sample_matern52_spectral(d, m_features, lengthscales, rng):
    ls = np.asarray(lengthscales).reshape(1, d)
    normal = rng.standard_normal((m_features, d))
    chi2 = rng.chisquare(df=5, size=(m_features, 1))
    omega = normal / np.sqrt(chi2 / 5.0) / ls
    phase = rng.uniform(0, 2 * np.pi, size=m_features)
    return omega, phase


def rff_features(x, omega, phase, m_features):
    proj = x @ omega.T + phase[None, :]
    return np.sqrt(2.0 / m_features) * np.cos(proj)


def eval_grid(surrogate_coef, init_pos, target_pos, error_init, grid):
    forces = grid * 1000.0
    disp = forces @ surrogate_coef.T
    p_init = init_pos[:, 0:2].flatten()
    p_target = target_pos[:, 0:2].flatten()
    dev = (p_init[None, :] + disp) - p_target[None, :]
    mae = np.abs(dev).sum(axis=1) / dev.shape[1]
    y = -mae / error_init
    return y.reshape(-1, 1), mae.reshape(-1, 1)


def eval_constraint_analytic(grid):
    # Fast fallback used when surrogate_tsaiwu cannot be deserialized in this env.
    r2 = grid[:, 0] ** 2 + grid[:, 17] ** 2
    return 0.6 - 0.5 * r2


def gp_predict_constraint(x_train, y_train, x_test, ls_c, alpha=1e-6):
    def rbf_kernel(x1, x2, ls):
        x1s, x2s = x1 / ls, x2 / ls
        sq = np.sum(x1s**2, 1)[:, None] + np.sum(x2s**2, 1)[None, :] - 2 * x1s @ x2s.T
        return np.exp(-0.5 * sq)

    k = rbf_kernel(x_train, x_train, ls_c) + alpha * np.eye(len(x_train))
    ks = rbf_kernel(x_train, x_test, ls_c)
    k_inv = np.linalg.inv(k)
    mu = ks.T @ k_inv @ y_train
    var = 1.0 - np.sum((ks.T @ k_inv) * ks.T, axis=1)
    sig = np.sqrt(np.maximum(var, 1e-12))
    return mu.ravel(), sig


def minmax_norm(values, mask):
    out = np.zeros_like(values)
    masked = values[mask]
    if masked.size == 0:
        return out
    lo, hi = masked.min(), masked.max()
    if hi - lo < 1e-12:
        out[mask] = 0.5
    else:
        out[mask] = (values[mask] - lo) / (hi - lo)
    return out


def run_trial(
    grid,
    y_all,
    mae_all,
    c_all,
    seed,
    *,
    obj_ls_active,
    b_coef,
    m_features,
    t0,
    lam0,
    lam_p,
    n_iters,
):
    rng = np.random.default_rng(seed)
    d = 18
    beta_c = 3.0
    lam = 1.0

    safe_idx = np.where(c_all >= 0.0)[0]
    if safe_idx.size == 0:
        raise RuntimeError("No safe points in grid.")

    # Objective RFF setup
    obj_ls = np.full(d, 0.6931)
    obj_ls[0] = obj_ls_active
    obj_ls[17] = obj_ls_active
    omega, phase = sample_matern52_spectral(d, m_features, obj_ls, rng)
    phi_grid = rff_features(grid, omega, phase, m_features)

    # Constraint GP setup
    c_ls = np.full(d, 0.6931)
    c_ls[0] = 0.50
    c_ls[17] = 0.50

    margins = np.clip(c_all[safe_idx], 0.0, None)
    probs = margins / (margins.sum() + 1e-12)

    # Fixed-size safe warmup (not swept on purpose)
    n_init = min(INIT_POINTS, safe_idx.size)
    init_idx = rng.choice(safe_idx, size=n_init, replace=False, p=probs if probs.sum() > 0 else None)

    y_obs = [float(y_all[i, 0] + rng.normal(0, math.sqrt(OBS_NOISE_VAR))) for i in init_idx]
    eps_list = [EPS_MAX] * n_init
    phi_mat = phi_grid[init_idx].copy()
    w_diag = np.array([1.0 / (EPS_MAX**2)] * n_init)

    n_c_init = min(CONSTRAINT_INIT_POINTS, safe_idx.size)
    c_train_idx = rng.choice(
        safe_idx,
        size=n_c_init,
        replace=False,
        p=probs if probs.sum() > 0 else None,
    )
    c_x = grid[c_train_idx].copy()
    c_y = c_all[c_train_idx].copy()

    ts = np.arange(1, n_iters + 20)
    beta_obj = (1.0 + b_coef * np.abs(np.log(ts))) ** 2

    mae_series = []
    cum_regret = 0.0

    for it in range(n_iters):
        n_obs = len(y_obs)
        phi_w = phi_mat * np.sqrt(w_diag[:, None])
        v_t = phi_w.T @ phi_w + lam * np.eye(m_features)
        v_t_inv = np.linalg.inv(v_t)
        y_arr = np.array(y_obs).reshape(-1, 1)
        nu_t = v_t_inv @ phi_mat.T @ (w_diag[:, None] * y_arr)

        mean = (phi_grid @ nu_t).ravel()
        var = np.maximum(lam * np.sum((phi_grid @ v_t_inv) * phi_grid, axis=1), 1e-12)
        sigma = np.sqrt(var)
        beta = beta_obj[min(n_obs - 1, len(beta_obj) - 1)]
        ucb = mean + np.sqrt(beta) * sigma

        mu_c, sig_c = gp_predict_constraint(c_x, c_y, grid, c_ls)
        lcb_c = mu_c - beta_c * sig_c
        safe_mask = lcb_c >= 0.0

        if not np.any(safe_mask):
            idx = int(np.argmax(lcb_c))
        else:
            a_bnd = -np.abs(mu_c / (sig_c + 1e-9))
            ucb_n = minmax_norm(ucb, safe_mask)
            bnd_n = minmax_norm(a_bnd, safe_mask)
            lam_t = float(lam0 * (t0 / (t0 + max(1, it + 1))) ** lam_p)
            score = (1.0 - lam_t) * ucb_n + lam_t * bnd_n
            score[~safe_mask] = -1e18
            idx = int(np.argmax(score))

        eps = min(EPS_MAX, math.sqrt(max(var[idx], 1e-12) / lam))
        y_new = float(y_all[idx, 0] + rng.normal(0, math.sqrt(OBS_NOISE_VAR)))
        mae_t = float(mae_all[idx, 0])

        mae_series.append(mae_t)
        cum_regret += mae_t - TARGET_MAE

        y_obs.append(y_new)
        eps_list.append(float(eps))
        phi_mat = np.vstack([phi_mat, phi_grid[idx]])
        w_diag = np.append(w_diag, 1.0 / (eps**2))
        c_x = np.vstack([c_x, grid[idx : idx + 1]])
        c_y = np.append(c_y, c_all[idx])

    tail = np.array(mae_series[-10:])
    return {
        "cum_regret": float(cum_regret),
        "tail_mean_abs": float(np.mean(np.abs(tail - TARGET_MAE))),
        "tail_hit_rate": float(np.mean(np.abs(tail - TARGET_MAE) <= 5e-4)),
        "final_mae": float(mae_series[-1]),
        "best_mae": float(np.min(mae_series)),
    }


def aggregate_metrics(trials):
    avg_regret = float(np.mean([t["cum_regret"] for t in trials]))
    std_regret = float(np.std([t["cum_regret"] for t in trials]))
    tail_abs = float(np.mean([t["tail_mean_abs"] for t in trials]))
    tail_hit = float(np.mean([t["tail_hit_rate"] for t in trials]))
    final_mae = float(np.mean([t["final_mae"] for t in trials]))
    best_mae = float(np.mean([t["best_mae"] for t in trials]))
    # Primary goal: minimize regret, secondary goal: keep tail near target.
    score = avg_regret + 200.0 * tail_abs + 2.0 * (1.0 - tail_hit)
    return {
        "avg_regret": avg_regret,
        "std_regret": std_regret,
        "tail_abs": tail_abs,
        "tail_hit": tail_hit,
        "avg_final_mae": final_mae,
        "avg_best_mae": best_mae,
        "score": score,
    }


def main():
    surrogate_coef = jload("surrogate_likeDu_v22.joblib").coef_
    tsai_wu = None
    tsai_mode = "analytic"
    try:
        tsai_wu = jload("surrogate_tsaiwu.joblib")
        tsai_mode = "surrogate_tsaiwu"
    except Exception as exc:
        print(f"[Warning] Failed to load surrogate_tsaiwu.joblib, fallback to analytic constraint: {exc}")

    init_pos = np.load("FuselageActuators/Shapes/Test/SolutionInputDP52.npy")
    target_pos = np.load("FuselageActuators/Shapes/Test/SolutionInputDP53.npy").astype(np.float64)
    p_init = init_pos[:, 0:2].flatten()
    p_target = target_pos[:, 0:2].flatten()
    error_init = float(np.abs(p_init - p_target).sum() / len(p_init))

    values = np.linspace(-1.0, 1.0, 21)
    g0, g1 = np.meshgrid(values, values)
    grid = np.zeros((441, 18))
    grid[:, 0] = g0.ravel()
    grid[:, 17] = g1.ravel()

    y_all, mae_all = eval_grid(surrogate_coef, init_pos, target_pos, error_init, grid)
    if tsai_wu is not None:
        fi = tsai_wu.predict(grid, return_std=False).reshape(-1)
        c_all = 1.0 - fi
    else:
        c_all = eval_constraint_analytic(grid)

    # Sweep ONLY requested parameters.
    param_space = list(
        product(
            [0.15, 0.2, 0.25, 0.3],   # obj_ls
            [0.5, 1.0, 2.0, 3.0],     # B
            [300, 400, 600],          # M_features
            [5.0, 10.0, 20.0],        # t0
            [0.5, 1.0, 1.5],          # lam0
            [1.0, 2.0],               # lam_p
        )
    )

    rng = np.random.default_rng(20260424)
    coarse_idx = rng.choice(len(param_space), size=12, replace=False)
    coarse_cfgs = [param_space[i] for i in coarse_idx]

    coarse_results = []
    for obj_ls, b_coef, m_feat, t0, lam0, lam_p in coarse_cfgs:
        cfg = {
            "obj_ls": obj_ls,
            "B": b_coef,
            "M_features": m_feat,
            "t0": t0,
            "lam0": lam0,
            "lam_p": lam_p,
        }
        trials = []
        for seed in [0, 1, 2]:
            trials.append(
                run_trial(
                    grid,
                    y_all,
                    mae_all,
                    c_all,
                    seed,
                    obj_ls_active=obj_ls,
                    b_coef=b_coef,
                    m_features=m_feat,
                    t0=t0,
                    lam0=lam0,
                    lam_p=lam_p,
                    n_iters=20,
                )
            )
        metrics = aggregate_metrics(trials)
        coarse_results.append({**cfg, **metrics})

    coarse_results.sort(key=lambda x: x["score"])
    top_cfgs = coarse_results[:4]

    final_results = []
    for cfg in top_cfgs:
        trials = []
        for seed in [0, 1, 2]:
            trials.append(
                run_trial(
                    grid,
                    y_all,
                    mae_all,
                    c_all,
                    seed,
                    obj_ls_active=cfg["obj_ls"],
                    b_coef=cfg["B"],
                    m_features=int(cfg["M_features"]),
                    t0=cfg["t0"],
                    lam0=cfg["lam0"],
                    lam_p=cfg["lam_p"],
                    n_iters=35,
                )
            )
        metrics = aggregate_metrics(trials)
        final_results.append({**cfg, **metrics, "trials": trials})

    final_results.sort(key=lambda x: x["score"])
    best = final_results[0]

    with open("regret_sweep_results.json", "w") as f:
        json.dump({"constraint_mode": tsai_mode, "top": final_results}, f, indent=2)

    with open("regret_sweep_best.json", "w") as f:
        json.dump(best, f, indent=2)

    print("Top configs (final stage):")
    for i, row in enumerate(final_results, 1):
        print(
            f"{i:02d}. obj_ls={row['obj_ls']:.2f}, B={row['B']:.2f}, M={int(row['M_features'])}, "
            f"t0={row['t0']:.1f}, lam0={row['lam0']:.2f}, lam_p={row['lam_p']:.1f} | "
            f"regret={row['avg_regret']:.3f}±{row['std_regret']:.3f}, "
            f"tail_abs={row['tail_abs']:.5f}, tail_hit={row['tail_hit']:.2f}, "
            f"final={row['avg_final_mae']:.5f}"
        )

    print("\nBest config:")
    print(
        f"obj_ls={best['obj_ls']}, B={best['B']}, M_features={int(best['M_features'])}, "
        f"t0={best['t0']}, lam0={best['lam0']}, lam_p={best['lam_p']}"
    )
    print(
        f"avg_regret={best['avg_regret']:.4f}, tail_abs={best['tail_abs']:.6f}, "
        f"tail_hit={best['tail_hit']:.3f}, avg_final_mae={best['avg_final_mae']:.6f}"
    )


if __name__ == "__main__":
    main()
