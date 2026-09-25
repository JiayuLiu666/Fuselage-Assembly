import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from joblib import load


ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "Experiments_constraints" / "Quantum_Discrete_cUCB" / "exp_set_1"


def grid_points(grid_size=21):
    values = np.linspace(-1.0, 1.0, grid_size)
    pairs = np.array(np.meshgrid(values, values)).T.reshape(-1, 2)
    grid = np.zeros((pairs.shape[0], 18), dtype=np.float64)
    grid[:, 0] = pairs[:, 0]
    grid[:, 17] = pairs[:, 1]
    return grid


def true_mae_for_grid(grid):
    surrogate = load(ROOT / "surrogate_likeDu_v22.joblib").coef_
    init_pos = np.load(ROOT / "FuselageActuators" / "Shapes" / "Test" / "SolutionInputDP52.npy")
    target_pos = np.load(ROOT / "FuselageActuators" / "Shapes" / "Test" / "SolutionInputDP53.npy").astype(np.float64)
    forces = grid * 1000.0
    disp = forces @ surrogate.T
    p_init = init_pos[:, 0:2].reshape(-1)
    p_target = target_pos[:, 0:2].reshape(-1)
    dev = (p_init[None, :] + disp) - p_target[None, :]
    return np.abs(dev).sum(axis=1) / dev.shape[1]


def safe_grid_optimum():
    grid = grid_points()
    mae = true_mae_for_grid(grid)
    tsai_wu = load(ROOT / "surrogate_tsaiwu.joblib")
    fi = tsai_wu.predict(grid, return_std=False).reshape(-1)
    safe = (1.0 - fi) >= 0.0
    safe_idx = np.where(safe)[0]
    best_local = safe_idx[np.argmin(mae[safe])]
    return {
        "safe_opt_mae": float(mae[best_local]),
        "safe_opt_index": int(best_local),
        "n_safe_grid": int(safe.sum()),
        "n_grid": int(grid.shape[0]),
    }


def load_trial_metrics(obs_noise, optimum):
    files = sorted(EXP_DIR.glob(f"{obs_noise}quan_training_data_*_.pth"))
    if not files:
        raise FileNotFoundError(f"No saved trial files found for obs_noise={obs_noise}")

    tsai_wu = load(ROOT / "surrogate_tsaiwu.joblib")
    trials = []
    for file in files:
        payload = torch.load(file, map_location="cpu")
        true_response = payload["true_response"].reshape(-1).numpy()
        actions = payload["actions"].reshape(-1, 18).numpy()
        if true_response.size == 0:
            continue

        c_vals = 1.0 - tsai_wu.predict(actions, return_std=False).reshape(-1)
        feasible = c_vals >= 0.0
        if feasible.any():
            best_mae = float(true_response[feasible].min())
            simple_regret = best_mae - optimum["safe_opt_mae"]
        else:
            best_mae = math.nan
            simple_regret = math.nan

        final_safe = bool(feasible[-1])
        final_simple_regret = (
            float(true_response[-1] - optimum["safe_opt_mae"]) if final_safe else math.nan
        )
        metadata = payload.get("metadata", {})
        trials.append(
            {
                "file": file.name,
                "steps": int(true_response.size),
                "best_mae": best_mae,
                "simple_regret": simple_regret,
                "final_mae": float(true_response[-1]),
                "final_simple_regret": final_simple_regret,
                "safe_queries": int(feasible.sum()),
                "unsafe_queries": int((~feasible).sum()),
                "final_total_budget": int(metadata.get("final_total_budget", -1)),
            }
        )
    return trials


def summarize_trials(trials):
    simple = np.array([t["simple_regret"] for t in trials], dtype=float)
    final_simple = np.array([t["final_simple_regret"] for t in trials], dtype=float)
    steps = np.array([t["steps"] for t in trials], dtype=float)
    unsafe = np.array([t["unsafe_queries"] for t in trials], dtype=float)
    return {
        "n_trials": int(len(trials)),
        "avg_simple_regret": float(np.nanmean(simple)),
        "max_simple_regret": float(np.nanmax(simple)),
        "avg_final_simple_regret": float(np.nanmean(final_simple)),
        "max_final_simple_regret": float(np.nanmax(final_simple)),
        "avg_steps": float(np.nanmean(steps)),
        "total_unsafe_queries": int(np.nansum(unsafe)),
    }


def run_config(config, obs_noise, log_dir):
    cmd = [
        sys.executable,
        str(ROOT / "quantum_safeset_discrete.py"),
        "--obs_noise",
        str(obs_noise),
        "--M_features",
        str(config["M_features"]),
        "--B",
        str(config["B"]),
        "--obj_ls",
        str(config["obj_ls"]),
        "--lam0",
        str(config["lam0"]),
        "--t0",
        str(config["t0"]),
        "--lam_p",
        str(config["lam_p"]),
        "--query_budget",
        str(config["query_budget"]),
        "--init_num_points",
        str(config["init_num_points"]),
    ]
    log_path = log_dir / f"{config['label']}.log"
    with log_path.open("w") as log_file:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    return str(log_path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Exact focused sweep for quantum_safeset_discrete.py.")
    parser.add_argument("--obs_noise", type=float, default=0.010000000000000002)
    parser.add_argument("--query_budget", type=int, default=20000)
    parser.add_argument("--init_num_points", type=int, default=5)
    args = parser.parse_args()

    configs = [
        {"label": "M400_B3_ls02_lam1_t10_p2", "M_features": 400, "B": 3.0, "obj_ls": 0.20, "lam0": 1.0, "t0": 10.0, "lam_p": 2.0},
        {"label": "M300_B3_ls02_lam1_t10_p2", "M_features": 300, "B": 3.0, "obj_ls": 0.20, "lam0": 1.0, "t0": 10.0, "lam_p": 2.0},
        {"label": "M600_B3_ls02_lam1_t10_p2", "M_features": 600, "B": 3.0, "obj_ls": 0.20, "lam0": 1.0, "t0": 10.0, "lam_p": 2.0},
        {"label": "M400_B1_ls02_lam1_t20_p2", "M_features": 400, "B": 1.0, "obj_ls": 0.20, "lam0": 1.0, "t0": 20.0, "lam_p": 2.0},
        {"label": "M400_B3_ls03_lam1_t10_p2", "M_features": 400, "B": 3.0, "obj_ls": 0.30, "lam0": 1.0, "t0": 10.0, "lam_p": 2.0},
        {"label": "M400_B3_ls02_lam05_t10_p2", "M_features": 400, "B": 3.0, "obj_ls": 0.20, "lam0": 0.5, "t0": 10.0, "lam_p": 2.0},
        {"label": "M400_B3_ls02_lam15_t10_p2", "M_features": 400, "B": 3.0, "obj_ls": 0.20, "lam0": 1.5, "t0": 10.0, "lam_p": 2.0},
        {"label": "M400_B1_ls02_lam1_t10_p1", "M_features": 400, "B": 1.0, "obj_ls": 0.20, "lam0": 1.0, "t0": 10.0, "lam_p": 1.0},
    ]
    for cfg in configs:
        cfg["query_budget"] = args.query_budget
        cfg["init_num_points"] = args.init_num_points

    optimum = safe_grid_optimum()
    log_dir = ROOT / "sweep_logs" / "quantum_safeset_exact"
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(
        f"Safe grid optimum: MAE={optimum['safe_opt_mae']:.12f} "
        f"at idx={optimum['safe_opt_index']} ({optimum['n_safe_grid']}/{optimum['n_grid']} safe)"
    )
    for i, cfg in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] Running {cfg['label']}", flush=True)
        log_path = run_config(cfg, args.obs_noise, log_dir)
        trials = load_trial_metrics(args.obs_noise, optimum)
        summary = summarize_trials(trials)
        row = {
            "config": cfg,
            "log": log_path,
            "optimum": optimum,
            "summary": summary,
            "trials": trials,
        }
        results.append(row)
        with (ROOT / "quantum_safeset_exact_sweep_results.json").open("w") as f:
            json.dump(results, f, indent=2)
        print(
            f"  max simple regret={summary['max_simple_regret']:.12g}, "
            f"avg steps={summary['avg_steps']:.1f}, unsafe={summary['total_unsafe_queries']}",
            flush=True,
        )

    ranked = sorted(
        results,
        key=lambda r: (
            r["summary"]["max_simple_regret"],
            r["summary"]["avg_simple_regret"],
            r["summary"]["total_unsafe_queries"],
            r["summary"]["avg_steps"],
        ),
    )
    with (ROOT / "quantum_safeset_exact_sweep_best.json").open("w") as f:
        json.dump(ranked[0], f, indent=2)

    print("\nRanked configs:")
    for row in ranked:
        cfg = row["config"]
        s = row["summary"]
        print(
            f"{cfg['label']}: max_sr={s['max_simple_regret']:.12g}, "
            f"avg_sr={s['avg_simple_regret']:.12g}, avg_steps={s['avg_steps']:.1f}, "
            f"unsafe={s['total_unsafe_queries']}"
        )

    best = ranked[0]["config"]
    print(
        "\nBest parameters: "
        f"M_features={best['M_features']}, B={best['B']}, obj_ls={best['obj_ls']}, "
        f"lam0={best['lam0']}, t0={best['t0']}, lam_p={best['lam_p']}"
    )


if __name__ == "__main__":
    main()
