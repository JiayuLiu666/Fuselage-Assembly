"""
Compare cumulative regret across actuator counts for continuous constraints.

Defaults:
  - actuator counts: 4, 6, 8
  - count 8 root: Experiments_constraint_continuous
  - noise variance: 0.1 ** 2
  - regret: cumsum(abs(F_OPT - true_response)), with rows expanded by queries

Example:
  python compare_actuator_count_cumulative_regret.py
  python compare_actuator_count_cumulative_regret.py --exp-sets 0 --trials 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-quan-fuselage")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_ROOTS = {
    4: Path("Experiments_constraint_continuous_actuators_4"),
    6: Path("Experiments_constraint_continuous_actuators_6"),
    8: Path("Experiments_constraint_continuous"),
}

METHOD_ORDER = ["Q-Safe BO", "C-Safe BO", "BO-ACL"]
COUNT_STYLE = {
    4: {"color": "#1f77b4", "marker": "o"},
    6: {"color": "#2ca02c", "marker": "s"},
    8: {"color": "#d62728", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cumulative regret for actuator counts 4, 6, and 8."
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[4, 6, 8],
        help="Actuator counts to compare. Defaults to 4 6 8.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1**2,
        help="Observation noise variance. Defaults to 0.1**2.",
    )
    parser.add_argument(
        "--f-opt",
        type=float,
        default=0.0,
        help="Reference optimum for regret. Defaults to 0.0.",
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=50_000,
        help="Maximum oracle-query budget to plot/summarize.",
    )
    parser.add_argument(
        "--x-lim",
        type=int,
        default=None,
        help="X-axis limit. Defaults to --max-budget.",
    )
    parser.add_argument(
        "--exp-sets",
        type=int,
        nargs="*",
        default=None,
        help="Optional exp_set filter, e.g. --exp-sets 0 1 2. Defaults to all.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        nargs="*",
        default=None,
        help="Optional trial filter, e.g. --trials 0 1 2 3 4. Defaults to all.",
    )
    parser.add_argument(
        "--methods",
        choices=METHOD_ORDER,
        nargs="+",
        default=METHOD_ORDER,
        help="Methods to include.",
    )
    parser.add_argument(
        "--root-4",
        type=Path,
        default=DEFAULT_ROOTS[4],
        help="Root directory for actuator count 4.",
    )
    parser.add_argument(
        "--root-6",
        type=Path,
        default=DEFAULT_ROOTS[6],
        help="Root directory for actuator count 6.",
    )
    parser.add_argument(
        "--root-8",
        type=Path,
        default=DEFAULT_ROOTS[8],
        help="Root directory for actuator count 8.",
    )
    parser.add_argument(
        "--output-prefix",
        default="actuator_count_cumulative_regret_noise_0p01",
        help="Prefix for output PNG and CSV files.",
    )
    return parser.parse_args()


def classify_training_file(file_path: Path) -> tuple[str, float, int] | None:
    """Return (method, noise, trial) for a saved continuous training file."""
    name = file_path.name
    path_str = str(file_path)

    if "quan_training_data" in name:
        method = "Q-Safe BO"
        token = "quan_training_data"
    elif "classic_safeset_training_data" in name:
        method = "C-Safe BO"
        token = "classic_safeset_training_data"
    elif "training_data" in name and "Classic_Continuous_ACL" in path_str:
        method = "BO-ACL"
        token = "training_data"
    else:
        return None

    pattern = rf"(?P<noise>.*?){re.escape(token)}_(?P<trial>\d+)_\.pth$"
    match = re.match(pattern, name)
    if match is None:
        return None

    return method, float(match.group("noise")), int(match.group("trial"))


def parse_exp_set(file_path: Path) -> int | None:
    for part in file_path.parts:
        match = re.match(r"exp_set_(\d+)$", part)
        if match:
            return int(match.group(1))
    return None


def to_numpy_1d(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)


def expand_by_queries(values, queries) -> np.ndarray:
    values = to_numpy_1d(values).astype(float)
    queries = to_numpy_1d(queries).astype(float)

    if len(values) != len(queries):
        raise ValueError(
            f"values and queries have different lengths: {len(values)} vs {len(queries)}"
        )

    queries_int = np.rint(queries).astype(np.int64)
    if not np.allclose(queries, queries_int):
        raise ValueError("queries contain non-integer values")

    keep = queries_int > 0
    return np.repeat(values[keep], queries_int[keep])


def compute_cumulative_regret(training_file: Path, f_opt: float) -> np.ndarray:
    data = torch.load(training_file, map_location="cpu", weights_only=False)
    expanded = expand_by_queries(data["true_response"], data["queries"])
    instantaneous = np.abs(float(f_opt) - expanded)
    return np.cumsum(instantaneous)


def discover_runs(
    roots: dict[int, Path],
    counts: list[int],
    methods: set[str],
    noise: float,
    exp_sets: list[int] | None,
    trials: list[int] | None,
) -> list[dict]:
    rows = []
    exp_filter = None if exp_sets is None else set(exp_sets)
    trial_filter = None if trials is None else set(trials)

    for count in counts:
        root = roots[count]
        if not root.exists():
            print(f"[WARN] actuator_count={count}: missing root {root}")
            continue

        for file_path in sorted(root.rglob("*training_data_*_.pth")):
            if "ckpt" in file_path.name:
                continue

            parsed = classify_training_file(file_path)
            if parsed is None:
                continue

            method, file_noise, trial = parsed
            if method not in methods or not np.isclose(file_noise, noise):
                continue

            exp_set = parse_exp_set(file_path)
            if exp_filter is not None and exp_set not in exp_filter:
                continue
            if trial_filter is not None and trial not in trial_filter:
                continue

            rows.append(
                {
                    "actuator_count": count,
                    "method": method,
                    "noise": file_noise,
                    "sigma": float(np.sqrt(file_noise)),
                    "exp_set": exp_set,
                    "trial": trial,
                    "path": file_path,
                }
            )

    return rows


def summarize_curves(curves: list[np.ndarray], max_budget: int) -> dict:
    curves = [curve for curve in curves if len(curve) > 0]
    if not curves:
        raise ValueError("no non-empty curves")

    min_len = min(len(curve) for curve in curves)
    if max_budget is not None:
        min_len = min(min_len, max_budget)
    arr = np.vstack([curve[:min_len] for curve in curves])
    mean = arr.mean(axis=0)
    stderr = arr.std(axis=0) / np.sqrt(arr.shape[0])

    return {
        "arr": arr,
        "mean": mean,
        "stderr": stderr,
        "lb": mean - stderr,
        "ub": mean + stderr,
        "min_len": min_len,
        "n_runs": arr.shape[0],
        "final_mean": float(mean[-1]),
        "final_stderr": float(stderr[-1]),
        "final_min": float(arr[:, -1].min()),
        "final_max": float(arr[:, -1].max()),
    }


def write_summary_csv(summary_rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "actuator_count",
        "method",
        "noise",
        "sigma",
        "n_runs",
        "steps",
        "exp_sets",
        "trials",
        "final_mean",
        "final_stderr",
        "final_min",
        "final_max",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_results(
    stats: dict[tuple[str, int], dict],
    methods: list[str],
    counts: list[int],
    x_lim: int,
    noise: float,
    output_path: Path,
) -> None:
    present_methods = [
        method for method in methods if any((method, count) in stats for count in counts)
    ]
    if not present_methods:
        raise ValueError("no plottable methods")

    fig, axes = plt.subplots(
        1,
        len(present_methods),
        figsize=(5.5 * len(present_methods), 4.6),
        squeeze=False,
    )

    subplot_tags = [f"({chr(ord('a') + i)})" for i in range(len(present_methods))]

    for idx, (ax, method) in enumerate(zip(axes[0, :], present_methods)):
        for count in counts:
            key = (method, count)
            if key not in stats:
                continue

            stat = stats[key]
            x = np.arange(1, stat["min_len"] + 1)
            style = COUNT_STYLE.get(
                count, {"color": None, "marker": None}
            )
            markevery = max(1, len(x) // 8)
            label = (
                f"{count} actuators "
            )

            ax.fill_between(
                x, stat["lb"], stat["ub"], color=style["color"], alpha=0.15
            )
            ax.plot(
                x,
                stat["mean"],
                color=style["color"],
                marker=style["marker"],
                markevery=markevery,
                markersize=5,
                linewidth=2,
                label=label,
            )

        ax.set_title(method)
        ax.set_xlabel("Iterations")
        if idx == 0:
            ax.set_ylabel("Cumulative regret")
        ax.set_xlim(0, x_lim)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)
        # (a) / (b) / (c) tag below each subplot
        ax.text(
            0.5,
            -0.18,
            subplot_tags[idx],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

    fig.suptitle(
        f"Cumulative Regret by Actuator Count, $\\sigma$ = {np.sqrt(noise):g}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    roots = {4: args.root_4, 6: args.root_6, 8: args.root_8}
    methods = [method for method in METHOD_ORDER if method in args.methods]
    method_set = set(methods)
    x_lim = args.x_lim if args.x_lim is not None else args.max_budget

    print(f"Noise variance : {args.noise:g} (sigma={np.sqrt(args.noise):g})")
    print(f"F_OPT          : {args.f_opt:g}")
    print(f"Max budget     : {args.max_budget}")
    print(f"Counts         : {args.counts}")
    print(f"Methods        : {methods}")
    if args.exp_sets is not None:
        print(f"exp_set filter : {args.exp_sets}")
    if args.trials is not None:
        print(f"trial filter   : {args.trials}")

    rows = discover_runs(
        roots=roots,
        counts=args.counts,
        methods=method_set,
        noise=args.noise,
        exp_sets=args.exp_sets,
        trials=args.trials,
    )
    print(f"\nDiscovered {len(rows)} matching training files")

    grouped_files: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped_files[(row["method"], row["actuator_count"])].append(row)

    stats = {}
    summary_rows = []

    for method in methods:
        for count in args.counts:
            group = grouped_files.get((method, count), [])
            if not group:
                print(f"  {method:16s} count={count}: NO DATA")
                continue

            curves = []
            for row in group:
                try:
                    curves.append(compute_cumulative_regret(row["path"], args.f_opt))
                except Exception as exc:
                    print(f"[WARN] failed to load {row['path']}: {exc}")

            if not curves:
                print(f"  {method:16s} count={count}: NO USABLE DATA")
                continue

            stat = summarize_curves(curves, args.max_budget)
            stats[(method, count)] = stat
            exp_sets = sorted({row["exp_set"] for row in group})
            trials = sorted({row["trial"] for row in group})
            print(
                f"  {method:16s} count={count}: "
                f"runs={stat['n_runs']:3d} steps={stat['min_len']:6d} "
                f"final={stat['final_mean']:.6g} +/- {stat['final_stderr']:.3g}"
            )
            summary_rows.append(
                {
                    "actuator_count": count,
                    "method": method,
                    "noise": args.noise,
                    "sigma": float(np.sqrt(args.noise)),
                    "n_runs": stat["n_runs"],
                    "steps": stat["min_len"],
                    "exp_sets": " ".join(str(item) for item in exp_sets),
                    "trials": " ".join(str(item) for item in trials),
                    "final_mean": stat["final_mean"],
                    "final_stderr": stat["final_stderr"],
                    "final_min": stat["final_min"],
                    "final_max": stat["final_max"],
                }
            )

    if not stats:
        raise SystemExit("No matching cumulative-regret curves were found.")

    png_path = Path(f"{args.output_prefix}.png")
    csv_path = Path(f"{args.output_prefix}.csv")
    plot_results(stats, methods, args.counts, x_lim, args.noise, png_path)
    write_summary_csv(summary_rows, csv_path)

    print(f"\nSaved plot    : {png_path}")
    print(f"Saved summary : {csv_path}")


if __name__ == "__main__":
    main()
