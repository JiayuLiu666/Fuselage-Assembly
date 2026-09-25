import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from joblib import load


DEFAULT_RUN = (
    "Experiments_constraint_continuous/"
    "Quantum_EXP_10_1.0_1.0_1.0_multi_gpu_noise_0.010000000000000002/"
    "exp_set_0/0.010000000000000002quan_training_data_4_.pth"
)
DEFAULT_INIT = "FuselageActuators/Shapes/Test/SolutionInputDP52.npy"
DEFAULT_TARGET = (
    "Experiments_constraint_continuous/"
    "Quantum_EXP_10_1.0_1.0_1.0_multi_gpu_noise_0.010000000000000002/"
    "exp_set_0/SolutionInputDP53.npy"
)
DEFAULT_SURROGATE = "surrogate_likeDu_v22.joblib"


def mae_between(a, b):
    return float(np.mean(np.abs((a - b).reshape(-1))))


def final_shape(init_shape, surrogate_coef, action):
    forces = np.asarray(action, dtype=np.float64).reshape(-1) * 1000.0
    disp = forces @ surrogate_coef.T
    return init_shape + disp.reshape((-1, 2))


def force_components(action):
    forces = np.asarray(action, dtype=np.float64).reshape(-1) * 1000.0
    angles = np.linspace(12.0, -192.0, 18)
    forces_y = forces * np.cos(np.deg2rad(angles))
    forces_z = forces * np.sin(np.deg2rad(angles))
    return forces, forces_y, forces_z


def best_so_far_at_budgets(mae, cum_queries, budgets):
    rows = []
    for budget in budgets:
        if budget == 0:
            rows.append(
                {
                    "label": "initial_no_force",
                    "source_index": -1,
                    "cum_queries": 0.0,
                    "mae": None,
                    "budget": 0.0,
                }
            )
            continue

        valid = np.where(cum_queries <= budget)[0]
        if valid.size == 0:
            idx = 0
        else:
            idx = valid[np.argmin(mae[valid])]
        rows.append(
            {
                "label": f"best_le_{int(budget)}_queries",
                "source_index": int(idx),
                "cum_queries": float(cum_queries[idx]),
                "mae": float(mae[idx]),
                "budget": float(budget),
            }
        )
    return rows


def record_improvements(mae, cum_queries):
    records = []
    best = np.inf
    for idx, value in enumerate(mae):
        if value < best:
            best = value
            records.append(
                {
                    "label": f"record_{len(records):02d}",
                    "source_index": int(idx),
                    "cum_queries": float(cum_queries[idx]),
                    "mae": float(value),
                    "budget": float(cum_queries[idx]),
                }
            )
    return records


def add_force_columns(row, action):
    forces, forces_y, forces_z = force_components(action)
    for i, value in enumerate(action):
        row[f"action_{i:02d}"] = float(value)
    for i, value in enumerate(forces):
        row[f"force_lb_{i:02d}"] = float(value)
    for i, value in enumerate(forces_y):
        row[f"force_y_lb_{i:02d}"] = float(value)
    for i, value in enumerate(forces_z):
        row[f"force_z_lb_{i:02d}"] = float(value)


def materialize_rows(selected, actions, error_init):
    out = []
    for row in selected:
        row = dict(row)
        if row["source_index"] < 0:
            action = np.zeros(18, dtype=np.float64)
            row["mae"] = float(error_init)
        else:
            action = actions[row["source_index"]]
        add_force_columns(row, action)
        out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Extract force configurations whose best-so-far shape gap decreases."
    )
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--init-shape", default=DEFAULT_INIT)
    parser.add_argument("--target-shape", default=DEFAULT_TARGET)
    parser.add_argument("--surrogate", default=DEFAULT_SURROGATE)
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=[0, 500, 1000, 10000, 50000],
        help="Cumulative oracle-query budgets. Use 0 for the no-force initial panel.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "Experiments_constraint_continuous/"
            "Quantum_EXP_10_1.0_1.0_1.0_multi_gpu_noise_0.010000000000000002/"
            "exp_set_0/shape_gap_reduction_extract"
        ),
    )
    parser.add_argument(
        "--skip-zero-query-prefix",
        action="store_true",
        help="Drop leading rows with zero query count before selecting best-so-far panels.",
    )
    args = parser.parse_args()

    run_path = Path(args.run)
    output_dir = Path(args.output_dir)
    data = torch.load(run_path, map_location="cpu")

    actions = data["actions"].detach().cpu().numpy().astype(np.float64)
    mae = data["true_response"].detach().cpu().numpy().reshape(-1).astype(np.float64)
    queries = data["queries"].detach().cpu().numpy().reshape(-1).astype(np.float64)
    source_offset = 0
    if args.skip_zero_query_prefix:
        nonzero = np.where(np.abs(queries) > 1e-12)[0]
        if nonzero.size:
            source_offset = int(nonzero[0])
            mae_for_selection = mae[source_offset:]
            queries_for_selection = queries[source_offset:]
        else:
            mae_for_selection = mae
            queries_for_selection = queries
    else:
        mae_for_selection = mae
        queries_for_selection = queries
        
    cum_queries = np.cumsum(queries_for_selection)
    init_shape = np.load(args.init_shape)
    target_shape = np.load(args.target_shape)
    surrogate_coef = load(args.surrogate).coef_

    raw_error_init = data.get("error_init")
    if isinstance(raw_error_init, torch.Tensor):
        error_init = float(raw_error_init.reshape(-1)[0])
    else:
        error_init = mae_between(init_shape, target_shape)

    panel_selection = best_so_far_at_budgets(mae_for_selection, cum_queries, args.budgets)
    record_selection = (
        [{"label": "initial_no_force", "source_index": -1, "cum_queries": 0.0, "mae": None, "budget": 0.0}]
        + record_improvements(mae_for_selection, cum_queries)
    )
    for row in panel_selection + record_selection:
        if row["source_index"] >= 0:
            row["source_index"] += source_offset

    panel_rows = materialize_rows(panel_selection, actions, error_init)
    record_rows = materialize_rows(record_selection, actions, error_init)

    write_csv(output_dir / "panel_force_configs.csv", panel_rows)
    write_csv(output_dir / "record_improvement_force_configs.csv", record_rows)

    panel_actions = np.array(
        [
            np.zeros(18, dtype=np.float64)
            if row["source_index"] < 0
            else actions[row["source_index"]]
            for row in panel_rows
        ]
    )
    panel_final_shapes = np.array(
        [final_shape(init_shape, surrogate_coef, action) for action in panel_actions]
    )

    np.savez(
        output_dir / "panel_shapes_and_forces.npz",
        init_shape=init_shape,
        target_shape=target_shape,
        final_shapes=panel_final_shapes,
        actions=panel_actions,
        forces_lb=panel_actions * 1000.0,
        mae=np.array([row["mae"] for row in panel_rows], dtype=np.float64),
        cum_queries=np.array([row["cum_queries"] for row in panel_rows], dtype=np.float64),
        source_indices=np.array([row["source_index"] for row in panel_rows], dtype=np.int64),
    )

    summary = {
        "run": str(run_path),
        "init_shape": args.init_shape,
        "target_shape": args.target_shape,
        "error_init": error_init,
        "total_saved_rows": int(len(mae)),
        "selection_source_offset": int(source_offset),
        "total_cum_queries": float(cum_queries[-1]),
        "best_mae": float(mae_for_selection.min()),
        "best_source_index": int(mae_for_selection.argmin() + source_offset),
        "best_cum_queries": float(cum_queries[mae_for_selection.argmin()]),
        "panel_rows": panel_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote {output_dir / 'panel_force_configs.csv'}")
    print(f"Wrote {output_dir / 'record_improvement_force_configs.csv'}")
    print(f"Wrote {output_dir / 'panel_shapes_and_forces.npz'}")
    print(f"Wrote {output_dir / 'summary.json'}")
    for row in panel_rows:
        print(
            f"{row['label']}: idx={row['source_index']} "
            f"cum_queries={row['cum_queries']:.0f} mae={row['mae']:.6f}"
        )


if __name__ == "__main__":
    main()
