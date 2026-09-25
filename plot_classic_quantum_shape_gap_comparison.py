import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_shape_gap_reduction_panels import (
    actuator_positions,
    draw_panel,
    load_panel_rows,
)


DEFAULT_QUANTUM_DIR = (
    "Experiments_constraint_continuous/"
    "Quantum_EXP_10_1.0_1.0_1.0_multi_gpu_noise_0.010000000000000002/"
    "exp_set_0/shape_gap_reduction_extract"
)
DEFAULT_CLASSIC_DIR = (
    "Experiments_constraint_continuous/"
    "Classic_Continuous_ACL_1.0_1.0_1.0_noise_0.010000000000000002/"
    "exp_set_0/shape_gap_reduction_extract_trial4"
)


def load_extract(extract_dir):
    extract_dir = Path(extract_dir)
    data = np.load(extract_dir / "panel_shapes_and_forces.npz")
    rows = load_panel_rows(extract_dir / "panel_force_configs.csv")
    return rows, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantum-dir", default=DEFAULT_QUANTUM_DIR)
    parser.add_argument("--classic-dir", default=DEFAULT_CLASSIC_DIR)
    parser.add_argument(
        "--output",
        default=(
            "Experiments_constraint_continuous/"
            "trial4_classic_quantum_shape_gap_comparison.png"
        ),
    )
    args = parser.parse_args()

    q_rows, q_data = load_extract(args.quantum_dir)
    c_rows, c_data = load_extract(args.classic_dir)
    if len(q_rows) != len(c_rows):
        raise ValueError("Quantum and classic extracts must have the same number of panels.")

    n_cols = len(q_rows)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(4.1 * n_cols, 8.4),
        constrained_layout=True,
    )

    q_target = q_data["target_shape"]
    c_target = c_data["target_shape"]
    q_actuators = actuator_positions(q_target, np.linspace(12.0, -192.0, 18))
    c_actuators = actuator_positions(c_target, np.linspace(12.0, -192.0, 18))

    for col in range(n_cols):
        draw_panel(
            axes[0, col],
            q_target,
            q_data["final_shapes"][col],
            q_rows[col],
            q_actuators,
            show_legend=(col == n_cols - 1),
        )
        draw_panel(
            axes[1, col],
            c_target,
            c_data["final_shapes"][col],
            c_rows[col],
            c_actuators,
            show_legend=False,
        )
        axes[1, col].text(
            0.5,
            -0.16,
            f"queries {float(c_rows[col]['cum_queries']):.0f}",
            transform=axes[1, col].transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )
        axes[0, col].text(
            0.5,
            -0.16,
            f"queries {float(q_rows[col]['cum_queries']):.0f}",
            transform=axes[0, col].transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

    axes[0, 0].text(
        -0.35,
        0.5,
        "Quantum BO\ntrial 4",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="center",
        fontsize=18,
    )
    axes[1, 0].text(
        -0.35,
        0.5,
        "Classic BO\ntrial 4",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="center",
        fontsize=18,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
