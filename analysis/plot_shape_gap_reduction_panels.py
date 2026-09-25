import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_EXTRACT_DIR = (
    "Experiments_constraint_continuous/"
    "Quantum_EXP_10_1.0_1.0_1.0_multi_gpu_noise_0.010000000000000002/"
    "exp_set_0/shape_gap_reduction_extract"
)


def load_panel_rows(csv_path):
    with Path(csv_path).open() as f:
        return list(csv.DictReader(f))


def actuator_positions(shape, angles):
    center = shape.mean(axis=0)
    rel = shape - center
    theta = np.rad2deg(np.arctan2(rel[:, 1], rel[:, 0]))
    out = []
    for angle in angles:
        diff = np.abs(((theta - angle + 180.0) % 360.0) - 180.0)
        out.append(shape[int(np.argmin(diff))])
    return np.asarray(out)


def row_forces(row):
    return np.array([float(row[f"force_lb_{i:02d}"]) for i in range(18)])


def draw_panel(ax, target, achieved, row, actuator_xy, show_legend=False):
    forces = row_forces(row)
    active = np.abs(forces) > 1e-9
    angles = np.linspace(12.0, -192.0, 18)
    force_y = forces * np.cos(np.deg2rad(angles))
    force_z = forces * np.sin(np.deg2rad(angles))

    ax.scatter(target[:, 0], target[:, 1], s=8, color="#1f77b4", label="Target shape")
    ax.scatter(achieved[:, 0], achieved[:, 1], s=8, color="#ff7f0e", label="Achieved shape")
    ax.scatter(
        actuator_xy[~active, 0],
        actuator_xy[~active, 1],
        s=16,
        color="#1f77b4",
        alpha=0.65,
        label="Unused actuators",
    )
    ax.scatter(
        actuator_xy[active, 0],
        actuator_xy[active, 1],
        s=20,
        color="#17becf",
        marker="*",
        label="Selected actuators",
    )

    if np.any(active):
        scale = 0.045
        ax.quiver(
            actuator_xy[active, 0],
            actuator_xy[active, 1],
            force_y[active] * scale,
            force_z[active] * scale,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="black",
            width=0.004,
            headwidth=4,
            label="Force vectors",
        )

    ax.set_title(f"MAE = {float(row['mae']):.3f} in", fontsize=16)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-170, 170)
    ax.set_ylim(-170, 170)
    ax.set_xlabel("Y [in]")
    ax.set_ylabel("Z [in]")
    ax.grid(False)
    if show_legend:
        ax.legend(loc="lower center", fontsize=8, frameon=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-dir", default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    extract_dir = Path(args.extract_dir)
    data = np.load(extract_dir / "panel_shapes_and_forces.npz")
    rows = load_panel_rows(extract_dir / "panel_force_configs.csv")
    target = data["target_shape"]
    final_shapes = data["final_shapes"]
    actuator_xy = actuator_positions(target, np.linspace(12.0, -192.0, 18))

    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4.4), constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]

    for i, (ax, row, achieved) in enumerate(zip(axes, rows, final_shapes)):
        draw_panel(ax, target, achieved, row, actuator_xy, show_legend=(i == 0))
        ax.text(
            0.5,
            -0.16,
            f"queries {float(row['cum_queries']):.0f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
        )

    output = Path(args.output) if args.output else extract_dir / "shape_gap_reduction_panels.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
