"""
Violation Rate Report — strictly by surrogate model.

Violation Rate = 1 - (queried safe points / total queried points)

A queried point is classified as **safe** if and only if (1 - FI) > 0,
where FI is the Tsai-Wu failure index evaluated **strictly** by the
surrogate model (surrogate_tsaiwu.joblib).  This does NOT use any
GP-based safety predictions, cached flags, or active_records —
every queried action is re-evaluated through the surrogate at analysis
time to ensure the result is unambiguous.
"""

import glob
import os
from pathlib import Path

import numpy as np
import torch
from joblib import load as joblib_load


# ── helpers ──────────────────────────────────────────────────────────
def compute_violation_rate_strict(actions_np, tsai_wu_model):
    """
    Compute violation rate strictly using the surrogate model.

    A point is 'safe' iff (1 - FI) > 0, where FI is predicted
    by tsai_wu_model.
    Violation rate = 1 - (safe_count / total_count).
    """
    if actions_np.size == 0:
        return 0, 0, float("nan")
    actions_2d = actions_np.reshape(-1, 18)
    fi = tsai_wu_model.predict(actions_2d, return_std=False).reshape(-1)
    c_val = 1.0 - fi
    safe_count = int(np.sum(c_val > 0.0))
    total_count = int(c_val.size)
    violation_rate = (
        1.0 - (safe_count / total_count)
        if total_count > 0
        else float("nan")
    )
    return safe_count, total_count, violation_rate


def classify_method(file_path: str) -> str:
    path_str = str(file_path)
    name = os.path.basename(path_str)
    if "Quantum_Discrete_cUCB_Real" in path_str and "quan_training_data" in name:
        return "Q-Safe-Set-UCB-Real"
    if "Quantum_Discrete_cUCB" in path_str and "quan_training_data" in name:
        return "Q-Safe-Set-UCB"
    if "Classic_Discrete_cUCB" in path_str and "training_data" in name:
        return "C-Safe-Set-UCB"
    if "Classic_ACL_Discrete" in path_str or "acl_training_data" in name:
        return "C-ACL"
    return "Unknown"


# ── main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load the Tsai-Wu surrogate model (strict safety oracle)
    tsai_wu_surrogate = joblib_load("surrogate_tsaiwu.joblib")

    data_dir = Path("Experiments_constraints")
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory {data_dir} not found!")

    files = glob.glob(
        str(data_dir / "**" / "*training_data_*_.pth"), recursive=True
    )
    if not files:
        print("No training data files found.")
        exit(0)

    violation_stats = {}

    for f in sorted(files):
        method = classify_method(f)
        if method == "Unknown":
            continue

        try:
            data = torch.load(
                f, map_location=torch.device("cpu"), weights_only=False
            )
        except Exception as exc:
            print(f"Failed to load {f}: {exc}")
            continue

        actions = data.get("actions")
        if actions is None:
            continue
        actions_np = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        )
        actions_np = actions_np.reshape(actions_np.shape[0], -1)
        if actions_np.shape[0] == 0:
            continue

        # Strictly evaluate safety via surrogate model
        safe_count, total_count, viol_rate = compute_violation_rate_strict(
            actions_np, tsai_wu_surrogate
        )

        violation_stats.setdefault(method, []).append(
            {
                "safe_count": safe_count,
                "total_count": total_count,
                "violation_rate": viol_rate,
                "filename": f,
            }
        )

    # ── Print Report ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("VIOLATION RATE REPORT  (Strictly by Surrogate Model)")
    print(
        "Violation Rate = 1 - (queried safe points / total queried points)"
    )
    print(
        "Safe condition: (1 - FI) > 0   [FI from surrogate_tsaiwu.joblib]"
    )
    print("=" * 70)

    for method, trials in sorted(violation_stats.items()):
        print(f"\n{'─' * 60}")
        print(f"Method: {method}  |  Trials: {len(trials)}")
        print(f"{'─' * 60}")

        viol_rates = np.asarray(
            [t["violation_rate"] for t in trials], dtype=float
        )
        safe_counts = np.asarray(
            [t["safe_count"] for t in trials], dtype=int
        )
        total_counts = np.asarray(
            [t["total_count"] for t in trials], dtype=int
        )

        valid = np.isfinite(viol_rates)
        avg_viol = (
            float(np.nanmean(viol_rates)) if np.any(valid) else float("nan")
        )
        std_viol = (
            float(np.nanstd(viol_rates)) if np.any(valid) else float("nan")
        )

        total_safe_all = int(np.sum(safe_counts))
        total_queried_all = int(np.sum(total_counts))
        overall_viol = (
            1.0 - (total_safe_all / total_queried_all)
            if total_queried_all > 0
            else float("nan")
        )

        print(
            f"  Average Violation Rate:   {avg_viol:.6f} ± {std_viol:.6f}"
        )
        print(
            f"  Overall Violation Rate:   {overall_viol:.6f} "
            f"({total_queried_all - total_safe_all} unsafe / "
            f"{total_queried_all} total)"
        )
        print()

        for i, t in enumerate(trials):
            unsafe_count = t["total_count"] - t["safe_count"]
            print(
                f"  Trial {i}: violation_rate={t['violation_rate']:.6f}  "
                f"(safe={t['safe_count']}, "
                f"unsafe={unsafe_count}, "
                f"total={t['total_count']})  "
                f"file={os.path.basename(t['filename'])}"
            )
        print()
