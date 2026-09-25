import numpy as np
import matplotlib.pyplot as plt

try:
    from .ACL_paper import run_bo_acl_simulation2
    from .experiment_env import compare_experiment_environments, sample_initial_indices
    from .quantum_safe_bo import run_quantum_safe_bo_simulation
    from .safe_set_bo import resolve_n_init
except ImportError:
    from ACL_paper import run_bo_acl_simulation2
    from experiment_env import compare_experiment_environments, sample_initial_indices
    from quantum_safe_bo import run_quantum_safe_bo_simulation
    from safe_set_bo import resolve_n_init


DEFAULT_SETTINGS = {
    "mode": "min",
    "xi": 0.25,
    "grid_size": 25,
    "x1_bounds": (-1.0, 1.0),
    "x2_bounds": (-1.0, 1.0),
    "init_percentage": None,
    "n_init": 1,
    "n_iter": 50,
    "obj_noise_std": 1e-6,
    "con_noise_std": 1e-6,
    "seed": 0,
    "allow_repeated_queries": True,
}

SAFE_BO_SETTINGS = {
    "beta_f": 1.0,
    "beta_c": 3.0,
    "lam0": 1.0,
}

QUANTUM_BO_SETTINGS = {
    "beta_f": 2.0,
    "beta_c": 3.0,
    "M_target": 256,
    "lam": 1.0,
    "lam0": 1.0,
    "t0": 10.0,
    "p": 2.0,
    "p_min": 0.6,
    "delta": 0.05,
    "use_quantum_query": True,
}

ACL_SETTINGS = {
    "beta_f": 2.0,
    "alpha_0": 0.7,
    "beta_w": 0.2,
    "eps": 1.0,
}


def _model_query_indices(result):
    return np.asarray(result["queried_idx"][result["n_init"] :], dtype=int)


def _model_query_safe_mask(result):
    model_idx = _model_query_indices(result)
    return result["zz"][model_idx] <= result["xi"]


def summarize_safe_study_result(result):
    safe_mask = _model_query_safe_mask(result)
    best_hist = np.asarray(result["best_safe_value_hist"], dtype=float)
    cumu_regret = np.asarray(result["queried_cumu_regret_hist"], dtype=float)

    best_safe_observed = np.nan
    if best_hist.size > 0 and not np.all(np.isnan(best_hist)):
        if result["mode"] == "max":
            best_safe_observed = float(np.nanmax(best_hist))
        else:
            best_safe_observed = float(np.nanmin(best_hist))

    final_cumulative_regret = np.nan
    if cumu_regret.size > 0:
        final_cumulative_regret = float(cumu_regret[-1])

    if "simple_regret" in result:
        simple_regret = (
            float(result["simple_regret"])
            if not np.isnan(result["simple_regret"])
            else np.nan
        )
    elif np.isnan(best_safe_observed):
        simple_regret = np.nan
    elif result["mode"] == "max":
        simple_regret = float(result["global_safe_opt"] - best_safe_observed)
    else:
        simple_regret = float(best_safe_observed - result["global_safe_opt"])

    safe_query_rate = np.nan
    violation_rate = np.nan
    if safe_mask.size > 0:
        safe_query_rate = float(np.mean(safe_mask))
        violation_rate = float(1.0 - safe_query_rate)

    return {
        "global_safe_opt": float(result["global_safe_opt"]),
        "simple_regret": simple_regret,
        "best_safe_observed": best_safe_observed,
        "final_cumulative_regret": final_cumulative_regret,
        "safe_query_rate": safe_query_rate,
        "violation_rate": violation_rate,
        "num_model_queries": int(safe_mask.size),
    }


def _global_safe_optimum_details(result):
    xx = np.asarray(result["xx"])
    yy = np.asarray(result["yy"])
    zz = np.asarray(result["zz"])
    xi = float(result["xi"])
    safe_mask = zz <= xi
    safe_idx = np.where(safe_mask)[0]
    if safe_idx.size == 0:
        return None

    safe_values = yy[safe_idx]
    if result["mode"] == "max":
        local_idx = int(np.argmax(safe_values))
    else:
        local_idx = int(np.argmin(safe_values))

    idx = int(safe_idx[local_idx])
    return {
        "idx": idx,
        "x": np.asarray(xx[idx], dtype=float),
        "f": float(yy[idx]),
        "h": float(zz[idx]),
        "safe": True,
    }


def _incumbent_details(result):
    xx = np.asarray(result["xx"])
    yy = np.asarray(result["yy"])
    zz = np.asarray(result["zz"])
    xi = float(result["xi"])
    queried_idx = np.asarray(result["queried_idx"], dtype=int)
    if queried_idx.size == 0:
        return None

    safe_queried_mask = zz[queried_idx] <= xi
    safe_queried_idx = queried_idx[safe_queried_mask]
    if safe_queried_idx.size == 0:
        return None

    safe_values = yy[safe_queried_idx]
    if result["mode"] == "max":
        local_idx = int(np.argmax(safe_values))
        value_gap = float(np.max(yy[zz <= xi]) - safe_values[local_idx])
    else:
        local_idx = int(np.argmin(safe_values))
        value_gap = float(safe_values[local_idx] - np.min(yy[zz <= xi]))

    idx = int(safe_queried_idx[local_idx])
    return {
        "idx": idx,
        "x": np.asarray(xx[idx], dtype=float),
        "f": float(yy[idx]),
        "h": float(zz[idx]),
        "safe": True,
        "value_gap": value_gap,
    }


def summarize_optimum_vs_incumbent(comparison):
    summary = {}
    for label in ("quantum_bo", "acl"):
        result = comparison[label]
        optimum = _global_safe_optimum_details(result)
        incumbent = _incumbent_details(result)
        if optimum is None or incumbent is None:
            summary[label] = {
                "optimal": optimum,
                "incumbent": incumbent,
                "point_delta": None,
                "euclidean_distance": np.nan,
                "objective_gap": np.nan,
            }
            continue

        point_delta = incumbent["x"] - optimum["x"]
        if result["mode"] == "max":
            objective_gap = float(optimum["f"] - incumbent["f"])
        else:
            objective_gap = float(incumbent["f"] - optimum["f"])

        summary[label] = {
            "optimal": optimum,
            "incumbent": incumbent,
            "point_delta": point_delta,
            "euclidean_distance": float(np.linalg.norm(point_delta)),
            "objective_gap": objective_gap,
        }
    return summary


def print_optimum_vs_incumbent(comparison):
    summary = summarize_optimum_vs_incumbent(comparison)
    print("\n=== Optimal Point vs Incumbent Point ===")
    for label in ("quantum_bo", "acl"):
        stats = summary[label]
        print(f"\n[{label}]")
        if stats["optimal"] is None:
            print("No true safe optimum exists in the environment.")
            continue
        if stats["incumbent"] is None:
            print("No safe incumbent was observed.")
            continue

        optimal = stats["optimal"]
        incumbent = stats["incumbent"]
        print(f"optimal_idx: {optimal['idx']}")
        print(f"optimal_x: {optimal['x']}")
        print(f"optimal_f: {optimal['f']}")
        print(f"incumbent_idx: {incumbent['idx']}")
        print(f"incumbent_x: {incumbent['x']}")
        print(f"incumbent_f: {incumbent['f']}")
        print(f"point_delta (incumbent - optimal): {stats['point_delta']}")
        print(f"euclidean_distance: {stats['euclidean_distance']}")
        print(f"objective_gap: {stats['objective_gap']}")


def run_comparison_experiment(
    base_settings=None,
    quantum_bo_settings=None,
    acl_settings=None,
):
    base = dict(DEFAULT_SETTINGS)
    if base_settings:
        base.update(base_settings)

    q_cfg = dict(QUANTUM_BO_SETTINGS)
    if quantum_bo_settings:
        q_cfg.update(quantum_bo_settings)

    acl_cfg = dict(ACL_SETTINGS)
    if acl_settings:
        acl_cfg.update(acl_settings)

    total_points = int(base["grid_size"]) ** 2
    n_init = resolve_n_init(
        total_points,
        n_init=base.get("n_init", 1),
        init_percentage=base.get("init_percentage"),
    )
    init_idx = sample_initial_indices(total_points, n_init=n_init, seed=base["seed"])

    common = dict(base)
    common["n_init"] = n_init
    common["init_idx"] = init_idx

    quantum_res = run_quantum_safe_bo_simulation(**common, **q_cfg)
    acl_res = run_bo_acl_simulation2(**common, **acl_cfg)

    env_comparison = compare_experiment_environments(quantum_res, acl_res)

    return {
        "settings": {
            "base": common,
            "quantum_bo": q_cfg,
            "acl": acl_cfg,
        },
        "environment_comparison": env_comparison,
        "quantum_bo": quantum_res,
        "acl": acl_res,
        "summary": {
            "quantum_bo": summarize_safe_study_result(quantum_res),
            "acl": summarize_safe_study_result(acl_res),
        },
    }


def print_comparison_summary(comparison):
    print("\n=== Shared Experiment Settings ===")
    for key, value in comparison["settings"]["base"].items():
        print(f"{key}: {value}")

    print("\n=== Environment Match ===")
    print(f"same_environment: {comparison['environment_comparison']['same']}")
    if not comparison["environment_comparison"]["same"]:
        print("differences:", comparison["environment_comparison"]["differences"])

    print("\n=== Performance Summary ===")
    for label in ("quantum_bo", "acl"):
        stats = comparison["summary"][label]
        print(f"\n[{label}]")
        for key, value in stats.items():
            print(f"{key}: {value}")


def plot_comparison(comparison):
    quantum_res = comparison["quantum_bo"]
    acl_res = comparison["acl"]

    quantum_idx = _model_query_indices(quantum_res)
    acl_idx = _model_query_indices(acl_res)

    quantum_rate_hist = np.cumsum(_model_query_safe_mask(quantum_res)) / np.arange(1, len(quantum_idx) + 1)
    acl_rate_hist = np.cumsum(_model_query_safe_mask(acl_res)) / np.arange(1, len(acl_idx) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        np.arange(1, len(quantum_res["queried_cumu_regret_hist"]) + 1),
        quantum_res["queried_cumu_regret_hist"],
        label="Q-Safe BO",
        color="tab:purple",
        linewidth=2,
    )
    axes[0].plot(
        np.arange(1, len(acl_res["queried_cumu_regret_hist"]) + 1),
        acl_res["queried_cumu_regret_hist"],
        label="BO-ACL",
        color="tab:blue",
        linewidth=2,
    )
    axes[0].set_title("Cumulative Regret")
    axes[0].set_xlabel("Model queries only")
    axes[0].set_ylabel("Cumulative regret")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        np.arange(1, len(quantum_rate_hist) + 1),
        quantum_rate_hist,
        label="Q-Safe BO",
        color="tab:purple",
        linewidth=2,
    )
    axes[1].plot(
        np.arange(1, len(acl_rate_hist) + 1),
        acl_rate_hist,
        label="BO-ACL",
        color="tab:blue",
        linewidth=2,
    )
    axes[1].set_title("Safe Query Rate")
    axes[1].set_xlabel("Model queries only")
    axes[1].set_ylabel("Safe queries / total queries")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def _plot_method_layout(ax, result, title):
    xx = np.asarray(result["xx"])
    zz = np.asarray(result["zz"])
    xi = result["xi"]
    grid_size = int(np.sqrt(len(xx)))

    x_axis = xx[:, 0].reshape(grid_size, grid_size)
    y_axis = xx[:, 1].reshape(grid_size, grid_size)
    z_axis = zz.reshape(grid_size, grid_size)

    init_idx = np.asarray(result["init_idx"], dtype=int)
    model_idx = _model_query_indices(result)
    init_points = xx[init_idx]
    model_points = xx[model_idx] if len(model_idx) > 0 else np.empty((0, 2))

    ax.contourf(x_axis, y_axis, z_axis <= xi, levels=2, alpha=0.25, cmap="Greens")
    ax.contour(x_axis, y_axis, z_axis, levels=[xi], colors="black", linewidths=2.0)
    ax.scatter(
        init_points[:, 0],
        init_points[:, 1],
        s=48,
        c="tab:blue",
        edgecolors="black",
        label="Initial points",
    )
    if len(model_points) > 0:
        ax.scatter(
            model_points[:, 0],
            model_points[:, 1],
            s=34,
            c="red",
            marker="x",
            linewidths=1.5,
            label="Query points",
        )
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="best")


def plot_environment_contours(comparison):
    result = comparison["quantum_bo"]
    xx = np.asarray(result["xx"])
    yy = np.asarray(result["yy"])
    zz = np.asarray(result["zz"])
    xi = result["xi"]
    grid_size = int(np.sqrt(len(xx)))

    x_axis = xx[:, 0].reshape(grid_size, grid_size)
    y_axis = xx[:, 1].reshape(grid_size, grid_size)
    y_true = yy.reshape(grid_size, grid_size)
    z_true = zz.reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    cs = axes[0].contourf(x_axis, y_axis, y_true, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=axes[0], shrink=0.85, label="Objective f(x)")
    axes[0].contour(x_axis, y_axis, z_true, levels=[xi], colors="white", linewidths=2.0)
    axes[0].set_title("Objective Contour with Constraint Boundary")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    hc = axes[1].contourf(x_axis, y_axis, z_true, levels=20, cmap="coolwarm")
    fig.colorbar(hc, ax=axes[1], shrink=0.85, label="Constraint h(x)")
    axes[1].contour(x_axis, y_axis, z_true, levels=[xi], colors="black", linewidths=2.0)
    axes[1].set_title("Constraint Contour and Boundary")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")

    plt.tight_layout()
    plt.show()


def plot_query_layout(comparison):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    _plot_method_layout(axes[0], comparison["quantum_bo"], "Q-Safe BO: Boundary, Initial, Queries")
    _plot_method_layout(axes[1], comparison["acl"], "BO-ACL: Boundary, Initial, Queries")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    comparison = run_comparison_experiment()
    print_comparison_summary(comparison)
    plot_comparison(comparison)
    plot_environment_contours(comparison)
    plot_query_layout(comparison)
