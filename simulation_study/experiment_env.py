import numpy as np
import torch

from itertools import product


def compute_global_safe_optimum(y_values, safe_mask, mode):
    true_safe_values = y_values[safe_mask]
    if true_safe_values.size == 0:
        return np.nan
    if mode == "max":
        return float(np.max(true_safe_values))
    if mode == "min":
        return float(np.min(true_safe_values))
    raise ValueError("mode must be 'max' or 'min'")


def simulation2_objective_np(x):
    """Paper Simulation 2 objective: g(x) = x1^2 - sin(4*x2^2)."""
    return np.square(x[:, 0]) - np.sin(4.0 * np.square(x[:, 1]))


def simulation2_constraint_np(x):
    """Paper Simulation 2 constraint metric: h(x) = x2 - x1^2."""
    return x[:, 1] - np.square(x[:, 0])

# def simulation2_constraint_np(x):
#     return np.minimum(x[:, 0], x[:, 1])


def simulation2_objective_torch(x):
    return torch.square(x[:, 0]) - torch.sin(4.0 * torch.square(x[:, 1]))

# def simulation2_constraint_torch(x):
#     return torch.minimum(x[:, 0], x[:, 1])


def simulation2_constraint_torch(x):
    return x[:, 1] - torch.square(x[:, 0])


def build_paper_sim2_environment(
    grid_size,
    xi=0.0,
    dtype=torch.float64,
    constraint_representation="margin",
    x1_bounds=(-1.0, 1.0),
    x2_bounds=(-1.0, 1.0),
):
    """
    Shared experiment environment used by both Safe BO and BO-ACL comparisons.

    constraint_representation:
    - "margin": C_true = h(x) - xi, safe iff C_true >= 0
    - "binary": C_true in {0, 1}, feasible iff h(x) >= xi
    """
    x1_axis = np.linspace(float(x1_bounds[0]), float(x1_bounds[1]), grid_size)
    x2_axis = np.linspace(float(x2_bounds[0]), float(x2_bounds[1]), grid_size)
    xx = np.array(list(product(x1_axis, x2_axis)))
    yy = simulation2_objective_np(xx)
    zz = simulation2_constraint_np(xx)

    X_grid = torch.tensor(xx, dtype=dtype)
    Y_true = torch.tensor(yy, dtype=dtype).view(-1, 1)
    Z_true = torch.tensor(zz, dtype=dtype).view(-1, 1)

    if constraint_representation == "margin":
        C_true = torch.tensor(zz - xi, dtype=dtype).view(-1, 1)
    elif constraint_representation == "binary":
        C_true = (Z_true >= xi).to(dtype)
    else:
        raise ValueError("constraint_representation must be 'margin' or 'binary'")

    safe_true_mask = zz >= xi
    environment = {
        "name": "paper_sim2",
        "grid_size": int(grid_size),
        "grid_bounds": {
            "x1": (float(x1_bounds[0]), float(x1_bounds[1])),
            "x2": (float(x2_bounds[0]), float(x2_bounds[1])),
        },
        "x1_bounds": (float(x1_bounds[0]), float(x1_bounds[1])),
        "x2_bounds": (float(x2_bounds[0]), float(x2_bounds[1])),
        "xi": float(xi),
        "objective": "x1^2 - sin(4*x2^2)",
        "constraint": "x2 - x1^2",
        "safe_condition": "h(x) >= xi",
        "constraint_representation": constraint_representation,
    }
    return {
        "environment": environment,
        "xx": xx,
        "yy": yy,
        "zz": zz,
        "X_grid": X_grid,
        "Y_true": Y_true,
        "Z_true": Z_true,
        "C_true": C_true,
        "safe_true_mask": safe_true_mask,
    }


def compare_experiment_environments(result_a, result_b):
    env_a = dict(result_a.get("environment", {}))
    env_b = dict(result_b.get("environment", {}))

    keys = sorted(set(env_a) | set(env_b))
    differences = {}
    for key in keys:
        if env_a.get(key) != env_b.get(key):
            differences[key] = {
                "left": env_a.get(key),
                "right": env_b.get(key),
            }

    return {
        "same": len(differences) == 0,
        "left": env_a,
        "right": env_b,
        "differences": differences,
    }


def sample_initial_indices(total_points, n_init, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(total_points, size=int(n_init), replace=False).tolist()


def sample_stratified_initial_indices(total_points, safe_mask, n_init, min_safe=3, seed=0):
    """
    Draw n_init indices guaranteeing at least min_safe come from the safe region.
    The remainder are drawn from the full grid (excluding already-chosen points).
    """
    rng = np.random.default_rng(seed)
    safe_idx   = np.where(safe_mask)[0]
    unsafe_idx = np.where(~safe_mask)[0]

    min_safe = min(min_safe, len(safe_idx), n_init)
    chosen_safe = rng.choice(safe_idx, size=min_safe, replace=False).tolist()

    remaining_pool = np.array([i for i in range(total_points) if i not in set(chosen_safe)])
    n_remaining = n_init - min_safe
    chosen_rest = rng.choice(remaining_pool, size=n_remaining, replace=False).tolist()

    combined = chosen_safe + chosen_rest
    rng.shuffle(combined)
    return combined
