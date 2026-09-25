"""
Run the cumulative regret comparison:
  Classic ACL  vs  Quantum cUCB  vs  Classic WGP-UCB
at noise levels sigma^2 = 0.1^2 and 0.2^2.
All curves in one figure, x-axis limited to 5000.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
from os import path
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Data directories ----
classic_bo_dir   = "Experiments/Classic_Discrete/exp_set_1/"
quantum_cbo_dir  = "Experiments_constraints/Quantum_Discrete_cUCB/exp_set_1/"
classic_acl_dir  = "Experiments_constraints/Classic_ACL_Discrete/exp_set_0/"

# ---- Optimal value ----
f_max = 0.048170337104871036

# ---- Noise levels ----
noise_levels = [0.1**2, 0.2**2]

# ---- Trials ----
n_trials = 5
run_list = np.arange(n_trials)

MAX_BUDGET = int(30000 + 100)
X_LIM = 5000

print(f"Device : {device}")
print(f"f_max  : {f_max}")
print(f"Noise  : {noise_levels}")


# ═══════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════

def load_data(filename, device):
    data = torch.load(filename, map_location=device)
    actions        = data["actions"].to(torch.float32).to(device)
    response       = data["response"].to(torch.float32).to(device)
    true_response  = data["true_response"].to(torch.float32).to(device)
    num_of_queries = data["queries"]
    eps            = data["uncertainty"]
    return actions, true_response, response, num_of_queries, eps


def compute_cumulative_regret(true_response, num_of_queries, f_max,
                              skip_second_element=False):
    if skip_second_element:
        queries   = num_of_queries[1:]
        true_resp = torch.cat((true_response[:1], true_response[2:]), dim=0)
    else:
        queries   = num_of_queries
        true_resp = true_response

    values        = true_resp.detach().cpu().numpy()
    track_queries = queries.detach().cpu().numpy().astype(np.int32)

    values_expanded = []
    for i in range(len(values)):
        values_expanded += list(np.repeat(values[i], track_queries[i]))

    values_expanded   = np.array(values_expanded)
    instantaneous     = np.squeeze(np.abs(f_max - values_expanded))
    cumulative_regret = np.cumsum(instantaneous)
    return cumulative_regret


def plot_mean_and_CI(ax, time_steps, mean, lb, ub,
                     color=None, marker=None, marker_size=8,
                     markevery=500, label=None, linestyle="-"):
    ax.fill_between(time_steps, ub, lb, color=color, alpha=0.15)
    line, = ax.plot(time_steps, mean, color=color,
                    marker=marker, markersize=marker_size,
                    markevery=markevery, label=label,
                    linestyle=linestyle, linewidth=2)
    return line


# ═══════════════════════════════════════════════════════
# Method configurations
# ═══════════════════════════════════════════════════════

methods = {
    "C-WGP-UCB": {
        "dir": classic_bo_dir,
        "pattern": "{noise}training_data_{trial}_.pth",
        "skip": True,
        "color": "r",
        "marker": "s",
    },
    "Q-cUCB": {
        "dir": quantum_cbo_dir,
        "pattern": "{noise}quan_training_data_{trial}_.pth",
        "skip": True,
        "color": "b",
        "marker": "v",
    },
    "C-ACL": {
        "dir": classic_acl_dir,
        "pattern": "{noise}acl_training_data_{trial}_.pth",
        "skip": False,
        "color": "g",
        "marker": "D",
    },
}

# ═══════════════════════════════════════════════════════
# Collect results
# ═══════════════════════════════════════════════════════

results = {}

for noise in noise_levels:
    noise_str = str(noise)
    results[noise] = {}

    for mname, cfg in methods.items():
        all_regrets = []
        min_len = MAX_BUDGET

        for trial in run_list:
            fpath = os.path.join(cfg["dir"],
                                 cfg["pattern"].format(noise=noise_str,
                                                       trial=trial))
            if not os.path.exists(fpath):
                continue

            try:
                _, true_resp, _, nq, _ = load_data(fpath, device)
                cr = compute_cumulative_regret(
                    true_resp, nq, f_max,
                    skip_second_element=cfg["skip"])
                min_len = min(min_len, len(cr))
                all_regrets.append(cr)
            except Exception as exc:
                print(f"[WARN] {fpath}: {exc}")

        if all_regrets:
            all_regrets = [a[:min_len] for a in all_regrets]
            arr    = np.array(all_regrets)
            mean   = arr.mean(axis=0)
            stderr = arr.std(axis=0) / np.sqrt(len(all_regrets))
            results[noise][mname] = dict(
                mean=mean, ub=mean + stderr, lb=mean - stderr,
                min_len=min_len, n_trials=len(all_regrets))
            sigma = np.sqrt(noise)
            print(f"  {mname:16s}  sigma={sigma:.1f}  "
                  f"trials={len(all_regrets)}  steps={min_len}")
        else:
            sigma = np.sqrt(noise)
            print(f"  {mname:16s}  sigma={sigma:.1f}  ** NO DATA **")


# ═══════════════════════════════════════════════════════
# Single combined plot  –  unique colour per curve
# ═══════════════════════════════════════════════════════

# (method, noise) → (colour, marker, linestyle)
curve_styles = {
    ("C-WGP-UCB", 0.1**2): {"color": "#d62728", "marker": "s", "ls": "-"},   # red solid
    ("C-WGP-UCB", 0.2**2): {"color": "#ff7f0e", "marker": "s", "ls": "--"},  # orange dashed
    ("Q-cUCB",    0.1**2): {"color": "#1f77b4", "marker": "v", "ls": "-"},   # blue solid
    ("Q-cUCB",    0.2**2): {"color": "#9467bd", "marker": "v", "ls": "--"},  # purple dashed
    ("C-ACL",     0.1**2): {"color": "#2ca02c", "marker": "D", "ls": "-"},   # green solid
    ("C-ACL",     0.2**2): {"color": "#17becf", "marker": "D", "ls": "--"},  # cyan dashed
}

fig, ax = plt.subplots(figsize=(14, 9))

for noise in noise_levels:
    sigma = np.sqrt(noise)

    for mname in methods:
        if mname not in results[noise]:
            continue
        d   = results[noise][mname]
        end = min(d["min_len"], X_LIM)
        inds = np.arange(1, end + 1)
        sty  = curve_styles[(mname, noise)]
        plot_mean_and_CI(
            ax, inds,
            d["mean"][:end], d["lb"][:end], d["ub"][:end],
            color=sty["color"], marker=sty["marker"],
            linestyle=sty["ls"],
            label=f"{mname} ($\\sigma={sigma:.1f}$)")

ax.legend(prop={"size": 14}, loc="upper left")
ax.set_xlim([0, X_LIM])
ax.set_ylim(bottom=-0.01)
ax.set_xlabel("Iterations", fontsize=16)
ax.set_ylabel("Cumulative Regret", fontsize=16)
ax.set_title("Cumulative Regret Comparison", fontsize=18)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=13)

plt.tight_layout()
out_file = "cumulative_regret_comparison.png"
plt.savefig(out_file, dpi=150, bbox_inches="tight")
print(f"\nSaved  {out_file}")
