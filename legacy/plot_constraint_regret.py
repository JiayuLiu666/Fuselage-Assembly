"""
Cumulative Regret Comparison: All Constraint BO Methods
=========================================================
Compares 4 methods (Quantum vs Classic × Safe-set cUCB vs PoF)
for two noise levels: 0.1**2 and 0.2**2.

Logic strictly follows analyze_discrete.ipynb.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ── helpers ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_wo_constraint(filename, device, quan=False):
    data = torch.load(filename, map_location=device)
    if quan:
        actions        = data['actions'].to(torch.float32).to(device)
        response       = data['response'].to(torch.float32).to(device)
        true_response  = data['true_response'].to(torch.float32)
        num_of_queries = data['queries']
        eps            = data['uncertainty']
    else:
        actions        = data['actions'].to(torch.float32).to(device)
        response       = data['response'].to(torch.float32).to(device)
        true_response  = data['true_response'].to(torch.float32).to(device)
        eps            = data['uncertainty']
        num_of_queries = data['queries']
    return actions, true_response, response, num_of_queries, eps


def compute_cumulative_regret(true_response, num_of_queries, f_max):
    """Exactly matches analyze_discrete.ipynb logic."""
    nq = num_of_queries[1:]
    tr = torch.cat((true_response[:1], true_response[2:]), dim=0)
    values = tr.detach().cpu().numpy()
    track_queries = nq.detach().cpu().numpy().astype(np.int32)
    values_new = []
    for i in range(len(values)):
        values_new += list(np.repeat(values[i], track_queries[i]))
    values = np.array(values_new)
    values = np.squeeze(np.abs(f_max - values))
    return np.cumsum(values)


def plot_mean_and_CI_with_marker(time_steps, mean, lb, ub,
                                 color_mean=None, color_shading=None,
                                 marker=None, marker_size=8):
    plt.fill_between(time_steps, ub, lb, color=color_shading, alpha=0.2)
    line, = plt.plot(time_steps, mean, color_mean,
                     marker=marker, markersize=marker_size, markevery=1000)
    return line


# ── configuration ────────────────────────────────────────────────────
f_max = 0.048170337104871036
n_trials = 5
run_list = np.arange(n_trials)

# Method directories
methods_config = [
    # (dir, label_prefix, color, marker)
    ('Experiments_constraints/Quantum_Discrete_cUCB/exp_set_1/',     'Q-SafeSet', 'b', 'v'),
    ('Experiments_constraints/Quantum_Discrete_cUCB_POF/exp_set_1/', 'Q-PoF',     'r', 's'),
    ('Experiments_constraints/Classic_Discrete_cUCB/exp_set_1/',     'C-SafeSet', 'g', 'D'),
    ('Experiments_constraints/Classic_Discrete_cUCB_POF/exp_set_1/', 'C-PoF',     'c', 'o'),
]

noise_levels = [0.1**2, 0.2**2]
noise_labels = ['σ=0.1', 'σ=0.2']

BIG = int(15000 + 100)

# ── collect all series ───────────────────────────────────────────────
all_series = []  # list of (regrets_list, min_len, label, color, marker)

for method_dir, label_prefix, color, marker in methods_config:
    for noise_idx, obs_noise in enumerate(noise_levels):
        regrets = []
        min_len = BIG
        label = f'{label_prefix} ({noise_labels[noise_idx]})'

        for itr in run_list:
            fname = method_dir + str(obs_noise) + 'quan_training_data_' + str(itr) + '_.pth'
            if os.path.exists(fname):
                _, tr, _, nq, _ = load_data_wo_constraint(fname, device, quan=True)
                cr = compute_cumulative_regret(tr, nq, f_max)
                min_len = min(min_len, len(cr))
                regrets.append(cr)
            else:
                pass  # silently skip missing files

        all_series.append((regrets, min_len, label, color, marker))

# ── plot ─────────────────────────────────────────────────────────────
lw = 2.0
plt.rc('font', size=16)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600
plt.figure(figsize=(9, 7))

marker_size = 8
lines = []

for regs, mlen, label, color, marker in all_series:
    if not regs:
        print(f"  [WARN] No data for '{label}' — skipping")
        continue
    arr = np.array([r[:mlen] for r in regs])
    mean   = np.mean(arr, axis=0)
    stderr = np.std(arr, axis=0) / np.sqrt(len(run_list))
    ub = mean + stderr
    lb = mean - stderr
    inds = np.arange(1, mlen + 1)
    line = plot_mean_and_CI_with_marker(
        inds, mean, lb, ub,
        color_mean=color, color_shading=color,
        marker=marker, marker_size=marker_size,
    )
    lines.append((line, label))

if lines:
    plt.legend([l for l, _ in lines], [lbl for _, lbl in lines], prop={'size': 12})

axes = plt.gca()
axes.set_xlim([0, 3000])
axes.set_ylim([-0.01, 1000])
axes.set_ylabel("Cumulative Regret")
axes.set_xlabel("Iterations")

plt.tight_layout()
output_path = 'constraint_regret_comparison.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
plt.show()
