import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data_wo_constraint(filename, device):
    data = torch.load(filename)
    actions = data['actions'].to(torch.float32).to(device)
    true_response = data['true_response'].to(torch.float32).to(device)
    response = data['response'].to(torch.float32).to(device)
    num_of_queries = data['queries'].to(device)
    eps = data['uncertainty']
    return actions, true_response, response, num_of_queries, eps

q_file  = 'Experiments_constraints/Quantum_Discrete_cUCB/exp_set_1/'
c_file  = 'Experiments_constraints/Classic_Discrete_cUCB/exp_set_1/'
# q_simulator_file = 'Experiments_constraints/Quantum_Discrete_cUCB/exp_set_1/'

def running_min_from_files(method, noise, itr):
    if method == 'Q':
        path = q_file + str(noise) + f'quan_training_data_{itr}_.pth'
        actions, true_resp, resp, n_queries, eps = load_data_wo_constraint(path, device=device)
    elif method == "RQ":
        path = q_real_file + str(noise) + f'quan_training_data_{itr}_.pth'
        actions, true_resp, resp, n_queries, eps = load_data_wo_constraint(path, device=device)
    else:
        path = c_file + str(noise) + f'training_data_{itr}_.pth'
        actions, true_resp, resp, n_queries, eps = load_data_wo_constraint(path, device=device)

    # n_queries = n_queries[1:]
    # eps = eps[1:]
    # resp = torch.cat((resp[:1], resp[2:]), dim=0)
    # true_resp = torch.cat((true_resp[:1], true_resp[2:]), dim=0)

    vals = true_resp.detach().cpu().numpy().reshape(-1)
    track = n_queries.detach().cpu().numpy().astype(np.int32).reshape(-1)
    expanded = np.repeat(vals, track)
    runmin   = np.minimum.accumulate(expanded)
    return runmin

# ---- collect data ----
noise_levels = [0.1**2, 0.2**2]
methods = ['Q', 'C']
num_runs = 5

curves = {}
for meth in methods:
    for noise in noise_levels:
        key = (meth, noise)
        curves[key] = []
        for itr in range(num_runs):
            curves[key].append(running_min_from_files(meth, noise, itr))

# ---- compute stats ----
stats = {}
for key, runs in curves.items():
    T = min(len(r) for r in runs)
    stack = np.stack([r[:T] for r in runs], axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    iters = np.arange(T)
    stats[key] = (iters, mean, std)

# ---- plot ----
style = {
    ('Q', 0.1**2): dict(label='Q-WGP-UCB (Simulator) (σ=0.1)', color='tab:blue'),
    ('RQ', 0.1**2): dict(label='Q-WGP-UCB (Real) (σ=0.1)', color='tab:green'),
    ('C', 0.1**2): dict(label='C-WGP-UCB (σ=0.1)', color='tab:red'),
    ('Q', 0.2**2): dict(label='Q-WGP-UCB (Simulator) (σ=0.2)', color='tab:cyan'),
    ('RQ', 0.2**2): dict(label='Q-WGP-UCB (Real) (σ=0.2)', color='tab:magenta'),
    ('C', 0.2**2): dict(label='C-WGP-UCB (σ=0.2)', color='tab:yellow'),
}

plt.rc('font', size=18)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600

plt.figure(figsize=(9, 7))
ax = plt.gca()

colors = ['b','g', 'r', 'c', 'm', 'y']
for (key, (iters, mean, std)), color in zip(stats.items(), colors):
    label = style[key]["label"]
    ax.plot(iters, mean, label=label, color=color, linewidth=2)
    ax.fill_between(iters, mean - std, mean + std, alpha=0.2, color=color)

ax.set_xlabel("Iterations")
ax.set_ylabel("Running Minimum MAE [in]")
ax.set_title("Running Minimum MAE [in] vs. Iterations")
ax.grid(alpha=0.3)

# vertical reference line
xmark = 2350
x0, x1 = ax.get_xlim()
if xmark > x1:
    ax.set_xlim(right=xmark + 50)
ax.axvline(x=xmark, linestyle='--', linewidth=2, color='k', alpha=0.8)

# ---- flip y-axis from 0.2 (top) to 0 (bottom) ----
ax.set_ylim(0.0, 0.2)   # reversed y-axis direction
ax.set_xlim(0, 3000)
# optional horizontal reference at 0.2
ax.axhline(0.2, linestyle='--', linewidth=1.5, color='gray', alpha=0.6)

ax.legend()
plt.tight_layout()
plt.show()