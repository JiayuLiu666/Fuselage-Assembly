import torch
import numpy as np
import matplotlib.pyplot as plt
from os import path
import os 
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── helper: load one experiment file ─────────────────────────────────────────
def load_data_wo_constraint(filename, device):
    data = torch.load(filename, map_location=device)
    actions          = data['actions'].to(torch.float32).to(device)
    true_response    = data['true_response'].to(torch.float32).to(device)
    response         = data['response'].to(torch.float32).to(device)
    num_of_queries   = data['queries'].to(device)
    eps              = data['uncertainty']
    error_init       = data['error_init']
    return actions, true_response, response, num_of_queries, eps, error_init


def plot_mean_and_CI_with_marker(time_steps, mean, lb, ub,
                                  color_mean=None, color_shading=None,
                                  marker=None, marker_size=12):
    fill = plt.fill_between(time_steps, ub, lb,
                            color=color_shading, alpha=.2)
    fill = None
    line = plt.plot(time_steps, mean, color_mean,
                    marker=marker, markersize=marker_size,
                    markevery=3000)
    line = line[0]
    return fill, line


# ─── True constraint oracle ────────────────────────────────────────────────────
from joblib import load as joblib_load
_tsai_wu_model = None

def get_tsai_wu():
    global _tsai_wu_model
    if _tsai_wu_model is None:
        _tsai_wu_model = joblib_load('surrogate_tsaiwu.joblib')
    return _tsai_wu_model

def is_feasible(action_np):
    """Return True if the action satisfies the Tsai-Wu constraint (FI < 1, i.e. c_val > 0)."""
    tsai_wu = get_tsai_wu()
    action_np = action_np.reshape(1, -1).astype(float)
    FI = tsai_wu.predict(action_np, return_std=False)
    c_val = 1.0 - float(FI)
    return c_val > 0.0


# ─── experiment parameters ────────────────────────────────────────────────────
lam    = 1.0
eta    = 1.0
Big_B  = 2.0
f_max  = 0

q_file_base = ('Experiments_constraint_continuous/'
               'Quantum_EXP_10_' + str(eta) + '_' + str(lam) +
               '_' + str(Big_B) + '_multi_gpu_noise_')
c_file_base = ('Experiments_constraint_continuous/'
               'Classic_EXP_10_' + str(eta) + '_' + str(lam) +
               '_' + str(1.0) + '_noise_')
c_acl_file_base = ('Experiments_constraint_continuous/'
               'Classic_Continuous_ACL_' + str(eta) + '_' + str(lam) +
               '_' + str(1.0) + '_noise_')

# ─── output containers ────────────────────────────────────────────────────────
all_regrets_1 = []   # quantum  noise=0.01 cumulative regret
all_regrets_2 = []   # classic  noise=0.01 cumulative regret
all_regrets_3 = []   # quantum  noise=0.04 cumulative regret
all_regrets_4 = []   # classic  noise=0.04 cumulative regret
all_regrets_5 = []   # classic ACL  noise=0.01 cumulative regret

min_len_1 = int(30000 + 100)
min_len_2 = int(30000 + 100)
min_len_3 = int(30000 + 100)
min_len_4 = int(30000 + 100)
min_len_5 = int(30000 + 100)

min_01_q = []   # best f(x) per seed, quantum  noise=0.01
min_01_c = []   # best f(x) per seed, classic  noise=0.01
min_02_q = []   # best f(x) per seed, quantum  noise=0.04
min_02_c = []   # best f(x) per seed, classic  noise=0.04
min_01_acl = [] # best f(x) per seed, classic ACL noise=0.01

# safe-query-rate accumulators
safe_rate_01_q = []   # safe_queried / total_queried, quantum  noise=0.01, per seed
safe_rate_01_c = []   # same for classic
safe_rate_02_q = []
safe_rate_02_c = []
safe_rate_01_acl = []

run_list = np.arange(5)

for itr in run_list:

    # ── Quantum, noise=0.01 ──────────────────────────────────────────────────
    noise_tag = '0.01'
    q_file    = q_file_base + noise_tag + '/exp_set_2/'
    q_actions, q_true_response, q_response, num_of_queries, eps, error_init = \
        load_data_wo_constraint(q_file + noise_tag + 'quan_training_data_' + str(itr) + '_.pth',
                                device=device)

    values = q_true_response.detach().cpu().numpy()
    # print('Quantum 0.1 f(x) {0}, Quantum obs: {1}, Uncertainty: {2}, Used Queries: {3}'.format(
    #     min(values).item(),
    #     (-q_response[np.argmin(values)] * error_init.item()).item(),
    #     eps[np.argmin(values)],
    #     num_of_queries[np.argmin(values)].item()))
    min_01_q.append(min(values).item())
    # print('Quantum 0.1 minimum Obs: {0}, Actual f(x): {1}, uncertainty: {2}, Used Queries: {3}'.format(
    #     torch.min(-q_response * error_init.item()),
    #     q_true_response[torch.argmin(-q_response * error_init.item())].item(),
    #     eps[torch.argmin(-q_response * error_init.item())],
    #     num_of_queries[np.argmin(values)].item()))

    # safe-query-rate (quantum, noise=0.01)
    actions_np = q_actions.detach().cpu().numpy()    # shape [N, 18]
    n_safe_q01 = sum(1 for a in actions_np if is_feasible(a))
    safe_rate_01_q.append(n_safe_q01 / len(actions_np))
    print(f'Quantum 0.1 safe_queried_rate: {n_safe_q01}/{len(actions_np)} = {n_safe_q01/len(actions_np):.4f}')

    track_queries = num_of_queries.detach().cpu().numpy().astype(np.int32)
    values_new = []
    for i in range(len(values)):
        values_new += list(np.repeat(values[i], track_queries[i]))
    values      = np.array(values_new)
    values      = np.squeeze(np.abs(f_max - values))
    values_acc  = np.cumsum(values)
    min_len_1   = np.min([min_len_1, len(values_acc)])
    all_regrets_1.append(values_acc)

    # ── Classic, noise=0.01 ──────────────────────────────────────────────────
    noise_tag_classic = str(0.1**2)
    c_file = c_file_base + noise_tag + '/exp_set_2/'
    c_actions, c_true_response, c_response, c_num_of_queries, c_eps, error_init = \
        load_data_wo_constraint(c_file + noise_tag_classic + 'training_data_' + str(itr) + '_.pth',
                                device=device)

    c_values = c_true_response.detach().cpu().numpy()
    # print('Classic 0.1 f(x): {0}, Classic Obs: {1}, Uncertainty: {2}, Used Queries: {3}'.format(
    #     min(c_values).item(),
    #     (-c_response[np.argmin(c_values)] * error_init.item()).item(),
    #     c_eps[np.argmin(c_values)],
    #     c_num_of_queries[np.argmin(c_values)].item()))
    min_01_c.append(min(c_values).item())
    # print('Classic 0.1 minimum Obs: {0}, Actual f(x): {1}, Uncertainty: {2}, Used Queries: {3}'.format(
    #     torch.min(-c_response * error_init.item()),
    #     c_true_response[torch.argmin(-c_response * error_init.item())].item(),
    #     c_eps[torch.argmin(-c_response * error_init.item())],
    #     c_num_of_queries[np.argmin(c_values)].item()))

    # safe-query-rate (classic, noise=0.01)
    c_actions_np = c_actions.detach().cpu().numpy()
    n_safe_c01   = sum(1 for a in c_actions_np if is_feasible(a))
    safe_rate_01_c.append(n_safe_c01 / len(c_actions_np))
    print(f'Classic 0.1 safe_queried_rate: {n_safe_c01}/{len(c_actions_np)} = {n_safe_c01/len(c_actions_np):.4f}')

    c_track_queries = c_num_of_queries.detach().cpu().numpy().astype(np.int32)
    c_values_new = []
    for i in range(len(c_values)):
        c_values_new += list(np.repeat(c_values[i], c_track_queries[i]))
    c_values     = np.array(c_values_new)
    c_values     = np.squeeze(np.abs(f_max - c_values))
    c_values_acc = np.cumsum(c_values)
    min_len_2    = np.min([min_len_2, len(c_values_acc)])
    all_regrets_2.append(c_values_acc)

    # print(f'  total classic queries (weighted): {sum(c_track_queries)}')

    # ── Quantum, noise=0.04 ──────────────────────────────────────────────────
    noise_tag = '0.04'
    q_file    = q_file_base + noise_tag + '/exp_set_2/'
    q_actions, q_true_response, q_response, num_of_queries, eps, error_init = \
        load_data_wo_constraint(q_file + noise_tag + 'quan_training_data_' + str(itr) + '_.pth',
                                device=device)

    values = q_true_response.detach().cpu().numpy()
    min_02_q.append(min(values).item())

    # safe-query-rate (quantum, noise=0.04)
    actions_np = q_actions.detach().cpu().numpy()
    n_safe_q04 = sum(1 for a in actions_np if is_feasible(a))
    safe_rate_02_q.append(n_safe_q04 / len(actions_np))
    print(f'Quantum 0.2 safe_queried_rate: {n_safe_q04}/{len(actions_np)} = {n_safe_q04/len(actions_np):.4f}')

    track_queries = num_of_queries.detach().cpu().numpy().astype(np.int32)
    values_new = []
    for i in range(len(values)):
        values_new += list(np.repeat(values[i], track_queries[i]))
    values      = np.array(values_new)
    values      = np.squeeze(np.abs(f_max - values))
    values_acc  = np.cumsum(values)
    min_len_3   = np.min([min_len_3, len(values_acc)])
    all_regrets_3.append(values_acc)

    # ── Classic, noise=0.04 ──────────────────────────────────────────────────
    noise_tag_2 = str(0.2**2)
    c_file = c_file_base + noise_tag + '/exp_set_2/'
    c_actions, c_true_response, c_response, c_num_of_queries, c_eps, error_init = \
        load_data_wo_constraint(c_file + noise_tag_2 + 'training_data_' + str(itr) + '_.pth',
                                device=device)

    c_values = c_true_response.detach().cpu().numpy()
    min_02_c.append(min(c_values).item())

    # safe-query-rate (classic, noise=0.04)
    c_actions_np = c_actions.detach().cpu().numpy()
    n_safe_c04   = sum(1 for a in c_actions_np if is_feasible(a))
    safe_rate_02_c.append(n_safe_c04 / len(c_actions_np))
    print(f'Classic 0.2 safe_queried_rate: {n_safe_c04}/{len(c_actions_np)} = {n_safe_c04/len(c_actions_np):.4f}')

    c_track_queries = c_num_of_queries.detach().cpu().numpy().astype(np.int32)
    c_values_new = []
    for i in range(len(c_values)):
        c_values_new += list(np.repeat(c_values[i], c_track_queries[i]))
    c_values     = np.array(c_values_new)
    c_values     = np.squeeze(np.abs(f_max - c_values))
    c_values_acc = np.cumsum(c_values)
    min_len_4    = np.min([min_len_4, len(c_values_acc)])
    all_regrets_4.append(c_values_acc)

    # ── Classic ACL, noise=0.01 ─────────────────────────────────────────────────
    noise_tag_acl = str(0.1**2)
    c_acl_file = c_acl_file_base + noise_tag_acl + '/exp_set_2/'
    c_acl_actions, c_acl_true_response, c_acl_response, c_acl_num_of_queries, c_acl_eps, error_init = \
        load_data_wo_constraint(c_acl_file + noise_tag_acl + 'training_data_' + str(itr) + '_.pth',
                                device=device)

    c_acl_values = c_acl_true_response.detach().cpu().numpy()
    min_01_acl.append(min(c_acl_values).item())
    c_acl_actions_np = c_acl_actions.detach().cpu().numpy()
    n_safe_acl01   = sum(1 for a in c_acl_actions_np if is_feasible(a))
    safe_rate_01_acl.append(n_safe_acl01 / len(c_acl_actions_np))
    print(f'Classic ACL 0.1 safe_queried_rate: {n_safe_acl01}/{len(c_acl_actions_np)} = {n_safe_acl01/len(c_acl_actions_np):.4f}')
    c_acl_track_queries = c_acl_num_of_queries.detach().cpu().numpy().astype(np.int32)
    c_acl_values_new = []
    for i in range(len(c_acl_values)):
        c_acl_values_new += list(np.repeat(c_acl_values[i], c_acl_track_queries[i]))
    c_acl_values     = np.array(c_acl_values_new)
    c_acl_values     = np.squeeze(np.abs(f_max - c_acl_values))
    c_acl_values_acc = np.cumsum(c_acl_values)
    min_len_5    = np.min([min_len_5, len(c_acl_values_acc)])
    all_regrets_5.append(c_acl_values_acc)



# ─── summary statistics for safe_queried_rate ─────────────────────────────────
print()
print('=== safe_queried_rate summary ===')
print(f'Quantum noise=0.01: mean={np.mean(safe_rate_01_q):.4f} ± {np.std(safe_rate_01_q):.4f}')
print(f'Classic noise=0.01: mean={np.mean(safe_rate_01_c):.4f} ± {np.std(safe_rate_01_c):.4f}')
print(f'Quantum noise=0.04: mean={np.mean(safe_rate_02_q):.4f} ± {np.std(safe_rate_02_q):.4f}')
print(f'Classic noise=0.04: mean={np.mean(safe_rate_02_c):.4f} ± {np.std(safe_rate_02_c):.4f}')
print(f'Classic ACL noise=0.01: mean={np.mean(safe_rate_01_acl):.4f} ± {np.std(safe_rate_01_acl):.4f}')

print('quantum 0.1 min mean {0} +- {1}:'.format(np.mean(min_01_q), np.std(min_01_q)))
print('classic 0.1 min mean {0} +- {1}'.format(np.mean(min_01_c), np.std(min_01_c)))
print('classic ACL 0.1 min mean {0} +- {1}'.format(np.mean(min_01_acl), np.std(min_01_acl)))

print('quantum 0.2 min mean {0} +- {1}'.format(np.mean(min_02_q), np.std(min_02_q)))
print('classic 0.2 min mean {0} +- {1}'.format(np.mean(min_02_c), np.std(min_02_c)))

print()
print('=== safe_queried_rate ===')
print(f'Quantum noise=0.01: mean={np.mean(safe_rate_01_q):.4f} +- {np.std(safe_rate_01_q):.4f}')
print(f'Classic noise=0.01: mean={np.mean(safe_rate_01_c):.4f} +- {np.std(safe_rate_01_c):.4f}')
print(f'Quantum noise=0.04: mean={np.mean(safe_rate_02_q):.4f} +- {np.std(safe_rate_02_q):.4f}')
print(f'Classic noise=0.04: mean={np.mean(safe_rate_02_c):.4f} +- {np.std(safe_rate_02_c):.4f}')
print(f'Classic ACL noise=0.01: mean={np.mean(safe_rate_01_acl):.4f} +- {np.std(safe_rate_01_acl):.4f}')

lw = 2.0
plt.rc('font', size=18)

# high-res on screen + save
plt.rcParams['figure.dpi'] = 150      # notebook/interactive
plt.rcParams['savefig.dpi'] = 600     # file output
plt.figure(figsize=(9, 7))

all_regrets_1 = [a[:min_len_1] for a in all_regrets_1]
all_regrets_1_np = np.array(all_regrets_1)
all_regrets_1_np_mean = np.mean(all_regrets_1_np, axis=0)
all_regrets_1_np_stderr = np.std(all_regrets_1_np, axis=0) / (np.sqrt(len(run_list)))
all_regrets_1_np_ub = all_regrets_1_np_mean + all_regrets_1_np_stderr
all_regrets_1_np_lb = all_regrets_1_np_mean - all_regrets_1_np_stderr

all_regrets_2 = [a[:min_len_2] for a in all_regrets_2]
all_regrets_2_np = np.array(all_regrets_2)
all_regrets_2_np_mean = np.mean(all_regrets_2_np, axis=0)
all_regrets_2_np_stderr = np.std(all_regrets_2_np, axis=0) / (np.sqrt(len(run_list)))
all_regrets_2_np_ub = all_regrets_2_np_mean + all_regrets_2_np_stderr
all_regrets_2_np_lb = all_regrets_2_np_mean - all_regrets_2_np_stderr

all_regrets_3 = [a[:min_len_3] for a in all_regrets_3]
all_regrets_3_np = np.array(all_regrets_3)
all_regrets_3_np_mean = np.mean(all_regrets_3_np, axis=0)
all_regrets_3_np_stderr = np.std(all_regrets_3_np, axis=0) / (np.sqrt(len(run_list)))
all_regrets_3_np_ub = all_regrets_3_np_mean + all_regrets_3_np_stderr
all_regrets_3_np_lb = all_regrets_3_np_mean - all_regrets_3_np_stderr

all_regrets_4 = [a[:min_len_4] for a in all_regrets_4]
all_regrets_4_np = np.array(all_regrets_4)
all_regrets_4_np_mean = np.mean(all_regrets_4_np, axis=0)
all_regrets_4_np_stderr = np.std(all_regrets_4_np, axis=0) / (np.sqrt(len(run_list)))
all_regrets_4_np_ub = all_regrets_4_np_mean + all_regrets_4_np_stderr
all_regrets_4_np_lb = all_regrets_4_np_mean - all_regrets_4_np_stderr

# all_regrets_5 = [a[:min_len_5] for a in all_regrets_5]
# all_regrets_5_np = np.array(all_regrets_5)
# all_regrets_5_np_mean = np.mean(all_regrets_5_np, axis=0)
# all_regrets_5_np_stderr = np.std(all_regrets_5_np, axis=0) / (np.sqrt(len(run_list)))
# all_regrets_5_np_ub = all_regrets_5_np_mean + all_regrets_5_np_stderr
# all_regrets_5_np_lb = all_regrets_5_np_mean - all_regrets_5_np_stderr

color_list = ["#625be6", "#edc709", '#2ca02c', '#d62728'] 
marker_list = ["v", "D", "s", "o", "^"]
inds = np.arange(1, 1e6+1)
marker_size = 8

fill1, line1 = plot_mean_and_CI_with_marker(inds[:min_len_1], all_regrets_1_np_mean[:min_len_1], all_regrets_1_np_ub[:min_len_1], all_regrets_1_np_lb[:min_len_1], color_mean=color_list[0], color_shading=color_list[0], marker=marker_list[0], marker_size=marker_size)
fill2, line2 = plot_mean_and_CI_with_marker(inds[:min_len_2], all_regrets_2_np_mean[:min_len_2], all_regrets_2_np_ub[:min_len_2], all_regrets_2_np_lb[:min_len_2], color_mean=color_list[2], color_shading=color_list[2], marker=marker_list[2], marker_size=marker_size)
fill3, line3 = plot_mean_and_CI_with_marker(inds[:min_len_3], all_regrets_3_np_mean[:min_len_3], all_regrets_3_np_ub[:min_len_3], all_regrets_3_np_lb[:min_len_3], color_mean=color_list[1], color_shading=color_list[1], marker=marker_list[1], marker_size=marker_size)
fill4, line4 = plot_mean_and_CI_with_marker(inds[:min_len_4], all_regrets_4_np_mean[:min_len_4], all_regrets_4_np_ub[:min_len_4], all_regrets_4_np_lb[:min_len_4], color_mean=color_list[3], color_shading=color_list[3], marker=marker_list[3], marker_size=marker_size)
# fill5, line5 = plot_mean_and_CI_with_marker(inds[:min_len_5], all_regrets_5_np_mean[:min_len_5], all_regrets_5_np_ub[:min_len_5], all_regrets_5_np_lb[:min_len_5], color_mean=color_list[4], color_shading=color_list[4], marker=marker_list[4], marker_size=marker_size)
plt.legend([line1, line3, line2, line4], ["Q-Safe-Set-UCB ($\sigma=0.1$)", "Q-Safe-Set-UCB ($\sigma=0.2$)", "C-WGP-UCB ($\sigma=0.1$)", "C-WGP-UCB ($\sigma=0.2$)"], \
    prop={'size':14})






axes = plt.gca()
axes.set_xlim([0, 30000])
axes.set_ylim([-0.01, 30000])

plt.ylabel("Cumulative Regret")
plt.xlabel("Iterations")
plt.show()

def load_data_wo_constraint(filename, device):
    data = torch.load(filename, map_location=device)
    actions        = data['actions'].to(torch.float32).to(device)
    true_response  = data['true_response'].to(torch.float32).to(device)
    response       = data['response'].to(torch.float32).to(device)
    num_of_queries = data['queries'].to(device)
    eps            = data['uncertainty']
    error_init     = data['error_init']
    return actions, true_response, response, num_of_queries, eps, error_init

import matplotlib.pyplot as plt
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Experiment parameters ────────────────────────────────────────────────
eta   = 1.0
lam   = 1.0
Big_B = 2.0

# Noise tags for filenames and directories
NOISE_01 = '0.01'
NOISE_04 = '0.04'

# Base paths (without noise tag and exp_set)
Q_BASE_01 = ('Experiments_constraint_continuous/'
             f'Quantum_EXP_10_{eta}_{lam}_{Big_B}_multi_gpu_noise_{NOISE_01}/')
Q_BASE_04 = ('Experiments_constraint_continuous/'
             f'Quantum_EXP_10_{eta}_{lam}_{Big_B}_multi_gpu_noise_{NOISE_04}/')
C_BASE_01 = ('Experiments_constraint_continuous/'
             f'Classic_EXP_10_{eta}_{lam}_1.0_noise_{NOISE_01}/')
C_BASE_04 = ('Experiments_constraint_continuous/'
             f'Classic_EXP_10_{eta}_{lam}_1.0_noise_{NOISE_04}/')

obs_noise = 0.1**2
exp_num   = 10    # exp_set_0 .. exp_set_6 for Classic; 10 for Quantum (0.9)
# Quantum B=2 noise dirs have 10 exp_sets (0-9); Classic has 7 (0-6)
# Use 7 to match the smaller set
run_list  = np.arange(5)  # 5 seeds per exp_set

all_min_means_quantum  = []  # List[List]: shape = [exp_num][5]
all_min_means_quantum2 = []  # Quantum noise=0.04
all_min_means_classic  = []  # Classic noise=0.01
all_min_means_classic2 = []  # Classic noise=0.04

all_min_actions_quantum  = []
all_min_actions_quantum2 = []
all_min_actions_classic  = []
all_min_actions_classic2 = []

for tri_num in range(exp_num):
    exp_set = f'exp_set_{tri_num}/'

    # ── Quantum noise=0.01 ─────────────────────────────────────────────
    q_min_mean, q_min_actions = [], []
    for itr in run_list:
        fname = Q_BASE_01 + exp_set + NOISE_01 + f'quan_training_data_{itr}_.pth'
        q_actions, q_true_response, q_response, num_of_queries, eps, error_init = \
            load_data_wo_constraint(fname, device=device)
        q_values  = q_true_response.detach().cpu().numpy()
        q_min_idx = np.argmin(q_values)
        q_min_mean.append(float(np.min(q_values)))
        q_min_actions.append(q_actions[q_min_idx].detach().cpu().numpy())
    all_min_means_quantum.append(q_min_mean)
    all_min_actions_quantum.append(q_min_actions)

    # ── Quantum noise=0.04 ─────────────────────────────────────────────
    q_min_mean2, q_min_actions2 = [], []
    for itr in run_list:
        fname = Q_BASE_04 + exp_set + NOISE_04 + f'quan_training_data_{itr}_.pth'
        q_actions2, q_true_response2, q_response2, nq2, eps2, ei2 = \
            load_data_wo_constraint(fname, device=device)
        q_values2  = q_true_response2.detach().cpu().numpy()
        q_min_idx2 = np.argmin(q_values2)
        q_min_mean2.append(float(np.min(q_values2)))
        q_min_actions2.append(q_actions2[q_min_idx2].detach().cpu().numpy())
    all_min_means_quantum2.append(q_min_mean2)
    all_min_actions_quantum2.append(q_min_actions2)

    # ── Classic noise=0.01 ─────────────────────────────────────────────
    C_NOISE_01 = str(0.1**2)
    c_min_mean, c_min_actions = [], []
    for itr in run_list:
        fname = C_BASE_01 + exp_set + C_NOISE_01 + f'training_data_{itr}_.pth'
        c_actions, c_true_response, c_response, cnq, ceps, cei = \
            load_data_wo_constraint(fname, device=device)
        c_values  = c_true_response.detach().cpu().numpy()
        c_min_idx = np.argmin(c_values)
        c_min_mean.append(float(np.min(c_values)))
        c_min_actions.append(c_actions[c_min_idx].detach().cpu().numpy())
    all_min_means_classic.append(c_min_mean)
    all_min_actions_classic.append(c_min_actions)

    # ── Classic noise=0.04 ─────────────────────────────────────────────
    C_NOISE_04 = str(0.2**2)
    c_min_mean2, c_min_actions2 = [], []
    for itr in run_list:
        fname = C_BASE_04 + exp_set + C_NOISE_04+ f'training_data_{itr}_.pth'
        c_actions2, c_true_response2, c_response2, cnq2, ceps2, cei2 = \
            load_data_wo_constraint(fname, device=device)
        c_values2  = c_true_response2.detach().cpu().numpy()
        c_min_idx2 = np.argmin(c_values2)
        c_min_mean2.append(float(np.min(c_values2)))
        c_min_actions2.append(c_actions2[c_min_idx2].detach().cpu().numpy())
    all_min_means_classic2.append(c_min_mean2)
    all_min_actions_classic2.append(c_min_actions2)

print(f'Loaded {exp_num} experiment sets, each with {len(run_list)} seeds.')
print('all_min_means_quantum  shape:', len(all_min_means_quantum), 'x', len(all_min_means_quantum[0]))
print('all_min_means_quantum2 shape:', len(all_min_means_quantum2), 'x', len(all_min_means_quantum2[0]))
print('all_min_means_classic  shape:', len(all_min_means_classic), 'x', len(all_min_means_classic[0]))
print('all_min_means_classic2 shape:', len(all_min_means_classic2), 'x', len(all_min_means_classic2[0]))

# Plot min f(x) per case comparison
lo, hi = None, None  # auto-scale
plt.figure(figsize=(9, 7))
ax = plt.gca()

positions = np.arange(1, exp_num + 1)
means_q01 = [np.mean(g) for g in all_min_means_quantum]
means_q04 = [np.mean(g) for g in all_min_means_quantum2]
means_c01 = [np.mean(g) for g in all_min_means_classic]
means_c04 = [np.mean(g) for g in all_min_means_classic2]

ax.plot(positions, means_q01, 'b-o', label='Quantum σ=0.1')
ax.plot(positions, means_q04, 'g-s', label='Quantum σ=0.2')
ax.plot(positions, means_c01, 'r-^', label='Classic σ=0.1')
ax.plot(positions, means_c04, 'm-D', label='Classic σ=0.2')

ax.set_xlabel('Experiment Set')
ax.set_ylabel('Min MAE')
ax.set_title('Min f(x) per Experiment Set')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Summary statistics across all experiment sets
import numpy as np

all_q01  = [v for group in all_min_means_quantum  for v in group]
all_q04  = [v for group in all_min_means_quantum2 for v in group]
all_c01  = [v for group in all_min_means_classic  for v in group]
all_c04  = [v for group in all_min_means_classic2 for v in group]

print(f'Quantum  σ=0.1  min f(x): mean={np.mean(all_q01):.5f} ± std={np.std(all_q01):.5f}')
print(f'Quantum  σ=0.2  min f(x): mean={np.mean(all_q04):.5f} ± std={np.std(all_q04):.5f}')
print(f'Classic  σ=0.1  min f(x): mean={np.mean(all_c01):.5f} ± std={np.std(all_c01):.5f}')
print(f'Classic  σ=0.2  min f(x): mean={np.mean(all_c04):.5f} ± std={np.std(all_c04):.5f}')

# Data already collected in Cell 8 (all_min_means_quantum, etc.)
# This cell intentionally left as a placeholder.
pass

all_min_means_quantum
all_min_means_classic

min_matrix_classic = np.array(all_min_means_classic)
row_mean_classic = np.mean(min_matrix_classic, axis=1)
print("Classic Mean:", np.round(row_mean_classic, 4))
print(np.std(row_mean_classic))
min_matrix_quantum = np.array(all_min_means_quantum)
row_min_quantum = np.mean(min_matrix_quantum, axis=1)
print('Quantum mean: ', np.round(row_min_quantum, 4))
print(np.std(row_min_quantum))
min_matrix_quantum = np.array(all_min_means_quantum2)
row_min_quantum = np.mean(min_matrix_quantum, axis=1)
print('Quantum mean: ', np.round(row_min_quantum, 4))
print(np.std(row_min_quantum))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

positions = np.arange(1, exp_num + 1)  # case numbers: 1..10

# ── Means per experiment set ──────────────────────────────────────
means_classic_01 = [np.mean(g) for g in all_min_means_classic]   # noise=0.01
means_classic_04 = [np.mean(g) for g in all_min_means_classic2]  # noise=0.04
means_quantum_01 = [np.mean(g) for g in all_min_means_quantum]   # noise=0.01
means_quantum_04 = [np.mean(g) for g in all_min_means_quantum2]  # noise=0.04

W = 0.35   # box width

plt.rcParams['figure.dpi']  = 150
plt.rcParams['savefig.dpi'] = 600

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# ─────────────────────────────────────────────────────────────────
# TOP panel  ──  noise = 0.01  (sigma = 0.1)
# ─────────────────────────────────────────────────────────────────
ax = axes[0]

# Classic noise=0.01 — drawn first (behind)
ax.boxplot(
    all_min_means_classic,
    positions    = positions,
    widths       = W,
    patch_artist = True,
    boxprops     = dict(facecolor='lightskyblue', color='black', alpha=0.85),
    medianprops  = dict(color='navy', linewidth=2),
    whiskerprops = dict(color='steelblue'),
    capprops     = dict(color='steelblue'),
    flierprops   = dict(marker='o', markersize=4, linestyle='none',
                        markerfacecolor='steelblue', alpha=0.5),
)
# Quantum noise=0.01 — drawn on top (semi-transparent)
ax.boxplot(
    all_min_means_quantum,
    positions    = positions,
    widths       = W,
    patch_artist = True,
    boxprops     = dict(facecolor='mistyrose', color='black', alpha=0.65),
    medianprops  = dict(color='darkred', linewidth=2),
    whiskerprops = dict(color='firebrick'),
    capprops     = dict(color='firebrick'),
    flierprops   = dict(marker='s', markersize=4, linestyle='none',
                        markerfacecolor='firebrick', alpha=0.5),
)

ax.plot(positions, means_classic_01, color='steelblue', marker='o',
        linestyle='--', linewidth=1.5, label=r'Classic mean')
ax.plot(positions, means_quantum_01, color='firebrick', marker='s',
        linestyle='--', linewidth=1.5, label=r'Quantum mean')

legend_patches = [
    Patch(facecolor='lightskyblue', edgecolor='black', label='Classic'),
    Patch(facecolor='mistyrose',    edgecolor='black', label='Quantum'),
]
ax.legend(handles=legend_patches + ax.lines[:2], fontsize=11,
          loc='upper right', framealpha=0.85)
ax.set_ylabel('MAE [in]', fontsize=13)
ax.set_title(r'Noise level $\sigma$ = 0.10  (obs noise = 0.01)', fontsize=13)
ax.set_xticks(positions)
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, linestyle='--', alpha=0.4)

# ─────────────────────────────────────────────────────────────────
# BOTTOM panel  ──  noise = 0.04  (sigma = 0.2)
# ─────────────────────────────────────────────────────────────────
ax = axes[1]

ax.boxplot(
    all_min_means_classic2,
    positions    = positions,
    widths       = W,
    patch_artist = True,
    boxprops     = dict(facecolor='lightskyblue', color='black', alpha=0.85),
    medianprops  = dict(color='navy', linewidth=2),
    whiskerprops = dict(color='steelblue'),
    capprops     = dict(color='steelblue'),
    flierprops   = dict(marker='o', markersize=4, linestyle='none',
                        markerfacecolor='steelblue', alpha=0.5),
)
ax.boxplot(
    all_min_means_quantum2,
    positions    = positions,
    widths       = W,
    patch_artist = True,
    boxprops     = dict(facecolor='mistyrose', color='black', alpha=0.65),
    medianprops  = dict(color='darkred', linewidth=2),
    whiskerprops = dict(color='firebrick'),
    capprops     = dict(color='firebrick'),
    flierprops   = dict(marker='s', markersize=4, linestyle='none',
                        markerfacecolor='firebrick', alpha=0.5),
)

ax.plot(positions, means_classic_04, color='steelblue', marker='o',
        linestyle='--', linewidth=1.5, label='Classic mean')
ax.plot(positions, means_quantum_04, color='firebrick', marker='s',
        linestyle='--', linewidth=1.5, label='Quantum mean')

legend_patches2 = [
    Patch(facecolor='lightskyblue', edgecolor='black', label='Classic'),
    Patch(facecolor='mistyrose',    edgecolor='black', label='Quantum'),
]
ax.legend(handles=legend_patches2 + ax.lines[:2], fontsize=11,
          loc='upper right', framealpha=0.85)
ax.set_xlabel('Case Number', fontsize=13)
ax.set_ylabel('MAE [in]',   fontsize=13)
ax.set_title(r'Noise level $\sigma$ = 0.20  (obs noise = 0.04)', fontsize=13)
ax.set_xticks(positions)
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, linestyle='--', alpha=0.4)

# ─────────────────────────────────────────────────────────────────
plt.suptitle('Unconstrained Classic BO vs Safe-Set Quantum BO – Min MAE per Case, by Noise Level',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('fig1.png', dpi=600, bbox_inches='tight')
plt.show()
