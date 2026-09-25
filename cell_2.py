import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_wo_constraint(filename, device, quan=False):
    data = torch.load(filename)
    if quan == True:      
        actions = data['actions'].to(torch.float32).to(device)
        response = data['response'].to(torch.float32).to(device)
        true_response = data['true_response'].to(torch.float32)
        num_of_queries =  data['queries']
        eps = data['uncertainty']
        return actions, true_response, response, num_of_queries, eps
    else:
        actions = data['actions'].to(torch.float32).to(device)
        response = data['response'].to(torch.float32).to(device)
        true_response = data['true_response'].to(torch.float32).to(device)
        eps = data['uncertainty']
        num_of_queries = data['queries']
        return actions, true_response, response, num_of_queries, eps
    
def plot_mean_and_CI_with_marker(time_steps, mean, lb, ub, color_mean=None, color_shading=None, marker=None, marker_size=12):
    fill = plt.fill_between(time_steps, ub, lb,
                        color=color_shading, alpha=.2)
    fill = None
    line = plt.plot(time_steps, mean, color_mean, marker=marker, markersize=marker_size, markevery=1000)
    line = line[0]
    return fill, line


q_file  = 'Experiments_constraints/Quantum_Discrete_cUCB/exp_set_1/'
c_file  = 'Experiments_constraints/Classic_Discrete_cUCB/exp_set_1/'
c_acl_file = 'Experiments_constraints/Classic_ACL_Discrete/exp_set_1/'
f_max = 0.048170337104871036

all_regrets_1 = []   # quantum noise=0.01
all_regrets_2 = []   # classic noise=0.01
all_regrets_3 = []   # quantum noise=0.04
all_regrets_4 = []   # classic noise=0.04
all_regrets_5 = []   # classic ACL noise=0.01
all_regrets_6 = []   # classic ACL noise=0.04

# running-minimum containers
all_runmin_1 = []
all_runmin_2 = []
all_runmin_3 = []
all_runmin_4 = []
all_runmin_5 = []
all_runmin_6 = []

min_len_1 = int(30000+100)
min_len_2 = int(30000+100)
min_len_3 = int(30000+100)
min_len_4 = int(30000+100)
min_len_5 = int(30000+100)
min_len_6 = int(30000+100)

runmin_len_1 = int(30000+100)
runmin_len_2 = int(30000+100)
runmin_len_3 = int(30000+100)
runmin_len_4 = int(30000+100)
runmin_len_5 = int(30000+100)
runmin_len_6 = int(30000+100)

min_01_q = []
min_01_c = []
min_02_q = []
min_02_c = []
min_01_acl = []
min_02_acl = []

run_list = np.arange(5)
for itr in run_list:
    # ── Quantum, noise=0.01 ──
    obs_noise = 0.1**2
    q_actions_s, q_true_response_s, q_response_s, num_of_queries_s, eps_s = load_data_wo_constraint(q_file + str(obs_noise) + 'quan_training_data_'+ str(itr)+'_.pth', device=device, quan=True)
    values_s = q_true_response_s.detach().cpu().numpy()
    min_01_q.append(min(values_s).item())
    track_queries_s = num_of_queries_s.detach().cpu().numpy().astype(np.int32)
    values_new_s = []
    for ii in range(len(values_s)):
        values_new_s += list(np.repeat(values_s[ii], track_queries_s[ii]))
    values_s = np.array(values_new_s)
    values_s = np.squeeze(np.abs(f_max - values_s))
    values_acc_s = np.cumsum(values_s)
    min_len_1 = np.min([min_len_1, len(values_acc_s)])
    all_regrets_1.append(values_acc_s)
    values_runmin = np.minimum.accumulate(values_s)
    runmin_len_1 = np.min([runmin_len_1, len(values_runmin)])
    all_runmin_1.append(values_runmin)
    
    # ── Classic, noise=0.01 ──
    obs_noise = 0.1**2
    c_actions, c_true_response, c_response, c_num_of_queries, c_eps = load_data_wo_constraint(c_file + str(obs_noise) + 'training_data_' + str(itr) +'_.pth', device=device)
    c_values = c_true_response.detach().cpu().numpy()
    min_01_c.append(min(c_values).item())
    c_track_queries = c_num_of_queries.detach().cpu().numpy().astype(np.int32)
    c_values_new = []
    for ii in range(len(c_values)):
        c_values_new += list(np.repeat(c_values[ii], c_track_queries[ii]))
    c_values = np.array(c_values_new)
    c_values = np.squeeze(np.abs(f_max - c_values))
    c_values_acc = np.cumsum(c_values)
    min_len_2 = np.min([min_len_2, len(c_values_acc)])
    all_regrets_2.append(c_values_acc)
    c_values_runmin = np.minimum.accumulate(c_values)
    runmin_len_2 = np.min([runmin_len_2, len(c_values_runmin)])
    all_runmin_2.append(c_values_runmin)
    
    # ── Quantum, noise=0.04 ──
    obs_noise = 0.2**2
    q_actions, q_true_response, q_response, num_of_queries, eps = load_data_wo_constraint(q_file + str(obs_noise) + 'quan_training_data_'+ str(itr)+'_.pth', device=device, quan=True)
    values = q_true_response.detach().cpu().numpy()
    min_02_q.append(min(values).item())
    track_queries = num_of_queries.detach().cpu().numpy().astype(np.int32)
    values_new = []
    for ii in range(len(values)):
        values_new += list(np.repeat(values[ii], track_queries[ii]))
    values = np.array(values_new)
    values = np.squeeze(np.abs(f_max - values))
    values_acc = np.cumsum(values)
    min_len_3 = np.min([min_len_3, len(values_acc)])
    all_regrets_3.append(values_acc)
    values_runmin = np.minimum.accumulate(values)
    runmin_len_3 = np.min([runmin_len_3, len(values_runmin)])
    all_runmin_3.append(values_runmin)
    
    # ── Classic, noise=0.04 ──
    obs_noise = 0.2**2
    c_actions, c_true_response, c_response, c_num_of_queries, c_eps = load_data_wo_constraint(c_file + str(obs_noise) + 'training_data_' + str(itr) +'_.pth', device=device)
    c_values = c_true_response.detach().cpu().numpy()
    min_02_c.append(min(c_values).item())
    c_track_queries = c_num_of_queries.detach().cpu().numpy().astype(np.int32)
    c_values_new = []
    for ii in range(len(c_values)):
        c_values_new += list(np.repeat(c_values[ii], c_track_queries[ii]))
    c_values = np.array(c_values_new)
    c_values = np.squeeze(np.abs(f_max - c_values))
    c_values_acc = np.cumsum(c_values)
    min_len_4 = np.min([min_len_4, len(c_values_acc)])
    all_regrets_4.append(c_values_acc)
    c_values_runmin = np.minimum.accumulate(c_values)
    runmin_len_4 = np.min([runmin_len_4, len(c_values_runmin)])
    all_runmin_4.append(c_values_runmin)

    # ── Classic ACL, noise=0.01 ──
    obs_noise = 0.1**2
    acl_actions, acl_true_response, acl_response, acl_num_of_queries, acl_eps = load_data_wo_constraint(c_acl_file + str(obs_noise) + 'acl_training_data_' + str(itr) +'_.pth', device=device)
    acl_values = acl_true_response.detach().cpu().numpy()
    min_01_acl.append(min(acl_values).item())
    acl_track_queries = acl_num_of_queries.detach().cpu().numpy().astype(np.int32)
    acl_values_new = []
    for ii in range(len(acl_values)):
        acl_values_new += list(np.repeat(acl_values[ii], acl_track_queries[ii]))
    acl_values = np.array(acl_values_new)
    acl_values = np.squeeze(np.abs(f_max - acl_values))
    acl_values_acc = np.cumsum(acl_values)
    min_len_5 = np.min([min_len_5, len(acl_values_acc)])
    all_regrets_5.append(acl_values_acc)
    acl_values_runmin = np.minimum.accumulate(acl_values)
    runmin_len_5 = np.min([runmin_len_5, len(acl_values_runmin)])
    all_runmin_5.append(acl_values_runmin)

    # ── Classic ACL, noise=0.04 ──
    obs_noise = 0.2**2
    acl_actions, acl_true_response, acl_response, acl_num_of_queries, acl_eps = load_data_wo_constraint(c_acl_file + str(obs_noise) + 'acl_training_data_' + str(itr) +'_.pth', device=device)
    acl_values = acl_true_response.detach().cpu().numpy()
    min_02_acl.append(min(acl_values).item())
    acl_track_queries = acl_num_of_queries.detach().cpu().numpy().astype(np.int32)
    acl_values_new = []
    for ii in range(len(acl_values)):
        acl_values_new += list(np.repeat(acl_values[ii], acl_track_queries[ii]))
    acl_values = np.array(acl_values_new)
    acl_values = np.squeeze(np.abs(f_max - acl_values))
    acl_values_acc = np.cumsum(acl_values)
    min_len_6 = np.min([min_len_6, len(acl_values_acc)])
    all_regrets_6.append(acl_values_acc)
    acl_values_runmin = np.minimum.accumulate(acl_values)
    runmin_len_6 = np.min([runmin_len_6, len(acl_values_runmin)])
    all_runmin_6.append(acl_values_runmin)

print('quantum 0.1 min mean', np.mean(min_01_q), '+-', np.std(min_01_q))
print('classic 0.1 min mean', np.mean(min_01_c), '+-', np.std(min_01_c))
print('classic ACL 0.1 min mean', np.mean(min_01_acl), '+-', np.std(min_01_acl))
print('quantum 0.2 min mean', np.mean(min_02_q), '+-', np.std(min_02_q))
print('classic 0.2 min mean', np.mean(min_02_c), '+-', np.std(min_02_c))
print('classic ACL 0.2 min mean', np.mean(min_02_acl), '+-', np.std(min_02_acl))
