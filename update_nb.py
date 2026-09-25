import json

with open('/data/liuj35/quan_fuselage/analyze_discrete.ipynb', 'r') as f:
    nb = json.load(f)

# Update cell 2
c2 = "".join(nb['cells'][2]['source'])

to_insert_1 = """
q_real_file = 'Experiments_constraints/Quantum_Discrete_cUCB_Real/exp_set_0/'
"""
c2 = c2.replace("c_acl_file = 'Experiments_constraints/Classic_ACL_Discrete/exp_set_1/'\n", "c_acl_file = 'Experiments_constraints/Classic_ACL_Discrete/exp_set_1/'\n" + to_insert_1)

to_insert_2 = """
all_regrets_7 = []
all_runmin_7 = []
min_len_7 = int(30000+100)
runmin_len_7 = int(30000+100)
min_02_q_real = []
"""
c2 = c2.replace("min_02_acl = []\n", "min_02_acl = []\n" + to_insert_2)

to_insert_3 = """
    # ── Quantum Real, noise=0.04 ──
    obs_noise = 0.2**2
    try:
        q_real_actions, q_real_true_response, q_real_response, q_real_num_of_queries, q_real_eps = load_data_wo_constraint(q_real_file + str(obs_noise) + 'quan_training_data_'+ str(itr)+'_.pth', device=device, quan=True)
        q_real_values = q_real_true_response.detach().cpu().numpy()
        min_02_q_real.append(min(q_real_values).item())
        q_real_track_queries = q_real_num_of_queries.detach().cpu().numpy().astype(np.int32)
        q_real_values_new = []
        for ii in range(len(q_real_values)):
            q_real_values_new += list(np.repeat(q_real_values[ii], q_real_track_queries[ii]))
        q_real_values = np.array(q_real_values_new)
        q_real_values = np.squeeze(np.abs(f_max - q_real_values))
        q_real_values_acc = np.cumsum(q_real_values)
        min_len_7 = np.min([min_len_7, len(q_real_values_acc)])
        all_regrets_7.append(q_real_values_acc)
        q_real_values_runmin = np.minimum.accumulate(q_real_values)
        runmin_len_7 = np.min([runmin_len_7, len(q_real_values_runmin)])
        all_runmin_7.append(q_real_values_runmin)
    except Exception as e:
        print(f"Skipping Quantum Real for itr {itr} due to {e}")
"""
c2 = c2.replace("print('quantum 0.1 min mean'", to_insert_3 + "\nprint('quantum 0.1 min mean'")
c2 = c2.replace("print('classic ACL 0.2 min mean', np.mean(min_02_acl), '+-', np.std(min_02_acl))\n", "print('classic ACL 0.2 min mean', np.mean(min_02_acl), '+-', np.std(min_02_acl))\nprint('quantum real 0.2 min mean', np.mean(min_02_q_real), '+-', np.std(min_02_q_real))\n")

nb['cells'][2]['source'] = [line + '\n' for line in c2.split('\n')[:-1]]


# Update cell 3
c3 = "".join(nb['cells'][3]['source'])

to_insert_4 = """
all_regrets_7 = [a[:min_len_7] for a in all_regrets_7]
all_regrets_7_np = np.array(all_regrets_7)
all_regrets_7_np_mean = np.mean(all_regrets_7_np, axis=0)
all_regrets_7_np_stderr = np.std(all_regrets_7_np, axis=0) / np.sqrt(max(1, len(all_regrets_7_np)))
all_regrets_7_np_ub = all_regrets_7_np_mean + all_regrets_7_np_stderr
all_regrets_7_np_lb = all_regrets_7_np_mean - all_regrets_7_np_stderr
"""
c3 = c3.replace("color_list = [", to_insert_4 + "\ncolor_list = [")
c3 = c3.replace("color_list = [\"#625be6\", \"#edc709\", '#2ca02c', '#d62728', '#9467bd', '#17becf']", "color_list = [\"#625be6\", \"#edc709\", '#2ca02c', '#d62728', '#9467bd', '#17becf', '#ff7f0e']")
c3 = c3.replace("marker_list = [\"v\", \"D\", \"s\", \"o\", \"^\", \"P\"]", "marker_list = [\"v\", \"D\", \"s\", \"o\", \"^\", \"P\", \"X\"]")

to_insert_5 = "fill7, line7 = plot_mean_and_CI_with_marker(inds[:min_len_7], all_regrets_7_np_mean[:min_len_7], all_regrets_7_np_ub[:min_len_7], all_regrets_7_np_lb[:min_len_7], color_mean=color_list[6], color_shading=color_list[6], marker=marker_list[6], marker_size=marker_size)\n"
c3 = c3.replace("plt.legend([line1, line3, line2,", to_insert_5 + "\nplt.legend([line1, line3, line7, line2,")
c3 = c3.replace("[\"Q-cUCB ($\sigma=0.1$)\", \"Q-cUCB ($\sigma=0.2$)\",", "[\"Q-cUCB ($\sigma=0.1$)\", \"Q-cUCB ($\sigma=0.2$)\", \"Q-cUCB (Real) ($\sigma=0.2$)\",")
c3 = c3.replace("prop={'size':12}", "prop={'size':11}")
c3 = c3.replace("line5, line6]", "line5, line6]")

nb['cells'][3]['source'] = [line + '\n' for line in c3.split('\n')[:-1]]

# Update cell 4
c4 = "".join(nb['cells'][4]['source'])

to_insert_6 = """
all_runmin_7_t = [a[:runmin_len_7] for a in all_runmin_7]
all_runmin_7_np = np.array(all_runmin_7_t)
all_runmin_7_np_mean = np.mean(all_runmin_7_np, axis=0)
all_runmin_7_np_stderr = np.std(all_runmin_7_np, axis=0) / np.sqrt(max(1, len(all_runmin_7_np)))
all_runmin_7_np_ub = all_runmin_7_np_mean + all_runmin_7_np_stderr
all_runmin_7_np_lb = all_runmin_7_np_mean - all_runmin_7_np_stderr
"""
c4 = c4.replace("color_list = [", to_insert_6 + "\ncolor_list = [")
c4 = c4.replace("color_list = [\"#625be6\", \"#edc709\", '#2ca02c', '#d62728', '#9467bd', '#17becf']", "color_list = [\"#625be6\", \"#edc709\", '#2ca02c', '#d62728', '#9467bd', '#17becf', '#ff7f0e']")
c4 = c4.replace("marker_list = [\"v\", \"D\", \"s\", \"o\", \"^\", \"P\"]", "marker_list = [\"v\", \"D\", \"s\", \"o\", \"^\", \"P\", \"X\"]")

to_insert_7 = "fill7, line7 = plot_mean_and_CI_with_marker(inds[:runmin_len_7], all_runmin_7_np_mean, all_runmin_7_np_ub, all_runmin_7_np_lb, color_mean=color_list[6], color_shading=color_list[6], marker=marker_list[6], marker_size=marker_size)\n"
c4 = c4.replace("plt.legend([line1, line3, line2,", to_insert_7 + "\nplt.legend([line1, line3, line7, line2,")
c4 = c4.replace("[\"Q-cUCB ($\sigma=0.1$)\", \"Q-cUCB ($\sigma=0.2$)\",", "[\"Q-cUCB ($\sigma=0.1$)\", \"Q-cUCB ($\sigma=0.2$)\", \"Q-cUCB (Real) ($\sigma=0.2$)\",")
c4 = c4.replace("prop={'size':12}", "prop={'size':11}")

nb['cells'][4]['source'] = [line + '\n' for line in c4.split('\n')[:-1]]

with open('/data/liuj35/quan_fuselage/analyze_discrete.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Done updating")
