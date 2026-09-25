lw = 2.0
plt.rc('font', size=16)
# high-res on screen + save
plt.rcParams['figure.dpi'] = 150      # notebook/interactive
plt.rcParams['savefig.dpi'] = 600     # file output
plt.figure(figsize=(9, 7))

all_regrets_1 = [a[:min_len_1] for a in all_regrets_1]
all_regrets_1_np = np.array(all_regrets_1)
all_regrets_1_np_mean = np.mean(all_regrets_1_np, axis=0)
all_regrets_1_np_stderr = np.std(all_regrets_1_np, axis=0) / np.sqrt(len(run_list))
all_regrets_1_np_ub = all_regrets_1_np_mean + all_regrets_1_np_stderr
all_regrets_1_np_lb = all_regrets_1_np_mean - all_regrets_1_np_stderr

all_regrets_2 = [a[:min_len_2] for a in all_regrets_2]
all_regrets_2_np = np.array(all_regrets_2)
all_regrets_2_np_mean = np.mean(all_regrets_2_np, axis=0)
all_regrets_2_np_stderr = np.std(all_regrets_2_np, axis=0) / np.sqrt(len(run_list))
all_regrets_2_np_ub = all_regrets_2_np_mean + all_regrets_2_np_stderr
all_regrets_2_np_lb = all_regrets_2_np_mean - all_regrets_2_np_stderr

all_regrets_3 = [a[:min_len_3] for a in all_regrets_3]
all_regrets_3_np = np.array(all_regrets_3)
all_regrets_3_np_mean = np.mean(all_regrets_3_np, axis=0)
all_regrets_3_np_stderr = np.std(all_regrets_3_np, axis=0) / np.sqrt(len(run_list))
all_regrets_3_np_ub = all_regrets_3_np_mean + all_regrets_3_np_stderr
all_regrets_3_np_lb = all_regrets_3_np_mean - all_regrets_3_np_stderr

all_regrets_4 = [a[:min_len_4] for a in all_regrets_4]
all_regrets_4_np = np.array(all_regrets_4)
all_regrets_4_np_mean = np.mean(all_regrets_4_np, axis=0)
all_regrets_4_np_stderr = np.std(all_regrets_4_np, axis=0) / np.sqrt(len(run_list))
all_regrets_4_np_ub = all_regrets_4_np_mean + all_regrets_4_np_stderr
all_regrets_4_np_lb = all_regrets_4_np_mean - all_regrets_4_np_stderr

all_regrets_5 = [a[:min_len_5] for a in all_regrets_5]
all_regrets_5_np = np.array(all_regrets_5)
all_regrets_5_np_mean = np.mean(all_regrets_5_np, axis=0)
all_regrets_5_np_stderr = np.std(all_regrets_5_np, axis=0) / np.sqrt(len(run_list))
all_regrets_5_np_ub = all_regrets_5_np_mean + all_regrets_5_np_stderr
all_regrets_5_np_lb = all_regrets_5_np_mean - all_regrets_5_np_stderr

all_regrets_6 = [a[:min_len_6] for a in all_regrets_6]
all_regrets_6_np = np.array(all_regrets_6)
all_regrets_6_np_mean = np.mean(all_regrets_6_np, axis=0)
all_regrets_6_np_stderr = np.std(all_regrets_6_np, axis=0) / np.sqrt(len(run_list))
all_regrets_6_np_ub = all_regrets_6_np_mean + all_regrets_6_np_stderr
all_regrets_6_np_lb = all_regrets_6_np_mean - all_regrets_6_np_stderr

color_list = ["#625be6", "#edc709", '#2ca02c', '#d62728', '#9467bd', '#17becf'] 
marker_list = ["v", "D", "s", "o", "^", "P"]
inds = np.arange(1, 1e6+1)
marker_size = 8

fill1, line1 = plot_mean_and_CI_with_marker(inds[:min_len_1], all_regrets_1_np_mean[:min_len_1], all_regrets_1_np_ub[:min_len_1], all_regrets_1_np_lb[:min_len_1], color_mean=color_list[0], color_shading=color_list[0], marker=marker_list[0], marker_size=marker_size)
fill2, line2 = plot_mean_and_CI_with_marker(inds[:min_len_2], all_regrets_2_np_mean[:min_len_2], all_regrets_2_np_ub[:min_len_2], all_regrets_2_np_lb[:min_len_2], color_mean=color_list[2], color_shading=color_list[2], marker=marker_list[2], marker_size=marker_size)
fill3, line3 = plot_mean_and_CI_with_marker(inds[:min_len_3], all_regrets_3_np_mean[:min_len_3], all_regrets_3_np_ub[:min_len_3], all_regrets_3_np_lb[:min_len_3], color_mean=color_list[1], color_shading=color_list[1], marker=marker_list[1], marker_size=marker_size)
fill4, line4 = plot_mean_and_CI_with_marker(inds[:min_len_4], all_regrets_4_np_mean[:min_len_4], all_regrets_4_np_ub[:min_len_4], all_regrets_4_np_lb[:min_len_4], color_mean=color_list[3], color_shading=color_list[3], marker=marker_list[3], marker_size=marker_size)
fill5, line5 = plot_mean_and_CI_with_marker(inds[:min_len_5], all_regrets_5_np_mean[:min_len_5], all_regrets_5_np_ub[:min_len_5], all_regrets_5_np_lb[:min_len_5], color_mean=color_list[4], color_shading=color_list[4], marker=marker_list[4], marker_size=marker_size)
fill6, line6 = plot_mean_and_CI_with_marker(inds[:min_len_6], all_regrets_6_np_mean[:min_len_6], all_regrets_6_np_ub[:min_len_6], all_regrets_6_np_lb[:min_len_6], color_mean=color_list[5], color_shading=color_list[5], marker=marker_list[5], marker_size=marker_size)

plt.legend([line1, line3, line2, line4, line5, line6],
    ["Q-cUCB ($\sigma=0.1$)", "Q-cUCB ($\sigma=0.2$)",
     "C-cUCB ($\sigma=0.1$)", "C-cUCB ($\sigma=0.2$)",
     "C-ACL ($\sigma=0.1$)", "C-ACL ($\sigma=0.2$)"],
    prop={'size':12}, ncol=2)

axes = plt.gca()
axes.set_xlim([0, 10000])
axes.set_ylim([0, 3000])
plt.ylabel("Cumulative Regret")
plt.xlabel("Iterations")
plt.show()