import json

with open('/data/liuj35/quan_fuselage/analyze_discrete.ipynb', 'r') as f:
    nb = json.load(f)

# Update cell 5
c5 = "".join(nb['cells'][5]['source'])

to_insert_6 = """
all_runmin_7_t = [a[:runmin_len_7] for a in all_runmin_7]
all_runmin_7_np = np.array(all_runmin_7_t)
all_runmin_7_np_mean = np.mean(all_runmin_7_np, axis=0)
all_runmin_7_np_stderr = np.std(all_runmin_7_np, axis=0) / np.sqrt(max(1, len(all_runmin_7_np)))
all_runmin_7_np_ub = all_runmin_7_np_mean + all_runmin_7_np_stderr
all_runmin_7_np_lb = all_runmin_7_np_mean - all_runmin_7_np_stderr
"""
c5 = c5.replace("color_list = [", to_insert_6 + "\ncolor_list = [")
c5 = c5.replace("color_list = [\"#625be6\", \"#edc709\", '#2ca02c', '#d62728', '#9467bd', '#17becf']", "color_list = [\"#625be6\", \"#edc709\", '#2ca02c', '#d62728', '#9467bd', '#17becf', '#ff7f0e']")
c5 = c5.replace("marker_list = [\"v\", \"D\", \"s\", \"o\", \"^\", \"P\"]", "marker_list = [\"v\", \"D\", \"s\", \"o\", \"^\", \"P\", \"X\"]")

to_insert_7 = "fill7, line7 = plot_mean_and_CI_with_marker(inds[:runmin_len_7], all_runmin_7_np_mean, all_runmin_7_np_ub, all_runmin_7_np_lb, color_mean=color_list[6], color_shading=color_list[6], marker=marker_list[6], marker_size=marker_size)\n"
c5 = c5.replace("plt.legend([line1, line3, line2,", to_insert_7 + "plt.legend([line1, line3, line7, line2,")
c5 = c5.replace("[\"Q-cUCB ($\sigma=0.1$)\", \"Q-cUCB ($\sigma=0.2$)\",", "[\"Q-cUCB ($\sigma=0.1$)\", \"Q-cUCB ($\sigma=0.2$)\", \"Q-cUCB (Real) ($\sigma=0.2$)\",")
c5 = c5.replace("prop={'size':12}", "prop={'size':11}")

nb['cells'][5]['source'] = [line + '\n' for line in c5.split('\n')[:-1]]

with open('/data/liuj35/quan_fuselage/analyze_discrete.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Done updating cell 5")
