import sys

with open('simulated_study/compare_safe_methods.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'from quantum_safe_bo  import run_quantum_safe_bo_simulation' in line:
        new_lines.append(line)
        new_lines.append('from quantum_safe_bo_real import run_quantum_safe_bo_simulation as run_quantum_safe_bo_simulation_real\n')
        continue
    
    if "print(f'\\nQuantum Safe BO done | total_oracle={res_qsafe['total_oracle_queries']} | '" in line:
        new_lines.append(line)
        # We need to skip the next line which is the simple_regret print
        continue
        
    if "f\"simple_regret={res_qsafe['simple_regret']:.4f}\")\n" in line:
        new_lines.append(line)
        # Now append the new method call
        new_lines.append('''
# ============================================================
# 2b. Quantum Safe BO (Real Hardware)
# ============================================================
res_qsafe_real = run_quantum_safe_bo_simulation_real(
    mode=MODE, xi=XI, grid_size=GRID_SIZE,
    n_init=N_INIT, oracle_budget=ORACLE_BUDGET,
    obj_noise_std=OBJ_NOISE, con_noise_std=CON_NOISE,
    beta_f=1.0, beta_c=3.0, lam0=1,
    seed=SEED, init_idx=INIT_IDX.copy(),
    M_rff=256, lam_rff=1.0, lengthscale_rff=np.array([1.0, 0.25]), v_kernel_rff=1.0,
    use_quantum_query=True,
)
print(f"\\nQuantum Safe BO (Real) done | total_oracle={res_qsafe_real.get('total_oracle_queries', 'N/A')} | "
      f"simple_regret={res_qsafe_real.get('simple_regret', float('nan')):.4f}")
''')
        continue

    if "r_qsafe = quantum_cumu_regret(res_qsafe)[:budget]" in line:
        new_lines.append(line)
        new_lines.append("r_qsafe_real = quantum_cumu_regret(res_qsafe_real)[:budget]\n")
        continue

    if "x_qsafe = np.arange(1, len(r_qsafe) + 1)" in line:
        new_lines.append(line)
        new_lines.append("x_qsafe_real = np.arange(1, len(r_qsafe_real) + 1)\n")
        continue
        
    if "print(f'  Quantum Safe BO  : {r_qsafe[-1]:.4f}  (oracle calls={len(r_qsafe)})')" in line:
        new_lines.append(line)
        new_lines.append("print(f'  Quantum Safe BO (R): {r_qsafe_real[-1]:.4f}  (oracle calls={len(r_qsafe_real)})')\n")
        continue

    if "print(f'  Quantum Safe BO  : {safe_rate(res_qsafe):.3f}')" in line:
        new_lines.append(line)
        new_lines.append("print(f'  Quantum Safe BO (R): {safe_rate(res_qsafe_real):.3f}')\n")
        continue

    if "'Quantum Safe BO' : '#9C27B0'," in line:
        new_lines.append(line)
        new_lines.append("    'Quantum Safe BO (Real)' : '#E91E63',\n") # Pink-red
        continue

    if "'Quantum Safe BO' : '--'," in line:
        new_lines.append(line)
        new_lines.append("    'Quantum Safe BO (Real)' : '--',\n") # also dashed
        continue

    if "ax.plot(x_qsafe, r_qsafe, label='Quantum Safe BO'" in line:
        new_lines.append(line)
        new_lines.append("ax.plot(x_qsafe_real, r_qsafe_real, label='Quantum Safe BO (Real)', color=COLORS['Quantum Safe BO (Real)'], ls=STYLES['Quantum Safe BO (Real)'], lw=2)\n")
        continue

    if "fig, axes = plt.subplots(3, 4, figsize=(20, 13))" in line:
        new_lines.append("fig, axes = plt.subplots(3, 5, figsize=(25, 13))\n")
        continue
        
    if "('Quantum Safe BO',  res_qsafe, COLORS['Quantum Safe BO'])," in line:
        new_lines.append(line)
        new_lines.append("    ('Quantum Safe BO (Real)',  res_qsafe_real, COLORS['Quantum Safe BO (Real)']),\n")
        continue
        
    if "for label, res in [('Safe BO', res_safe), ('Quantum Safe BO', res_qsafe)," in line:
        new_lines.append("for label, res in [('Safe BO', res_safe), ('Quantum Safe BO', res_qsafe), ('Quantum Safe BO (R)', res_qsafe_real),\n")
        continue

    new_lines.append(line)

with open('simulated_study/compare_safe_methods.py', 'w') as f:
    f.writelines(new_lines)
