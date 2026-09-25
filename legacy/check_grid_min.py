import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import numpy as np
from os import path
import sys

sys.path.append("/data/liuj35/quan_fuselage")
from fuselageENV import ClassicFuselageEnv
from quantum_cbo_discrete import sample_sobol_on_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = ClassicFuselageEnv(ip="129.161.91.97", obs_noise=0.01, min_shots=20)

file = "SolutionInputDP52.inp"
folder = path.join("FuselageActuators", "AnsysFiles", "Test")
filepath = path.join(folder, file)
input_filename = filepath

search_space = sample_sobol_on_grid(n=1681, seed=0, device=device)
env.reset(input_filename, seed=0)

init_np = search_space.cpu().numpy()
forces_init = (init_np * 1000).astype(np.float64)
u_init = forces_init @ env.surrogate.T
dev_init = (env.p_init_flat[None, :] + u_init) - env.p_target_flat[None, :]
mae_init = np.abs(dev_init).sum(axis=1)

print("Total space min MAE:", np.min(mae_init))

from joblib import load
from quantum_cbo_discrete import eval_tsai_wu
tsai_wu_model = load("surrogate_tsaiwu.joblib")
c_val = np.zeros(search_space.shape[0])
for i in range(search_space.shape[0]):
    c_val[i] = eval_tsai_wu(tsai_wu_model, search_space[i].cpu().numpy())

safe_mask = c_val >= 0
if np.any(safe_mask):
    print("Safe space min MAE:", np.min(mae_init[safe_mask]))
else:
    print("No safe points found!")
