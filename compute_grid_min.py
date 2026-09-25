import torch
import numpy as np
from joblib import load

# recreate grid
import sys
sys.path.append("/data/liuj35/quan_fuselage")
from quantum_cbo_discrete import sample_sobol_on_grid

search_space = sample_sobol_on_grid(n=1681, seed=0, device=torch.device('cpu'))

# load target
p_init = np.load('/data/liuj35/quan_fuselage/FuselageActuators/Shapes/Test/SolutionInputDP52.npy')
p_target = np.load('/data/liuj35/quan_fuselage/FuselageActuators/Shapes/Test/SolutionInputDP53.npy')

p_init_flat = p_init[:,0:2].flatten()
p_target_flat = p_target[:,0:2].flatten()
n_dev = len(p_init_flat) # 354 points * 2 coords? wait, in env n_dev is len(deviations) which is 354. But dev is already 354. n_dev is 354.

# load surrogate
surrogate = load('/data/liuj35/quan_fuselage/surrogate_likeDu_v22.joblib').coef_

forces_init = (search_space.numpy() * 1000).astype(np.float32)
angles = np.linspace(12, -192, 18)
forces_Y = forces_init * np.cos(np.deg2rad(angles))
forces_Z = forces_init * np.sin(np.deg2rad(angles))

u_init = np.dot(surrogate, forces_init.T).T
dev_init = (p_init_flat[None, :] + u_init) - p_target_flat[None, :]
mae_init = np.sum(np.abs(dev_init), axis=1) / n_dev

print("Absolute lowest true MAE across the entire 1681 discrete points is:")
print(np.min(mae_init))
