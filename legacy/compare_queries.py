import torch
import glob

q_files = glob.glob('Experiments_constraint_continuous/*/exp_set_0/*quan_training_data_0_.pth')
c_files = glob.glob('Experiments_constraint_continuous/*/exp_set_0/*classic_safeset_training_data_0_.pth')

quan_file = q_files[0]
class_file = c_files[0]

q_data = torch.load(quan_file, map_location=torch.device('cpu'), weights_only=False)
c_data = torch.load(class_file, map_location=torch.device('cpu'), weights_only=False)

q_q = q_data['queries'].numpy()
c_q = c_data['queries'].numpy()

print(f"Quantum queries head 15: {q_q[:15].flatten()}")
print(f"Classic queries head 15: {c_q[:15].flatten()}")
print(f"Quantum mean queries per step: {q_q[q_q>0].mean()}")
print(f"Classic mean queries per step: {c_q[c_q>0].mean()}")
