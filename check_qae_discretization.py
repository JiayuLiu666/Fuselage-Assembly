import numpy as np
stddev = 0.2
mean = 0.023
low = mean - 3 * stddev
high = mean + 3 * stddev 

# 6 qubits = 64 bins
num_qubits = 6
bins = 2**num_qubits
bin_width = (high - low) / (bins - 1)

print(f"Domain: [{low:.4f}, {high:.4f}]")
print(f"Bin width: {bin_width:.4f}")
print(f"Max discretization error: {bin_width / 2:.4f}")
