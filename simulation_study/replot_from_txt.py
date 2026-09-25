import sys, os, re, ast, argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--txt', default='results_20260521_111827.txt')
args = parser.parse_args()

with open(args.txt) as f:
    text = f.read()

# --- Parse metadata (SEED / GRID / N_INIT / noise) ---
meta = {}
m = re.search(r'SEED=(\d+)\s+GRID=(\d+).*N_INIT=(\d+)\s+ORACLE_BUDGET=(\d+)\s+OBJ_NOISE=([\d.]+)', text)
if m:
    meta = dict(seed=int(m.group(1)), grid=int(m.group(2)),
                n_init=int(m.group(3)), budget=int(m.group(4)), noise=float(m.group(5)))
print('Parsed metadata:', meta)

# --- Parse the cumulative-regret curves ---
# Lines look like:  "  <label>: [<floats>]"
curves = {}
for label, arr in re.findall(r'^\s{2}([\w\- ()]+):\s*(\[[^\]]*\])', text, flags=re.MULTILINE):
    curves[label.strip()] = np.array(ast.literal_eval(arr))

print('Parsed curves:', {k: len(v) for k, v in curves.items()})

# Map the txt labels -> canonical display names used in the figure
NAME_MAP = {
    'C-Safe BO'        : 'C-Safe BO',
    'Q-Safe BO'        : 'Q-Safe BO',
    'Quantum Safe (R)' : 'Q-Safe BO (Real)',
    'Unconstrained BO' : 'Unconstrained BO',
    'BO-ACL'           : 'BO-ACL',
}

COLORS = {
    'C-Safe BO'        : '#2196F3',
    'Q-Safe BO'        : '#9C27B0',
    'Q-Safe BO (Real)' : '#E91E63',
    'Unconstrained BO' : '#FF5722',
    'BO-ACL'           : '#4CAF50',
}
STYLES = {
    'C-Safe BO'        : '-',
    'Q-Safe BO'        : '--',
    'Q-Safe BO (Real)' : '--',
    'Unconstrained BO' : ':',
    'BO-ACL'           : '-.',
}

# Plot order (unconstrained BO included)
ORDER = ['C-Safe BO', 'Q-Safe BO', 'Quantum Safe (R)', 'Unconstrained BO', 'BO-ACL']

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
_grid = meta.get('grid', '?')
_noise = meta.get('noise', '?')
fig.suptitle(
    f'Cumulative Regret Comparison  (grid={_grid}², noise={_noise})',
    fontsize=13, fontweight='bold'
)

for txt_label in ORDER:
    if txt_label not in curves:
        print(f'  [skip] {txt_label} not found in txt')
        continue
    disp = NAME_MAP[txt_label]
    r = curves[txt_label]
    x = np.arange(1, len(r) + 1)
    ax.plot(x, r, label=f'{disp:16s} (final: {r[-1]:.2f})',
            color=COLORS[disp], ls=STYLES[disp], lw=2)

ax.set_xlabel('Iterations', fontsize=11)
ax.set_ylabel('Cumulative Regret', fontsize=11)
ax.set_title('Cumulative Regret vs Iterations')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_regret.png', dpi=600, bbox_inches='tight')
print('Saved -> comparison_regret.png')
