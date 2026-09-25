import json

def read_cells(nb_file):
    with open(nb_file, 'r') as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            print(f"--- Cell {i} ---")
            print(source)

read_cells('compare_cumulative_regret.ipynb')
