import json

def get_cell(nb_file, keyword, outfile):
    with open(nb_file, 'r') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        source = ''.join(cell.get('source', []))
        if keyword in source:
            with open(outfile, 'w') as out:
                out.write(source)
            print("Wrote to " + outfile)
            return source

get_cell('analyze_discrete.ipynb', 'Running Minimum', 'cell_analyze.py')
