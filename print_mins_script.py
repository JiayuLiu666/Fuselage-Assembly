import torch
import glob
from pathlib import Path
import os
import numpy as np

def classify_training_file(file_path):
    name = os.path.basename(file_path)
    if "quan_training_data" in name:
        return "Q-Safe-Set-UCB"
    elif "classic_safeset_training_data" in name:
        return "C-Safe-Set-UCB"
    elif "acl_training_data" in name:
        return "C-ACL"
    return "Unknown"

files = glob.glob("**/*training_data_*_.pth", recursive=True)

stats = {}

for f in files:
    method = classify_training_file(f)
    if method == "Unknown":
        continue
    
    try:
        data = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
        
        tr = data.get('true_response', data.get('Y_true'))
        queries = data.get('queries', data.get('stages', data.get('num_oracle_queries')))
        unc = data.get('uncertainty', data.get('eps', data.get('eps_list')))
        
        if tr is None or queries is None:
            continue
            
        tr = tr.numpy().flatten()
        queries = queries.numpy().flatten()
        if unc is not None:
            if isinstance(unc, torch.Tensor):
                unc = unc.numpy().flatten()
            elif isinstance(unc, list):
                unc = np.array(unc)
        
        active = queries > 0
        tr_active = tr[active]
        queries_active = queries[active]
        
        if len(tr_active) == 0:
            continue
            
        cumsum_queries = np.cumsum(queries_active)
        
        min_idx = np.argmin(tr_active)
        min_val = tr_active[min_idx]
        cum_queries_at_min = cumsum_queries[min_idx]
        
        precision_at_min = unc[active][min_idx] if unc is not None else "N/A"
        
        if method not in stats:
            stats[method] = []
            
        stats[method].append({
            'min_val': min_val,
            'cum_queries': cum_queries_at_min,
            'precision': precision_at_min,
            'total_queries': cumsum_queries[-1],
            'total_steps': np.sum(active),
            'filename': f
        })
        
    except Exception as e:
        pass

for method, trials in stats.items():
    print(f"=== {method} ===")
    avg_min_val = np.mean([t['min_val'] for t in trials])
    avg_cum_q = np.mean([t['cum_queries'] for t in trials])
    prcs = [t['precision'] for t in trials if t['precision'] != "N/A"]
    avg_prec = np.mean(prcs) if prcs else "N/A"
    print(f"Average Minimum Found: {avg_min_val:.6f}")
    print(f"Average Queries to Min: {avg_cum_q:.1f}")
    if isinstance(avg_prec, float):
         print(f"Average Precision at Min: {avg_prec:.6f}")
    else:
         print(f"Average Precision at Min: {avg_prec}")
    print(f"Total Trials: {len(trials)}")
    
    best_trial = min(trials, key=lambda x: x['min_val'])
    print(f"Best Trial Min: {best_trial['min_val']:.6f} at query {best_trial['cum_queries']:.0f} (Precision: {best_trial['precision']})")
    print("")
