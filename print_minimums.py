import torch
import glob
from pathlib import Path
import os
import numpy as np

def classify_training_file(file_path):
    """Classify the method based on the filename."""
    name = os.path.basename(file_path)
    if "quan_training_data" in name:
        return "Q-Safe-Set-UCB"
    elif "classic_safeset_training_data" in name:
        return "C-Safe-Set-UCB"
    elif "acl_training_data" in name:
        return "C-ACL"
    return "Unknown"

if __name__ == "__main__":
    # Define the directory to search
    data_dir = Path("Experiments_constraint_continuous")
    
    if not data_dir.exists():
        print(f"Directory {data_dir} not found!")
        exit(1)
        
    print(f"Scanning for data in: {data_dir}...\n")
    
    # Locate all .pth files recursively
    files = glob.glob(str(data_dir / "**" / "*training_data_*_.pth"), recursive=True)
    
    stats = {}

    for f in files:
        method = classify_training_file(f)
        if method == "Unknown":
            continue
        
        try:
            # Load PyTorch object directly from CPU
            data = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
            
            # Extract attributes handling backward compatibility with older keys
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
            
            # Only consider rows where actual queries occurred
            active = (queries > 0)
            tr_active = tr[active]
            queries_active = queries[active]
            
            if len(tr_active) == 0:
                continue
                
            cumsum_queries = np.cumsum(queries_active)
            
            # Identify the absolute minimum performance threshold met
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
                'filename': f
            })
            
        except Exception as e:
            print(f"Failed to process {f}: {e}")

    # Display best overall metric for each method
    for method, trials in stats.items():
        print(f"=============================")
        print(f" {method} (Total Trials: {len(trials)})")
        print(f"=============================")
        
        # Calculate averages
        avg_min_val = np.mean([t['min_val'] for t in trials])
        avg_cum_q = np.mean([t['cum_queries'] for t in trials])
        prcs = [t['precision'] for t in trials if t['precision'] != "N/A"]
        
        print(f"Average Minimum Found:  {avg_min_val:.6f}")
        print(f"Average Queries to Min: {avg_cum_q:.1f}")
        
        # Determine the best single trial
        best_trial = min(trials, key=lambda x: x['min_val'])
        
        print(f"\n--- Best Trial ---")
        print(f"Best Minimum f(x):      {best_trial['min_val']:.6f}")
        print(f"Queries Reached At:     {best_trial['cum_queries']:.0f}")
        if isinstance(best_trial['precision'], (float, int, np.floating)):
            print(f"Precision at Minimum:   {best_trial['precision']:.6f}")
        else:
            print(f"Precision at Minimum:   {best_trial['precision']}")
        print(f"File Source:            {best_trial['filename']}\n")
