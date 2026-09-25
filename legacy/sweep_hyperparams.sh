#!/bin/bash
# Hyperparameter sweep for quantum_safeset_discrete.py
# Uses classic_safeset_discrete.py as fast proxy (identical GP logic)
# Run from conda quantum env: conda activate quantum && bash sweep_hyperparams.sh

set -e
SCRIPT="classic_safeset_discrete.py"
RESULTS_DIR="sweep_results"
mkdir -p "$RESULTS_DIR"

BUDGET=20000

echo "========================================"
echo "Hyperparameter Sweep ($(date))"
echo "========================================"

run_config() {
    local noise=$1 ls=$2 B=$3 t0=$4 lam_p=$5 init=$6 c_init=$7
    local tag="n${noise}_ls${ls}_B${B}_t${t0}_p${lam_p}_i${init}_c${c_init}"
    local outfile="${RESULTS_DIR}/${tag}.log"
    
    echo ""
    echo ">>> noise=$noise, obj_ls=$ls, B=$B, t0=$t0, lam_p=$lam_p, init=$init, c_init=$c_init"
    
    python -u $SCRIPT \
        --obs_noise $noise \
        --obj_ls $ls \
        --B $B \
        --t0 $t0 \
        --lam_p $lam_p \
        --initial_num_points $init \
        --n_constraint_init $c_init \
        --query_budget $BUDGET \
        2>&1 | tee "$outfile"
    
    echo "--- Saved to $outfile ---"
}

# ══════════════════════════════════════════════
# Noise = 0.01 (obs_noise = 0.1²)
# ══════════════════════════════════════════════
echo ""
echo "===== NOISE = 0.01 ====="

# Baseline
run_config 0.01 0.20 1.0 5.0 2.0 1 1
run_config 0.01 0.50 1.0 5.0 2.0 1 1

# More init points
run_config 0.01 0.20 1.0 5.0 2.0 5 5
run_config 0.01 0.50 1.0 5.0 2.0 5 5

# Slower lambda decay
run_config 0.01 0.20 1.0 10.0 2.0 5 5
run_config 0.01 0.50 1.0 10.0 2.0 5 5
run_config 0.01 0.20 1.0 20.0 2.0 5 5
run_config 0.01 0.50 1.0 20.0 2.0 5 5

# Gentler decay power
run_config 0.01 0.20 1.0 10.0 1.0 5 5
run_config 0.01 0.50 1.0 10.0 1.0 5 5

# Different B
run_config 0.01 0.20 3.0 10.0 2.0 5 5
run_config 0.01 0.50 0.5 10.0 2.0 5 5

# Best combo candidates
run_config 0.01 0.20 1.0 15.0 1.5 5 5
run_config 0.01 0.30 1.0 10.0 2.0 5 5

# ══════════════════════════════════════════════
# Noise = 0.04 (obs_noise = 0.2²)
# ══════════════════════════════════════════════
echo ""
echo "===== NOISE = 0.04 ====="

# Baseline
run_config 0.04 0.20 1.0 5.0 2.0 1 1
run_config 0.04 0.50 1.0 5.0 2.0 1 1

# More init points
run_config 0.04 0.20 1.0 5.0 2.0 5 5
run_config 0.04 0.50 1.0 5.0 2.0 5 5

# Slower lambda decay
run_config 0.04 0.20 1.0 10.0 2.0 5 5
run_config 0.04 0.50 1.0 10.0 2.0 5 5
run_config 0.04 0.20 1.0 20.0 2.0 5 5
run_config 0.04 0.50 1.0 20.0 2.0 5 5

# Gentler decay power
run_config 0.04 0.20 1.0 10.0 1.0 5 5
run_config 0.04 0.50 1.0 10.0 1.0 5 5

# Different B
run_config 0.04 0.20 3.0 10.0 2.0 5 5
run_config 0.04 0.50 0.5 10.0 2.0 5 5

# Best combo candidates
run_config 0.04 0.50 0.8 15.0 1.5 5 5
run_config 0.04 0.30 1.0 10.0 2.0 5 5

echo ""
echo "========================================"
echo "SWEEP COMPLETE ($(date))"
echo "Results in: $RESULTS_DIR/"
echo "========================================"

# Parse and summarize results
echo ""
echo "SUMMARY TABLE:"
echo "noise  obj_ls  B    t0    lam_p init c_init  | converged (f(x)≤0.08) / total"
echo "--------------------------------------------------------------------------"
for f in ${RESULTS_DIR}/*.log; do
    tag=$(basename "$f" .log)
    conv=$(grep -c "f(x): 0.07" "$f" 2>/dev/null || echo 0)
    trials=$(grep -c "\[Trial" "$f" 2>/dev/null || echo 0)
    echo "$tag  →  $conv / $trials converged"
done
