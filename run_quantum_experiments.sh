#!/bin/bash
# Experiment: vary --initial_num_points to find best setting
# Fixed: eps_max=0.04, B=0.5, n_constraint_init=10, obs_noise=0.1^2=0.01

set -e

for N in 5 10 15 20; do
    echo "=============================================="
    echo "  Running with --initial_num_points=$N"
    echo "=============================================="
    python -u quantum_safeset_discrete.py \
        --initial_num_points $N \
        --obs_noise 0.01 \
        --B 0.5 \
        --n_constraint_init 10 \
        --eps_max 0.04 \
        2>&1 | tee "experiment_init_${N}.log"
    echo ""
done

echo "===== All experiments done ====="
# Quick summary: extract best f(x) per trial from each log
for N in 5 10 15 20; do
    echo "--- init_num_points=$N ---"
    grep "Trial" "experiment_init_${N}.log"
done
