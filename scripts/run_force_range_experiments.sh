#!/usr/bin/env bash
# Run continuous constrained experiments at actuator_count=8 for narrower force ranges.
# Force 1000 already exists in Experiments_constraint_continuous.
# Runs the 3 methods sequentially per force scale (each script uses all GPUs).
set -u

cd "$(dirname "$0")/.."   # run from the repository root
source "$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")/etc/profile.d/conda.sh"
conda activate quantum

LOGDIR="force_range_logs"
mkdir -p "$LOGDIR"

ACTUATORS=8
SCRIPTS=(quantum_safeset_continuous.py classic_safeset_continuous.py classic_acl_continuous.py)

for FS in 500 200; do
  for SCRIPT in "${SCRIPTS[@]}"; do
    NAME="${SCRIPT%.py}_force_${FS}"
    echo "===== $(date '+%F %T')  START  $NAME ====="
    python "$SCRIPT" --actuator_count "$ACTUATORS" --force_scale "$FS" \
        > "$LOGDIR/${NAME}.log" 2>&1
    echo "===== $(date '+%F %T')  DONE   $NAME  (exit $?) ====="
  done
done

echo "===== $(date '+%F %T')  ALL FORCE-RANGE EXPERIMENTS COMPLETE ====="
