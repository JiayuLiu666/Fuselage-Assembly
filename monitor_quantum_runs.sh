#!/usr/bin/env bash

set -u

POLL_SECONDS="${1:-30}"
ALERT_LOG="${2:-/data/liuj35/quan_fuselage/nohup_logs/quantum_monitor_alerts.log}"
STATUS_DIR="/tmp/quantum_run_monitor"

mkdir -p "$(dirname "$ALERT_LOG")"
mkdir -p "$STATUS_DIR"
touch "$ALERT_LOG"

RUN_NAMES=(
  "obs_0p01_gpu1"
  "obs_0p04_gpu0"
)

RUN_PIDS=(
  "2928617"
  "2929844"
)

RUN_LOGS=(
  "/data/liuj35/quan_fuselage/nohup_logs/quantum_bo_continuous_constrain_quantum_obs_0p01_gpu1_20260331_135418.log"
  "/data/liuj35/quan_fuselage/nohup_logs/quantum_bo_continuous_constrain_quantum_obs_0p04_gpu0_20260331_135534.log"
)

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

append_alert() {
  local name="$1"
  local kind="$2"
  local pid="$3"
  local log_file="$4"
  {
    echo "[$(timestamp)] $name $kind pid=$pid"
    if [[ -f "$log_file" ]]; then
      echo "--- tail: $log_file ---"
      tail -n 60 "$log_file"
      echo "--- end tail ---"
    else
      echo "log file missing: $log_file"
    fi
    echo
  } >> "$ALERT_LOG"
}

while true; do
  for idx in "${!RUN_NAMES[@]}"; do
    name="${RUN_NAMES[$idx]}"
    pid="${RUN_PIDS[$idx]}"
    log_file="${RUN_LOGS[$idx]}"
    exit_flag="$STATUS_DIR/${name}.exit_alerted"
    err_flag="$STATUS_DIR/${name}.error_alerted"

    if ! ps -p "$pid" > /dev/null 2>&1; then
      if [[ ! -f "$exit_flag" ]]; then
        append_alert "$name" "PROCESS_EXITED" "$pid" "$log_file"
        touch "$exit_flag"
      fi
      continue
    fi

    if [[ -f "$log_file" ]]; then
      if grep -Eq "Traceback|CRITICAL -|ERROR |IndexError|TypeError|RuntimeError|Exception" "$log_file"; then
        if [[ ! -f "$err_flag" ]]; then
          append_alert "$name" "ERROR_DETECTED" "$pid" "$log_file"
          touch "$err_flag"
        fi
      fi
    fi
  done

  sleep "$POLL_SECONDS"
done
