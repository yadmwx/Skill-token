#!/usr/bin/env bash
set -u

REPO_DIR=/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
cd "$REPO_DIR"
LOG=experiment_results/skill_depth/E04_fixed_layers/resume_wait_a100.log
mkdir -p "$(dirname "$LOG")"
echo "[$(date --iso-8601=seconds)] waiting for A100 memory release" >> "$LOG"

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "${used:-81920}" -lt 10000 ]; then
    echo "[$(date --iso-8601=seconds)] A100 memory free (${used} MiB); resuming E04 queue" >> "$LOG"
    exec bash scripts/queue_e04_fixed_layer_a100.sh >> "$LOG" 2>&1
  fi
  echo "[$(date --iso-8601=seconds)] A100 still occupied (${used} MiB); retrying in 60s" >> "$LOG"
  sleep 60
done
