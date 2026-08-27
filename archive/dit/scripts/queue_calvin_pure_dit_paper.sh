#!/usr/bin/env bash
# Run the paper-comparable CALVIN ablation sequentially on one GPU:
# identical pure-DiT baseline and latent-skill-token training, then the
# official 1,000-chain ABC->D evaluation for each final checkpoint.
# The short paper smoke budget is deliberate: increase to 400005 only after
# an early result justifies the extra compute.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation}
LOG_DIR="$REPO_DIR/train_logs"
QUEUE_LOG="$LOG_DIR/queue_calvin_pure_dit_paper.log"
mkdir -p "$LOG_DIR"

find_checkpoint() {
  local variant=$1
  find "$REPO_DIR/outputs" /data/xiaguanxiao/archives/calvin_pure_dit_intermediates -maxdepth 1 -type d \
    -name "*CALVIN-ABC-pureDIT-${variant}-skilltoken-*--20000_chkpt" \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_variant() {
  local variant=$1
  local checkpoint
  checkpoint=$(find_checkpoint "$variant")
  if test -n "$checkpoint"; then
    echo "[$(date '+%F %T')] reusing existing ${variant} 20k checkpoint: ${checkpoint}" >> "$QUEUE_LOG"
  else
    echo "[$(date '+%F %T')] starting formal ${variant} training" >> "$QUEUE_LOG"
    bash "$REPO_DIR/scripts/train_calvin_pure_dit_skill_ablation.sh" "$variant" 20000 >> "$QUEUE_LOG" 2>&1
    checkpoint=$(find_checkpoint "$variant")
  fi
  test -n "$checkpoint" || { echo "Missing final ${variant} checkpoint" >> "$QUEUE_LOG"; exit 1; }

  echo "[$(date '+%F %T')] starting 1,000-chain ${variant} CALVIN evaluation" >> "$QUEUE_LOG"
  bash "$REPO_DIR/scripts/eval_calvin_pure_dit_standard.sh" "$checkpoint" "$variant" >> "$QUEUE_LOG" 2>&1
  echo "[$(date '+%F %T')] finished formal ${variant}" >> "$QUEUE_LOG"
}

run_variant baseline
run_variant skill
echo "[$(date '+%F %T')] full CALVIN pure-DiT paper run complete" >> "$QUEUE_LOG"
