#!/usr/bin/env bash
# Wait for the CALVIN RLDS download, then verify both pure-DiT variants can train.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation}
DATA_DIR=${DATA_DIR:-/data/xiaguanxiao/rlds/calvin_abc/1.0.0}
LOG_DIR="$REPO_DIR/train_logs"
QUEUE_LOG="$LOG_DIR/queue_calvin_pure_dit_smoke.log"
mkdir -p "$LOG_DIR"

count_files() {
  find -L "$DATA_DIR" -maxdepth 1 -type f | grep -c 'calvin_abc-.*\.tfrecord' || true
}

while true; do
  count=$(count_files)
  if [ "$count" -ge 544 ] && [ -f "$DATA_DIR/dataset_info.json" ]; then
    break
  fi
  echo "[$(date '+%F %T')] waiting for CALVIN RLDS: files=${count}/544" >> "$QUEUE_LOG"
  sleep 120
done

while pgrep -f 'vla-scripts/finetune.py' >/dev/null; do
  echo "[$(date '+%F %T')] another finetune job is using the A100; delaying CALVIN smoke" >> "$QUEUE_LOG"
  sleep 300
done

echo "[$(date '+%F %T')] data ready; starting baseline smoke" >> "$QUEUE_LOG"
bash "$REPO_DIR/scripts/train_calvin_pure_dit_skill_ablation.sh" baseline 20 >> "$QUEUE_LOG" 2>&1
echo "[$(date '+%F %T')] baseline smoke finished; starting skill smoke" >> "$QUEUE_LOG"
bash "$REPO_DIR/scripts/train_calvin_pure_dit_skill_ablation.sh" skill 20 >> "$QUEUE_LOG" 2>&1
echo "[$(date '+%F %T')] both CALVIN pure-DiT smoke runs finished" >> "$QUEUE_LOG"
