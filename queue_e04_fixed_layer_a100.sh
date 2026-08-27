#!/usr/bin/env bash
set -u

REPO_DIR=/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
cd "$REPO_DIR"
mkdir -p experiment_results/skill_depth/E04_fixed_layers
QUEUE_LOG=experiment_results/skill_depth/E04_fixed_layers/queue_a100.log
echo "[$(date --iso-8601=seconds)] E04 A100-only queue start" >> "$QUEUE_LOG"

# Pre-registered representative layers for the five Figure 3(b) depth groups.
# The 1-based convention matches E02/E03 artifacts; final is the last probed layer (16).
for spec in "1:8" "1:9" "5:7" "5:8" "5:9" "9:7" "9:8" "9:9" "13:7" "13:8" "13:9" "16:7" "16:8" "16:9"; do
  fixed=${spec%%:*}
  seed=${spec##*:}
  status="experiment_results/flowmlp_ablation_routing_only_seed${seed}_a100_fixed${fixed}.status"
  if [ -f "$status" ] && grep -q '^COMPLETE ' "$status"; then
    echo "[$(date --iso-8601=seconds)] skip complete fixed=$fixed seed=$seed" >> "$QUEUE_LOG"
    continue
  fi
  echo "[$(date --iso-8601=seconds)] start fixed=$fixed seed=$seed" >> "$QUEUE_LOG"
  FIXED_LAYER_INDEX="$fixed" SEED="$seed" bash scripts/launch_e04_fixed_layer_a100.sh >> "$QUEUE_LOG" 2>&1 || \
    echo "[$(date --iso-8601=seconds)] failed fixed=$fixed seed=$seed" >> "$QUEUE_LOG"
  echo "[$(date --iso-8601=seconds)] done fixed=$fixed seed=$seed" >> "$QUEUE_LOG"
done
python scripts/aggregate_e04_fixed_layer.py \
  --log_dir train_logs \
  --output_dir experiment_results/skill_depth/E04_fixed_layers \
  >> "$QUEUE_LOG" 2>&1 || echo "[$(date --iso-8601=seconds)] aggregate pending/failed" >> "$QUEUE_LOG"
echo "[$(date --iso-8601=seconds)] E04 A100-only queue end" >> "$QUEUE_LOG"
