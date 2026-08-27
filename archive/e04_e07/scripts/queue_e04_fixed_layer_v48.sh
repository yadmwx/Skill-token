#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}
cd "$REPO_DIR"
mkdir -p experiment_results/skill_depth/E04_fixed_layers_v48
QUEUE_LOG=experiment_results/skill_depth/E04_fixed_layers_v48/queue_v48.log
DEVICE_TAG=${DEVICE_TAG:-v48-4090}
SCRATCH_ROOT=${SCRATCH_ROOT:-/root/autodl-tmp/skill_depth/E04_fixed_layers_v48/runs}

# Refuse to create a duplicate trainer/evaluator. This queue is only launched
# after the audited canonical process has exited.
if pgrep -af 'vla-scripts/finetune.py|experiments/robot/libero/run_libero_eval.py' >/dev/null; then
  echo "[$(date --iso-8601=seconds)] refuse start: trainer/evaluator already active" >> "$QUEUE_LOG"
  exit 4
fi

prepare_output_links() {
  local fixed=$1 seed=$2 run_id name target
  run_id="FlowMLP-ablation-routing_only-seed${seed}-${DEVICE_TAG}-fixed${fixed}"
  mkdir -p "$SCRATCH_ROOT"
  for name in "$run_id" "${run_id}--5000_chkpt" "${run_id}--10000_chkpt"; do
    if [ ! -e "outputs/$name" ] && [ ! -L "outputs/$name" ]; then
      target="$SCRATCH_ROOT/$name"
      mkdir -p "$target"
      ln -s "$target" "outputs/$name"
    fi
  done
}
echo "[$(date --iso-8601=seconds)] E04 ${DEVICE_TAG}-only queue start" >> "$QUEUE_LOG"

for spec in "1:7" "1:8" "1:9" "5:7" "5:8" "5:9" "9:7" "9:8" "9:9" "13:7" "13:8" "13:9" "24:7" "24:8" "24:9"; do
  fixed=${spec%%:*}
  seed=${spec##*:}
  status="experiment_results/flowmlp_ablation_routing_only_seed${seed}_${DEVICE_TAG}_fixed${fixed}.status"
  if [ -f "$status" ] && grep -q '^COMPLETE ' "$status"; then
    echo "[$(date --iso-8601=seconds)] skip complete fixed=$fixed seed=$seed" >> "$QUEUE_LOG"
    continue
  fi
  echo "[$(date --iso-8601=seconds)] start fixed=$fixed seed=$seed" >> "$QUEUE_LOG"
  prepare_output_links "$fixed" "$seed"
  FIXED_LAYER_INDEX="$fixed" SEED="$seed" DEVICE_TAG="$DEVICE_TAG" \
    bash scripts/launch_e04_fixed_layer_v48.sh >> "$QUEUE_LOG" 2>&1 || \
    echo "[$(date --iso-8601=seconds)] failed fixed=$fixed seed=$seed" >> "$QUEUE_LOG"
  echo "[$(date --iso-8601=seconds)] done fixed=$fixed seed=$seed" >> "$QUEUE_LOG"
done

/autodl-fs/data/skill_depth/envs/vla-flow/bin/python scripts/aggregate_e04_fixed_layer.py \
  --device_tag "$DEVICE_TAG" \
  --log_dir train_logs \
  --output_dir experiment_results/skill_depth/E04_fixed_layers_v48 \
  >> "$QUEUE_LOG" 2>&1 || echo "[$(date --iso-8601=seconds)] aggregate pending/failed" >> "$QUEUE_LOG"
echo "[$(date --iso-8601=seconds)] E04 ${DEVICE_TAG}-only queue end" >> "$QUEUE_LOG"
