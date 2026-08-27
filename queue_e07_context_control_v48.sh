#!/usr/bin/env bash
# E07's sole formal queue: seven missing configurations, sequentially on V48G.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}
cd "$REPO_DIR"
OUT=experiment_results/skill_depth/E07_context_control
QUEUE_LOG=$OUT/queue_v48.log
DEVICE_TAG=${DEVICE_TAG:-v48-4090}
SCRATCH_ROOT=${SCRATCH_ROOT:-/root/autodl-tmp/skill_depth/E07_context_control/runs}
mkdir -p "$OUT" "$SCRATCH_ROOT"

# Never attach a second trainer/evaluator to an active machine.
if pgrep -af 'vla-scripts/finetune.py|experiments/robot/libero/run_libero_eval.py' >/dev/null; then
  echo "[$(date --iso-8601=seconds)] refuse start: trainer/evaluator already active" >> "$QUEUE_LOG"
  exit 4
fi

prepare_output_links() {
  local variant=$1 seed=$2 run_id name target
  run_id="FlowMLP-ablation-${variant}-seed${seed}-${DEVICE_TAG}-fixed-1"
  for name in "$run_id" "${run_id}--5000_chkpt" "${run_id}--10000_chkpt"; do
    if [ ! -e "outputs/$name" ] && [ ! -L "outputs/$name" ]; then
      target="$SCRATCH_ROOT/$name"
      mkdir -p "$target"
      ln -s "$target" "outputs/$name"
    fi
  done
}

echo "[$(date --iso-8601=seconds)] E07 canonical queue start" >> "$QUEUE_LOG"
# seed8 R0/R2 are audit-validated legacy V48 runs.  These are the only missing
# runs; each attempt has its own output id and is never overwritten.
for spec in no_skill:7 no_skill:9 continuous_context:7 continuous_context:8 continuous_context:9 routing_only:7 routing_only:9; do
  variant=${spec%%:*}; seed=${spec##*:}
  run_id="FlowMLP-ablation-${variant}-seed${seed}-${DEVICE_TAG}-fixed-1"
  status="experiment_results/flowmlp_ablation_${variant}_seed${seed}_${DEVICE_TAG}_fixed-1.status"
  if [ -f "$status" ] && grep -q '^COMPLETE ' "$status"; then
    echo "[$(date --iso-8601=seconds)] skip complete variant=$variant seed=$seed" >> "$QUEUE_LOG"
    continue
  fi
  echo "[$(date --iso-8601=seconds)] start variant=$variant seed=$seed run_id=$run_id" >> "$QUEUE_LOG"
  prepare_output_links "$variant" "$seed"
  VARIANT="$variant" SEED="$seed" DEVICE_TAG="$DEVICE_TAG" \
    bash scripts/launch_e07_context_control_v48.sh >> "$QUEUE_LOG" 2>&1
  echo "[$(date --iso-8601=seconds)] complete variant=$variant seed=$seed" >> "$QUEUE_LOG"
done
echo "[$(date --iso-8601=seconds)] E07 canonical queue end" >> "$QUEUE_LOG"
