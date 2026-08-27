#!/usr/bin/env bash
# Sole clean E07 queue: R0/R1/R2 x seeds 7/8/9, sequentially on A100 40GB.
set -euo pipefail

echo "DISABLED: E07 was removed from the experiment queue by user instruction on 2026-07-27."
exit 0

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}
E07_ROOT=${E07_ROOT:-/root/autodl-tmp/e07_context_control_clean}
QUEUE_LOG=${QUEUE_LOG:-$E07_ROOT/queue.log}
DEVICE_TAG=${DEVICE_TAG:-a100-pcie40gb}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_STEPS=${MAX_STEPS:-10000}
PROTOCOL_TAG=${PROTOCOL_TAG:-clean-vlmbase-${MAX_STEPS}updates-b${BATCH_SIZE}ga${GRAD_ACCUM}}

mkdir -p "$E07_ROOT"
cd "$REPO_DIR"

exec 9>"$E07_ROOT/canonical_queue.lock"
if ! flock -n 9; then
  echo "[$(date --iso-8601=seconds)] refuse start: canonical queue lock is held" >>"$QUEUE_LOG"
  exit 4
fi

# Refuse to share the GPU with any trainer, evaluator, or unrelated compute job.
if pgrep -af 'vla-scripts/finetune.py|experiments/robot/libero/run_libero_eval.py' >/dev/null; then
  echo "[$(date --iso-8601=seconds)] refuse start: E07 trainer/evaluator/queue already active" >>"$QUEUE_LOG"
  exit 4
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "[$(date --iso-8601=seconds)] refuse start: GPU has an active compute process" >>"$QUEUE_LOG"
  exit 4
fi

printf '[%s] clean E07 canonical queue start protocol=%s\n' \
  "$(date --iso-8601=seconds)" "$PROTOCOL_TAG" >>"$QUEUE_LOG"

for spec in \
  no_skill:7 no_skill:8 no_skill:9 \
  continuous_context:7 continuous_context:8 continuous_context:9 \
  routing_only:7 routing_only:8 routing_only:9
do
  variant=${spec%%:*}
  seed=${spec##*:}
  status_glob="$E07_ROOT/status/flowmlp_ablation_${variant}_seed${seed}_${DEVICE_TAG}_fixed-1_${PROTOCOL_TAG}_attempt"*
  if compgen -G "$status_glob" >/dev/null && grep -l '^COMPLETE ' $status_glob >/dev/null 2>&1; then
    printf '[%s] skip complete variant=%s seed=%s\n' \
      "$(date --iso-8601=seconds)" "$variant" "$seed" >>"$QUEUE_LOG"
    continue
  fi

  attempt_id=$(date -u +%Y%m%dT%H%M%SZ)
  run_id="FlowMLP-E07-${variant}-seed${seed}-${DEVICE_TAG}-${PROTOCOL_TAG}-attempt${attempt_id}"
  printf '[%s] start variant=%s seed=%s attempt=%s\n' \
    "$(date --iso-8601=seconds)" "$variant" "$seed" "$attempt_id" >>"$QUEUE_LOG"

  VARIANT="$variant" SEED="$seed" ATTEMPT_ID="$attempt_id" \
    RUN_ID_OVERRIDE="$run_id" DEVICE_TAG="$DEVICE_TAG" \
    BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
    MAX_STEPS="$MAX_STEPS" PROTOCOL_TAG="$PROTOCOL_TAG" \
    E07_ROOT="$E07_ROOT" \
    bash scripts/launch_e07_context_control_a100_clean.sh >>"$QUEUE_LOG" 2>&1

  printf '[%s] complete variant=%s seed=%s attempt=%s\n' \
    "$(date --iso-8601=seconds)" "$variant" "$seed" "$attempt_id" >>"$QUEUE_LOG"
done

printf '[%s] clean E07 canonical queue end\n' "$(date --iso-8601=seconds)" >>"$QUEUE_LOG"
