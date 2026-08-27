#!/usr/bin/env bash
set -u

: "${EXPERIMENT_SPECS:?space-separated variant:seed entries are required}"
: "${REPO_DIR:?set REPO_DIR}"

cd "$REPO_DIR" || exit 2
mkdir -p train_logs experiment_results
QUEUE_LOG="train_logs/flowmlp_ablation_queue_${DEVICE_TAG:-gpu}.log"
echo "[$(date --iso-8601=seconds)] queue start: $EXPERIMENT_SPECS" | tee -a "$QUEUE_LOG"

failed=0
for spec in $EXPERIMENT_SPECS; do
  variant=${spec%%:*}
  seed=${spec##*:}
  echo "[$(date --iso-8601=seconds)] start $variant seed=$seed" | tee -a "$QUEUE_LOG"
  if VARIANT="$variant" SEED="$seed" bash scripts/run_flowmlp_ablation.sh; then
    echo "[$(date --iso-8601=seconds)] complete $variant seed=$seed" | tee -a "$QUEUE_LOG"
  else
    rc=$?
    failed=$((failed + 1))
    echo "[$(date --iso-8601=seconds)] failed $variant seed=$seed rc=$rc" | tee -a "$QUEUE_LOG"
  fi
done

python scripts/summarize_flowmlp_ablation.py train_logs \
  --output "experiment_results/flowmlp_ablation_${DEVICE_TAG:-gpu}.csv" | tee -a "$QUEUE_LOG"
echo "[$(date --iso-8601=seconds)] queue done failures=$failed" | tee -a "$QUEUE_LOG"
exit "$failed"
