#!/usr/bin/env bash

set -euo pipefail

WAIT_USER="${WAIT_USER:-xyjh}"
WAIT_PATTERN="${WAIT_PATTERN:-scripts/run_train_v2_stage1.sh}"
POLL_SECONDS="${POLL_SECONDS:-300}"

while pgrep -u "$WAIT_USER" -f "$WAIT_PATTERN" >/dev/null; do
  echo "[queue] waiting for $WAIT_USER:$WAIT_PATTERN"
  sleep "$POLL_SECONDS"
done

while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; do
  echo "[queue] waiting for the GPU to become idle"
  sleep 60
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export CONDA_PREFIX="${CONDA_PREFIX:-/home/xiaguanxiao/miniconda3/envs/vla-flow}"
export PATH="$CONDA_PREFIX/bin:/usr/local/bin:/usr/bin:/bin"

exec ./train.sh libero_object_no_noops \
  --action_head_type DIT \
  --data_root_dir data/libero \
  --max_steps 1000 \
  --save_freq 1000 \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --dit_num_blocks 12 \
  --dit_num_inference_steps 5 \
  --dit_num_inference_samples 8 \
  --dit_supervised_anchor_weight 1.0 \
  --dit_anchor_blend 0.0 \
  --use_adaptive_bridge False \
  --bridge_mode adaptive \
  --dit_anchor_gripper_weight 1.0 \
  --dit_anchor_gripper_bce_weight 0.2 \
  --dit_detach_flow_conditioning True \
  --run_id_note VLA-Adapter-DIT12-residual-anchor-object-1000-queued-20260620
