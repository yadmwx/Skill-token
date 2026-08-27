#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

PYTHON=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python

MASKAWARE_RUN_ID="${MASKAWARE_RUN_ID:-configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-tokenfix56-prompt-taskonly-promptcompact-gripperhead1000-20260627}"
MASKAWARE_QUEUE_LOG="${MASKAWARE_QUEUE_LOG:-train_logs/queue_dit12_tokenfix56_prompt_taskonly_promptcompact_gripperhead1000_20260627.log}"
OUT_PREFIX="${OUT_PREFIX:-train_logs/dit_task_token_modes_maskaware_$(date +%Y%m%d_%H%M%S)}"
START_ANCHORINIT="${START_ANCHORINIT:-false}"
ANCHORINIT_MODE="${ANCHORINIT_MODE:-vision_prompt}"

mkdir -p train_logs

echo "[recovery] start $(date '+%F %T')"
echo "[recovery] maskaware_run_id=${MASKAWARE_RUN_ID}"

echo "[recovery] current eval successes from ${MASKAWARE_QUEUE_LOG}:"
grep -E "successes=|Total successes:" "$MASKAWARE_QUEUE_LOG" 2>/dev/null || true

echo "[recovery] active jobs:"
pgrep -af "run_libero_eval.py|finetune.py|tmp_queue_dit12" || true

echo "[recovery] gpu:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 || true

wait_for_idle_gpu() {
  while true; do
    local used util active_jobs
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    active_jobs=$(ps -eo args= | awk '/[f]inetune.py|[r]un_libero_eval.py/ {n++} END {print n + 0}')
    echo "[recovery] wait gpu_used=${used}MiB gpu_util=${util}% active_jobs=${active_jobs}"
    if [ "$active_jobs" -eq 0 ] && [ "$used" -lt 9000 ] && [ "$util" -lt 20 ]; then
      break
    fi
    sleep 60
  done
}

find_probe_checkpoint() {
  local step ckpt
  for step in 1000 500; do
    ckpt="outputs/${MASKAWARE_RUN_ID}--${step}_chkpt"
    if [ -d "$ckpt" ]; then
      echo "$ckpt"
      return 0
    fi
  done
  return 1
}

if ckpt="$(find_probe_checkpoint)"; then
  echo "[recovery] probing checkpoint=${ckpt}"
  wait_for_idle_gpu
  bash scripts/tmp_probe_dit_task_token_modes_20260627.sh "$ckpt" "$OUT_PREFIX"
else
  echo "[recovery] no mask-aware checkpoint found for ${MASKAWARE_RUN_ID}"
fi

if [ "$START_ANCHORINIT" = "true" ]; then
  echo "[recovery] starting anchor-init queue with DIT_TASK_TOKEN_MODE=${ANCHORINIT_MODE}"
  wait_for_idle_gpu
  nohup env DIT_TASK_TOKEN_MODE="$ANCHORINIT_MODE" bash scripts/tmp_queue_dit12_mlp_anchorinit_prompt_taskonly_20260627.sh \
    > "train_logs/nohup_mlp_anchorinit_${ANCHORINIT_MODE}_20260627.out" 2>&1 &
  echo "[recovery] anchor-init pid=$!"
else
  echo "[recovery] START_ANCHORINIT=false; not starting training"
fi

echo "[recovery] done $(date '+%F %T')"
