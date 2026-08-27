#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
mkdir -p experiments/logs train_logs

QUEUE_LOG=train_logs/queue_pure_dit_bridge_eval_20260624.log
CKPT=outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-fullaction-pure-gripfix-nozeroinit-object-1000-20260623--1000_chkpt

echo "[queue] waiting for GPU availability for pure DIT bridge eval" >> "$QUEUE_LOG"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  active_jobs=$( { pgrep -af 'finetune.py|run_libero_eval.py|tmp_queue_dit12_pure_clean_base1000' | grep -v tmp_queue_pure_dit_bridge_eval || true; } | wc -l | tr -d ' ')
  ts=$(date '+%F %T')
  echo "[$ts] gpu_used=${used}MiB gpu_util=${util}% active_jobs=${active_jobs}" >> "$QUEUE_LOG"
  if [ "$used" -le 7000 ] && [ "$util" -le 20 ] && [ "$active_jobs" -eq 0 ]; then
    break
  fi
  sleep 60
done

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

COMMON=(
  "$CKPT"
  --task_suite libero_object --task_ids 0 --num_trials 3 --action_head_type DIT
  --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps 20 --dit_num_inference_samples 1
  --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_anchor_blend_was_set True
  --dit_detach_flow_conditioning True --dit_disable_inference_anchor True --dit_pure_inference True
  --dit_use_state_conditioning False --dit_zero_init_adaln False --dit_zero_init_output False
  --debug_action_scale 1.0 --debug_non_gripper_action_scale 1.0
  --debug_gripper_scale 1.0 --debug_gripper_bias 0.0
  --debug_flip_raw_gripper_output False --debug_raw_gripper_threshold 0.5
)

run_case() {
  local name="$1"
  shift
  local case_log="train_logs/pure_dit_bridge_${name}_20260624.log"
  echo "===== PURE_DIT_BRIDGE_CASE ${name} START $(date '+%F %T') =====" >> "$QUEUE_LOG"
  : > "$case_log"
  ./eval.sh "${COMMON[@]}" "$@" >> "$case_log" 2>&1
  cat "$case_log" >> "$QUEUE_LOG"
  echo "===== PURE_DIT_BRIDGE_CASE ${name} END $(date '+%F %T') =====" >> "$QUEUE_LOG"
}

run_case adaptive --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1
run_case uniform --use_adaptive_bridge True --bridge_mode uniform --fixed_layer_index -1
run_case fixed_auto_mid --use_adaptive_bridge False --bridge_mode fixed --fixed_layer_index -1
run_case fixed_16 --use_adaptive_bridge False --bridge_mode fixed --fixed_layer_index 16
run_case fixed_23 --use_adaptive_bridge False --bridge_mode fixed --fixed_layer_index 23

echo "[queue] all done" >> "$QUEUE_LOG"
