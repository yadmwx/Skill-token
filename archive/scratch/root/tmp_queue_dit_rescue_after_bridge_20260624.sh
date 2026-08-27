#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
mkdir -p train_logs

WATCH_LOG=train_logs/queue_dit_rescue_after_bridge_20260624.log
BRIDGE_LOG=train_logs/queue_pure_dit_bridge_eval_20260624.log
CLEAN_SCRIPT=scripts/tmp_queue_dit12_pure_clean_base1000_20260624.sh
CKPT=outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-fullaction-pure-gripfix-nozeroinit-object-1000-20260623--1000_chkpt

launch_clean_training() {
  echo "[watch] launching clean pure DIT base training" >> "$WATCH_LOG"
  nohup bash "$CLEAN_SCRIPT" > train_logs/queue_dit12_pure_clean_base1000_20260624.nohup.log 2>&1 &
  echo "[watch] clean training queued at $(date '+%F %T')" >> "$WATCH_LOG"
}

eval_old_ckpt_samples() {
  local samples="$1"
  local trials="$2"
  local out_log="$3"
  ./eval.sh "$CKPT" \
    --task_suite libero_object --task_ids 0 --num_trials "$trials" --action_head_type DIT \
    --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps 20 --dit_num_inference_samples "$samples" \
    --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_anchor_blend_was_set True \
    --dit_detach_flow_conditioning True --dit_disable_inference_anchor True --dit_pure_inference True \
    --dit_use_state_conditioning False --dit_zero_init_adaln False --dit_zero_init_output False \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
    --debug_action_scale 1.0 --debug_non_gripper_action_scale 1.0 \
    --debug_gripper_scale 1.0 --debug_gripper_bias 0.0 \
    --debug_flip_raw_gripper_output False --debug_raw_gripper_threshold 0.5 \
    >> "$out_log" 2>&1
}

parse_successes() {
  local log_file="$1"
  grep 'Total successes:' "$log_file" 2>/dev/null | tail -n 1 | awk '{
    for (i = 1; i <= NF; i++) {
      if ($i == "successes:") {
        print $(i + 1)
        exit
      }
    }
  }'
}

echo "[watch] started $(date '+%F %T')" >> "$WATCH_LOG"
echo "[watch] waiting for bridge queue to finish" >> "$WATCH_LOG"
while pgrep -af 'tmp_queue_pure_dit_bridge_eval_20260624.sh' >/dev/null 2>&1; do
  sleep 60
done

echo "[watch] bridge queue finished or not running at $(date '+%F %T')" >> "$WATCH_LOG"
if [ ! -f "$BRIDGE_LOG" ]; then
  echo "[watch] missing bridge log; launching clean pure DIT training" >> "$WATCH_LOG"
  launch_clean_training
  exit 0
fi

best_successes=0
best_case=""
for case_name in adaptive uniform fixed_auto_mid fixed_16 fixed_23; do
  case_log="train_logs/pure_dit_bridge_${case_name}_20260624.log"
  if [ ! -f "$case_log" ]; then
    continue
  fi
  successes=$(parse_successes "$case_log")
  successes=${successes:-0}
  echo "[watch] bridge case ${case_name} successes=${successes}/3" >> "$WATCH_LOG"
  if [ "$successes" -gt "$best_successes" ]; then
    best_successes="$successes"
    best_case="$case_name"
  fi
done
echo "[watch] best bridge case=${best_case:-none} successes=${best_successes}/3" >> "$WATCH_LOG"

if [ "$best_successes" -ge 2 ]; then
  echo "[watch] bridge found >=2/3; launching 10-trial verification for ${best_case}" >> "$WATCH_LOG"
  COMMON=(
    "$CKPT"
    --task_suite libero_object --task_ids 0 --num_trials 10 --action_head_type DIT
    --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps 20 --dit_num_inference_samples 1
    --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_anchor_blend_was_set True
    --dit_detach_flow_conditioning True --dit_disable_inference_anchor True --dit_pure_inference True
    --dit_use_state_conditioning False --dit_zero_init_adaln False --dit_zero_init_output False
    --debug_action_scale 1.0 --debug_non_gripper_action_scale 1.0
    --debug_gripper_scale 1.0 --debug_gripper_bias 0.0
    --debug_flip_raw_gripper_output False --debug_raw_gripper_threshold 0.5
  )
  case "$best_case" in
    adaptive) BRIDGE_ARGS=(--use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1) ;;
    uniform) BRIDGE_ARGS=(--use_adaptive_bridge True --bridge_mode uniform --fixed_layer_index -1) ;;
    fixed_auto_mid) BRIDGE_ARGS=(--use_adaptive_bridge False --bridge_mode fixed --fixed_layer_index -1) ;;
    fixed_16) BRIDGE_ARGS=(--use_adaptive_bridge False --bridge_mode fixed --fixed_layer_index 16) ;;
    fixed_23) BRIDGE_ARGS=(--use_adaptive_bridge False --bridge_mode fixed --fixed_layer_index 23) ;;
    *) echo "[watch] unknown best case; cannot verify" >> "$WATCH_LOG"; exit 1 ;;
  esac
  ./eval.sh "${COMMON[@]}" "${BRIDGE_ARGS[@]}" >> "train_logs/verify_pure_dit_bridge_${best_case}_10trials_20260624.log" 2>&1
  echo "[watch] verification finished for ${best_case}" >> "$WATCH_LOG"
  exit 0
fi

echo "[watch] no bridge case reached 2/3; trying pure DIT multi-sample adaptive eval before training" >> "$WATCH_LOG"
for samples in 4 8; do
  sample_log="train_logs/pure_dit_old_adaptive_samples${samples}_eval3_20260624.log"
  : > "$sample_log"
  echo "[watch] eval old pure DIT adaptive samples=${samples}" >> "$WATCH_LOG"
  eval_old_ckpt_samples "$samples" 3 "$sample_log"
  successes=$(parse_successes "$sample_log")
  successes=${successes:-0}
  echo "[watch] old pure DIT adaptive samples=${samples} successes=${successes}/3" >> "$WATCH_LOG"
  if [ "$successes" -ge 2 ]; then
    verify_log="train_logs/verify_pure_dit_old_adaptive_samples${samples}_10trials_20260624.log"
    : > "$verify_log"
    echo "[watch] samples=${samples} reached >=2/3; launching 10-trial verification" >> "$WATCH_LOG"
    eval_old_ckpt_samples "$samples" 10 "$verify_log"
    echo "[watch] samples=${samples} verification finished" >> "$WATCH_LOG"
    exit 0
  fi
done

echo "[watch] old pure DIT multi-sample eval did not reach 2/3" >> "$WATCH_LOG"
launch_clean_training
