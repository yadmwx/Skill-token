#!/usr/bin/env bash
set -euo pipefail

CKPT='outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-fullaction-pure-gripfix-nozeroinit-object-1000-20260623--1000_chkpt'

COMMON=(
  "$CKPT"
  --task_suite libero_object --task_ids 0 --num_trials 3 --action_head_type DIT
  --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps 20 --dit_num_inference_samples 1
  --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0
  --dit_detach_flow_conditioning True --dit_disable_inference_anchor True --dit_pure_inference True
  --dit_zero_init_adaln False --dit_zero_init_output False
  --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1
)

run_case() {
  local name="$1"
  shift
  echo "===== SWEEP_CASE ${name} START $(date '+%F %T') ====="
  ./eval.sh "${COMMON[@]}" "$@"
  echo "===== SWEEP_CASE ${name} END $(date '+%F %T') ====="
}

run_case grip_thr_0p9 --debug_raw_gripper_threshold 0.9
run_case grip_thr_1p0 --debug_raw_gripper_threshold 1.0
run_case grip_thr_1p1 --debug_raw_gripper_threshold 1.1
run_case scale_0p75_thr_1p0 --debug_non_gripper_action_scale 0.75 --debug_raw_gripper_threshold 1.0
