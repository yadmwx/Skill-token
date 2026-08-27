#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/autodl-fs/data/skill_depth/code/action_expert_prefix_trial
cd "$REPO_DIR"
mkdir -p train_logs

PY=/autodl-fs/data/skill_depth/envs/vla-flow/bin/python
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:/autodl-fs/data/skill_depth/code/action_expert_prefix_trial/LIBERO:/autodl-fs/data/skill_depth/code/LIBERO:/autodl-fs/data/skill_depth/code/action_expert_prefix_trial/robosuite:/autodl-fs/data/skill_depth/code/robosuite:${PYTHONPATH:-}
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export HF_HOME=/autodl-fs/data/skill_depth/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/autodl-fs/data/skill_depth/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LABEL=dit12_h896_std_action_prefix_scratch_b4ga4_12000_v48_20260708
RUN_ID=DIT12-h896-std-action-prefix-scratch-b4ga4-12000-v48-20260708
QUEUE_LOG=train_logs/eval_queue_${LABEL}.log

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"; }

success_count(){
  local log_file=$1
  grep -A3 'Total successes:' "$log_file" | grep -Eo '[0-9]+' | head -1 || echo 0
}

wait_gpu(){
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    jobs=$(ps -eo args= | awk '/[r]un_libero_eval.py/ {n++} END {print n + 0}')
    log "gpu used=${used}MiB free=${free}MiB util=${util}% eval_jobs=${jobs}"
    if [ "$free" -gt 42000 ] && [ "$util" -lt 20 ] && [ "$jobs" -eq 0 ]; then break; fi
    sleep 60
  done
}

run_eval(){
  local step=$1
  local chkpt=$2
  local eval_label=${LABEL}_step${step}_hardtasks_eval5_fixedpy
  local eval_log=train_logs/${eval_label}.log
  local eval_status=train_logs/${eval_label}.status
  log "launch hard subset eval step=${step} chkpt=${chkpt}"
  wait_gpu
  set +e
  "$PY" experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$chkpt" \
    --action_head_type DIT --task_suite_name libero_object --task_ids 3,4,7,9 --num_trials_per_task 5 \
    --use_depth_interface False --depth_interface_mode none --depth_interface_max_layers 64 --depth_interface_add_proprio True \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 --flow_ratio 1.0 \
    --dit_hidden_dim 896 --dit_num_blocks 12 --dit_num_inference_steps 20 --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_anchor_blend_was_set True \
    --dit_inference_residual_scale 1.0 --dit_anchor_gripper_weight 1.0 --dit_anchor_gripper_bce_weight 0.0 \
    --dit_flow_xyz_loss_weight 1.0 --dit_flow_rot_loss_weight 1.0 --dit_flow_gripper_loss_weight 1.0 --dit_flow_gripper_bce_weight 0.0 \
    --dit_sample_t_mode_flow beta --dit_detach_flow_conditioning False \
    --dit_disable_inference_anchor True --dit_pure_inference True \
    --dit_use_state_conditioning True --dit_state_include_task_tokens True --dit_state_use_chunk_pos True \
    --dit_state_proprio_mode concat --dit_fuse_state_into_action_tokens True \
    --dit_condition_mode task_only --dit_condition_injection_mode action_expert_prefix \
    --dit_include_prompt_tokens True --dit_task_token_mode vision_prompt \
    --dit_zero_init_adaln True --dit_zero_init_output False \
    --use_proprio True --num_images_in_input 2 --use_film False --use_minivlm True --use_pro_version True \
    --use_wandb False --center_crop True --seed 7 \
    > "$eval_log" 2>&1
  rc=$?
  echo "$rc" > "$eval_status"
  set -e
  log "eval finished step=${step} rc=${rc} successes=$(success_count "$eval_log")"
  grep -n 'Final results\|Total episodes\|Total successes\|Overall success rate\|Current task success rate\|Current total success\|Success:' "$eval_log" | tail -120 >> "$QUEUE_LOG" || true
}

log "fixed eval queue start"
for step in 8000 10000 12000; do
  chkpt=$(find outputs -maxdepth 1 -type d -name "*${RUN_ID}--${step}_chkpt" | sort | tail -1)
  if [ -n "$chkpt" ]; then
    run_eval "$step" "$chkpt"
  else
    log "missing checkpoint step=${step}"
  fi
done
log "fixed eval queue done"
