#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/skill_depth
CODE=${ROOT}/code/action_expert_prefix_trial
cd "$CODE"
mkdir -p train_logs

PY=/autodl-fs/data/skill_depth/envs/vla-flow/bin/python
TORCHRUN=/autodl-fs/data/skill_depth/envs/vla-flow/bin/torchrun
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${CODE}:${ROOT}/code/LIBERO:${ROOT}/code/LIBERO/libero:${ROOT}/code/robosuite:${PYTHONPATH:-}
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

LABEL=dit12_h896_std_action_prefix_pure_automodel_frommlp5000_b4ga4_16000_v48_20260704
RUN_ID=DIT12-h896-std-action-prefix-pure-automodel-frommlp5000-b4ga4-16000-v48-20260704
TRAIN_LOG=train_logs/${LABEL}.log
QUEUE_LOG=train_logs/queue_${LABEL}.log
STATUS=train_logs/${LABEL}.status
BASE_MLP=/autodl-fs/data/skill_depth/outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-MLP-object-5000-20260618_213642--5000_chkpt
RESUME_NO_LORA=/autodl-fs/data/skill_depth/outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-MLP-object-5000-20260618_213642--5000_chkpt_nolora_resume

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"; }

wait_gpu(){
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    jobs=$(ps -eo args= | awk '/[v]la-scripts\/finetune.py|[r]un_libero_eval.py/ {n++} END {print n + 0}')
    log "gpu used=${used}MiB free=${free}MiB util=${util}% jobs=${jobs}"
    if [ "$free" -gt 38000 ] && [ "$util" -lt 20 ] && [ "$jobs" -eq 0 ]; then break; fi
    sleep 60
  done
}

ensure_no_lora_resume(){
  rm -rf "$RESUME_NO_LORA"
  mkdir -p "$RESUME_NO_LORA"
  find "$BASE_MLP" -maxdepth 1 -mindepth 1 ! -name lora_adapter -exec ln -s {} "$RESUME_NO_LORA"/ \;
}

success_count(){
  local log_file=$1
  grep -A3 'Total successes:' "$log_file" | grep -Eo '[0-9]+' | head -1 || echo 0
}

run_eval(){
  local step=$1
  local chkpt=$2
  local eval_label=${LABEL}_step${step}_hardtasks_eval5
  local eval_log=train_logs/${eval_label}.log
  local eval_status=train_logs/${eval_label}.status
  if [ -f "$eval_status" ] && [ "$(cat "$eval_status")" = 0 ]; then
    log "eval already completed step=${step} successes=$(success_count "$eval_log")"
    return 0
  fi
  log "launch eval step=${step} chkpt=${chkpt}"
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
    --dit_detach_flow_conditioning False --dit_disable_inference_anchor True --dit_pure_inference True \
    --dit_use_state_conditioning True --dit_state_include_task_tokens True --dit_state_use_chunk_pos True \
    --dit_state_proprio_mode concat --dit_fuse_state_into_action_tokens True \
    --dit_condition_mode task_only --dit_condition_injection_mode action_expert_prefix \
    --dit_include_prompt_tokens True --dit_task_token_mode vision_prompt \
    --dit_zero_init_adaln True --dit_zero_init_output False \
    --use_proprio True --num_images_in_input 2 --use_film False --use_minivlm False --use_pro_version True \
    --use_wandb False --center_crop True --seed 7 \
    > "$eval_log" 2>&1
  rc=$?
  echo "$rc" > "$eval_status"
  set -e
  log "eval finished step=${step} rc=${rc} successes=$(success_count "$eval_log")"
  grep -n 'Final results\|Total episodes\|Total successes\|Overall success rate\|Current task success rate\|Current total success\|Success:' "$eval_log" | tail -120 >> "$QUEUE_LOG" || true
  return "$rc"
}

log "queue start"
if [ ! -d "$BASE_MLP" ]; then
  log "missing base checkpoint $BASE_MLP"
  exit 2
fi
ensure_no_lora_resume

if [ -f "$STATUS" ] && [ "$(cat "$STATUS")" = 0 ]; then
  log "training already completed; skip"
else
  : > "$TRAIN_LOG"
  : > "$STATUS"
  log "launch train label=${LABEL}"
  wait_gpu
  set +e
  "$TORCHRUN" --standalone --nnodes 1 --nproc_per_node 1 \
    vla-scripts/finetune.py \
    --config_file_path "$BASE_MLP" \
    --vlm_path "$BASE_MLP" \
    --data_root_dir /root/skill_depth/data/libero \
    --dataset_name libero_object_no_noops \
    --run_root_dir /autodl-fs/data/skill_depth/outputs \
    --train_from checkpoint \
    --resum_vla_path "$RESUME_NO_LORA" \
    --resume_step 5000 \
    --resume_load_training_state False \
    --vlm_training freeze_lora \
    --action_head_type DIT \
    --use_lora False --lora_rank 32 --lora_dropout 0.0 --freeze_vlm True --merge_lora_during_training False \
    --use_minivlm False --use_pro_version True --use_proprio True --num_images_in_input 2 --use_film False \
    --learning_rate 0.0002 --use_constant_lr True --lr_warmup_steps 200 \
    --max_steps 16000 --save_freq 1000 --save_latest_checkpoint_only False --image_aug True \
    --use_wandb False --wandb_log_freq 50 \
    --dit_hidden_dim 896 --dit_num_blocks 12 --dit_num_inference_steps 20 --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_inference_residual_scale 1.0 \
    --dit_anchor_gripper_weight 1.0 --dit_anchor_gripper_bce_weight 0.0 \
    --dit_flow_xyz_loss_weight 1.0 --dit_flow_rot_loss_weight 1.0 --dit_flow_gripper_loss_weight 1.0 \
    --dit_endpoint_loss_weight 1.0 --dit_flow_gripper_bce_weight 0.0 --dit_sample_t_mode_flow beta \
    --dit_detach_flow_conditioning False --dit_use_state_conditioning True --dit_state_include_task_tokens True \
    --dit_state_use_chunk_pos True --dit_state_proprio_mode concat --dit_fuse_state_into_action_tokens True \
    --dit_condition_mode task_only --dit_condition_injection_mode action_expert_prefix \
    --dit_include_prompt_tokens True --dit_task_token_mode vision_prompt \
    --dit_zero_init_adaln True --dit_zero_init_output False \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 --ddp_find_unused_params True --flow_ratio 1.0 \
    --run_id_note "$RUN_ID" --run_id_override configs+libero_object_no_noops+b4+lr-0.0002+lora-r32+dropout-0.0--image_aug--${RUN_ID} \
    --batch_size 4 --grad_accumulation_steps 4 \
    > "$TRAIN_LOG" 2>&1
  rc=$?
  echo "$rc" > "$STATUS"
  set -e
  log "train finished rc=${rc}"
  tail -100 "$TRAIN_LOG" >> "$QUEUE_LOG" || true
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
fi

best=0
for step in 10000 12000 14000 16000; do
  chkpt=$(find /autodl-fs/data/skill_depth/outputs -maxdepth 1 -type d -name "*${RUN_ID}--${step}_chkpt" | sort | tail -1)
  if [ -n "$chkpt" ]; then
    run_eval "$step" "$chkpt" || true
    got=$(success_count "train_logs/${LABEL}_step${step}_hardtasks_eval5.log")
    if [ "$got" -gt "$best" ]; then best=$got; fi
    log "step=${step} hard_successes=${got}/20 best=${best}/20"
  else
    log "missing checkpoint step=${step}"
  fi
done
log "queue done best=${best}/20"
