#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
cd "$REPO_DIR"
mkdir -p train_logs

PY=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python
TORCHRUN=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/torchrun
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:/home/xiaguanxiao/code/VLA/LIBERO/libero:/home/xiaguanxiao/code/starVLA/robosuite:${PYTHONPATH:-}
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

QUEUE_LABEL=dit_pure_std_prefix_rescue_20260704
QUEUE_LOG=train_logs/queue_${QUEUE_LABEL}.log
BASE_MLP=outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-MLP-object-5000-20260618_213642--5000_chkpt
BASE_H896=outputs/configs+libero_object_no_noops+b4+lr-0.0002+lora-r32+dropout-0.0--image_aug--DIT12-h896-std-action-prefix-pure-frommlp5000-b4ga4-10000-20260703--10000_chkpt

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"; }

wait_gpu(){
  local need_free=${1:-70000}
  while true; do
    local used free util jobs
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    jobs=$(ps -eo args= | awk '/[v]la-scripts\/finetune.py|[r]un_libero_eval.py/ {n++} END {print n + 0}')
    log "gpu used=${used}MiB free=${free}MiB util=${util}% vla_jobs=${jobs} need_free=${need_free}"
    if [ "$free" -gt "$need_free" ] && [ "$util" -lt 20 ] && [ "$jobs" -eq 0 ]; then break; fi
    sleep 60
  done
}

success_count(){
  local log_file=$1
  grep -A3 'Total successes:' "$log_file" | grep -Eo '[0-9]+' | head -1 || echo 0
}

run_eval(){
  local label=$1
  local chkpt=$2
  local hidden=$3
  local eval_log=train_logs/${label}_hardtasks_eval5.log
  local eval_status=train_logs/${label}_hardtasks_eval5.status
  if [ -f "$eval_status" ] && [ "$(cat "$eval_status")" = 0 ]; then
    log "eval already completed label=$label successes=$(success_count "$eval_log")"
    return 0
  fi
  log "launch hard subset eval label=$label chkpt=$chkpt hidden=$hidden"
  wait_gpu 60000
  set +e
  "$PY" experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$chkpt" \
    --action_head_type DIT --task_suite_name libero_object --task_ids 3,4,7,9 --num_trials_per_task 5 \
    --use_depth_interface False --depth_interface_mode none --depth_interface_max_layers 64 --depth_interface_add_proprio True \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 --flow_ratio 1.0 \
    --dit_hidden_dim "$hidden" --dit_num_blocks 12 --dit_num_inference_steps 20 --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_anchor_blend_was_set True \
    --dit_inference_residual_scale 1.0 --dit_anchor_gripper_weight 1.0 --dit_anchor_gripper_bce_weight 0.0 \
    --dit_flow_xyz_loss_weight 1.0 --dit_flow_rot_loss_weight 1.0 --dit_flow_gripper_loss_weight 1.0 --dit_flow_gripper_bce_weight 0.0 \
    --dit_detach_flow_conditioning False --dit_disable_inference_anchor True --dit_pure_inference True \
    --dit_use_state_conditioning True --dit_state_include_task_tokens True --dit_state_use_chunk_pos True \
    --dit_state_proprio_mode concat --dit_fuse_state_into_action_tokens True \
    --dit_condition_mode task_only --dit_condition_injection_mode action_expert_prefix \
    --dit_include_prompt_tokens True --dit_task_token_mode vision_prompt \
    --dit_zero_init_adaln True --dit_zero_init_output False \
    --use_proprio True --num_images_in_input 2 --use_film False --use_minivlm True --use_pro_version True \
    --use_wandb False --center_crop True --seed 7 \
    > "$eval_log" 2>&1
  local rc=$?
  echo "$rc" > "$eval_status"
  set -e
  log "eval finished label=$label rc=$rc successes=$(success_count "$eval_log")"
  grep -n 'Final results\|Total episodes\|Total successes\|Overall success rate\|Current task success rate\|Current total success\|Success:' "$eval_log" | tail -120 >> "$QUEUE_LOG" || true
  return "$rc"
}

train_h896_continue(){
  local label=dit12_h896_std_action_prefix_pure_continue10000to16000_lr1e4_b4ga4_20260704
  local run_id=DIT12-h896-std-action-prefix-pure-continue10000to16000-lr1e4-b4ga4-20260704
  local train_log=train_logs/${label}.log
  local status=train_logs/${label}.status
  if [ -f "$status" ] && [ "$(cat "$status")" = 0 ]; then
    log "h896 continuation already completed"
    return 0
  fi
  : > "$train_log"
  : > "$status"
  log "launch h896 continuation from 10000 to 16000 lr=1e-4"
  wait_gpu 70000
  set +e
  "$TORCHRUN" --standalone --nnodes 1 --nproc_per_node 1 \
    vla-scripts/finetune.py \
    --config_file_path pretrained_models/configs \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --data_root_dir data/libero \
    --dataset_name libero_object_no_noops \
    --run_root_dir outputs \
    --train_from checkpoint \
    --resum_vla_path "$BASE_H896" \
    --resume_step 10000 \
    --resume_load_training_state False \
    --vlm_training freeze_lora \
    --action_head_type DIT \
    --use_lora True --lora_rank 32 --lora_dropout 0.0 --freeze_vlm True --merge_lora_during_training True \
    --use_minivlm True --use_pro_version True --use_proprio True --num_images_in_input 2 --use_film False \
    --learning_rate 0.0001 --use_constant_lr True --lr_warmup_steps 100 \
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
    --run_id_note "$run_id" --run_id_override configs+libero_object_no_noops+b4+lr-0.0001+lora-r32+dropout-0.0--image_aug--${run_id} \
    --batch_size 4 --grad_accumulation_steps 4 \
    > "$train_log" 2>&1
  local rc=$?
  echo "$rc" > "$status"
  set -e
  log "h896 continuation finished rc=$rc"
  tail -80 "$train_log" >> "$QUEUE_LOG" || true
  return "$rc"
}

train_h1024_from_mlp(){
  local label=dit12_h1024_std_action_prefix_pure_frommlp5000_b4ga4_12000_20260704
  local run_id=DIT12-h1024-std-action-prefix-pure-frommlp5000-b4ga4-12000-20260704
  local train_log=train_logs/${label}.log
  local status=train_logs/${label}.status
  if [ -f "$status" ] && [ "$(cat "$status")" = 0 ]; then
    log "h1024 training already completed"
    return 0
  fi
  : > "$train_log"
  : > "$status"
  log "launch h1024 standard prefix from MLP-5000 to 12000"
  wait_gpu 70000
  set +e
  "$TORCHRUN" --standalone --nnodes 1 --nproc_per_node 1 \
    vla-scripts/finetune.py \
    --config_file_path pretrained_models/configs \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --data_root_dir data/libero \
    --dataset_name libero_object_no_noops \
    --run_root_dir outputs \
    --train_from checkpoint \
    --resum_vla_path "$BASE_MLP" \
    --resume_step 5000 \
    --resume_load_training_state False \
    --vlm_training freeze_lora \
    --action_head_type DIT \
    --use_lora True --lora_rank 32 --lora_dropout 0.0 --freeze_vlm True --merge_lora_during_training True \
    --use_minivlm True --use_pro_version True --use_proprio True --num_images_in_input 2 --use_film False \
    --learning_rate 0.0002 --use_constant_lr True --lr_warmup_steps 200 \
    --max_steps 12000 --save_freq 1000 --save_latest_checkpoint_only False --image_aug True \
    --use_wandb False --wandb_log_freq 50 \
    --dit_hidden_dim 1024 --dit_num_blocks 12 --dit_num_inference_steps 20 --dit_num_inference_samples 1 \
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
    --run_id_note "$run_id" --run_id_override configs+libero_object_no_noops+b4+lr-0.0002+lora-r32+dropout-0.0--image_aug--${run_id} \
    --batch_size 4 --grad_accumulation_steps 4 \
    > "$train_log" 2>&1
  local rc=$?
  echo "$rc" > "$status"
  set -e
  log "h1024 training finished rc=$rc"
  tail -80 "$train_log" >> "$QUEUE_LOG" || true
  return "$rc"
}

log "queue start head=$(git rev-parse --short HEAD)"

best=9
train_h896_continue || log "h896 continuation failed rc=$?"
for step in 12000 14000 16000; do
  chkpt=$(find outputs -maxdepth 1 -type d -name "*DIT12-h896-std-action-prefix-pure-continue10000to16000-lr1e4-b4ga4-20260704--${step}_chkpt" | sort | tail -1)
  if [ -n "$chkpt" ]; then
    label=dit12_h896_std_action_prefix_pure_continue10000to16000_lr1e4_b4ga4_20260704_step${step}
    run_eval "$label" "$chkpt" 896 || true
    got=$(success_count "train_logs/${label}_hardtasks_eval5.log")
    if [ "$got" -gt "$best" ]; then best=$got; fi
    log "h896 step=${step} hard_successes=${got}/20 best=${best}/20"
  else
    log "missing h896 checkpoint step=${step}"
  fi
done

if [ "$best" -lt 16 ]; then
  log "best h896 hard subset below 16/20; launch h1024 fallback"
  train_h1024_from_mlp || log "h1024 training failed rc=$?"
  chkpt=$(find outputs -maxdepth 1 -type d -name "*DIT12-h1024-std-action-prefix-pure-frommlp5000-b4ga4-12000-20260704--12000_chkpt" | sort | tail -1)
  if [ -n "$chkpt" ]; then
    label=dit12_h1024_std_action_prefix_pure_frommlp5000_b4ga4_12000_20260704
    run_eval "$label" "$chkpt" 1024 || true
    got=$(success_count "train_logs/${label}_hardtasks_eval5.log")
    if [ "$got" -gt "$best" ]; then best=$got; fi
    log "h1024 hard_successes=${got}/20 best=${best}/20"
  else
    log "missing h1024 checkpoint"
  fi
fi

log "queue done best=${best}/20"
