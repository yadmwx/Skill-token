#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-online}"

PYTHON=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python
TORCHRUN=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/torchrun

MLP_CKPT=outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-MLP-object-5000-20260618_213642--5000_chkpt
MLP_RESUME_STEP=5000
ANCHORINIT_TRAIN_STEPS="${ANCHORINIT_TRAIN_STEPS:-1000}"
ANCHORINIT_MAX_STEPS=$((MLP_RESUME_STEP + ANCHORINIT_TRAIN_STEPS))
ANCHORINIT_SAVE_FREQ="${ANCHORINIT_SAVE_FREQ:-500}"
ANCHORINIT_BATCH_SIZE="${ANCHORINIT_BATCH_SIZE:-4}"
ANCHORINIT_GRAD_ACCUM="${ANCHORINIT_GRAD_ACCUM:-4}"
DIT_TASK_TOKEN_MODE="${DIT_TASK_TOKEN_MODE:-vision_prompt}"
DIT_CONDITION_MODE="${DIT_CONDITION_MODE:-task_only}"
DIT_CONDITION_INJECTION_MODE="${DIT_CONDITION_INJECTION_MODE:-cross_attn}"
DIT_CLIP_NORMALIZED_ACTIONS="${DIT_CLIP_NORMALIZED_ACTIONS:-False}"
DEBUG_DIT_GROUP_ACTION_TOKENS_TO_CHUNK="${DEBUG_DIT_GROUP_ACTION_TOKENS_TO_CHUNK:-False}"
RUN_OFFLINE_PROBE="${RUN_OFFLINE_PROBE:-True}"
GPU_IDLE_MEM_LIMIT="${GPU_IDLE_MEM_LIMIT:-13000}"
if [ "$DIT_TASK_TOKEN_MODE" = "vision_prompt" ]; then
  RUN_ID=configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-mlp-anchorinit-prompt-taskonly-${ANCHORINIT_TRAIN_STEPS}from${MLP_RESUME_STEP}-20260627
else
  RUN_ID=configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-mlp-anchorinit-${DIT_TASK_TOKEN_MODE}-taskonly-${ANCHORINIT_TRAIN_STEPS}from${MLP_RESUME_STEP}-20260627
fi
if [ "$DIT_CONDITION_MODE" != "task_only" ]; then
  RUN_ID="${RUN_ID}-cond-${DIT_CONDITION_MODE}"
fi
if [ "$DIT_CONDITION_INJECTION_MODE" != "cross_attn" ]; then
  RUN_ID="${RUN_ID}-inject-${DIT_CONDITION_INJECTION_MODE}"
fi
if [ "$DEBUG_DIT_GROUP_ACTION_TOKENS_TO_CHUNK" = "True" ]; then
  RUN_ID="${RUN_ID}-groupaction"
fi
if [ "$DIT_CLIP_NORMALIZED_ACTIONS" = "True" ]; then
  RUN_ID="${RUN_ID}-clipnorm"
fi
LOG_SUFFIX="${DIT_TASK_TOKEN_MODE}_${DIT_CONDITION_MODE}"
if [ "$DIT_CONDITION_INJECTION_MODE" != "cross_attn" ]; then
  LOG_SUFFIX="${LOG_SUFFIX}_${DIT_CONDITION_INJECTION_MODE}"
fi
if [ "$DEBUG_DIT_GROUP_ACTION_TOKENS_TO_CHUNK" = "True" ]; then
  LOG_SUFFIX="${LOG_SUFFIX}_groupaction"
fi
if [ "$DIT_CLIP_NORMALIZED_ACTIONS" = "True" ]; then
  LOG_SUFFIX="${LOG_SUFFIX}_clipnorm"
fi
QUEUE_LOG=train_logs/queue_dit12_mlp_anchorinit_${LOG_SUFFIX}_20260627.log
TRAIN_LOG=train_logs/dit12_mlp_anchorinit_${LOG_SUFFIX}_20260627.log

mkdir -p train_logs

wait_for_gpu() {
  echo "[queue] waiting for GPU availability" >> "$QUEUE_LOG"
  while true; do
    local used util active_jobs ts
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    active_jobs=$(ps -eo args= | awk '/[f]inetune.py|[r]un_libero_eval.py/ {n++} END {print n + 0}')
    ts=$(date '+%F %T')
    echo "[$ts] gpu_used=${used}MiB gpu_util=${util}% active_jobs=${active_jobs}" >> "$QUEUE_LOG"
    if [ "$active_jobs" -eq 0 ] && [ "$used" -lt "$GPU_IDLE_MEM_LIMIT" ] && [ "$util" -lt 20 ]; then
      break
    fi
    sleep 60
  done
}

parse_successes() {
  grep -E "Total successes:" "$1" | tail -1 | sed -E 's/.*Total successes:[[:space:]]*([0-9]+).*/\1/'
}

eval_ckpt() {
  local ckpt="$1"
  local trials="$2"
  local seed="$3"
  local out_log="$4"
  ./eval.sh "$ckpt" \
    --task_suite libero_object --task_ids 0 --num_trials "$trials" --action_head_type DIT \
    --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps 20 --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 1.0 --dit_anchor_blend 0.0 --dit_disable_inference_anchor False --dit_pure_inference False \
    --dit_anchor_gripper_weight 1.0 --dit_anchor_gripper_bce_weight 0.0 \
    --dit_gripper_head_weight 1.0 --dit_gripper_head_override True \
    --dit_condition_mode "$DIT_CONDITION_MODE" --dit_include_prompt_tokens True \
    --dit_condition_injection_mode "$DIT_CONDITION_INJECTION_MODE" \
    --dit_task_token_mode "$DIT_TASK_TOKEN_MODE" \
    --dit_clip_normalized_actions "$DIT_CLIP_NORMALIZED_ACTIONS" \
    --debug_dit_group_action_tokens_to_chunk "$DEBUG_DIT_GROUP_ACTION_TOKENS_TO_CHUNK" \
    --dit_zero_init_adaln False --dit_zero_init_output False \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
    --seed "$seed" > "$out_log" 2>&1
}

probe_ckpt() {
  local ckpt="$1"
  local step="$2"
  local out_log="train_logs/dit12_mlp_anchorinit_${LOG_SUFFIX}_step${step}_offline_probe_20260627.log"
  echo "[queue] offline_probe step=${step}" >> "$QUEUE_LOG"
  "$PYTHON" scripts/tmp_probe_offline_action_error_20260627.py \
    --checkpoint "$ckpt" \
    --action_head_type DIT \
    --data_root_dir data/libero \
    --dataset_name libero_object_no_noops \
    --task_suite libero_object \
    --unnorm_key libero_object_no_noops \
    --num_batches 4 \
    --batch_size 2 \
    --image_aug False \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_minivlm True \
    --use_adaptive_bridge True \
    --bridge_mode adaptive \
    --fixed_layer_index -1 \
    --dit_num_blocks 12 \
    --dit_num_inference_steps 20 \
    --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 1.0 \
    --dit_anchor_blend 0.0 \
    --dit_anchor_gripper_weight 1.0 \
    --dit_anchor_gripper_bce_weight 0.0 \
    --dit_flow_xyz_loss_weight 1.0 \
    --dit_flow_rot_loss_weight 1.0 \
    --dit_flow_gripper_loss_weight 1.0 \
    --dit_flow_gripper_bce_weight 0.0 \
    --dit_gripper_head_weight 1.0 \
    --dit_gripper_head_override True \
    --dit_clip_normalized_actions "$DIT_CLIP_NORMALIZED_ACTIONS" \
    --dit_detach_flow_conditioning False \
    --dit_disable_inference_anchor False \
    --dit_pure_inference False \
    --dit_zero_init_adaln False \
    --dit_zero_init_output False \
    --dit_condition_mode "$DIT_CONDITION_MODE" \
    --dit_condition_injection_mode "$DIT_CONDITION_INJECTION_MODE" \
    --dit_include_prompt_tokens True \
    --dit_task_token_mode "$DIT_TASK_TOKEN_MODE" \
    --debug_dit_group_action_tokens_to_chunk "$DEBUG_DIT_GROUP_ACTION_TOKENS_TO_CHUNK" \
    --seed 7 > "$out_log" 2>&1
  cat "$out_log" >> "$QUEUE_LOG"
}

train_from_mlp_anchor() {
  echo "[queue] launching DIT12 from MLP checkpoint with MLP-initialized frozen anchor" >> "$QUEUE_LOG"
  : > "$TRAIN_LOG"
  "$TORCHRUN" --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --config_file_path pretrained_models/configs \
    --data_root_dir data/libero \
    --dataset_name libero_object_no_noops \
    --run_root_dir outputs \
    --train_from checkpoint \
    --resum_vla_path "$MLP_CKPT" \
    --resume_step "$MLP_RESUME_STEP" \
    --resume_load_training_state False \
    --vlm_training freeze_lora \
    --action_head_type DIT \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_minivlm True \
    --merge_lora_during_training False \
    --image_aug True \
    --num_steps_before_decay 20000 \
    --max_steps "$ANCHORINIT_MAX_STEPS" \
    --save_freq "$ANCHORINIT_SAVE_FREQ" \
    --save_latest_checkpoint_only False \
    --batch_size "$ANCHORINIT_BATCH_SIZE" \
    --grad_accumulation_steps "$ANCHORINIT_GRAD_ACCUM" \
    --learning_rate 2e-4 \
    --use_constant_lr True \
    --use_pro_version True \
    --eval_on_checkpoint False \
    --ddp_find_unused_params True \
    --wandb_entity 323660583-huazhong-university-of-science-and-technology \
    --wandb_project libero_object_no_noops \
    --run_id_note VLA-Adapter-DIT12-mlp-anchorinit-prompt-taskonly-1000-20260627 \
    --run_id_override "$RUN_ID" \
    --dit_num_blocks 12 \
    --flow_ratio 1.0 \
    --dit_num_inference_steps 20 \
    --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 1.0 \
    --dit_anchor_blend 0.0 \
    --dit_anchor_gripper_weight 1.0 \
    --dit_anchor_gripper_bce_weight 0.0 \
    --dit_flow_xyz_loss_weight 1.0 \
    --dit_flow_rot_loss_weight 1.0 \
    --dit_flow_gripper_loss_weight 1.0 \
    --dit_flow_gripper_bce_weight 0.0 \
    --dit_gripper_head_weight 1.0 \
    --dit_gripper_head_override True \
    --dit_anchor_init_checkpoint "$MLP_CKPT" \
    --dit_freeze_anchor_head True \
    --dit_sample_t_mode_flow uniform \
    --dit_detach_flow_conditioning False \
    --dit_condition_mode "$DIT_CONDITION_MODE" \
    --dit_condition_injection_mode "$DIT_CONDITION_INJECTION_MODE" \
    --dit_include_prompt_tokens True \
    --dit_task_token_mode "$DIT_TASK_TOKEN_MODE" \
    --dit_clip_normalized_actions "$DIT_CLIP_NORMALIZED_ACTIONS" \
    --debug_dit_group_action_tokens_to_chunk "$DEBUG_DIT_GROUP_ACTION_TOKENS_TO_CHUNK" \
    --dit_zero_init_adaln False \
    --dit_zero_init_output False \
    --use_adaptive_bridge True \
    --bridge_mode adaptive \
    --fixed_layer_index -1 \
    > "$TRAIN_LOG" 2>&1
}

eval_anchorinit_checkpoints() {
  local step seed ckpt log successes
  for step in $((MLP_RESUME_STEP + ANCHORINIT_SAVE_FREQ)) "$ANCHORINIT_MAX_STEPS"; do
    ckpt="outputs/${RUN_ID}--${step}_chkpt"
    [ -d "$ckpt" ] || continue
    if [ "$RUN_OFFLINE_PROBE" = "True" ]; then
      probe_ckpt "$ckpt" "$step"
    fi
    for seed in 0 3 7; do
      log="train_logs/dit12_mlp_anchorinit_${LOG_SUFFIX}_step${step}_seed${seed}_eval3_20260627.log"
      echo "[queue] eval mlp_anchorinit step=${step} seed=${seed}" >> "$QUEUE_LOG"
      : > "$log"
      eval_ckpt "$ckpt" 3 "$seed" "$log"
      cat "$log" >> "$QUEUE_LOG"
      successes=$(parse_successes "$log")
      echo "[queue] mlp_anchorinit step=${step} seed=${seed} successes=${successes:-NA}/3" >> "$QUEUE_LOG"
    done
  done
}

echo "[queue] start $(date '+%F %T') dit_task_token_mode=${DIT_TASK_TOKEN_MODE} dit_condition_mode=${DIT_CONDITION_MODE} dit_condition_injection_mode=${DIT_CONDITION_INJECTION_MODE} dit_clip_normalized_actions=${DIT_CLIP_NORMALIZED_ACTIONS} debug_group_action_tokens=${DEBUG_DIT_GROUP_ACTION_TOKENS_TO_CHUNK} batch_size=${ANCHORINIT_BATCH_SIZE} grad_accum=${ANCHORINIT_GRAD_ACCUM} run_offline_probe=${RUN_OFFLINE_PROBE}" >> "$QUEUE_LOG"
"$PYTHON" -m py_compile prismatic/models/ditx_blocks.py prismatic/models/ditx_vla_adapter.py prismatic/models/action_heads.py prismatic/models/flow_matching_head.py prismatic/extern/hf/modeling_prismatic.py vla-scripts/finetune.py experiments/robot/openvla_utils.py experiments/robot/libero/run_libero_eval.py scripts/tmp_probe_offline_action_error_20260627.py
wait_for_gpu
train_from_mlp_anchor
wait_for_gpu
eval_anchorinit_checkpoints
echo "[queue] all done $(date '+%F %T')" >> "$QUEUE_LOG"
