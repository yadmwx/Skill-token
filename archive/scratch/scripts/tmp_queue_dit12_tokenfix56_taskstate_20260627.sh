#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
mkdir -p train_logs

PYTHON=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python
TORCHRUN=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/torchrun
export PYTHONPATH=/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

RUN_ID=configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-tokenfix56-taskstate-sqrtpos1000-20260627
QUEUE_LOG=train_logs/queue_dit12_tokenfix56_taskstate_sqrtpos1000_20260627.log
TRAIN_LOG=train_logs/dit12_tokenfix56_taskstate_sqrtpos1000_20260627.log

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

wait_for_gpu() {
  echo "[queue] waiting for GPU availability" >> "$QUEUE_LOG"
  while true; do
    local used util active_jobs statecond_jobs ts
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    active_jobs=$(ps -eo args= | awk '/[f]inetune.py|[r]un_libero_eval.py/ {n++} END {print n + 0}')
    statecond_jobs=$(ps -eo args= | awk '/[t]mp_queue_dit12_tokenfix56_statecond_20260627.sh/ {n++} END {print n + 0}')
    ts=$(date '+%F %T')
    echo "[$ts] gpu_used=${used}MiB gpu_util=${util}% active_jobs=${active_jobs} statecond_jobs=${statecond_jobs}" >> "$QUEUE_LOG"
    if [ "$used" -le 7000 ] && [ "$util" -le 20 ] && [ "$active_jobs" -eq 0 ] && [ "$statecond_jobs" -eq 0 ]; then
      break
    fi
    sleep 60
  done
}

eval_ckpt() {
  local ckpt="$1"
  local trials="$2"
  local steps="$3"
  local seed="$4"
  local out_log="$5"
  ./eval.sh "$ckpt" \
    --task_suite libero_object --task_ids 0 --num_trials "$trials" --action_head_type DIT \
    --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps "$steps" --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_anchor_blend_was_set True \
    --dit_detach_flow_conditioning False --dit_disable_inference_anchor True --dit_pure_inference True \
    --dit_use_state_conditioning True --dit_state_scale_mode sqrt_group --dit_state_proprio_mode concat \
    --dit_state_use_chunk_pos True --dit_state_include_task_tokens True \
    --dit_zero_init_adaln False --dit_zero_init_output False \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
    --debug_action_scale 1.0 --debug_non_gripper_action_scale 1.0 \
    --debug_gripper_scale 1.0 --debug_gripper_bias 0.0 \
    --debug_flip_raw_gripper_output False --debug_raw_gripper_threshold 0.5 \
    --seed "$seed" \
    >> "$out_log" 2>&1
}

train_taskstate_from_base() {
  echo "[queue] launching clean DIT12 task+state-conditioned training with exact 56 action tokens" >> "$QUEUE_LOG"
  "$TORCHRUN" --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --config_file_path pretrained_models/configs \
    --data_root_dir data/libero \
    --dataset_name libero_object_no_noops \
    --run_root_dir outputs \
    --train_from vlm_base \
    --vlm_training freeze_lora \
    --action_head_type DIT \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --use_minivlm True \
    --merge_lora_during_training False \
    --image_aug True \
    --num_steps_before_decay 20000 \
    --max_steps 1000 \
    --save_freq 500 \
    --save_latest_checkpoint_only False \
    --batch_size 4 \
    --grad_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --use_constant_lr True \
    --use_pro_version True \
    --eval_on_checkpoint False \
    --ddp_find_unused_params True \
    --wandb_entity 323660583-huazhong-university-of-science-and-technology \
    --wandb_project libero_object_no_noops \
    --run_id_note VLA-Adapter-DIT12-tokenfix56-taskstate-sqrtpos1000-20260627 \
    --run_id_override "$RUN_ID" \
    --dit_num_blocks 12 \
    --flow_ratio 1.0 \
    --dit_num_inference_steps 20 \
    --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 0.0 \
    --dit_anchor_blend 0.0 \
    --dit_anchor_gripper_weight 1.0 \
    --dit_anchor_gripper_bce_weight 0.0 \
    --dit_flow_xyz_loss_weight 1.0 \
    --dit_flow_rot_loss_weight 1.0 \
    --dit_flow_gripper_loss_weight 1.0 \
    --dit_flow_gripper_bce_weight 0.0 \
    --dit_sample_t_mode_flow uniform \
    --dit_detach_flow_conditioning False \
    --dit_use_state_conditioning True \
    --dit_state_scale_mode sqrt_group \
    --dit_state_proprio_mode concat \
    --dit_state_use_chunk_pos True \
    --dit_state_include_task_tokens True \
    --dit_zero_init_adaln False \
    --dit_zero_init_output False \
    --use_adaptive_bridge True \
    --bridge_mode adaptive \
    --fixed_layer_index -1 \
    >> "$TRAIN_LOG" 2>&1
}

eval_taskstate_checkpoints() {
  local step seed ckpt log verify_log successes
  for step in 500 1000; do
    ckpt="outputs/${RUN_ID}--${step}_chkpt"
    [ -d "$ckpt" ] || continue
    for seed in 0 3 7; do
      log="train_logs/dit12_tokenfix56_taskstate_step${step}_steps20_seed${seed}_eval3_20260627.log"
      echo "[queue] eval taskstate step=${step} seed=${seed}" >> "$QUEUE_LOG"
      : > "$log"
      eval_ckpt "$ckpt" 3 20 "$seed" "$log"
      cat "$log" >> "$QUEUE_LOG"
      successes=$(parse_successes "$log")
      successes=${successes:-0}
      echo "[queue] taskstate step=${step} seed=${seed} successes=${successes}/3" >> "$QUEUE_LOG"
      if [ "$successes" -ge 2 ]; then
        verify_log="train_logs/verify_dit12_tokenfix56_taskstate_step${step}_steps20_seed${seed}_10trials_20260627.log"
        echo "[queue] taskstate candidate reached >=2/3; launching 10-trial verification step=${step} seed=${seed}" >> "$QUEUE_LOG"
        : > "$verify_log"
        eval_ckpt "$ckpt" 10 20 "$seed" "$verify_log"
        cat "$verify_log" >> "$QUEUE_LOG"
        successes=$(parse_successes "$verify_log")
        successes=${successes:-0}
        echo "[queue] taskstate verify step=${step} seed=${seed} successes=${successes}/10" >> "$QUEUE_LOG"
      fi
    done
  done
}

echo "[queue] start $(date '+%F %T')" >> "$QUEUE_LOG"
"$PYTHON" -m py_compile prismatic/extern/hf/modeling_prismatic.py prismatic/vla/datasets/datasets.py prismatic/models/flow_matching_head.py experiments/robot/openvla_utils.py vla-scripts/finetune.py experiments/robot/libero/run_libero_eval.py
wait_for_gpu
train_taskstate_from_base
wait_for_gpu
eval_taskstate_checkpoints
echo "[queue] all done $(date '+%F %T')" >> "$QUEUE_LOG"
