#!/bin/bash
set -e
cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
mkdir -p train_logs
QUEUE_LOG=train_logs/queue_dit12_statecond_sqrtgroup_nozeroinit_20260623.log
TRAIN_LOG=train_logs/dit12_statecond_sqrtgroup_nozeroinit_20260623.log
OUT_BASE=outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-statecond-sqrtgroup-nozeroinit-object-1000-20260623
CHKPT_DIR="${OUT_BASE}--1000_chkpt"

echo "[queue] waiting for statecond sqrt_group queue to finish" >> "$QUEUE_LOG"
while pgrep -f 'tmp_queue_dit12_statecond_sqrtgroup_20260623.sh' >/dev/null; do
  ts=$(date '+%F %T')
  echo "[$ts] waiting for tmp_queue_dit12_statecond_sqrtgroup_20260623.sh" >> "$QUEUE_LOG"
  sleep 120
done

echo "[queue] waiting for GPU availability" >> "$QUEUE_LOG"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  ts=$(date '+%F %T')
  echo "[$ts] gpu_used=${used}MiB" >> "$QUEUE_LOG"
  if [ "$used" -le 2000 ]; then
    break
  fi
  sleep 120
done

echo "[queue] launching DIT12 state-conditioning sqrt_group training with non-zero-init DiT head" >> "$QUEUE_LOG"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
echo "[queue] using LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> "$QUEUE_LOG"
/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 \
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
  --save_freq 1000 \
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
  --run_id_note VLA-Adapter-DIT12-statecond-sqrtgroup-nozeroinit-object-1000-20260623 \
  --dit_num_blocks 12 \
  --flow_ratio 1.0 \
  --dit_num_inference_steps 20 \
  --dit_num_inference_samples 1 \
  --dit_supervised_anchor_weight 0.0 \
  --dit_anchor_blend 0.0 \
  --dit_anchor_gripper_weight 1.0 \
  --dit_anchor_gripper_bce_weight 0.0 \
  --dit_detach_flow_conditioning True \
  --dit_use_state_conditioning True \
  --dit_state_scale_mode sqrt_group \
  --dit_zero_init_adaln False \
  --dit_zero_init_output False \
  --use_adaptive_bridge True \
  --bridge_mode adaptive \
  --fixed_layer_index -1 \
  >> "$TRAIN_LOG" 2>&1

echo "[queue] training finished, starting eval" >> "$QUEUE_LOG"
./eval.sh "$CHKPT_DIR" \
  --task_suite libero_object \
  --task_ids 0 \
  --num_trials 3 \
  --action_head_type DIT \
  --dit_num_blocks 12 \
  --flow_ratio 1.0 \
  --dit_num_inference_steps 20 \
  --dit_num_inference_samples 1 \
  --dit_supervised_anchor_weight 0.0 \
  --dit_anchor_blend 0.0 \
  --dit_anchor_gripper_weight 1.0 \
  --dit_anchor_gripper_bce_weight 0.0 \
  --dit_detach_flow_conditioning True \
  --dit_use_state_conditioning True \
  --dit_state_scale_mode sqrt_group \
  --dit_zero_init_adaln False \
  --dit_zero_init_output False \
  --use_adaptive_bridge True \
  --bridge_mode adaptive \
  --fixed_layer_index -1 \
  >> "$QUEUE_LOG" 2>&1

echo "[queue] all done" >> "$QUEUE_LOG"
