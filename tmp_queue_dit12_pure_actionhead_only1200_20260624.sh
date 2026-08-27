#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
mkdir -p train_logs

QUEUE_LOG=train_logs/queue_dit12_pure_actionhead_only1200_20260624.log
TRAIN_LOG=train_logs/dit12_pure_actionhead_only1200_20260624.log
RESUME_CHKPT=outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-fullaction-pure-gripfix-nozeroinit-object-1000-20260623--1000_chkpt
RUN_ID=configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-fullaction-pure-actionhead-only1200-20260624
FINAL_CHKPT="outputs/${RUN_ID}--1200_chkpt"

echo "[queue] waiting for GPU availability" >> "$QUEUE_LOG"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  ts=$(date '+%F %T')
  echo "[$ts] gpu_used=${used}MiB gpu_util=${util}%" >> "$QUEUE_LOG"
  if [ "$used" -le 7000 ] && [ "$util" -le 20 ]; then
    break
  fi
  sleep 60
done

echo "[queue] short DIT action-head-only training from 1000 to 1200" >> "$QUEUE_LOG"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path "$RESUME_CHKPT" \
  --data_root_dir data/libero \
  --dataset_name libero_object_no_noops \
  --run_root_dir outputs \
  --train_from checkpoint \
  --resum_vla_path "$RESUME_CHKPT" \
  --resume_step 1000 \
  --resume_load_training_state False \
  --train_action_head_only True \
  --freeze_proprio_projector_for_action_head_only True \
  --vlm_training freeze_lora \
  --action_head_type DIT \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_minivlm True \
  --merge_lora_during_training False \
  --image_aug True \
  --num_steps_before_decay 20000 \
  --max_steps 1200 \
  --save_freq 1200 \
  --save_latest_checkpoint_only False \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --use_constant_lr True \
  --use_pro_version True \
  --eval_on_checkpoint False \
  --ddp_find_unused_params True \
  --wandb_entity 323660583-huazhong-university-of-science-and-technology \
  --wandb_project libero_object_no_noops \
  --run_id_note VLA-Adapter-DIT12-fullaction-pure-actionhead-only1200-20260624 \
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
  --dit_flow_gripper_loss_weight 4.0 \
  --dit_flow_gripper_bce_weight 0.2 \
  --dit_detach_flow_conditioning True \
  --dit_use_state_conditioning False \
  --dit_zero_init_adaln False \
  --dit_zero_init_output False \
  --use_adaptive_bridge True \
  --bridge_mode adaptive \
  --fixed_layer_index -1 \
  >> "$TRAIN_LOG" 2>&1

for threshold in 0.5 0.9; do
  echo "[queue] pure eval threshold=${threshold}" >> "$QUEUE_LOG"
  ./eval.sh "$FINAL_CHKPT" \
    --task_suite libero_object --task_ids 0 --num_trials 3 --action_head_type DIT \
    --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps 20 --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 \
    --dit_detach_flow_conditioning True --dit_disable_inference_anchor True --dit_pure_inference True \
    --dit_zero_init_adaln False --dit_zero_init_output False \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
    --debug_raw_gripper_threshold "$threshold" \
    >> "$QUEUE_LOG" 2>&1
done

echo "[queue] all done" >> "$QUEUE_LOG"
