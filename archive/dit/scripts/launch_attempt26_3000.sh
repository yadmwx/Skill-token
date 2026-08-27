#!/usr/bin/env bash
set -uo pipefail

RUN=/root/autodl-tmp/depthcurriculum_seed9_20260816_attempt26_multidepthdit_shared
SRC=/root/autodl-tmp/e07_a100_src
mkdir -p "$RUN/logs" "$RUN/outputs"
cd "$SRC"

export PYTHONPATH="$SRC"
export VLA_DISABLE_CUDNN=1
export WANDB_MODE=disabled

/root/miniconda3/bin/torchrun --standalone --nnodes=1 --nproc-per-node=1 vla-scripts/finetune.py \
  --config_file_path /root/autodl-tmp/e07_a100_hf/e07_base_ckpt \
  --vlm_path /root/autodl-tmp/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --data_root_dir /root/autodl-tmp/e07_a100_data \
  --dataset_name libero_object_no_noops \
  --run_root_dir "$RUN/outputs" \
  --action_head_type DIT \
  --train_from vlm_base \
  --vlm_training freeze_lora \
  --freeze_vlm True \
  --use_lora True \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --ddp_find_unused_params True \
  --max_steps 3000 \
  --save_freq 1000 \
  --learning_rate 2e-4 \
  --use_constant_lr True \
  --use_cosine_schedule False \
  --dit_num_blocks 12 \
  --dit_num_inference_steps 10 \
  --dit_num_inference_samples 1 \
  --dit_supervised_anchor_weight 0.0 \
  --dit_anchor_blend 0.0 \
  --dit_flow_rot_loss_weight 2.0 \
  --dit_detach_flow_conditioning False \
  --dit_condition_mode full \
  --dit_condition_injection_mode cross_attn \
  --dit_include_prompt_tokens True \
  --dit_task_token_mode vision_prompt \
  --dit_zero_init_adaln True \
  --dit_zero_init_output True \
  --use_adaptive_bridge False \
  --fixed_layer_index 24 \
  --dit_balanced_train_layers 1,5,9,13,24 \
  --use_minivlm True \
  --use_pro_version True \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --seed 9 \
  --run_id_override DIT-pureflow-canonical-a26-multidepth \
  --run_id_note a26-multidepth \
  > "$RUN/logs/train.log" 2>&1
code=$?
printf '%s\n' "$code" > "$RUN/train.exit_code"
exit "$code"
