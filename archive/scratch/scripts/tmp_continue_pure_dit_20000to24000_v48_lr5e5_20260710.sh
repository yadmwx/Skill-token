#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/autodl-fs/data/skill_depth/code/action_expert_prefix_trial
cd "$REPO_DIR"
mkdir -p train_logs

PY=/autodl-fs/data/skill_depth/envs/vla-flow/bin/python
unset LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:/autodl-fs/data/skill_depth/code/action_expert_prefix_trial/LIBERO:/autodl-fs/data/skill_depth/code/LIBERO:/autodl-fs/data/skill_depth/code/action_expert_prefix_trial/robosuite:/autodl-fs/data/skill_depth/code/robosuite:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export HF_HOME=/autodl-fs/data/skill_depth/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/autodl-fs/data/skill_depth/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SRC=outputs/configs+libero_object_no_noops+b8+lr-0.0001+lora-r32+dropout-0.0--image_aug--DIT12-h896-std-action-prefix-scratch-continue12000to20000-v48-lr1e4-20260710--20000_chkpt
RUN_ID=DIT12-h896-std-action-prefix-scratch-continue20000to24000-v48-lr5e5-20260710

"$PY" /autodl-fs/data/skill_depth/envs/vla-flow/bin/torchrun --standalone --nnodes 1 --nproc_per_node 1 vla-scripts/finetune.py \
  --config_file_path /autodl-fs/data/skill_depth/pretrained_models/configs \
  --vlm_path /autodl-fs/data/skill_depth/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --data_root_dir /autodl-fs/data/skill_depth/data/libero --dataset_name libero_object_no_noops --run_root_dir outputs \
  --resume_load_training_state False --train_from checkpoint --resum_vla_path "$SRC" --resume_step 20000 \
  --vlm_training freeze_lora --action_head_type DIT --use_lora True --lora_rank 32 --lora_dropout 0.0 \
  --freeze_vlm True --merge_lora_during_training True --use_minivlm True --use_pro_version True --use_proprio True \
  --num_images_in_input 2 --use_film False --learning_rate 0.00005 --use_constant_lr True --lr_warmup_steps 200 \
  --max_steps 24000 --save_freq 2000 --save_latest_checkpoint_only False --image_aug True --use_wandb False --wandb_log_freq 50 \
  --dit_hidden_dim 896 --dit_num_blocks 12 --dit_num_inference_steps 20 --dit_num_inference_samples 1 \
  --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_inference_residual_scale 1.0 \
  --dit_anchor_gripper_weight 1.0 --dit_anchor_gripper_bce_weight 0.0 --dit_flow_xyz_loss_weight 1.0 \
  --dit_flow_rot_loss_weight 1.0 --dit_flow_gripper_loss_weight 1.0 --dit_endpoint_loss_weight 1.0 \
  --dit_flow_gripper_bce_weight 0.0 --dit_sample_t_mode_flow beta --dit_detach_flow_conditioning False \
  --dit_use_state_conditioning True --dit_state_include_task_tokens True --dit_state_use_chunk_pos True \
  --dit_state_proprio_mode concat --dit_fuse_state_into_action_tokens True --dit_condition_mode task_only \
  --dit_condition_injection_mode action_expert_prefix --dit_include_prompt_tokens True --dit_task_token_mode vision_prompt \
  --dit_zero_init_adaln True --dit_zero_init_output False --use_adaptive_bridge True --bridge_mode adaptive \
  --fixed_layer_index -1 --ddp_find_unused_params True --flow_ratio 1.0 --run_id_note "$RUN_ID" \
  --run_id_override configs+libero_object_no_noops+b8+lr-0.00005+lora-r32+dropout-0.0--image_aug--$RUN_ID \
  --batch_size 8 --grad_accumulation_steps 2
