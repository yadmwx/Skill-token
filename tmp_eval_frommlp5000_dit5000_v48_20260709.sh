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

LABEL=dit12_h896_std_action_prefix_pure_frommlp5000_dit5000_v48_20260709
CHKPT=/autodl-fs/data/skill_depth/outputs/configs+libero_object_no_noops+b4+lr-0.0002+lora-r32+dropout-0.0--image_aug--DIT12-h896-std-action-prefix-pure-frommlp5000-b4ga4-16000-v48-20260704--5000_chkpt
LOG=train_logs/${LABEL}_hardtasks_eval5.log
STATUS=train_logs/${LABEL}_hardtasks_eval5.status

echo "[$(date '+%F %T')] eval start chkpt=$CHKPT" | tee "$LOG"
"$PY" experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CHKPT" \
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
  >> "$LOG" 2>&1
rc=$?
echo "$rc" > "$STATUS"
echo "[$(date '+%F %T')] eval done rc=$rc" | tee -a "$LOG"
exit "$rc"
