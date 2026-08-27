#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/autodl-fs/data/skill_depth/code/action_expert_prefix_trial
cd "$REPO_DIR"
PY=/autodl-fs/data/skill_depth/envs/vla-flow/bin/python
TORCHRUN=/autodl-fs/data/skill_depth/envs/vla-flow/bin/torchrun
LOCAL_SYNC_ROOT=/autodl-fs/data/skill_depth/code/action_expert_prefix_trial
MLP_CKPT=/autodl-fs/data/skill_depth/outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-MLP-object-5000-20260618_213642--5000_chkpt
QUEUE_LOG=train_logs/queue_v48_dit_prefix_after_flowmlp.log
mkdir -p train_logs outputs

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:/autodl-fs/data/skill_depth/code/action_expert_prefix_trial/LIBERO:/autodl-fs/data/skill_depth/code/robosuite:${PYTHONPATH:-}
export PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
export HF_HOME=/autodl-fs/data/skill_depth/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/autodl-fs/data/skill_depth/cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

log(){ echo "[$(date --iso-8601=seconds)] $*" | tee -a "$QUEUE_LOG"; }

wait_flowmlp(){
  while true; do
    a=$(tail -n 1 experiment_results/flowmlp_ablation_direct_only_seed9_v48.status 2>/dev/null || true)
    b=$(tail -n 1 experiment_results/flowmlp_ablation_routing_direct_seed9_v48.status 2>/dev/null || true)
    log "waiting flowmlp direct_only='${a}' routing_direct='${b}'"
    [[ "$a" == COMPLETE* && "$b" == COMPLETE* ]] && break
    sleep 120
  done
}

sync_unified_dit_source(){
  test -f prismatic/models/flow_matching_head.py.unified_local
  cp -f prismatic/models/flow_matching_head.py prismatic/models/flow_matching_head.py.before_unified_prefix_$(date +%Y%m%d_%H%M%S)
  cp -f prismatic/models/flow_matching_head.py.unified_local prismatic/models/flow_matching_head.py
  log "installed unified local flow_matching_head.py after FlowMLP queue"
}

wait_gpu(){
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    jobs=$(ps -eo args= | awk '/[v]la-scripts\/finetune.py|[r]un_libero_eval.py/ {n++} END {print n+0}')
    old_queue=$(ps -eo pid=,args= | awk '!/grep|awk|pgrep|queue_v48_dit_prefix/ && /tmp_continue_pure_dit_12000to20000_v48_lr1e4_20260710.sh|queue_dit12_h896_std_action_prefix_scratch_continue12000to20000_v48_lr1e4_20260710/ {n++} END {print n+0}')
    log "gpu free=${free}MiB jobs=${jobs} old_dit_queue=${old_queue}"
    [[ "$free" -gt 42000 && "$jobs" -eq 0 && "$old_queue" -eq 0 ]] && break
    sleep 60
  done
}

run_one(){
  local variant=$1 use_skill=$2
  local label="DIT-prefix-${variant}-from-MLP5000-v48"
  local run_id="DIT-prefix-${variant}-from-MLP5000-v48-20260712"
  local train_log="train_logs/${label}.train.log"
  local eval_log="train_logs/${label}.eval.log"
  local status="train_logs/${label}.status"
  if [[ -f "$status" && "$(cat "$status")" == COMPLETE* ]]; then
    log "skip completed ${variant}"
    return
  fi
  wait_gpu
  checkpoint=$(find outputs -maxdepth 1 -type d -name "*${run_id}--10000_chkpt" | sort | tail -1)
  if [[ -n "$checkpoint" ]]; then
    log "reuse existing ${variant} checkpoint=${checkpoint}"
  else
    log "start ${variant} use_skill=${use_skill}"
    cp -f prismatic/models/flow_matching_head.py "prismatic/models/flow_matching_head.py.before_prefix_${variant}_$(date +%Y%m%d_%H%M%S)"
    "$TORCHRUN" --standalone --nnodes 1 --nproc_per_node 1 vla-scripts/finetune.py \
    --config_file_path /autodl-fs/data/skill_depth/pretrained_models/configs \
    --vlm_path /autodl-fs/data/skill_depth/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --data_root_dir /autodl-fs/data/skill_depth/data/libero --dataset_name libero_object_no_noops --run_root_dir outputs \
    --train_from checkpoint --resum_vla_path "$MLP_CKPT" --resume_step 5000 --resume_load_training_state False \
    --vlm_training freeze_lora --action_head_type DIT --use_lora True --lora_rank 64 --lora_dropout 0.0 \
    --freeze_vlm True --merge_lora_during_training False --use_minivlm True --use_pro_version True \
    --use_proprio True --num_images_in_input 2 --use_film False --learning_rate 0.0002 --use_constant_lr True \
    --lr_warmup_steps 200 --max_steps 10000 --save_freq 5000 --save_latest_checkpoint_only False --image_aug True \
    --dit_num_blocks 12 --dit_num_inference_steps 20 \
    --dit_num_inference_samples 1 --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 \
    --dit_inference_residual_scale 1.0 --dit_flow_xyz_loss_weight 1.0 --dit_flow_rot_loss_weight 1.0 \
    --dit_flow_gripper_loss_weight 1.0 --dit_sample_t_mode_flow beta --dit_detach_flow_conditioning False \
    --dit_use_state_conditioning True --dit_state_include_task_tokens True --dit_state_use_chunk_pos True \
    --dit_state_proprio_mode concat --dit_condition_mode task_only --dit_condition_injection_mode action_expert_prefix \
    --dit_include_prompt_tokens True --dit_task_token_mode vision_prompt --dit_use_latent_skill_token "$use_skill" \
    --dit_num_skill_tokens 16 --dit_skill_token_dim 128 --dit_skill_temperature 1.0 \
    --dit_zero_init_adaln True --dit_zero_init_output False --use_adaptive_bridge True --bridge_mode adaptive \
    --fixed_layer_index -1 --ddp_find_unused_params True --flow_ratio 1.0 --seed 7 \
    --run_id_note "$run_id" --run_id_override "$run_id" --batch_size 8 --grad_accumulation_steps 2 \
    > "$train_log" 2>&1
    checkpoint=$(find outputs -maxdepth 1 -type d -name "*${run_id}--10000_chkpt" | sort | tail -1)
  fi
  test -n "$checkpoint"
  wait_gpu
  "$PY" experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint "$checkpoint" \
    --action_head_type DIT --task_suite_name libero_object --task_ids 3,4,7,9 --num_trials_per_task 10 \
    --use_depth_interface False --depth_interface_mode none --use_adaptive_bridge True --bridge_mode adaptive \
    --fixed_layer_index -1 --flow_ratio 1.0 --dit_num_blocks 12 \
    --dit_num_inference_steps 20 --dit_num_inference_samples 1 --dit_supervised_anchor_weight 0.0 \
    --dit_anchor_blend 0.0 --dit_inference_residual_scale 1.0 --dit_flow_xyz_loss_weight 1.0 \
    --dit_flow_rot_loss_weight 1.0 --dit_flow_gripper_loss_weight 1.0 --dit_disable_inference_anchor True \
    --dit_pure_inference True --dit_use_state_conditioning True --dit_state_include_task_tokens True \
    --dit_state_use_chunk_pos True --dit_state_proprio_mode concat --dit_condition_mode task_only \
    --dit_condition_injection_mode action_expert_prefix --dit_include_prompt_tokens True \
    --dit_task_token_mode vision_prompt --dit_use_latent_skill_token "$use_skill" --dit_num_skill_tokens 16 \
    --dit_skill_token_dim 128 --dit_skill_temperature 1.0 --use_proprio True --num_images_in_input 2 \
    --use_minivlm True --use_pro_version True --use_wandb False --center_crop True --seed 7 \
    > "$eval_log" 2>&1
  echo "COMPLETE $(date --iso-8601=seconds) checkpoint=$checkpoint" > "$status"
  log "complete ${variant}"
}

log "queue start"
wait_flowmlp
sync_unified_dit_source
run_one baseline False
run_one skill True
log "queue done"
