#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/autodl-fs/data/skill_depth/code/action_expert_prefix_trial
cd "$REPO_DIR"
mkdir -p train_logs
PY=/autodl-fs/data/skill_depth/envs/vla-flow/bin/python
TORCHRUN=/autodl-fs/data/skill_depth/envs/vla-flow/bin/torchrun
export CUDA_VISIBLE_DEVICES=0 PYOPENGL_PLATFORM=osmesa MUJOCO_GL=osmesa TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTHONPATH=.:$REPO_DIR/LIBERO:$REPO_DIR/robosuite:${PYTHONPATH:-}
export HF_HOME=/autodl-fs/data/skill_depth/cache/huggingface HUGGINGFACE_HUB_CACHE=/autodl-fs/data/skill_depth/cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

LABEL=flowmlp_directskill_from10000_15k_v48_20260711
RUN_ID=FlowMLP-directskill-from10000-15k-v48-20260711
TRAIN_LOG=train_logs/${LABEL}.log
QUEUE_LOG=train_logs/queue_${LABEL}.log
STATUS=train_logs/${LABEL}.status
BASE_CKPT=/autodl-fs/data/skill_depth/code/action_expert_prefix_trial/outputs/configs+libero_object_no_noops+b8+lr-0.0002+lora-r64+dropout-0.0--image_aug--FlowMLP-skilltoken-frommlp5000-10k-v48-20260711--10000_chkpt

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"; }
wait_gpu(){
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    jobs=$(ps -eo comm=,args= | awk '$1 ~ /python/ && $0 ~ /vla-scripts\/finetune.py|run_libero_eval.py/ {n++} END {print n+0}')
    log "gpu free=${free}MiB util=${util}% vla_jobs=${jobs}"
    if [ "$free" -gt 42000 ] && [ "$util" -lt 20 ] && [ "$jobs" -eq 0 ]; then break; fi
    sleep 60
  done
}
success_count(){ grep -A3 'Total successes:' "$1" | grep -Eo '[0-9]+' | head -1 || echo 0; }

log "queue start"
wait_gpu
set +e
"$TORCHRUN" --standalone --nnodes 1 --nproc_per_node 1 vla-scripts/finetune.py \
  --config_file_path /autodl-fs/data/skill_depth/pretrained_models/configs \
  --vlm_path /autodl-fs/data/skill_depth/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --data_root_dir /autodl-fs/data/skill_depth/data/libero --dataset_name libero_object_no_noops --run_root_dir outputs \
  --train_from checkpoint --resum_vla_path "$BASE_CKPT" --resume_step 10000 --resume_load_training_state False \
  --vlm_training freeze_lora --action_head_type FlowMLP --use_lora True --lora_rank 64 --lora_dropout 0.0 \
  --freeze_vlm True --merge_lora_during_training False --use_minivlm True --use_pro_version True \
  --use_proprio True --num_images_in_input 2 --use_film False --learning_rate 0.0001 --use_constant_lr True \
  --lr_warmup_steps 200 --max_steps 15000 --save_freq 5000 --save_latest_checkpoint_only False --image_aug True \
  --flowmlp_use_latent_skill_token True --flowmlp_num_skill_tokens 16 --flowmlp_skill_token_dim 128 \
  --flowmlp_skill_temperature 1.0 --flowmlp_skill_entropy_weight 0.0 --flowmlp_num_inference_steps 5 \
  --flowmlp_num_inference_samples 8 --flowmlp_supervised_anchor_weight 0.0 --flowmlp_anchor_blend 0.0 \
  --flowmlp_detach_flow_conditioning False --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
  --ddp_find_unused_params True --run_id_note "$RUN_ID" \
  --run_id_override configs+libero_object_no_noops+b8+lr-0.0001+lora-r64+dropout-0.0--image_aug--${RUN_ID} \
  --batch_size 8 --grad_accumulation_steps 2 > "$TRAIN_LOG" 2>&1
rc=$?
echo "$rc" > "$STATUS"
set -e
log "train finished rc=$rc"
tail -120 "$TRAIN_LOG" >> "$QUEUE_LOG" || true
[ "$rc" -eq 0 ] || exit "$rc"
CHKPT=$(find outputs -maxdepth 1 -type d -name "*${RUN_ID}--15000_chkpt" | sort | tail -1)
[ -n "$CHKPT" ] || { log "missing checkpoint"; exit 2; }
log "checkpoint ready: $CHKPT"
wait_gpu
EVAL_LOG=train_logs/${LABEL}_step15000_hardtasks_eval5.log
set +e
"$PY" experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CHKPT" --action_head_type FlowMLP --task_suite_name libero_object --task_ids 3,4,7,9 --num_trials_per_task 5 \
  --use_depth_interface False --depth_interface_mode none --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
  --flowmlp_use_latent_skill_token True --flowmlp_num_skill_tokens 16 --flowmlp_skill_token_dim 128 --flowmlp_skill_temperature 1.0 \
  --flowmlp_skill_entropy_weight 0.0 --flowmlp_num_inference_steps 5 --flowmlp_num_inference_samples 8 \
  --flowmlp_supervised_anchor_weight 0.0 --flowmlp_anchor_blend 0.0 --flowmlp_detach_flow_conditioning False \
  --use_proprio True --num_images_in_input 2 --use_film False --use_minivlm True --use_pro_version True --use_wandb False --center_crop True --seed 7 \
  > "$EVAL_LOG" 2>&1
eval_rc=$?
echo "$eval_rc" > "${EVAL_LOG}.status"
grep -n 'Final results\|Total episodes\|Total successes\|Overall success rate\|Current task success rate' "$EVAL_LOG" | tail -120 >> "$QUEUE_LOG" || true
log "eval finished rc=$eval_rc successes=$(success_count "$EVAL_LOG")"
exit "$eval_rc"
