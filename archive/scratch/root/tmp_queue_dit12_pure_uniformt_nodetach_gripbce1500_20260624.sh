#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
mkdir -p train_logs

PREV_PREFIX=dit12_pure_uniformt_nodetach1500
QUEUE_LOG=train_logs/queue_dit12_pure_uniformt_nodetach_gripbce1500_20260624.log
TRAIN_LOG=train_logs/dit12_pure_uniformt_nodetach_gripbce_resume1000_1500_20260624.log
RESUME_STEP=1000
RESUME_CHKPT=outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-pure-uniformt-nodetach1500-20260624--1000_chkpt
RUN_ID=configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-pure-uniformt-nodetach-gripbce-resume1000-1500-20260624

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

best_previous_verify_successes() {
  local best=0
  local log_file successes
  shopt -s nullglob
  for log_file in train_logs/verify_${PREV_PREFIX}_step*_10trials_20260624.log; do
    successes=$(parse_successes "$log_file")
    successes=${successes:-0}
    if [ "$successes" -gt "$best" ]; then
      best="$successes"
    fi
  done
  shopt -u nullglob
  echo "$best"
}

eval_step() {
  local step="$1"
  local trials="$2"
  local out_log="$3"
  local samples="${4:-1}"
  local eval_ckpt="outputs/${RUN_ID}--${step}_chkpt"
  ./eval.sh "$eval_ckpt" \
    --task_suite libero_object --task_ids 0 --num_trials "$trials" --action_head_type DIT \
    --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps 20 --dit_num_inference_samples "$samples" \
    --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_anchor_blend_was_set True \
    --dit_detach_flow_conditioning False --dit_disable_inference_anchor True --dit_pure_inference True \
    --dit_use_state_conditioning False --dit_zero_init_adaln False --dit_zero_init_output False \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
    --debug_action_scale 1.0 --debug_non_gripper_action_scale 1.0 \
    --debug_gripper_scale 1.0 --debug_gripper_bias 0.0 \
    --debug_flip_raw_gripper_output False --debug_raw_gripper_threshold 0.5 \
    >> "$out_log" 2>&1
}

eval_existing_ckpt() {
  local ckpt="$1"
  local trials="$2"
  local samples="$3"
  local out_log="$4"
  ./eval.sh "$ckpt" \
    --task_suite libero_object --task_ids 0 --num_trials "$trials" --action_head_type DIT \
    --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps 20 --dit_num_inference_samples "$samples" \
    --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_anchor_blend_was_set True \
    --dit_detach_flow_conditioning False --dit_disable_inference_anchor True --dit_pure_inference True \
    --dit_use_state_conditioning False --dit_zero_init_adaln False --dit_zero_init_output False \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
    --debug_action_scale 1.0 --debug_non_gripper_action_scale 1.0 \
    --debug_gripper_scale 1.0 --debug_gripper_bias 0.0 \
    --debug_flip_raw_gripper_output False --debug_raw_gripper_threshold 0.5 \
    >> "$out_log" 2>&1
}

echo "[queue] waiting for current uniformt/nodetach queue to finish" >> "$QUEUE_LOG"
while true; do
  ts=$(date '+%F %T')
  active_lines=$(
    pgrep -af 'tmp_queue_dit12_pure_uniformt_nodetach1500_20260624|finetune.py|run_libero_eval.py' \
      | awk -v self="$$" '$1 != self' \
      | grep -v 'tmp_queue_dit12_pure_uniformt_nodetach_gripbce1500_20260624' \
      || true
  )
  active=$(printf '%s\n' "$active_lines" | sed '/^$/d' | wc -l | tr -d ' ')
  if [ "$active" -eq 0 ]; then
    break
  fi
  echo "[$ts] active_jobs=${active}" >> "$QUEUE_LOG"
  sleep 120
done

best_verify=$(best_previous_verify_successes)
echo "[queue] previous best 10-trial verification successes=${best_verify}/10" >> "$QUEUE_LOG"
if [ "$best_verify" -ge 6 ]; then
  echo "[queue] previous run exceeded 50% in 10-trial verification; not launching fallback" >> "$QUEUE_LOG"
  exit 0
fi

echo "[queue] waiting for GPU availability" >> "$QUEUE_LOG"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  active_jobs=$( { pgrep -af 'finetune.py|run_libero_eval.py' || true; } | wc -l | tr -d ' ')
  ts=$(date '+%F %T')
  echo "[$ts] gpu_used=${used}MiB gpu_util=${util}% active_jobs=${active_jobs}" >> "$QUEUE_LOG"
  if [ "$used" -le 7000 ] && [ "$util" -le 20 ] && [ "$active_jobs" -eq 0 ]; then
    break
  fi
  sleep 60
done

for samples in 4 8; do
  preflight_log="train_logs/verify_dit12_pure_uniformt_nodetach1500_step1000_samples${samples}_10trials_20260624.log"
  echo "[queue] preflight pure DIT multi-sample verification samples=${samples}" >> "$QUEUE_LOG"
  : > "$preflight_log"
  eval_existing_ckpt "$RESUME_CHKPT" 10 "$samples" "$preflight_log"
  cat "$preflight_log" >> "$QUEUE_LOG"
  successes=$(parse_successes "$preflight_log")
  successes=${successes:-0}
  echo "[queue] preflight samples=${samples} successes=${successes}/10" >> "$QUEUE_LOG"
  if [ "$successes" -ge 6 ]; then
    echo "[queue] preflight exceeded 50%; not launching fallback training" >> "$QUEUE_LOG"
    exit 0
  fi
done

echo "[queue] launching pure DIT12 resume-from-1000 with gripper BCE, no anchor, no gate" >> "$QUEUE_LOG"
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
  --resume_step "$RESUME_STEP" \
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
  --max_steps 1500 \
  --save_freq 250 \
  --save_latest_checkpoint_only False \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --use_constant_lr True \
  --use_pro_version True \
  --eval_on_checkpoint False \
  --ddp_find_unused_params True \
  --wandb_entity 323660583-huazhong-university-of-science-and-technology \
  --wandb_project libero_object_no_noops \
  --run_id_note VLA-Adapter-DIT12-pure-uniformt-nodetach-gripbce1500-20260624 \
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
  --dit_flow_gripper_loss_weight 2.0 \
  --dit_flow_gripper_bce_weight 0.2 \
  --dit_sample_t_mode_flow uniform \
  --dit_detach_flow_conditioning False \
  --dit_use_state_conditioning False \
  --dit_zero_init_adaln False \
  --dit_zero_init_output False \
  --use_adaptive_bridge True \
  --bridge_mode adaptive \
  --fixed_layer_index -1 \
  >> "$TRAIN_LOG" 2>&1

for step in 1250 1500; do
  eval_ckpt="outputs/${RUN_ID}--${step}_chkpt"
  if [ ! -d "$eval_ckpt" ]; then
    continue
  fi
  eval_log="train_logs/dit12_pure_uniformt_nodetach_gripbce_resume1000_1500_step${step}_eval3_20260624.log"
  verify_log="train_logs/verify_dit12_pure_uniformt_nodetach_gripbce_resume1000_1500_step${step}_10trials_20260624.log"
  echo "[queue] eval step=${step}" >> "$QUEUE_LOG"
  : > "$eval_log"
  eval_step "$step" 3 "$eval_log"
  cat "$eval_log" >> "$QUEUE_LOG"
  successes=$(parse_successes "$eval_log")
  successes=${successes:-0}
  echo "[queue] step=${step} successes=${successes}/3" >> "$QUEUE_LOG"
  if [ "$successes" -ge 2 ]; then
    echo "[queue] step=${step} reached >=2/3; launching 10-trial verification" >> "$QUEUE_LOG"
    : > "$verify_log"
    eval_step "$step" 10 "$verify_log"
    cat "$verify_log" >> "$QUEUE_LOG"
  fi
done

echo "[queue] all done" >> "$QUEUE_LOG"
