#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
mkdir -p train_logs

BASE_CKPT=outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-pure-uniformt-nodetach1500-20260624--1000_chkpt
QUEUE_LOG=train_logs/queue_dit12_pure_multisample_rescue_20260627.log
PREV_QUEUE_PATTERN='tmp_queue_dit12_pure_seed_lowlr_rescue_20260626.sh'

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

wait_for_previous_queue() {
  echo "[queue] waiting for previous DIT rescue queue to finish" >> "$QUEUE_LOG"
  while pgrep -af "$PREV_QUEUE_PATTERN" | grep -v "tmp_queue_dit12_pure_multisample_rescue_20260627" >/dev/null 2>&1; do
    echo "[$(date '+%F %T')] previous queue still active" >> "$QUEUE_LOG"
    sleep 120
  done
}

wait_for_gpu() {
  echo "[queue] waiting for GPU availability" >> "$QUEUE_LOG"
  while true; do
    local used util active_jobs ts
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    active_jobs=$(ps -eo args= | awk '/[f]inetune.py|[r]un_libero_eval.py|[t]mp_probe_dit_checkpoint_actions/ {n++} END {print n + 0}')
    ts=$(date '+%F %T')
    echo "[$ts] gpu_used=${used}MiB gpu_util=${util}% active_jobs=${active_jobs}" >> "$QUEUE_LOG"
    if [ "$used" -le 7000 ] && [ "$util" -le 20 ] && [ "$active_jobs" -eq 0 ]; then
      break
    fi
    sleep 60
  done
}

eval_ckpt() {
  local ckpt="$1"
  local trials="$2"
  local steps="$3"
  local samples="$4"
  local seed="$5"
  local out_log="$6"
  ./eval.sh "$ckpt" \
    --task_suite libero_object --task_ids 0 --num_trials "$trials" --action_head_type DIT \
    --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps "$steps" --dit_num_inference_samples "$samples" \
    --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_anchor_blend_was_set True \
    --dit_detach_flow_conditioning False --dit_disable_inference_anchor True --dit_pure_inference True \
    --dit_use_state_conditioning False --dit_zero_init_adaln False --dit_zero_init_output False \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
    --debug_action_scale 1.0 --debug_non_gripper_action_scale 1.0 \
    --debug_gripper_scale 1.0 --debug_gripper_bias 0.0 \
    --debug_flip_raw_gripper_output False --debug_raw_gripper_threshold 0.5 \
    --debug_dit_group_action_tokens_to_chunk False \
    --seed "$seed" \
    >> "$out_log" 2>&1
}

run_multisample_sweep() {
  local steps samples seed log successes verify_log
  for samples in 4 8; do
    for steps in 20 30; do
      for seed in 0 1 2 3 7 11 23; do
        wait_for_gpu
        log="train_logs/dit12_pure_multisample_step1000_steps${steps}_samples${samples}_seed${seed}_eval3_20260627.log"
        echo "[queue] preflight multisample ckpt=step1000 steps=${steps} samples=${samples} seed=${seed}" >> "$QUEUE_LOG"
        : > "$log"
        eval_ckpt "$BASE_CKPT" 3 "$steps" "$samples" "$seed" "$log"
        cat "$log" >> "$QUEUE_LOG"
        successes=$(parse_successes "$log")
        successes=${successes:-0}
        echo "[queue] preflight multisample steps=${steps} samples=${samples} seed=${seed} successes=${successes}/3" >> "$QUEUE_LOG"
        if [ "$successes" -ge 2 ]; then
          wait_for_gpu
          verify_log="train_logs/verify_dit12_pure_multisample_step1000_steps${steps}_samples${samples}_seed${seed}_10trials_20260627.log"
          echo "[queue] multisample candidate reached >=2/3; launching 10-trial verification steps=${steps} samples=${samples} seed=${seed}" >> "$QUEUE_LOG"
          : > "$verify_log"
          eval_ckpt "$BASE_CKPT" 10 "$steps" "$samples" "$seed" "$verify_log"
          cat "$verify_log" >> "$QUEUE_LOG"
          successes=$(parse_successes "$verify_log")
          successes=${successes:-0}
          echo "[queue] multisample verify steps=${steps} samples=${samples} seed=${seed} successes=${successes}/10" >> "$QUEUE_LOG"
          if [ "$successes" -ge 6 ]; then
            echo "[queue] multisample pure DIT exceeded 50%; stopping" >> "$QUEUE_LOG"
            exit 0
          fi
        fi
      done
    done
  done
}

echo "[queue] start $(date '+%F %T')" >> "$QUEUE_LOG"
wait_for_previous_queue
run_multisample_sweep
echo "[queue] all done $(date '+%F %T')" >> "$QUEUE_LOG"
