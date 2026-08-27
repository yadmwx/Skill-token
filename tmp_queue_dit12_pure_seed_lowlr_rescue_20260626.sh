#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
mkdir -p train_logs

PYTHON=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python
TORCHRUN=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/torchrun
export PYTHONPATH=/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
export TOKENIZERS_PARALLELISM=false

BASE_CKPT=outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-pure-uniformt-nodetach1500-20260624--1000_chkpt
RUN_ID=configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-pure-uniformt-nodetach-lowlr-resume1000-1300-20260626
QUEUE_LOG=train_logs/queue_dit12_pure_seed_lowlr_rescue_20260626.log
TRAIN_LOG=train_logs/dit12_pure_uniformt_nodetach_lowlr_resume1000_1300_20260626.log

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
    local used util active_jobs ts
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    active_jobs=$(ps -eo args= | awk '/[f]inetune.py|[r]un_libero_eval.py|[t]mp_probe_dit_checkpoint_actions/ {n++} END {print n + 0}')
    ts=$(date '+%F %T')
    echo "[$ts] gpu_used=${used}MiB gpu_util=${util}% active_jobs=${active_jobs}" >> "$QUEUE_LOG"
    if [ "$used" -le 7000 ] && [ "$util" -le 20 ]; then
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
  local group_action_tokens="${6:-False}"
  ./eval.sh "$ckpt" \
    --task_suite libero_object --task_ids 0 --num_trials "$trials" --action_head_type DIT \
    --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps "$steps" --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 0.0 --dit_anchor_blend 0.0 --dit_anchor_blend_was_set True \
    --dit_detach_flow_conditioning False --dit_disable_inference_anchor True --dit_pure_inference True \
    --dit_use_state_conditioning False --dit_zero_init_adaln False --dit_zero_init_output False \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
    --debug_action_scale 1.0 --debug_non_gripper_action_scale 1.0 \
    --debug_gripper_scale 1.0 --debug_gripper_bias 0.0 \
    --debug_flip_raw_gripper_output False --debug_raw_gripper_threshold 0.5 \
    --debug_dit_group_action_tokens_to_chunk "$group_action_tokens" \
    --seed "$seed" \
    >> "$out_log" 2>&1
}

run_probe() {
  local out_log=train_logs/probe_dit12_pure_uniformt_nodetach_step1000_actions_20260626.log
  echo "[queue] probing pure DIT action distribution on best checkpoint" >> "$QUEUE_LOG"
  : > "$out_log"
  "$PYTHON" scripts/tmp_probe_dit_checkpoint_actions_20260624.py \
    --checkpoint "$BASE_CKPT" \
    --episodes 5 \
    --task_suite libero_object \
    --task_id 0 \
    --dit_num_inference_steps 20 \
    --dit_num_inference_samples 1 \
    --dit_detach_flow_conditioning False \
    --dit_zero_init_adaln False \
    --dit_zero_init_output False \
    --use_adaptive_bridge True \
    --bridge_mode adaptive \
    --compare_flow False \
    --seed 7 \
    >> "$out_log" 2>&1 || true
  cat "$out_log" >> "$QUEUE_LOG"
}

preflight_seed_sweep() {
  local steps seed log successes verify_log
  for steps in 10 20 30; do
    for seed in 0 1 2 3 7 11 23; do
      log="train_logs/dit12_pure_uniformt_nodetach_step1000_steps${steps}_seed${seed}_eval3_20260626.log"
      echo "[queue] preflight ckpt=step1000 steps=${steps} seed=${seed}" >> "$QUEUE_LOG"
      : > "$log"
      eval_ckpt "$BASE_CKPT" 3 "$steps" "$seed" "$log"
      cat "$log" >> "$QUEUE_LOG"
      successes=$(parse_successes "$log")
      successes=${successes:-0}
      echo "[queue] preflight steps=${steps} seed=${seed} successes=${successes}/3" >> "$QUEUE_LOG"
      if [ "$successes" -ge 2 ]; then
        verify_log="train_logs/verify_dit12_pure_uniformt_nodetach_step1000_steps${steps}_seed${seed}_10trials_20260626.log"
        echo "[queue] candidate reached >=2/3; launching 10-trial verification steps=${steps} seed=${seed}" >> "$QUEUE_LOG"
        : > "$verify_log"
        eval_ckpt "$BASE_CKPT" 10 "$steps" "$seed" "$verify_log"
        cat "$verify_log" >> "$QUEUE_LOG"
        successes=$(parse_successes "$verify_log")
        successes=${successes:-0}
        echo "[queue] verify steps=${steps} seed=${seed} successes=${successes}/10" >> "$QUEUE_LOG"
        if [ "$successes" -ge 6 ]; then
          echo "[queue] pure DIT exceeded 50%; stopping rescue queue" >> "$QUEUE_LOG"
          exit 0
        fi
      fi
    done
  done
}

preflight_grouped_action_token_sweep() {
  local steps seed log successes verify_log
  for steps in 10 20 30; do
    for seed in 0 1 2 3 7 11 23; do
      log="train_logs/dit12_pure_uniformt_nodetach_step1000_groupctx_steps${steps}_seed${seed}_eval3_20260626.log"
      echo "[queue] preflight grouped action-token context ckpt=step1000 steps=${steps} seed=${seed}" >> "$QUEUE_LOG"
      : > "$log"
      eval_ckpt "$BASE_CKPT" 3 "$steps" "$seed" "$log" True
      cat "$log" >> "$QUEUE_LOG"
      successes=$(parse_successes "$log")
      successes=${successes:-0}
      echo "[queue] preflight groupctx steps=${steps} seed=${seed} successes=${successes}/3" >> "$QUEUE_LOG"
      if [ "$successes" -ge 2 ]; then
        verify_log="train_logs/verify_dit12_pure_uniformt_nodetach_step1000_groupctx_steps${steps}_seed${seed}_10trials_20260626.log"
        echo "[queue] grouped action-token context reached >=2/3; launching 10-trial verification steps=${steps} seed=${seed}" >> "$QUEUE_LOG"
        : > "$verify_log"
        eval_ckpt "$BASE_CKPT" 10 "$steps" "$seed" "$verify_log" True
        cat "$verify_log" >> "$QUEUE_LOG"
        successes=$(parse_successes "$verify_log")
        successes=${successes:-0}
        echo "[queue] groupctx verify steps=${steps} seed=${seed} successes=${successes}/10" >> "$QUEUE_LOG"
        if [ "$successes" -ge 6 ]; then
          echo "[queue] grouped pure DIT exceeded 50%; stopping rescue queue" >> "$QUEUE_LOG"
          exit 0
        fi
      fi
    done
  done
}

train_lowlr_resume() {
  echo "[queue] launching low-LR pure DIT resume from best step1000, no anchor, no gate" >> "$QUEUE_LOG"
  export CUDA_VISIBLE_DEVICES=0
  if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  fi

  "$TORCHRUN" --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
    --config_file_path "$BASE_CKPT" \
    --data_root_dir data/libero \
    --dataset_name libero_object_no_noops \
    --run_root_dir outputs \
    --train_from checkpoint \
    --resum_vla_path "$BASE_CKPT" \
    --resume_step 1000 \
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
    --max_steps 1300 \
    --save_freq 100 \
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
    --run_id_note VLA-Adapter-DIT12-pure-uniformt-nodetach-lowlr-resume1000-1300-20260626 \
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
    --dit_use_state_conditioning False \
    --dit_zero_init_adaln False \
    --dit_zero_init_output False \
    --use_adaptive_bridge True \
    --bridge_mode adaptive \
    --fixed_layer_index -1 \
    >> "$TRAIN_LOG" 2>&1
}

eval_lowlr_checkpoints() {
  local step seed log successes verify_log ckpt
  for step in 1100 1200 1300; do
    ckpt="outputs/${RUN_ID}--${step}_chkpt"
    [ -d "$ckpt" ] || continue
    for seed in 7 0 11; do
      log="train_logs/dit12_pure_uniformt_nodetach_lowlr_resume1000_1300_step${step}_seed${seed}_eval3_20260626.log"
      echo "[queue] eval lowlr step=${step} seed=${seed}" >> "$QUEUE_LOG"
      : > "$log"
      eval_ckpt "$ckpt" 3 20 "$seed" "$log"
      cat "$log" >> "$QUEUE_LOG"
      successes=$(parse_successes "$log")
      successes=${successes:-0}
      echo "[queue] lowlr step=${step} seed=${seed} successes=${successes}/3" >> "$QUEUE_LOG"
      if [ "$successes" -ge 2 ]; then
        verify_log="train_logs/verify_dit12_pure_uniformt_nodetach_lowlr_resume1000_1300_step${step}_seed${seed}_10trials_20260626.log"
        echo "[queue] lowlr candidate reached >=2/3; launching 10-trial verification step=${step} seed=${seed}" >> "$QUEUE_LOG"
        : > "$verify_log"
        eval_ckpt "$ckpt" 10 20 "$seed" "$verify_log"
        cat "$verify_log" >> "$QUEUE_LOG"
        successes=$(parse_successes "$verify_log")
        successes=${successes:-0}
        echo "[queue] lowlr verify step=${step} seed=${seed} successes=${successes}/10" >> "$QUEUE_LOG"
        if [ "$successes" -ge 6 ]; then
          echo "[queue] lowlr pure DIT exceeded 50%" >> "$QUEUE_LOG"
          exit 0
        fi
      fi
    done
  done
}

echo "[queue] start $(date '+%F %T')" >> "$QUEUE_LOG"
wait_for_gpu
run_probe
preflight_seed_sweep
preflight_grouped_action_token_sweep
train_lowlr_resume
eval_lowlr_checkpoints
echo "[queue] all done $(date '+%F %T')" >> "$QUEUE_LOG"
