#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

PYTHON=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python

CHECKPOINT="${1:?usage: $0 CHECKPOINT [OUT_LOG_PREFIX]}"
OUT_PREFIX="${2:-train_logs/dit_task_token_modes_probe_$(date +%Y%m%d_%H%M%S)}"
DIT_CLIP_NORMALIZED_ACTIONS="${DIT_CLIP_NORMALIZED_ACTIONS:-False}"

mkdir -p train_logs

for mode in vision_prompt vision_only prompt_only last_prompt; do
  for probe_case in task_only:False full:False full:True; do
    condition_mode="${probe_case%%:*}"
    group_action_tokens="${probe_case##*:}"
    suffix="nogroup"
    if [ "$group_action_tokens" = "True" ]; then
      suffix="group"
    fi
    out_log="${OUT_PREFIX}_${mode}_${condition_mode}_${suffix}_clip${DIT_CLIP_NORMALIZED_ACTIONS}.log"
    echo "[probe] mode=${mode} condition_mode=${condition_mode} group_action_tokens=${group_action_tokens} clip=${DIT_CLIP_NORMALIZED_ACTIONS} checkpoint=${CHECKPOINT} -> ${out_log}"
    "$PYTHON" scripts/tmp_probe_offline_action_error_20260627.py \
      --checkpoint "$CHECKPOINT" \
      --action_head_type DIT \
      --num_batches 4 \
      --batch_size 2 \
      --num_images_in_input 2 \
      --use_proprio True \
      --use_minivlm True \
      --use_adaptive_bridge True \
      --bridge_mode adaptive \
      --dit_num_blocks 12 \
      --dit_num_inference_steps 20 \
      --dit_num_inference_samples 1 \
      --dit_supervised_anchor_weight 1.0 \
      --dit_anchor_blend 0.0 \
      --dit_anchor_gripper_weight 1.0 \
      --dit_anchor_gripper_bce_weight 0.0 \
      --dit_gripper_head_weight 1.0 \
      --dit_gripper_head_override True \
      --dit_condition_mode "$condition_mode" \
      --dit_include_prompt_tokens True \
      --dit_task_token_mode "$mode" \
      --dit_clip_normalized_actions "$DIT_CLIP_NORMALIZED_ACTIONS" \
      --debug_dit_group_action_tokens_to_chunk "$group_action_tokens" \
      --dit_zero_init_adaln False \
      --dit_zero_init_output False \
      > "$out_log" 2>&1
    grep -E "SPLIT_INFO|PRED_OOR_FRAC|TEACHER_FLOW|MAE_NON_GRIP|MAE_GRIP|MAE_DIM|BIAS_DIM" "$out_log" || true
  done
done
