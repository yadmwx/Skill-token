#!/usr/bin/env bash
set -euo pipefail

cd /home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export TOKENIZERS_PARALLELISM=false

RUN_ID=${RUN_ID:-configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0--image_aug--VLA-Adapter-DIT12-tokenfix56-prompt-taskonly-anchor-gripperhead1000-20260627}
STEP=${STEP:-500}
TRIALS=${TRIALS:-3}
SEEDS=${SEEDS:-"0 3 7"}
CKPT=${CKPT:-outputs/${RUN_ID}--${STEP}_chkpt}
OUT_DIR=${OUT_DIR:-train_logs}

mkdir -p "$OUT_DIR"

if [ ! -d "$CKPT" ]; then
  echo "Checkpoint does not exist: $CKPT" >&2
  exit 1
fi

parse_successes() {
  grep -E "Total successes:" "$1" | tail -1 | sed -E 's/.*Total successes:[[:space:]]*([0-9]+).*/\1/'
}

run_mode() {
  local mode="$1"
  local seed="$2"
  local log="$OUT_DIR/dit_anchor_modes_step${STEP}_${mode}_seed${seed}_eval${TRIALS}_20260627.log"
  local blend=0.0
  local disable_anchor=False
  local pure=False

  case "$mode" in
    anchor_residual)
      blend=0.0
      disable_anchor=False
      pure=False
      ;;
    anchor_only)
      blend=1.0
      disable_anchor=False
      pure=False
      ;;
    residual_only)
      blend=0.0
      disable_anchor=True
      pure=True
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      exit 2
      ;;
  esac

  echo "[anchor-modes] step=$STEP mode=$mode seed=$seed trials=$TRIALS" | tee -a "$OUT_DIR/dit_anchor_modes_summary_20260627.log"
  ./eval.sh "$CKPT" \
    --task_suite libero_object --task_ids 0 --num_trials "$TRIALS" --action_head_type DIT \
    --dit_num_blocks 12 --flow_ratio 1.0 --dit_num_inference_steps 20 --dit_num_inference_samples 1 \
    --dit_supervised_anchor_weight 1.0 --dit_anchor_blend "$blend" --dit_disable_inference_anchor "$disable_anchor" --dit_pure_inference "$pure" \
    --dit_anchor_gripper_weight 1.0 --dit_anchor_gripper_bce_weight 0.0 \
    --dit_gripper_head_weight 1.0 --dit_gripper_head_override True \
    --dit_condition_mode task_only --dit_include_prompt_tokens True \
    --dit_zero_init_adaln False --dit_zero_init_output False \
    --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
    --seed "$seed" > "$log" 2>&1

  local successes
  successes=$(parse_successes "$log")
  echo "[anchor-modes] step=$STEP mode=$mode seed=$seed successes=${successes:-NA}/$TRIALS log=$log" | tee -a "$OUT_DIR/dit_anchor_modes_summary_20260627.log"
}

for seed in $SEEDS; do
  run_mode anchor_residual "$seed"
  run_mode anchor_only "$seed"
  run_mode residual_only "$seed"
done
