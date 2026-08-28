#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/e07_a100_src_mlp_diag}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the completed MLP checkpoint directory}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/eval_libero10_mlp_5000_50}"
TRIALS="${TRIALS:-5}"
SEED="${SEED:-9}"

[[ -d "$CHECKPOINT" ]] || { echo "checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -f "$CHECKPOINT/dataset_statistics.json" ]] || { echo "checkpoint lacks dataset_statistics.json" >&2; exit 3; }
python - "$CHECKPOINT/dataset_statistics.json" <<'PY'
import json, sys
keys = json.load(open(sys.argv[1])).keys()
assert "libero_10_no_noops" in keys, f"wrong normalization statistics: {list(keys)}"
PY

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}" MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled

/root/miniconda3/bin/python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CHECKPOINT" \
  --model_family openvla --action_head_type MLP \
  --task_suite_name libero_10 --task_ids 0,1,2,3,4,5,6,7,8,9 \
  --num_trials_per_task "$TRIALS" --seed "$SEED" \
  --num_open_loop_steps 8 --use_minivlm True --use_pro_version True \
  --use_proprio True --num_images_in_input 2 --use_film False \
  --center_crop True --allow_unnorm_key_fallback False \
  --local_log_dir "$OUT_DIR"
