#!/usr/bin/env bash
# Usage: bash scripts/eval_calvin_pure_dit_standard.sh /path/to/checkpoint baseline|skill
set -euo pipefail

CHECKPOINT=${1:?usage: $0 CHECKPOINT baseline\|skill}
VARIANT=${2:?usage: $0 CHECKPOINT baseline\|skill}
case "$VARIANT" in baseline) USE_SKILL=False ;; skill) USE_SKILL=True ;; *) exit 2 ;; esac

REPO_DIR=${REPO_DIR:-/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation}
PYTHON_BIN=${PYTHON_BIN:-/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python}
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/calvin/calvin_models:$REPO_DIR/calvin/calvin_env:${PYTHONPATH:-}"
export CALVIN_ROOT="$REPO_DIR/calvin" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false

# `task_D_D/validation/.hydra/merged_config.yaml` is the official CALVIN-D
# evaluation environment.  Do not use `calvin_debug_dataset` for paper metrics.
"$PYTHON_BIN" vla-scripts/evaluate_calvin.py \
  --pretrained_checkpoint "$CHECKPOINT" --calvin_path "$REPO_DIR/calvin" --calvin_dataset task_D_D \
  --use_l1_regression False --use_diffusion False --use_flow_matching True --flow_matching_head_type ditx \
  --use_pro_version True --use_proprio True --num_images_in_input 2 \
  --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
  --dit_num_blocks 12 --dit_num_inference_steps 5 --dit_num_inference_samples 1 \
  --dit_pure_inference True --dit_disable_inference_anchor True \
  --dit_use_latent_skill_token "$USE_SKILL" --dit_num_skill_tokens 16 --dit_skill_token_dim 128 --dit_skill_temperature 1.0 \
  --num_sequences 1000 --ep_len 360 --use_wandb False --seed 7
