#!/usr/bin/env bash
# One E07 configuration.  The caller owns queueing and duplicate prevention.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}
export PY=${PY:-/autodl-fs/data/skill_depth/envs/vla-flow/bin/python}
export TORCHRUN=${TORCHRUN:-/autodl-fs/data/skill_depth/envs/vla-flow/bin/torchrun}
export VLM_PATH=${VLM_PATH:-$REPO_DIR/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}
export DATA_ROOT=${DATA_ROOT:-/autodl-fs/data/skill_depth/data/libero}
export CONFIG_PATH=${CONFIG_PATH:-/autodl-fs/data/skill_depth/pretrained_models/configs}
export BASE_CKPT=${BASE_CKPT:-/autodl-fs/data/skill_depth/outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-MLP-object-5000-20260618_213642--5000_chkpt}
export DEVICE_TAG=${DEVICE_TAG:-v48-4090}
export BATCH_SIZE=${BATCH_SIZE:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-2}
export MAX_STEPS=${MAX_STEPS:-10000}
export EVAL_TRIALS=${EVAL_TRIALS:-10}
export HF_HOME=${HF_HOME:-/autodl-fs/data/skill_depth/cache/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export USE_ADAPTIVE_BRIDGE=True
export BRIDGE_MODE=adaptive
export FIXED_LAYER_INDEX=-1

: "${VARIANT:?set VARIANT to no_skill, continuous_context, or routing_only}"
: "${SEED:?set SEED}"
case "$VARIANT" in no_skill|continuous_context|routing_only) ;; *) echo "invalid E07 variant: $VARIANT" >&2; exit 2;; esac

cd "$REPO_DIR"
VARIANT="$VARIANT" SEED="$SEED" DEVICE_TAG="$DEVICE_TAG" \
  BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" MAX_STEPS="$MAX_STEPS" \
  EVAL_TRIALS="$EVAL_TRIALS" USE_ADAPTIVE_BRIDGE=True BRIDGE_MODE=adaptive \
  FIXED_LAYER_INDEX=-1 bash scripts/run_flowmlp_ablation.sh
