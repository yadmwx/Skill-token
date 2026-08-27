#!/usr/bin/env bash
# Launch one clean E07 configuration on the AutoDL A100 40GB host.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_DIR=${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}
export PY=${PY:-/root/miniconda3/bin/python}
export TORCHRUN=${TORCHRUN:-/root/miniconda3/bin/torchrun}
export VLM_PATH=${VLM_PATH:-/root/autodl-tmp/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}
export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/e07_a100_data}
export CONFIG_PATH=${CONFIG_PATH:-/root/autodl-tmp/e07_a100_hf/e07_base_ckpt}

export E07_ROOT=${E07_ROOT:-/root/autodl-tmp/e07_context_control_clean}
export RUN_ROOT_DIR=${RUN_ROOT_DIR:-$E07_ROOT/outputs}
export TRAIN_LOG_DIR=${TRAIN_LOG_DIR:-$E07_ROOT/logs}
export EXPERIMENT_STATUS_DIR=${EXPERIMENT_STATUS_DIR:-$E07_ROOT/status}
export TMPDIR=${TMPDIR:-$E07_ROOT/tmp}
export HF_HOME=${HF_HOME:-/root/autodl-tmp/e07_pristine_hf}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$E07_ROOT/cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

export TRAIN_FROM=vlm_base
export DEVICE_TAG=${DEVICE_TAG:-a100-pcie40gb}
export BATCH_SIZE=${BATCH_SIZE:-16}
export GRAD_ACCUM=${GRAD_ACCUM:-1}
export MAX_STEPS=${MAX_STEPS:-10000}
export EVAL_TRIALS=${EVAL_TRIALS:-10}
export PROTOCOL_TAG=${PROTOCOL_TAG:-clean-vlmbase-${MAX_STEPS}updates-b${BATCH_SIZE}ga${GRAD_ACCUM}}
export USE_ADAPTIVE_BRIDGE=True
export BRIDGE_MODE=adaptive
export FIXED_LAYER_INDEX=-1

: "${VARIANT:?set VARIANT to no_skill, continuous_context, or routing_only}"
: "${SEED:?set SEED to 7, 8, or 9}"
case "$VARIANT" in
  no_skill|continuous_context|routing_only) ;;
  *) echo "invalid E07 variant: $VARIANT" >&2; exit 2 ;;
esac
case "$SEED" in
  7|8|9) ;;
  *) echo "invalid E07 seed: $SEED" >&2; exit 2 ;;
esac

mkdir -p \
  "$RUN_ROOT_DIR" "$TRAIN_LOG_DIR" "$EXPERIMENT_STATUS_DIR" \
  "$TMPDIR" "$XDG_CACHE_HOME"

cd "$REPO_DIR"
bash scripts/run_flowmlp_ablation.sh
