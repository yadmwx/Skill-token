#!/usr/bin/env bash
set -euo pipefail

# Clean VLM-base cold-start run for the representation-centric soft prototype router.
# This wrapper intentionally refuses checkpoint initialization or resume.
: "${REPO_DIR:?set REPO_DIR}"
: "${PY:?set PY}"
: "${TORCHRUN:?set TORCHRUN}"
: "${VLM_PATH:?set VLM_PATH}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT on the persistent data disk}"

if pgrep -af 'vla-scripts/finetune.py|run_libero_eval.py' >/dev/null; then
  echo "Refusing to start: another training or evaluation process is active." >&2
  pgrep -af 'vla-scripts/finetune.py|run_libero_eval.py' >&2 || true
  exit 3
fi

mkdir -p "$RUN_ROOT"/{outputs,logs,status}

export VARIANT=routing_only
export SEED=9
export DEVICE_TAG=a100-softproto
export TRAIN_FROM=vlm_base
export BATCH_SIZE=${BATCH_SIZE:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-2}
export MAX_STEPS=${MAX_STEPS:-10000}
export SAVE_FREQ=${SAVE_FREQ:-1000}
export EVAL_TRIALS=${EVAL_TRIALS:-10}
export LR=${LR:-0.0002}
export RUN_ROOT_DIR="$RUN_ROOT/outputs"
export TRAIN_LOG_DIR="$RUN_ROOT/logs"
export EXPERIMENT_STATUS_DIR="$RUN_ROOT/status"

export SKILL_ASSIGNMENT_MODE=soft
export SKILL_ROUTING_MODE=prototype_soft
export SKILL_LAYER_TEMPERATURE=1.0
export SKILL_TEMPERATURE_START=2.0
export SKILL_TEMPERATURE_ANNEAL_STEPS=4000
export SKILL_BALANCE_WEIGHT=0.01
export SKILL_Z_LOSS_WEIGHT=0.0001
export SKILL_MI_WEIGHT=0.01
export SKILL_TEMPLATE_DIVERSITY_WEIGHT=0.002
export ROUTER_LR_SCALE=0.25

export ROUTING_ANCHOR_LAYER=-1
export ROUTING_ADAPTIVE_MIX=1.0
export ADAPTIVE_LAYER_ALIGNMENT=False
export ADAPTIVE_NUM_LAYERS=25
export ADAPTIVE_ALIGNMENT_BOTTLENECK=64

export FLOW_TIME_EMBEDDING_MODE=continuous
export FLOW_TIME_SAMPLING_MODE=openpi_beta
export FLOW_FLOAT32_PATH=True
export FLOW_ZERO_INIT_OUTPUT=False
export FLOW_NUM_INFERENCE_STEPS=10
export FLOW_NUM_INFERENCE_SAMPLES=8

export PROTOCOL_TAG=vlmbase-softproto-10000updates
export RUN_ID_OVERRIDE=FlowMLP-routing-only-softproto-seed9-a100-vlmbase-10000

if [[ -e "$RUN_ROOT/outputs/$RUN_ID_OVERRIDE" ]] \
  || [[ -e "$RUN_ROOT/logs/$RUN_ID_OVERRIDE.train.log" ]] \
  || [[ -e "$RUN_ROOT/status/$RUN_ID_OVERRIDE.status" ]]; then
  echo "Refusing to reuse an existing canonical attempt: $RUN_ID_OVERRIDE" >&2
  echo "Set RUN_ROOT to a fresh attempt directory after auditing the existing files." >&2
  exit 4
fi

exec bash "$REPO_DIR/scripts/run_flowmlp_ablation.sh"
