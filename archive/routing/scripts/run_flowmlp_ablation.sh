#!/usr/bin/env bash
set -euo pipefail

: "${REPO_DIR:?set REPO_DIR}"
: "${PY:?set PY}"
: "${TORCHRUN:?set TORCHRUN}"
: "${VLM_PATH:?set VLM_PATH}"
: "${DATA_ROOT:?set DATA_ROOT}"
CONFIG_PATH=${CONFIG_PATH:-$REPO_DIR/pretrained_models/configs}

VARIANT=${VARIANT:?set VARIANT to no_skill, continuous_context, direct_only, routing_only, or routing_direct}
SEED=${SEED:-7}
DEVICE_TAG=${DEVICE_TAG:-gpu}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
MAX_STEPS=${MAX_STEPS:-10000}
EVAL_TRIALS=${EVAL_TRIALS:-10}
LR=${LR:-0.0002}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-outputs}
TRAIN_LOG_DIR=${TRAIN_LOG_DIR:-train_logs}
EXPERIMENT_STATUS_DIR=${EXPERIMENT_STATUS_DIR:-experiment_results}
USE_ADAPTIVE_BRIDGE=${USE_ADAPTIVE_BRIDGE:-True}
BRIDGE_MODE=${BRIDGE_MODE:-adaptive}
FIXED_LAYER_INDEX=${FIXED_LAYER_INDEX:--1}
TRAIN_FROM=${TRAIN_FROM:-vlm_base}
SOURCE_STEP=${SOURCE_STEP:-5000}
LOAD_ACTION_HEAD_FROM_CHECKPOINT=${LOAD_ACTION_HEAD_FROM_CHECKPOINT:-}
SKILL_ASSIGNMENT_MODE=${SKILL_ASSIGNMENT_MODE:-hard_gumbel}
SKILL_ROUTING_MODE=${SKILL_ROUTING_MODE:-legacy}
SKILL_LAYER_TEMPERATURE=${SKILL_LAYER_TEMPERATURE:-1.0}
SKILL_TEMPERATURE_START=${SKILL_TEMPERATURE_START:--1.0}
SKILL_TEMPERATURE_ANNEAL_STEPS=${SKILL_TEMPERATURE_ANNEAL_STEPS:-0}
SKILL_BALANCE_WEIGHT=${SKILL_BALANCE_WEIGHT:-0.0}
SKILL_Z_LOSS_WEIGHT=${SKILL_Z_LOSS_WEIGHT:-0.0}
SKILL_MI_WEIGHT=${SKILL_MI_WEIGHT:-0.0}
SKILL_TEMPLATE_DIVERSITY_WEIGHT=${SKILL_TEMPLATE_DIVERSITY_WEIGHT:-0.0}
ROUTER_LR_SCALE=${ROUTER_LR_SCALE:-1.0}
SAVE_FREQ=${SAVE_FREQ:-5000}
ROUTING_ANCHOR_LAYER=${ROUTING_ANCHOR_LAYER:--1}
ROUTING_ADAPTIVE_MIX=${ROUTING_ADAPTIVE_MIX:-1.0}
ADAPTIVE_LAYER_ALIGNMENT=${ADAPTIVE_LAYER_ALIGNMENT:-False}
ADAPTIVE_NUM_LAYERS=${ADAPTIVE_NUM_LAYERS:-25}
ADAPTIVE_ALIGNMENT_BOTTLENECK=${ADAPTIVE_ALIGNMENT_BOTTLENECK:-64}
FLOW_TIME_EMBEDDING_MODE=${FLOW_TIME_EMBEDDING_MODE:-legacy}
FLOW_TIME_SAMPLING_MODE=${FLOW_TIME_SAMPLING_MODE:-uniform}
FLOW_FLOAT32_PATH=${FLOW_FLOAT32_PATH:-False}
FLOW_ZERO_INIT_OUTPUT=${FLOW_ZERO_INIT_OUTPUT:-True}
FLOW_NUM_INFERENCE_STEPS=${FLOW_NUM_INFERENCE_STEPS:-5}
FLOW_NUM_INFERENCE_SAMPLES=${FLOW_NUM_INFERENCE_SAMPLES:-8}

TRAIN_SOURCE_ARGS=(--train_from "$TRAIN_FROM")
case "$TRAIN_FROM" in
  vlm_base)
    PROTOCOL_TAG=${PROTOCOL_TAG:-vlmbase-${MAX_STEPS}updates}
    ;;
  checkpoint_init)
    : "${BASE_CKPT:?set BASE_CKPT for TRAIN_FROM=checkpoint_init}"
    LOAD_ACTION_HEAD_FROM_CHECKPOINT=${LOAD_ACTION_HEAD_FROM_CHECKPOINT:-False}
    TRAIN_SOURCE_ARGS+=(
      --resum_vla_path "$BASE_CKPT"
      --resume_step "$SOURCE_STEP"
      --resume_load_training_state False
      --load_action_head_from_checkpoint "$LOAD_ACTION_HEAD_FROM_CHECKPOINT"
    )
    if [[ "$LOAD_ACTION_HEAD_FROM_CHECKPOINT" == "True" || "$LOAD_ACTION_HEAD_FROM_CHECKPOINT" == "true" ]]; then
      HEAD_INIT_TAG=headload
    else
      HEAD_INIT_TAG=headfresh
    fi
    PROTOCOL_TAG=${PROTOCOL_TAG:-ckpt${SOURCE_STEP}init-${HEAD_INIT_TAG}-${MAX_STEPS}updates}
    ;;
  checkpoint)
    : "${BASE_CKPT:?set BASE_CKPT for TRAIN_FROM=checkpoint}"
    LOAD_ACTION_HEAD_FROM_CHECKPOINT=${LOAD_ACTION_HEAD_FROM_CHECKPOINT:-True}
    TRAIN_SOURCE_ARGS+=(
      --resum_vla_path "$BASE_CKPT"
      --resume_step "$SOURCE_STEP"
      --load_action_head_from_checkpoint "$LOAD_ACTION_HEAD_FROM_CHECKPOINT"
    )
    PROTOCOL_TAG=${PROTOCOL_TAG:-resume${SOURCE_STEP}to${MAX_STEPS}}
    ;;
  *)
    echo "unknown TRAIN_FROM=$TRAIN_FROM" >&2
    exit 2
    ;;
esac

case "$VARIANT" in
  no_skill)           USE_SKILL=False; USE_CONTINUOUS_CONTEXT=False; USE_ROUTING=False; USE_DIRECT=False ;;
  continuous_context) USE_SKILL=False; USE_CONTINUOUS_CONTEXT=True;  USE_ROUTING=True;  USE_DIRECT=False ;;
  direct_only)        USE_SKILL=True;  USE_CONTINUOUS_CONTEXT=False; USE_ROUTING=False; USE_DIRECT=True ;;
  routing_only)       USE_SKILL=True;  USE_CONTINUOUS_CONTEXT=False; USE_ROUTING=True;  USE_DIRECT=False ;;
  routing_direct)     USE_SKILL=True;  USE_CONTINUOUS_CONTEXT=False; USE_ROUTING=True;  USE_DIRECT=True ;;
  *) echo "unknown VARIANT=$VARIANT" >&2; exit 2 ;;
esac

cd "$REPO_DIR"
mkdir -p "$RUN_ROOT_DIR" "$TRAIN_LOG_DIR" "$EXPERIMENT_STATUS_DIR"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=".:$REPO_DIR/LIBERO:${PYTHONPATH:-}"
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

FIXED_TAG="fixed${FIXED_LAYER_INDEX}"
RUN_ID="FlowMLP-ablation-${VARIANT}-seed${SEED}-${DEVICE_TAG}-${FIXED_TAG}-${PROTOCOL_TAG}"
RUN_ID=${RUN_ID_OVERRIDE:-$RUN_ID}
LABEL="flowmlp_ablation_${VARIANT}_seed${SEED}_${DEVICE_TAG}_${FIXED_TAG}_${PROTOCOL_TAG}"
ATTEMPT_ID=${ATTEMPT_ID:-$(date +%Y%m%dT%H%M%S)}
LOG_LABEL="${LABEL}_attempt${ATTEMPT_ID}"
TRAIN_LOG="${TRAIN_LOG_DIR}/${LOG_LABEL}.train.log"
EVAL_LOG="${TRAIN_LOG_DIR}/${LOG_LABEL}.eval.log"
STATUS="${EXPERIMENT_STATUS_DIR}/${LABEL}_attempt${ATTEMPT_ID}.status"

echo "RUNNING $(date --iso-8601=seconds) $VARIANT seed=$SEED device=$DEVICE_TAG" > "$STATUS"
set +e
"$TORCHRUN" --standalone --nnodes 1 --nproc_per_node 1 vla-scripts/finetune.py \
  --config_file_path "$CONFIG_PATH" --vlm_path "$VLM_PATH" \
  --data_root_dir "$DATA_ROOT" --dataset_name libero_object_no_noops --run_root_dir "$RUN_ROOT_DIR" \
  "${TRAIN_SOURCE_ARGS[@]}" \
  --vlm_training freeze_lora --action_head_type FlowMLP --use_lora True --lora_rank 64 --lora_dropout 0.0 \
  --freeze_vlm True --merge_lora_during_training False --use_minivlm True --use_pro_version True \
  --use_proprio True --num_images_in_input 2 --use_film False --learning_rate "$LR" --use_constant_lr True \
  --lr_warmup_steps 200 --max_steps "$MAX_STEPS" --save_freq "$SAVE_FREQ" --save_latest_checkpoint_only True --image_aug True \
  --flowmlp_use_latent_skill_token "$USE_SKILL" \
  --flowmlp_use_continuous_context "$USE_CONTINUOUS_CONTEXT" \
  --flowmlp_skill_use_layer_routing "$USE_ROUTING" \
  --flowmlp_skill_use_direct_conditioning "$USE_DIRECT" \
  --flowmlp_continuous_context_use_direct_conditioning False \
  --flowmlp_num_skill_tokens 16 --flowmlp_skill_token_dim 128 --flowmlp_skill_temperature 1.0 \
  --flowmlp_skill_entropy_weight 0.0 --flowmlp_skill_assignment_mode "$SKILL_ASSIGNMENT_MODE" \
  --flowmlp_skill_routing_mode "$SKILL_ROUTING_MODE" \
  --flowmlp_skill_layer_temperature "$SKILL_LAYER_TEMPERATURE" \
  --flowmlp_skill_temperature_start "$SKILL_TEMPERATURE_START" \
  --flowmlp_skill_temperature_anneal_steps "$SKILL_TEMPERATURE_ANNEAL_STEPS" \
  --flowmlp_skill_balance_weight "$SKILL_BALANCE_WEIGHT" \
  --flowmlp_skill_z_loss_weight "$SKILL_Z_LOSS_WEIGHT" \
  --flowmlp_skill_mi_weight "$SKILL_MI_WEIGHT" \
  --flowmlp_skill_template_diversity_weight "$SKILL_TEMPLATE_DIVERSITY_WEIGHT" \
  --flowmlp_router_lr_scale "$ROUTER_LR_SCALE" \
  --flowmlp_routing_anchor_layer "$ROUTING_ANCHOR_LAYER" --flowmlp_routing_adaptive_mix "$ROUTING_ADAPTIVE_MIX" \
  --flowmlp_adaptive_layer_alignment "$ADAPTIVE_LAYER_ALIGNMENT" \
  --flowmlp_adaptive_num_layers "$ADAPTIVE_NUM_LAYERS" \
  --flowmlp_adaptive_alignment_bottleneck "$ADAPTIVE_ALIGNMENT_BOTTLENECK" \
  --flowmlp_time_embedding_mode "$FLOW_TIME_EMBEDDING_MODE" \
  --flowmlp_time_sampling_mode "$FLOW_TIME_SAMPLING_MODE" \
  --flowmlp_float32_path "$FLOW_FLOAT32_PATH" \
  --flowmlp_zero_init_output "$FLOW_ZERO_INIT_OUTPUT" \
  --flowmlp_num_inference_steps "$FLOW_NUM_INFERENCE_STEPS" \
  --flowmlp_num_inference_samples "$FLOW_NUM_INFERENCE_SAMPLES" \
  --flowmlp_supervised_anchor_weight 0.0 --flowmlp_anchor_blend 0.0 --flowmlp_detach_flow_conditioning False \
  --use_adaptive_bridge "$USE_ADAPTIVE_BRIDGE" --bridge_mode "$BRIDGE_MODE" --fixed_layer_index "$FIXED_LAYER_INDEX" --ddp_find_unused_params True \
  --seed "$SEED" --run_id_note "$RUN_ID" --run_id_override "$RUN_ID" \
  --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUM" > "$TRAIN_LOG" 2>&1
train_rc=$?
set -e
if [ "$train_rc" -ne 0 ]; then
  echo "TRAIN_FAILED $(date --iso-8601=seconds) rc=$train_rc" > "$STATUS"
  exit "$train_rc"
fi

# Checkpoints may be symlinked to the experiment scratch volume on disk-constrained V48 runs.
# `finetune.py` normally appends `--${MAX_STEPS}_chkpt`, but an explicit
# `run_id_override` is also allowed to be the final checkpoint directory name.
# Accept the latter only when the expected saved modules are present, so a
# similarly named partial directory cannot be evaluated accidentally.
CHKPT=$(find "$RUN_ROOT_DIR" -maxdepth 1 \( -type d -o -type l \) -name "${RUN_ID}--${MAX_STEPS}_chkpt" | head -1)
if [ -z "$CHKPT" ] \
  && [ -f "$RUN_ROOT_DIR/$RUN_ID/action_head--latest_checkpoint.pt" ] \
  && [ -f "$RUN_ROOT_DIR/$RUN_ID/training_state--latest_checkpoint.pt" ]; then
  CHKPT="$RUN_ROOT_DIR/$RUN_ID"
fi
if [ -z "$CHKPT" ]; then
  echo "CHECKPOINT_MISSING $(date --iso-8601=seconds)" > "$STATUS"
  exit 3
fi

set +e
"$PY" experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CHKPT" --action_head_type FlowMLP \
  --task_suite_name libero_object --task_ids 3,4,7,9 --num_trials_per_task "$EVAL_TRIALS" \
  --use_depth_interface False --depth_interface_mode none --use_adaptive_bridge "$USE_ADAPTIVE_BRIDGE" --bridge_mode "$BRIDGE_MODE" --fixed_layer_index "$FIXED_LAYER_INDEX" \
  --flowmlp_use_latent_skill_token "$USE_SKILL" \
  --flowmlp_use_continuous_context "$USE_CONTINUOUS_CONTEXT" \
  --flowmlp_skill_use_layer_routing "$USE_ROUTING" \
  --flowmlp_skill_use_direct_conditioning "$USE_DIRECT" \
  --flowmlp_continuous_context_use_direct_conditioning False \
  --flowmlp_num_skill_tokens 16 --flowmlp_skill_token_dim 128 --flowmlp_skill_temperature 1.0 \
  --flowmlp_skill_entropy_weight 0.0 --flowmlp_skill_assignment_mode "$SKILL_ASSIGNMENT_MODE" \
  --flowmlp_skill_routing_mode "$SKILL_ROUTING_MODE" \
  --flowmlp_skill_layer_temperature "$SKILL_LAYER_TEMPERATURE" \
  --flowmlp_skill_temperature_start "$SKILL_TEMPERATURE_START" \
  --flowmlp_skill_temperature_anneal_steps "$SKILL_TEMPERATURE_ANNEAL_STEPS" \
  --flowmlp_skill_balance_weight "$SKILL_BALANCE_WEIGHT" \
  --flowmlp_skill_z_loss_weight "$SKILL_Z_LOSS_WEIGHT" \
  --flowmlp_skill_mi_weight "$SKILL_MI_WEIGHT" \
  --flowmlp_skill_template_diversity_weight "$SKILL_TEMPLATE_DIVERSITY_WEIGHT" \
  --flowmlp_routing_anchor_layer "$ROUTING_ANCHOR_LAYER" --flowmlp_routing_adaptive_mix "$ROUTING_ADAPTIVE_MIX" \
  --flowmlp_adaptive_layer_alignment "$ADAPTIVE_LAYER_ALIGNMENT" \
  --flowmlp_adaptive_num_layers "$ADAPTIVE_NUM_LAYERS" \
  --flowmlp_adaptive_alignment_bottleneck "$ADAPTIVE_ALIGNMENT_BOTTLENECK" \
  --flowmlp_time_embedding_mode "$FLOW_TIME_EMBEDDING_MODE" \
  --flowmlp_time_sampling_mode "$FLOW_TIME_SAMPLING_MODE" \
  --flowmlp_float32_path "$FLOW_FLOAT32_PATH" \
  --flowmlp_zero_init_output "$FLOW_ZERO_INIT_OUTPUT" \
  --flowmlp_num_inference_steps "$FLOW_NUM_INFERENCE_STEPS" \
  --flowmlp_num_inference_samples "$FLOW_NUM_INFERENCE_SAMPLES" \
  --flowmlp_supervised_anchor_weight 0.0 --flowmlp_anchor_blend 0.0 --flowmlp_detach_flow_conditioning False \
  --use_proprio True --num_images_in_input 2 --use_film False --use_minivlm True --use_pro_version True \
  --use_wandb False --center_crop True --seed "$SEED" > "$EVAL_LOG" 2>&1
eval_rc=$?
set -e
if [ "$eval_rc" -eq 0 ]; then
  echo "COMPLETE $(date --iso-8601=seconds) checkpoint=$CHKPT" > "$STATUS"
else
  echo "EVAL_FAILED $(date --iso-8601=seconds) rc=$eval_rc checkpoint=$CHKPT" > "$STATUS"
fi
exit "$eval_rc"
