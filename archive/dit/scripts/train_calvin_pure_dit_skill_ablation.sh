#!/usr/bin/env bash
# Usage: bash scripts/train_calvin_pure_dit_skill_ablation.sh baseline|skill [max_steps]
set -euo pipefail

VARIANT=${1:?usage: $0 baseline\|skill [max_steps]}
MAX_STEPS=${2:-400005}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUMULATION_STEPS=${GRAD_ACCUMULATION_STEPS:-1}
case "$VARIANT" in
  baseline) USE_SKILL=False ;;
  skill) USE_SKILL=True ;;
  *) echo "variant must be baseline or skill" >&2; exit 2 ;;
esac

REPO_DIR=${REPO_DIR:-/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation}
DATA_ROOT=${DATA_ROOT:-/data/xiaguanxiao/rlds}
VLM_ROOT=${VLM_ROOT:-$REPO_DIR/pretrained_models}
PYTHON_BIN=${PYTHON_BIN:-/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python}
TORCHRUN=${TORCHRUN:-/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/torchrun}

cd "$REPO_DIR"
mkdir -p train_logs outputs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$REPO_DIR:$REPO_DIR/calvin/calvin_models:$REPO_DIR/calvin/calvin_env:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

RUN_ID="CALVIN-ABC-pureDIT-${VARIANT}-skilltoken-$(date +%Y%m%d_%H%M%S)"
LOG="train_logs/${RUN_ID}.log"

"$TORCHRUN" --standalone --nnodes=1 --nproc_per_node=1 vla-scripts/finetune.py \
  --train_from vlm_base \
  --vlm_path "$VLM_ROOT/prism-qwen25-extra-dinosiglip-224px-0_5b" \
  --config_file_path "$VLM_ROOT/configs" \
  --data_root_dir "$DATA_ROOT" --dataset_name calvin_abc --run_root_dir outputs \
  --action_head_type DIT --flow_ratio 1.0 \
  --use_lora True --lora_rank 64 --lora_dropout 0.0 --vlm_training freeze_lora \
  --freeze_vlm True --merge_lora_during_training False \
  --use_minivlm True --use_pro_version True --use_proprio True --num_images_in_input 2 --use_film False \
  --use_adaptive_bridge True --bridge_mode adaptive --fixed_layer_index -1 \
  --dit_num_blocks 12 --dit_num_inference_steps 5 --dit_num_inference_samples 1 \
  --dit_use_latent_skill_token "$USE_SKILL" --dit_num_skill_tokens 16 --dit_skill_token_dim 128 --dit_skill_temperature 1.0 \
  --learning_rate 0.0002 --lr_warmup_steps 200 --num_steps_before_decay "$MAX_STEPS" --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_SIZE" --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS" --image_aug True --save_freq 5000 --save_latest_checkpoint_only False \
  --ddp_find_unused_params True --run_id_note "$RUN_ID" > "$LOG" 2>&1
