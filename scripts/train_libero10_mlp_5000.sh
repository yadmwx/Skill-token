#!/usr/bin/env bash
set -euo pipefail

# Diagnostic control: standard 24-block L1 MLP head on LIBERO-10.
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/e07_a100_src_mlp_diag}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/e07_a100_data}"
HF_ROOT="${HF_ROOT:-/root/autodl-tmp/e07_a100_hf}"
VLM_PATH="${VLM_PATH:-/root/autodl-tmp/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/libero10_mlp_5000}"
RUN_ID="libero10-mlp-l1-5000"
MAX_STEPS="${MAX_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-9}"

case "$BATCH_SIZE" in
  8) GRAD_ACCUMULATION_STEPS=2 ;;
  4) GRAD_ACCUMULATION_STEPS=4 ;;
  *) echo "BATCH_SIZE must be 8 or 4" >&2; exit 2 ;;
esac

for path in "$REPO_ROOT/vla-scripts/finetune.py" "$DATA_ROOT/libero_10_no_noops" "$HF_ROOT/e07_base_ckpt" "$VLM_PATH"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 3; }
done
if pgrep -f '[v]la-scripts/finetune.py' >/dev/null; then
  echo "another finetune.py process is already running" >&2
  exit 4
fi
available_kb="$(df -Pk "$(dirname "$RUN_ROOT")" | awk 'NR==2 {print $4}')"
if (( available_kb < 8 * 1024 * 1024 )); then
  echo "at least 8 GiB free is required on the run volume" >&2
  exit 5
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs"
cd "$REPO_ROOT"
if [[ -n "$(git status --short)" && "${ALLOW_DIRTY:-0}" != 1 ]]; then
  echo "refusing to train from a dirty worktree" >&2
  exit 6
fi

export PRISMATIC_DINO_CHECKPOINT="${PRISMATIC_DINO_CHECKPOINT:-$HF_ROOT/dino_pytorch_model_224.bin}"
export PRISMATIC_SIGLIP_CHECKPOINT="${PRISMATIC_SIGLIP_CHECKPOINT:-$HF_ROOT/siglip_timm_pytorch.pth}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled

cmd=(/root/miniconda3/bin/torchrun --standalone --nnodes 1 --nproc_per_node 1 vla-scripts/finetune.py
  --config_file_path "$HF_ROOT/e07_base_ckpt"
  --vlm_path "$VLM_PATH"
  --data_root_dir "$DATA_ROOT"
  --dataset_name libero_10_no_noops
  --run_root_dir "$RUN_ROOT/outputs"
  --train_from vlm_base --vlm_training freeze_lora
  --action_head_type MLP
  --use_lora True --lora_rank 64 --lora_dropout 0.0 --freeze_vlm True
  --merge_lora_during_training False
  --use_minivlm True --use_pro_version True --use_proprio True
  --num_images_in_input 2 --use_film False
  --learning_rate 0.0002 --use_constant_lr True --lr_warmup_steps 200
  --max_steps "$MAX_STEPS" --save_freq "$MAX_STEPS" --save_latest_checkpoint_only True
  --image_aug True --batch_size "$BATCH_SIZE"
  --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS"
  --gradient_clipping_norm 1.0 --ddp_find_unused_params True
  --eval_on_checkpoint False --seed "$SEED"
  --run_id_note "$RUN_ID" --run_id_override "$RUN_ID")

{
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_status_count=%s\n' "$(git status --porcelain | wc -l)"
  printf 'dataset=libero_10_no_noops\naction_head=MLP\nobjective=L1\n'
  printf 'seed=%s\nmax_steps=%s\neffective_global_batch=16\n' "$SEED" "$MAX_STEPS"
  printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'
} > "$RUN_ROOT/provenance.txt"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
"${cmd[@]}"
