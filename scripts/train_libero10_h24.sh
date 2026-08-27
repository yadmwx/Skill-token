#!/usr/bin/env bash
set -euo pipefail

# Reproducible FlowMLP/H24 LIBERO-10 mainline. Environment variables may override paths.
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/e07_a100_src}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/e07_a100_data}"
HF_ROOT="${HF_ROOT:-/root/autodl-tmp/e07_a100_hf}"
VLM_PATH="${VLM_PATH:-/root/autodl-tmp/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/libero10_mainline_h24_15000}"
MAX_STEPS="${MAX_STEPS:-15000}"
SEED="${SEED:-9}"

for path in "$REPO_ROOT/vla-scripts/finetune.py" "$DATA_ROOT/libero_10_no_noops" "$HF_ROOT/e07_base_ckpt" "$VLM_PATH"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 2
  fi
done
if pgrep -f '[v]la-scripts/finetune.py' >/dev/null; then
  echo "another finetune.py process is already running" >&2
  exit 3
fi
available_kb="$(df -Pk "$(dirname "$RUN_ROOT")" | awk 'NR==2 {print $4}')"
if (( available_kb < 12 * 1024 * 1024 )); then
  echo "at least 12 GiB free is required on the run volume" >&2
  exit 4
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/outputs"
cd "$REPO_ROOT"
if [[ -n "$(git status --short)" && "${ALLOW_DIRTY:-0}" != 1 ]]; then
  echo "refusing to train from a dirty worktree; commit first or set ALLOW_DIRTY=1" >&2
  exit 5
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
  --train_from vlm_base
  --vlm_training freeze_lora
  --action_head_type FlowMLP
  --use_lora True --lora_rank 64 --lora_dropout 0.0 --freeze_vlm True
  --merge_lora_during_training False
  --use_minivlm True --use_pro_version True --use_proprio True
  --num_images_in_input 2 --use_film False
  --learning_rate 0.0002 --use_constant_lr True --lr_warmup_steps 200
  --max_steps "$MAX_STEPS" --save_freq 5000 --save_latest_checkpoint_only True
  --image_aug True
  --flowmlp_use_latent_skill_token False
  --flowmlp_use_continuous_context False
  --flowmlp_skill_use_layer_routing False
  --flowmlp_skill_use_direct_conditioning False
  --flowmlp_time_embedding_mode legacy
  --flowmlp_time_sampling_mode uniform
  --flowmlp_float32_path False
  --flowmlp_zero_init_output True
  --flowmlp_num_inference_steps 5 --flowmlp_num_inference_samples 1
  --flowmlp_supervised_anchor_weight 0.0 --flowmlp_anchor_blend 0.0
  --flowmlp_detach_flow_conditioning False
  --use_adaptive_bridge False --bridge_mode fixed --fixed_layer_index 24
  --ddp_find_unused_params True --seed "$SEED"
  --batch_size 8 --grad_accumulation_steps 2
  --eval_on_checkpoint False
  --run_id_note libero10-mainline-h24-15000
  --run_id_override libero10-mainline-h24-15000)

{
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_status=%s\n' "$(git status --porcelain | wc -l)"
  printf 'dataset=libero_10_no_noops\nseed=%s\nmax_steps=%s\n' "$SEED" "$MAX_STEPS"
  printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'
} > "$RUN_ROOT/provenance.txt"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi
"${cmd[@]}"
