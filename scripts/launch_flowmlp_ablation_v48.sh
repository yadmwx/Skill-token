#!/usr/bin/env bash
set -euo pipefail
export REPO_DIR=/autodl-fs/data/skill_depth/code/action_expert_prefix_trial
export PY=/autodl-fs/data/skill_depth/envs/vla-flow/bin/python
export TORCHRUN=/autodl-fs/data/skill_depth/envs/vla-flow/bin/torchrun
export VLM_PATH=/autodl-fs/data/skill_depth/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b
export CONFIG_PATH=/autodl-fs/data/skill_depth/pretrained_models/configs
export DATA_ROOT=/autodl-fs/data/skill_depth/data/libero
export BASE_CKPT="/autodl-fs/data/skill_depth/outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-MLP-object-5000-20260618_213642--5000_chkpt"
export DEVICE_TAG=v48
export BATCH_SIZE=8 GRAD_ACCUM=2 MAX_STEPS=10000 EVAL_TRIALS=10
export HF_HOME=/autodl-fs/data/skill_depth/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/autodl-fs/data/skill_depth/cache/huggingface/hub
export EXPERIMENT_SPECS="direct_only:7 routing_direct:7 no_skill:8 routing_only:8 direct_only:9 routing_direct:9"
cd "$REPO_DIR"
exec bash scripts/queue_flowmlp_ablation.sh
