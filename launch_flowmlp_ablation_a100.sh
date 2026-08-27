#!/usr/bin/env bash
set -euo pipefail
export REPO_DIR=/home/xiaguanxiao/code/Skill-Conditioned-Representation-Depth-for-Vision-Language-Robot-Manipulation
export PY=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/python
export TORCHRUN=/home/xiaguanxiao/miniconda3/envs/vla-flow/bin/torchrun
export VLM_PATH="$REPO_DIR/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
export DATA_ROOT="$REPO_DIR/data/libero"
export BASE_CKPT="$REPO_DIR/outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-MLP-object-5000-20260618_213642--5000_chkpt"
export DEVICE_TAG=a100
export BATCH_SIZE=16 GRAD_ACCUM=1 MAX_STEPS=10000 EVAL_TRIALS=10
export HF_HOME=/home/xiaguanxiao/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/home/xiaguanxiao/.cache/huggingface/hub
export EXPERIMENT_SPECS="no_skill:7 routing_only:7 direct_only:8 routing_direct:8 no_skill:9 routing_only:9"
cd "$REPO_DIR"
exec bash scripts/queue_flowmlp_ablation.sh
