#!/usr/bin/env bash
# Prepare deploy: data/libero and pretrained_models. See SETUP_DEPLOY.md.
set -e
cd "$(dirname "$0")/.."

cmd="${1:-help}"
if [ "$cmd" = "libero" ]; then
  mkdir -p data/libero
  if [ ! -d "modified_libero_rlds" ]; then
    git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
  fi
  for d in modified_libero_rlds/libero_*_no_noops; do
    [ -d "$d" ] && cp -rn "$d" data/libero/ || true
  done
  echo "LIBERO data ready in data/libero"
elif [ "$cmd" = "pretrained" ]; then
  mkdir -p pretrained_models pretrained_models/configs
  if [ ! -f "pretrained_models/configs/config.json" ]; then
    echo "Downloading config.json from VLA-Adapter/LIBERO-Spatial..."
    curl -sL -o pretrained_models/configs/config.json \
      "https://huggingface.co/VLA-Adapter/LIBERO-Spatial/raw/main/config.json"
    echo "config.json saved to pretrained_models/configs/"
  else
    echo "pretrained_models/configs/config.json already exists."
  fi
  echo ""
  echo "Also download:"
  echo "  - VLM base: https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b -> pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
  echo "  - VLA-Adapter checkpoint (optional): https://huggingface.co/VLA-Adapter -> pretrained_models/"
elif [ "$cmd" = "from_old" ]; then
  OLD="${VLA_OLD_DIR:-$HOME/code/VLA-old}"
  if [ ! -d "$OLD" ]; then
    echo "Error: 原项目目录不存在: $OLD"
    echo "Set VLA_OLD_DIR to your old project path, e.g.: export VLA_OLD_DIR=/path/to/VLA-old"
    exit 1
  fi
  echo "Syncing from: $OLD"
  mkdir -p pretrained_models pretrained_models/configs
  if [ ! -f "pretrained_models/configs/config.json" ]; then
    if [ -f "$OLD/pretrained_models/configs/config.json" ]; then
      cp "$OLD/pretrained_models/configs/config.json" pretrained_models/configs/
      echo "Copied pretrained_models/configs/config.json"
    else
      latest_backup=$(ls -t "$OLD/pretrained_models/configs/config.json.back."* 2>/dev/null | head -1)
      if [ -n "$latest_backup" ]; then
        cp "$latest_backup" pretrained_models/configs/config.json
        echo "Copied config from backup: $latest_backup"
      else
        echo "No config.json in $OLD; run: $0 pretrained"
      fi
    fi
  fi
  if [ ! -d "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b" ] && [ -d "$OLD/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b" ]; then
    echo "Copying VLM base (prism-qwen25-extra-dinosiglip-224px-0_5b, ~5GB)..."
    cp -rn "$OLD/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b" pretrained_models/
    echo "Done."
  fi
  if [ ! -d "data/libero" ] && [ -d "$OLD/data/libero" ]; then
    echo "Copying data/libero..."
    mkdir -p data
    cp -rn "$OLD/data/libero" data/
    echo "Done."
  fi
  for sub in configuration_prismatic.py modeling_prismatic.py; do
    if [ ! -f "pretrained_models/configs/$sub" ] && [ -f "$OLD/prismatic/extern/hf/$sub" ]; then
      cp "$OLD/prismatic/extern/hf/$sub" pretrained_models/configs/
      echo "Copied pretrained_models/configs/$sub"
    fi
  done
  # 复制 processor 相关文件（从 outputs 任一 checkpoint），使 pretrained_models/configs 可被 AutoProcessor 加载
  if [ ! -f "pretrained_models/configs/preprocessor_config.json" ]; then
    pf=$(find "$OLD/outputs" -name "preprocessor_config.json" 2>/dev/null | head -1)
    src_chkpt=""
    [ -n "$pf" ] && src_chkpt=$(dirname "$pf")
    if [ -n "$src_chkpt" ] && [ -d "$src_chkpt" ]; then
      for f in preprocessor_config.json processor_config.json tokenizer_config.json tokenizer.json vocab.json merges.txt special_tokens_map.json added_tokens.json; do
        [ -f "$src_chkpt/$f" ] && cp "$src_chkpt/$f" pretrained_models/configs/ && echo "Copied $f"
      done
      [ -f "$src_chkpt/processing_prismatic.py" ] && cp "$src_chkpt/processing_prismatic.py" pretrained_models/configs/ && echo "Copied processing_prismatic.py"
    fi
  fi
  echo "from_old done."
else
  echo "Usage: $0 libero|pretrained|from_old"
  echo "  from_old - sync pretrained_models + config + data from VLA-old (set VLA_OLD_DIR if not $HOME/code/VLA-old)"
  echo "See SETUP_DEPLOY.md for full steps."
fi
