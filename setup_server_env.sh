#!/bin/bash
# VLA-Adapter Server Environment Setup Script
# This script helps you quickly set up the vla-adapter environment on a remote server

set -e  # Exit on error

echo "=========================================="
echo "VLA-Adapter Environment Setup"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install conda first."
    exit 1
fi

# Set environment name
ENV_NAME="vla-adapter"
PYTHON_VERSION="3.10.16"

echo "Step 1: Creating conda environment..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} already exists. Skipping creation."
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
        conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    fi
else
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo "Step 3: Installing PyTorch..."
echo "Please select your CUDA version:"
echo "1) CUDA 11.8"
echo "2) CUDA 12.1"
echo "3) CPU only"
read -p "Enter choice [1-3]: " cuda_choice

case $cuda_choice in
    1)
        echo "Installing PyTorch with CUDA 11.8..."
        pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        echo "Installing PyTorch (CPU only)..."
        pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        echo "Invalid choice. Installing with CUDA 12.1 by default..."
        pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
        ;;
esac

echo "Step 4: Installing base requirements..."
pip install packaging ninja

echo "Step 5: Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping..."
fi

echo "Step 6: Installing special packages from git..."
echo "Installing transformers from custom fork..."
pip install git+https://github.com/moojink/transformers-openvla-oft.git

echo "Installing dlimp from custom fork..."
pip install git+https://github.com/moojink/dlimp_openvla

echo "Step 7: Installing Flash Attention..."
echo "This may take a while..."
pip install "flash-attn==2.5.5" --no-build-isolation || {
    echo "Warning: Flash Attention installation failed. You may need to install it manually."
    echo "Try: pip cache remove flash_attn"
    echo "Or download the wheel file from: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5"
}

echo "Step 8: Installing local packages..."
if [ -d "." ] && [ -f "pyproject.toml" ]; then
    echo "Installing vla-adapter..."
    pip install -e .
else
    echo "Warning: vla-adapter repository not found. Please clone it first."
fi

if [ -d "LIBERO" ]; then
    echo "Installing LIBERO..."
    pip install -e LIBERO
else
    echo "Warning: LIBERO directory not found. Skipping..."
fi

if [ -d "calvin/calvin_env" ]; then
    echo "Installing calvin_env..."
    pip install -e calvin/calvin_env
else
    echo "Warning: calvin_env directory not found. Skipping..."
fi

echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; print(torch.__version__)'"
echo "  python -c 'import transformers; print(transformers.__version__)'"
echo "  python -c 'import vla_adapter; print(\"VLA-Adapter installed successfully\")'"

