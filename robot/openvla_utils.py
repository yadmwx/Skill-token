"""Utils for evaluating VLA-Adapter or fine-tuned VLA-Adapter policies."""

import filecmp
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import requests
import tensorflow as tf
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# Apply JSON numpy patch for serialization
json_numpy.patch()

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead, DiffusionActionHead, FlowMatchingMLPActionHead
from prismatic.models.adaptive_depth_interface import SkillAdaptiveDepthInterface
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

# Initialize important constants
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224  # Standard image size expected by OpenVLA

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    if os.path.exists(model_path):
        return False

    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # Read and update the config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    # Write back the updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint versions of both
    modeling_prismatic.py and configuration_prismatic.py:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    # Find current files
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def load_action_queries_into_model(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], context: str = "") -> None:
    """Restore action_queries explicitly for checkpoints that save them outside PEFT adapters."""
    target = model
    if hasattr(target, "module"):
        target = target.module
    if hasattr(target, "base_model") and hasattr(target.base_model, "model"):
        target = target.base_model.model
    weight = state_dict["weight"].to(device=target.action_queries.weight.device, dtype=target.action_queries.weight.dtype)
    target.action_queries.weight.data.copy_(weight)
    prefix = f"{context} " if context else ""
    weight_cpu = weight.detach().float().cpu()
    print(
        f"{prefix}loaded action_queries "
        f"shape={tuple(weight_cpu.shape)} "
        f"abs_mean={float(weight_cpu.abs().mean()):.6f} "
        f"norm={float(weight_cpu.norm()):.6f} "
        f"min={float(weight_cpu.min()):.6f} "
        f"max={float(weight_cpu.max()):.6f}"
    )


def load_action_queries_from_model_safetensors(checkpoint_dir: str) -> Optional[Dict[str, torch.Tensor]]:
    """Load legacy action_queries from model.safetensors when no explicit component file exists."""
    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(model_path):
        return None
    try:
        from safetensors.torch import safe_open
    except ImportError:
        return None

    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        key = None
        for candidate in (
            "action_queries.weight",
            "base_model.model.action_queries.weight",
            "model.action_queries.weight",
        ):
            if candidate in keys:
                key = candidate
                break
        if key is None:
            matches = [k for k in keys if k.endswith("action_queries.weight")]
            if not matches:
                return None
            key = matches[0]
        return {"weight": f.get_tensor(key), "source_key": key}

def load_component_state_dict_v1(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v

    return new_state_dict


def get_vla(cfg: Any) -> torch.nn.Module:
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: Configuration object

    Returns:
        torch.nn.Module: The initialized VLA model
    """
    print("Instantiating pretrained VLA policy...")

    # If loading a locally stored pretrained checkpoint, check whether config or model files
    # need to be synced so that any changes the user makes to the VLA modeling code will
    # actually go into effect
    # If loading a pretrained checkpoint from Hugging Face Hub, we just assume that the policy
    # will be used as is, with its original modeling logic
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # If checkpoint has lora_adapter but no root config.json (e.g. older saves), copy from training config
        config_path = os.path.join(cfg.pretrained_checkpoint, "config.json")
        if not os.path.exists(config_path):
            fallback = "pretrained_models/configs/config.json"
            if os.path.exists(fallback):
                shutil.copy2(fallback, config_path)
                print(f"Created config.json in checkpoint from {fallback} for evaluation.")
            else:
                raise FileNotFoundError(
                    f"Checkpoint has no config.json and fallback not found at {fallback}. "
                    "Re-save the checkpoint with current finetune.py or add config.json to the checkpoint."
                )

        # Update config.json and sync model files
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)

    # Check if this is a LoRA checkpoint (has lora_adapter directory but no model files in root)
    adapter_dir = os.path.join(cfg.pretrained_checkpoint, "lora_adapter")
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        adapter_config_path = os.path.join(adapter_dir, "config.json")
    # Older checkpoints may have adapter_model.safetensors without metadata;
    # create a minimal config so they still take the LoRA loading path.
    if os.path.isdir(adapter_dir) and not os.path.exists(adapter_config_path):
        if os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")):
            fallback_config = "pretrained_models/configs/config.json"
            if os.path.exists(fallback_config):
                with open(fallback_config, "r") as f:
                    adapter_config = json.load(f)
                adapter_config["_name_or_path"] = fallback_config.replace("/config.json", "")
                with open(adapter_config_path, "w") as f:
                    json.dump(adapter_config, f, indent=2)
                print(f"Created lora_adapter/config.json from {fallback_config} for LoRA evaluation.")
    has_lora_adapter = os.path.isdir(adapter_dir) and os.path.exists(adapter_config_path)
    
    # Check if root directory has model files
    root_has_model = any(
        os.path.exists(os.path.join(cfg.pretrained_checkpoint, f))
        for f in ["model.safetensors", "pytorch_model.bin", "model.ckpt.index"]
    )
    
    # Load the model
    if has_lora_adapter and not root_has_model:
        # This is a LoRA checkpoint - need to load base model first, then LoRA adapter
        print("Detected LoRA checkpoint. Loading base model and LoRA adapter...")
        
        # PEFT saves adapter metadata as adapter_config.json. Older checkpoints
        # may have used config.json, so retain that as a fallback.
        adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            adapter_config_path = os.path.join(adapter_dir, "config.json")
        base_model_path = None
        use_config_only = False
        
        try:
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
                # Try to infer base model path from _name_or_path or other fields
                name_or_path = adapter_config.get("_name_or_path") or adapter_config.get(
                    "base_model_name_or_path", ""
                )
                
                # Common patterns: config path -> model path
                if "pretrained_models/configs" in name_or_path:
                    # When _name_or_path points to configs/config.json, we need to load from VLM checkpoint
                    # This is the case for minivlm models
                    use_config_only = True
                    config_candidates = [
                        "/autodl-fs/data/skill_depth/pretrained_models/configs/config.json",
                        "pretrained_models/configs/config.json",
                    ]
                    vlm_candidates = [
                        "/autodl-fs/data/skill_depth/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
                        "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
                    ]
                    config_path = next((p for p in config_candidates if os.path.exists(p)), config_candidates[-1])
                    vlm_path = next((p for p in vlm_candidates if os.path.exists(p)), vlm_candidates[-1])
                    
                    if not os.path.exists(config_path):
                        raise ValueError(f"Config file not found: {config_path}")
                    if not os.path.exists(vlm_path):
                        raise ValueError(f"VLM checkpoint not found: {vlm_path}")
                    
                    try:
                        print(f"Loading base model from VLM checkpoint: {vlm_path}")
                        # Load VLM weights first (use load, not load_vla, for VLM checkpoints)
                        from prismatic.models import load
                        vlm = load(
                            vlm_path,
                            hf_token=None,
                            load_for_training=False,
                        )
                        
                        # Create VLA model from config
                        print(f"Creating VLA model from config: {config_path}")
                        config = AutoConfig.from_pretrained(config_path)
                        base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
                        
                        # Convert VLM state dict keys to VLA format
                        replace_map = [
                            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
                            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
                            ("llm_backbone.llm", "language_model"),
                            ("projector.projector.0", "projector.fc1"),
                            ("projector.projector.2", "projector.fc2"),
                            ("projector.projector.4", "projector.fc3"),
                            ("gamma", "scale_factor"),
                        ]
                        
                        def rename_state_dict_keys(state_dict, replace_map):
                            new_state_dict = {}
                            for k, v in state_dict.items():
                                new_k = k
                                for old, new in replace_map:
                                    if old in new_k:
                                        new_k = new_k.replace(old, new)
                                new_state_dict[new_k] = v
                            return new_state_dict
                        
                        old_state_dict = vlm.state_dict()
                        converted_state_dict = rename_state_dict_keys(old_state_dict, replace_map)
                        
                        # Load converted weights into VLA model
                        print("Loading VLM weights into VLA model...")
                        missing_keys, unexpected_keys = base_vla.load_state_dict(converted_state_dict, strict=False)
                        if missing_keys:
                            print(f"Warning: Missing keys when loading VLM weights: {missing_keys[:10]}...")
                        if unexpected_keys:
                            print(f"Warning: Unexpected keys when loading VLM weights: {unexpected_keys[:10]}...")
                    except Exception as e:
                        print(f"Error loading VLM checkpoint: {e}")
                        raise
                else:
                    # Try to find actual model directory
                    possible_paths = [
                        "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
                        "pretrained_models/configs/prism-qwen25-extra-dinosiglip-224px-0_5b",
                        name_or_path.replace("/configs/config.json", "").replace("/configs", ""),
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                            # Check if this path has model weights
                            has_weights = any(
                                os.path.exists(os.path.join(path, f))
                                for f in ["model.safetensors", "pytorch_model.bin", "model.ckpt.index"]
                            )
                            if has_weights:
                                base_model_path = path
                                break
        except Exception as e:
            print(f"Warning: Could not determine base model path from adapter config: {e}")
        
        # If still not found and not using config-only, try default paths
        if base_model_path is None and not use_config_only:
            default_paths = [
                "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b",
                "pretrained_models/configs/prism-qwen25-extra-dinosiglip-224px-0_5b",
            ]
            for path in default_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                    # Check if this path has model weights
                    has_weights = any(
                        os.path.exists(os.path.join(path, f))
                        for f in ["model.safetensors", "pytorch_model.bin", "model.ckpt.index"]
                    )
                    if has_weights:
                        base_model_path = path
                        break
        
        # Load base model if not already loaded from config
        if not use_config_only:
            if base_model_path is None:
                raise ValueError(
                    f"Could not determine base model path for LoRA checkpoint. "
                    f"Please ensure the base model exists at one of the expected locations, "
                    f"or merge LoRA weights using: python vla-scripts/merge_lora_weights_and_save.py"
                )
            
            print(f"Loading base model from: {base_model_path}")
            # Load base model
            base_vla = AutoModelForVision2Seq.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from: {adapter_dir}")
        from peft import PeftModel, LoraConfig

        # Check if adapter_config.json exists
        adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
        
        if not os.path.exists(adapter_config_path):
            # adapter_config.json is missing - this happens when save_pretrained saves model config instead of PEFT config
            # Create a minimal adapter_config.json with default LoRA settings
            print(f"Warning: adapter_config.json not found, creating one with default LoRA settings...")
            
            # Read base model path from config.json if available
            config_json_path = os.path.join(adapter_dir, "config.json")
            base_model_path = "pretrained_models/configs/config.json"
            if os.path.exists(config_json_path):
                with open(config_json_path, "r") as f:
                    config_data = json.load(f)
                    base_model_path = config_data.get("_name_or_path", base_model_path)
            
            # Default LoRA config (based on finetune.py defaults: r=32, lora_alpha=2*r=64, target_modules="all-linear")
            adapter_config = {
                "peft_type": "LORA",
                "task_type": "FEATURE_EXTRACTION",
                "base_model_name_or_path": base_model_path,
                "inference_mode": False,
                "r": 32,  # Default lora_rank
                "target_modules": "all-linear",  # Based on finetune.py
                "lora_alpha": 64,  # 2 * lora_rank
                "lora_dropout": 0.0,
                "bias": "none",
                "init_lora_weights": "gaussian",
            }
            
            # Save the adapter config
            with open(adapter_config_path, "w") as f:
                json.dump(adapter_config, f, indent=2)
            print(f"Created adapter_config.json at {adapter_config_path}")
        
        vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        # Merge LoRA weights for inference (this is more efficient than keeping them separate)
        vla = vla.merge_and_unload()
    else:
        # Standard checkpoint (merged model or non-LoRA)
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.pretrained_checkpoint,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        )

    try:
        action_queries_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_queries")
    except AssertionError:
        action_queries_path = None
    if action_queries_path is not None:
        action_queries_state = load_component_state_dict(action_queries_path)
        load_action_queries_into_model(vla, action_queries_state, context="[eval]")
    else:
        action_queries_state = load_action_queries_from_model_safetensors(cfg.pretrained_checkpoint)
        if action_queries_state is not None:
            load_action_queries_into_model(vla, action_queries_state, context="[eval:model.safetensors]")
        else:
            action_queries = vla.action_queries.weight.detach().float().cpu()
            print(
                "[eval] no explicit action_queries checkpoint found and no action_queries.weight in model.safetensors; "
                "relying on model/adapters only. "
                f"Current action_queries shape={tuple(action_queries.shape)} "
                f"abs_mean={float(action_queries.abs().mean()):.6f} "
                f"norm={float(action_queries.norm()):.6f} "
                f"min={float(action_queries.min()):.6f} "
                f"max={float(action_queries.max()):.6f}"
            )

    # If using FiLM, wrap the vision backbone to allow for infusion of language inputs
    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)

    # Set number of images in model input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    # Move model to device if not using quantization
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats for action normalization
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla


def _apply_film_to_vla(vla: torch.nn.Module, cfg: Any) -> torch.nn.Module:
    """
    Apply FiLM (Feature-wise Linear Modulation) to the VLA vision backbone.

    Args:
        vla: The VLA model
        cfg: Configuration object with model parameters

    Returns:
        torch.nn.Module: VLA model with FiLM applied
    """
    from peft import LoraConfig, get_peft_model

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)

    # Create and apply FiLMed vision backbone
    new_vision_backbone = FiLMedPrismaticVisionBackbone(
        vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim,
    )
    vla.model.vision_backbone = new_vision_backbone

    # Load vision backbone checkpoint
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "vision_backbone")
    state_dict = torch.load(checkpoint_path, weights_only=True)
    vla.model.vision_backbone.load_state_dict(state_dict)

    # Use the model component instead of wrapper and convert to bfloat16
    vla = vla.model
    vla.vision_backbone = vla.vision_backbone.to(torch.bfloat16)

    return vla


def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.

    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """
    if model_is_on_hf_hub(checkpoint_path):
        # Download dataset stats directly from HF Hub
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )


def get_processor(cfg: Any) -> AutoProcessor:
    """
    Get the VLA model's Hugging Face processor.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        AutoProcessor: The model's processor
    """
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)


def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int) -> ProprioProjector:
    """
    Get proprioception projector for the VLA model.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model
        proprio_dim: Dimension of proprioception data

    Returns:
        ProprioProjector: The initialized proprio projector
    """
    # Initialize projector and move to device
    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=proprio_dim,
    ).to(DEVICE)
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_proprio_projector_name = {
            "VLA-Adapter/LIBERO-Spatial-Pro": "proprio_projector--checkpoint.pt",
            "VLA-Adapter/LIBERO-Object-Pro": "proprio_projector--checkpoint.pt",
            "VLA-Adapter/LIBERO-Goal-Pro": "proprio_projector--checkpoint.pt",
            "VLA-Adapter/LIBERO-Long-Pro": "proprio_projector--checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_proprio_projector_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        # Download proprio projector directly from HF Hub
        proprio_projector_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_proprio_projector_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(proprio_projector_path)
        proprio_projector.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
        state_dict = load_component_state_dict(checkpoint_path)
        proprio_projector.load_state_dict(state_dict)

    return proprio_projector


def get_noisy_action_projector(cfg: Any, llm_dim: int) -> NoisyActionProjector:
    """
    Get noisy action projector for diffusion-based action prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        NoisyActionProjector: The initialized noisy action projector
    """
    # Initialize projector and move to device
    noisy_action_projector = NoisyActionProjector(
        llm_dim=llm_dim,
    ).to(DEVICE)
    noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(DEVICE)
    noisy_action_projector.eval()

    # Find and load checkpoint
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "noisy_action_projector")
    state_dict = load_component_state_dict(checkpoint_path)
    noisy_action_projector.load_state_dict(state_dict)

    return noisy_action_projector


def get_action_head(cfg: Any, llm_dim: int, model: Optional[torch.nn.Module] = None) -> Union[L1RegressionActionHead, DiffusionActionHead, FlowMatchingMLPActionHead, torch.nn.Module]:
    """Load the action head checkpoint supporting both local checkpoints and Hugging Face Hub repos."""

    def _infer_num_dit_blocks(state_dict: dict) -> Optional[int]:
        prefix = "velocity_network.ditx.blocks."
        block_ids = []
        for key in state_dict.keys():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            block_str = suffix.split(".", 1)[0]
            if block_str.isdigit():
                block_ids.append(int(block_str))
        return (max(block_ids) + 1) if block_ids else None

    def _has_any_prefix(state_dict: dict, prefixes: Tuple[str, ...]) -> bool:
        return any(key.startswith(prefixes) for key in state_dict.keys())

    def _infer_legacy_dit_anchor_blend(cfg: Any, anchor_enabled: bool, current_blend: float) -> float:
        if not anchor_enabled or current_blend > 0.0:
            return current_blend
        if getattr(cfg, "dit_anchor_blend_was_set", False):
            return current_blend

        checkpoint_name = Path(str(getattr(cfg, "pretrained_checkpoint", ""))).name.lower()
        if "residual-anchor" in checkpoint_name:
            inferred_blend = 0.875
            print(
                "[compat] inferred legacy DIT residual-anchor checkpoint; "
                f"setting dit_anchor_blend={inferred_blend}"
            )
            return inferred_blend
        return current_blend

    skip_loading = getattr(cfg, "skip_action_head_loading", False)
    state_dict = None
    inferred_dit_num_blocks = None
    inferred_flowmlp_anchor_enabled = False
    inferred_dit_anchor_enabled = False
    if not skip_loading:
        if model_is_on_hf_hub(cfg.pretrained_checkpoint):
            model_path_to_action_head_name = {
                "VLA-Adapter/LIBERO-Spatial-Pro": "action_head--checkpoint.pt",
                "VLA-Adapter/LIBERO-Object-Pro": "action_head--checkpoint.pt",
                "VLA-Adapter/LIBERO-Goal-Pro": "action_head--checkpoint.pt",
                "VLA-Adapter/LIBERO-Long-Pro": "action_head--checkpoint.pt",
            }
            if cfg.pretrained_checkpoint not in model_path_to_action_head_name.keys():
                raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
            action_head_path = hf_hub_download(
                repo_id=cfg.pretrained_checkpoint,
                filename=model_path_to_action_head_name[cfg.pretrained_checkpoint],
            )
            state_dict = load_component_state_dict(action_head_path)
        else:
            checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
            state_dict = load_component_state_dict(checkpoint_path)

        inferred_dit_num_blocks = _infer_num_dit_blocks(state_dict)
        inferred_flowmlp_anchor_enabled = _has_any_prefix(state_dict, ("anchor_proj.",))
        inferred_dit_anchor_enabled = _has_any_prefix(state_dict, ("anchor_head.", "state_anchor_proj."))
        print(
            f"[compat] action_head checkpoint inference: "
            f"dit_blocks={inferred_dit_num_blocks}, "
            f"flowmlp_anchor={inferred_flowmlp_anchor_enabled}, "
            f"dit_anchor={inferred_dit_anchor_enabled}"
        )

    # Initialize appropriate action head based on configuration
    if cfg.use_flow_matching:
        head_type = getattr(cfg, "flow_matching_head_type", "ditx")
        if model is not None:
            NUM_PATCHES = model.vision_backbone.get_num_patches() * model.vision_backbone.get_num_images_in_input()
        else:
            num_images = getattr(cfg, "num_images_in_input", 2)
            NUM_PATCHES = 256 * num_images
            print(f"Warning: Model instance not provided, using default NUM_PATCHES={NUM_PATCHES} (num_images={num_images})")
        if head_type == "mlp":
            flowmlp_supervised_anchor_weight = getattr(cfg, "flowmlp_supervised_anchor_weight", 0.0)
            flowmlp_anchor_blend = getattr(cfg, "flowmlp_anchor_blend", 0.0)
            if inferred_flowmlp_anchor_enabled and flowmlp_supervised_anchor_weight <= 0.0 and flowmlp_anchor_blend <= 0.0:
                # Instantiate the anchor branch for legacy checkpoints even if eval defaults leave it disabled.
                flowmlp_supervised_anchor_weight = 1.0
            action_head = FlowMatchingMLPActionHead(
                input_dim=llm_dim,
                hidden_dim=getattr(cfg, "flow_matching_mlp_hidden_dim", 1024),
                output_dim=ACTION_DIM,
                num_task_tokens=NUM_PATCHES,
                num_layers=getattr(cfg, "flow_matching_mlp_num_layers", 3),
                time_dim=256,
                dropout=0.1,
                use_adaptive_bridge=getattr(cfg, "use_adaptive_bridge", True),
                bridge_mode=getattr(cfg, "bridge_mode", "adaptive"),
                fixed_layer_index=getattr(cfg, "fixed_layer_index", -1),
                use_latent_skill_token=(
                    getattr(cfg, "flowmlp_use_latent_skill_token", False)
                ),
                use_continuous_context=getattr(cfg, "flowmlp_use_continuous_context", False),
                skill_use_layer_routing=getattr(cfg, "flowmlp_skill_use_layer_routing", True),
                skill_use_direct_conditioning=getattr(
                    cfg, "flowmlp_skill_use_direct_conditioning", True
                ),
                continuous_context_use_direct_conditioning=getattr(
                    cfg, "flowmlp_continuous_context_use_direct_conditioning", False
                ),
                num_skill_tokens=getattr(cfg, "flowmlp_num_skill_tokens", 16),
                skill_token_dim=getattr(cfg, "flowmlp_skill_token_dim", 128),
                skill_temperature=getattr(cfg, "flowmlp_skill_temperature", 1.0),
                skill_entropy_weight=getattr(cfg, "flowmlp_skill_entropy_weight", 0.0),
                num_inference_steps=getattr(cfg, "flowmlp_num_inference_steps", 5),
                num_inference_samples=getattr(cfg, "flowmlp_num_inference_samples", 8),
                supervised_anchor_weight=flowmlp_supervised_anchor_weight,
                anchor_blend=flowmlp_anchor_blend,
                anchor_gripper_weight=getattr(cfg, "flowmlp_anchor_gripper_weight", 1.0),
                anchor_gripper_bce_weight=getattr(cfg, "flowmlp_anchor_gripper_bce_weight", 0.0),
                detach_flow_conditioning=getattr(cfg, "flowmlp_detach_flow_conditioning", False),
            )
        else:
            from prismatic.models.flow_matching_head import FlowMatchingActionHead

            dit_supervised_anchor_weight = getattr(cfg, "dit_supervised_anchor_weight", 0.0)
            dit_anchor_blend = getattr(cfg, "dit_anchor_blend", 0.0)
            dit_anchor_blend = _infer_legacy_dit_anchor_blend(
                cfg, inferred_dit_anchor_enabled, dit_anchor_blend
            )
            if inferred_dit_anchor_enabled and dit_supervised_anchor_weight <= 0.0 and dit_anchor_blend <= 0.0:
                # Instantiate the anchor branch for legacy checkpoints even if eval defaults leave it disabled.
                dit_supervised_anchor_weight = 1.0
            dit_flow_ratio = getattr(cfg, "flow_ratio", 1.0)
            action_head = FlowMatchingActionHead(
                input_dim=llm_dim,
                hidden_dim=llm_dim,
                action_dim=ACTION_DIM,
                num_task_tokens=NUM_PATCHES,
                num_blocks=inferred_dit_num_blocks or getattr(cfg, "dit_num_blocks", 12),
                num_heads=8,
                mlp_ratio=4.0,
                use_pro_version=cfg.use_pro_version,
                flow_ratio=dit_flow_ratio,
                time_dist=("lognorm", -0.4, 1.0),
                inference_default_mode="ode",
                num_inference_steps=getattr(cfg, "dit_num_inference_steps", 5),
                num_inference_samples=getattr(cfg, "dit_num_inference_samples", 8),
                supervised_anchor_weight=dit_supervised_anchor_weight,
                anchor_blend=dit_anchor_blend,
                inference_residual_scale=getattr(cfg, "dit_inference_residual_scale", 1.0),
                anchor_gripper_weight=getattr(cfg, "dit_anchor_gripper_weight", 1.0),
                anchor_gripper_bce_weight=getattr(cfg, "dit_anchor_gripper_bce_weight", 0.0),
                flow_xyz_loss_weight=getattr(cfg, "dit_flow_xyz_loss_weight", 1.0),
                flow_rot_loss_weight=getattr(cfg, "dit_flow_rot_loss_weight", 1.0),
                flow_gripper_loss_weight=getattr(cfg, "dit_flow_gripper_loss_weight", 1.0),
                flow_gripper_bce_weight=getattr(cfg, "dit_flow_gripper_bce_weight", 0.0),
                flow_gripper_bce_logit_scale=getattr(cfg, "dit_flow_gripper_bce_logit_scale", 1.0),
                flow_gripper_bce_balanced=getattr(cfg, "dit_flow_gripper_bce_balanced", False),
                gripper_head_weight=getattr(cfg, "dit_gripper_head_weight", 0.0),
                gripper_head_override=getattr(cfg, "dit_gripper_head_override", False),
                clip_normalized_actions=getattr(cfg, "dit_clip_normalized_actions", False),
                detach_flow_conditioning=getattr(cfg, "dit_detach_flow_conditioning", False),
                use_state_conditioning=getattr(cfg, "dit_use_state_conditioning", False),
                state_scale_mode=getattr(cfg, "dit_state_scale_mode", "none"),
                state_proprio_mode=getattr(cfg, "dit_state_proprio_mode", "concat"),
                state_use_chunk_pos=getattr(cfg, "dit_state_use_chunk_pos", False),
                state_include_task_tokens=getattr(cfg, "dit_state_include_task_tokens", False),
                condition_mode=getattr(cfg, "dit_condition_mode", "full"),
                condition_injection_mode=getattr(cfg, "dit_condition_injection_mode", "cross_attn"),
                include_prompt_tokens=getattr(cfg, "dit_include_prompt_tokens", False),
                task_token_mode=getattr(cfg, "dit_task_token_mode", "vision_prompt"),
                use_latent_skill_token=getattr(cfg, "dit_use_latent_skill_token", False),
                num_skill_tokens=getattr(cfg, "dit_num_skill_tokens", 16),
                skill_token_dim=getattr(cfg, "dit_skill_token_dim", 128),
                skill_temperature=getattr(cfg, "dit_skill_temperature", 1.0),
                dit_zero_init_adaln=getattr(cfg, "dit_zero_init_adaln", True),
                dit_zero_init_output=getattr(cfg, "dit_zero_init_output", True),
                use_adaptive_bridge=getattr(cfg, "use_adaptive_bridge", True),
                bridge_mode=getattr(cfg, "bridge_mode", "adaptive"),
                fixed_layer_index=getattr(cfg, "fixed_layer_index", -1),
            )
    elif cfg.use_diffusion:
        action_head = DiffusionActionHead(
            input_dim=llm_dim,
            hidden_dim=llm_dim,
            action_dim=ACTION_DIM,
            num_task_tokens=512,
            num_diffusion_steps=cfg.num_diffusion_steps if hasattr(cfg, "num_diffusion_steps") else 50,
            use_pro_version=cfg.use_pro_version,
        )
    elif cfg.use_l1_regression:
        action_head = L1RegressionActionHead(
            input_dim=llm_dim,
            hidden_dim=llm_dim,
            action_dim=ACTION_DIM,
            use_pro_version=cfg.use_pro_version,
        )
    else:
        raise ValueError("Either use_l1_regression, use_diffusion, or use_flow_matching must be True")

    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()
    if cfg.use_flow_matching and getattr(cfg, "flow_matching_head_type", "ditx") != "mlp":
        pure_inference = bool(getattr(cfg, "dit_pure_inference", False))
        disable_anchor = bool(getattr(cfg, "dit_disable_inference_anchor", False)) or pure_inference
        if hasattr(action_head, "pure_inference"):
            action_head.pure_inference = pure_inference
            if pure_inference:
                print("[diag] DIT pure inference enabled; bypassing inference anchor reconstruction.")
        if hasattr(action_head, "disable_inference_anchor"):
            action_head.disable_inference_anchor = disable_anchor
            if disable_anchor:
                print("[diag] DIT inference anchor disabled; sampling pure flow residual output.")
        if hasattr(action_head, "velocity_network"):
            if hasattr(action_head.velocity_network, "pure_inference"):
                action_head.velocity_network.pure_inference = pure_inference
                if pure_inference:
                    print("[diag] DIT pure inference enabled; bypassing adaptive_gated token gate and alpha scale.")
            debug_group_action_tokens = bool(getattr(cfg, "debug_dit_group_action_tokens_to_chunk", False))
            if hasattr(action_head.velocity_network, "debug_group_action_tokens_to_chunk"):
                action_head.velocity_network.debug_group_action_tokens_to_chunk = debug_group_action_tokens
                if debug_group_action_tokens:
                    print("[diag] DIT velocity network will group action-token context into chunk-level means.")
        if hasattr(action_head, "debug_use_state_conditioning"):
            action_head.debug_use_state_conditioning = bool(
                getattr(cfg, "debug_dit_use_state_conditioning", False)
            )
            if action_head.debug_use_state_conditioning:
                print("[diag] DIT inference will use chunk-aligned state conditioning instead of full token context.")
        if hasattr(action_head, "debug_state_scale_mode"):
            action_head.debug_state_scale_mode = str(getattr(cfg, "debug_dit_state_scale_mode", "none"))
            if action_head.debug_state_scale_mode != "none":
                print(f"[diag] DIT debug state scale mode enabled: {action_head.debug_state_scale_mode}")
        if getattr(cfg, "dit_use_state_conditioning", False):
            print("[config] DIT configured to use chunk-aligned state conditioning.")
        if getattr(cfg, "dit_state_scale_mode", "none") != "none":
            print(f"[config] DIT state conditioning scale mode: {getattr(cfg, 'dit_state_scale_mode', 'none')}")
        print(f"[config] DIT state proprio mode: {getattr(cfg, 'dit_state_proprio_mode', 'concat')}")
        print(f"[config] DIT state chunk pos: {getattr(cfg, 'dit_state_use_chunk_pos', False)}")
        print(f"[config] DIT state include task tokens: {getattr(cfg, 'dit_state_include_task_tokens', False)}")
        print(f"[config] DIT condition mode: {getattr(cfg, 'dit_condition_mode', 'full')}")
        print(f"[config] DIT condition injection mode: {getattr(cfg, 'dit_condition_injection_mode', 'cross_attn')}")
        print(f"[config] DIT inference residual scale: {getattr(cfg, 'dit_inference_residual_scale', 1.0)}")
        print(f"[config] DIT include prompt tokens: {getattr(cfg, 'dit_include_prompt_tokens', False)}")
        print(f"[config] DIT task token mode: {getattr(cfg, 'dit_task_token_mode', 'vision_prompt')}")
        print(f"[config] DIT clip normalized actions: {getattr(cfg, 'dit_clip_normalized_actions', False)}")
        print(
            "[config] DIT init flags: "
            f"zero_init_adaln={getattr(cfg, 'dit_zero_init_adaln', True)} "
            f"zero_init_output={getattr(cfg, 'dit_zero_init_output', True)}"
        )

    if not skip_loading:
        try:
            action_head.load_state_dict(state_dict)
        except (RuntimeError, FileNotFoundError) as exc:
            exc_text = str(exc)
            can_retry_dit_anchor = (
                isinstance(exc, RuntimeError)
                and cfg.use_flow_matching
                and getattr(cfg, "flow_matching_head_type", "ditx") != "mlp"
                and inferred_dit_anchor_enabled
                and ("anchor_head." in exc_text or "state_anchor_proj." in exc_text)
            )
            if can_retry_dit_anchor:
                print("[compat] retrying DIT action head load with anchor modules enabled")
                from prismatic.models.flow_matching_head import FlowMatchingActionHead

                action_head = FlowMatchingActionHead(
                    input_dim=llm_dim,
                    hidden_dim=llm_dim,
                    action_dim=ACTION_DIM,
                    num_task_tokens=NUM_PATCHES,
                    num_blocks=inferred_dit_num_blocks or getattr(cfg, "dit_num_blocks", 12),
                    num_heads=8,
                    mlp_ratio=4.0,
                    use_pro_version=cfg.use_pro_version,
                    flow_ratio=dit_flow_ratio,
                    time_dist=("lognorm", -0.4, 1.0),
                    inference_default_mode="ode",
                    num_inference_steps=getattr(cfg, "dit_num_inference_steps", 5),
                    num_inference_samples=getattr(cfg, "dit_num_inference_samples", 8),
                    supervised_anchor_weight=max(dit_supervised_anchor_weight, 1.0),
                    anchor_blend=dit_anchor_blend,
                    inference_residual_scale=getattr(cfg, "dit_inference_residual_scale", 1.0),
                    anchor_gripper_weight=getattr(cfg, "dit_anchor_gripper_weight", 1.0),
                    anchor_gripper_bce_weight=getattr(cfg, "dit_anchor_gripper_bce_weight", 0.0),
                    flow_xyz_loss_weight=getattr(cfg, "dit_flow_xyz_loss_weight", 1.0),
                    flow_rot_loss_weight=getattr(cfg, "dit_flow_rot_loss_weight", 1.0),
                    flow_gripper_loss_weight=getattr(cfg, "dit_flow_gripper_loss_weight", 1.0),
                    flow_gripper_bce_weight=getattr(cfg, "dit_flow_gripper_bce_weight", 0.0),
                    flow_gripper_bce_logit_scale=getattr(cfg, "dit_flow_gripper_bce_logit_scale", 1.0),
                    flow_gripper_bce_balanced=getattr(cfg, "dit_flow_gripper_bce_balanced", False),
                    gripper_head_weight=getattr(cfg, "dit_gripper_head_weight", 0.0),
                    gripper_head_override=getattr(cfg, "dit_gripper_head_override", False),
                    clip_normalized_actions=getattr(cfg, "dit_clip_normalized_actions", False),
                    detach_flow_conditioning=getattr(cfg, "dit_detach_flow_conditioning", False),
                    use_state_conditioning=getattr(cfg, "dit_use_state_conditioning", False),
                    state_scale_mode=getattr(cfg, "dit_state_scale_mode", "none"),
                    state_proprio_mode=getattr(cfg, "dit_state_proprio_mode", "concat"),
                    state_use_chunk_pos=getattr(cfg, "dit_state_use_chunk_pos", False),
                    state_include_task_tokens=getattr(cfg, "dit_state_include_task_tokens", False),
                    condition_mode=getattr(cfg, "dit_condition_mode", "full"),
                    condition_injection_mode=getattr(cfg, "dit_condition_injection_mode", "cross_attn"),
                    include_prompt_tokens=getattr(cfg, "dit_include_prompt_tokens", False),
                    task_token_mode=getattr(cfg, "dit_task_token_mode", "vision_prompt"),
                    use_latent_skill_token=getattr(cfg, "dit_use_latent_skill_token", False),
                    num_skill_tokens=getattr(cfg, "dit_num_skill_tokens", 16),
                    skill_token_dim=getattr(cfg, "dit_skill_token_dim", 128),
                    skill_temperature=getattr(cfg, "dit_skill_temperature", 1.0),
                    dit_zero_init_adaln=getattr(cfg, "dit_zero_init_adaln", True),
                    dit_zero_init_output=getattr(cfg, "dit_zero_init_output", True),
                    use_adaptive_bridge=getattr(cfg, "use_adaptive_bridge", True),
                bridge_mode=getattr(cfg, "bridge_mode", "adaptive"),
                    fixed_layer_index=getattr(cfg, "fixed_layer_index", -1),
                ).to(torch.bfloat16).to(DEVICE)
                action_head.eval()
                pure_inference = bool(getattr(cfg, "dit_pure_inference", False))
                action_head.pure_inference = pure_inference
                if hasattr(action_head, "velocity_network"):
                    action_head.velocity_network.pure_inference = pure_inference
                action_head.disable_inference_anchor = bool(getattr(cfg, "dit_disable_inference_anchor", False)) or pure_inference
                if pure_inference:
                    print("[diag] DIT pure inference enabled; bypassing inference-time gates/anchor reconstruction.")
                if action_head.disable_inference_anchor:
                    print("[diag] DIT inference anchor disabled; sampling pure flow residual output.")
                action_head.load_state_dict(state_dict)
            elif getattr(cfg, "allow_partial_action_head_load", False):
                print(
                    f"Warning: failed to load action head weights due to {exc}. "
                    "Continuing with randomly initialised DiffusionActionHead."
                )
            else:
                raise

    return action_head


def get_depth_interface(cfg: Any, llm_dim: int, model: Optional[torch.nn.Module] = None) -> Optional[SkillAdaptiveDepthInterface]:
    """Load the shared skill-conditioned depth interface for continuous action heads."""
    if not (getattr(cfg, "use_depth_interface", False) or getattr(cfg, "depth_interface_mode", "none") != "none"):
        return None

    if model is not None:
        num_patches = model.vision_backbone.get_num_patches() * model.vision_backbone.get_num_images_in_input()
    else:
        num_images = getattr(cfg, "num_images_in_input", 2)
        num_patches = 256 * num_images
        print(f"Warning: Model instance not provided, using default NUM_PATCHES={num_patches} (num_images={num_images})")

    depth_interface = SkillAdaptiveDepthInterface(
        input_dim=llm_dim,
        num_task_tokens=num_patches,
        mode=getattr(cfg, "depth_interface_mode", "skill_adaptive"),
        fixed_layer_index=getattr(cfg, "fixed_layer_index", -1),
        max_vlm_layers=getattr(cfg, "depth_interface_max_layers", 64),
        num_skill_tokens=getattr(cfg, "flowmlp_num_skill_tokens", 16),
        skill_token_dim=getattr(cfg, "flowmlp_skill_token_dim", 128),
        skill_temperature=getattr(cfg, "flowmlp_skill_temperature", 1.0),
        skill_entropy_weight=getattr(cfg, "flowmlp_skill_entropy_weight", 0.0),
        add_proprio_to_output=getattr(cfg, "depth_interface_add_proprio", True),
    ).to(torch.bfloat16).to(DEVICE)
    depth_interface.eval()

    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "depth_interface")
    state_dict = load_component_state_dict(checkpoint_path)
    depth_interface.load_state_dict(state_dict)
    return depth_interface


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    """
    Prepare images for VLA input by resizing and cropping as needed.

    Args:
        images: List of input images as numpy arrays
        cfg: Configuration object with parameters

    Returns:
        List[Image.Image]: Processed images ready for the model
    """
    processed_images = []

    for image in images:
        # Validate format
        check_image_format(image)

        # Resize if needed
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

        # Convert to PIL image
        pil_image = Image.fromarray(image).convert("RGB")

        # Apply center crop if configured
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)

        processed_images.append(pil_image)

    return processed_images


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    depth_interface: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    use_minivlm: bool = False,
) -> List[np.ndarray]:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task
        action_head: Optional action head for continuous actions
        depth_interface: Optional shared adaptive depth interface
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        List[np.ndarray]: Predicted actions
    """
    with torch.inference_mode():

        # Collect all input images
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # Process images
        all_images = prepare_images_for_vla(all_images, cfg)

        # Extract primary image and additional images
        primary_image = all_images.pop(0)

        # Build VLA prompt
        if not use_minivlm:
            prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        else:
            prompt = f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {task_label.lower()}?<|im_end|>\n<|im_start|>assistant\n'

        # Process primary image
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

        # Process additional wrist images if any
        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
            ]
            # Concatenate all images
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        # Process proprioception data if used
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]

        
        # Generate action
        if action_head is None:
            # Standard VLA output (single-image inputs, discrete actions)
            action, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
        else:
            # Custom action head for continuous actions
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                action_head=action_head,
                depth_interface=depth_interface,
                use_film=use_film,
            )

    # Extract subset of actions for open loop steps
    return [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]


def get_action_from_server(
    observation: Dict[str, Any], server_endpoint: str = "http://0.0.0.0:8777/act"
) -> Dict[str, Any]:
    """
    Get VLA action from remote inference server.

    Args:
        observation: Observation data to send to server
        server_endpoint: URL of the inference server

    Returns:
        Dict[str, Any]: Action response from server
    """
    response = requests.post(
        server_endpoint,
        json=observation,
    )
    return response.json()
