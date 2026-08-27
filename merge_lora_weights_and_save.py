
"""
Loads a checkpoint that only has a LoRA adapter (no merged model) and merges the adapter
into the base VLA-Adapter model. Saves the final checkpoint in the same directory.

Usage:
    python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint openvla/openvla-7b \
        --lora_finetuned_checkpoint_dir /PATH/TO/CHECKPOINT/DIR/
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load, load_vla



@dataclass
class ConvertConfig:
    # fmt: off

    base_checkpoint: Union[str, Path] = ""                   # Base model checkpoint path/dir (either openvla/openvla-7b or whichever model you fine-tuned / resumed training from)
    lora_finetuned_checkpoint_dir: Union[str, Path] = ""     # Checkpoint directory containing the LoRA adapter
    vlm_path: Union[str, Path] = "" 
    use_minivla: bool = False                        # 


    # fmt: on


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if cfg.use_minivla:
        if not cfg.vlm_path:
            cfg.vlm_path = "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
        
        print(f"Loading VLM from: {cfg.vlm_path}")
        from prismatic.models import load
        vlm = load(
            cfg.vlm_path,
            hf_token=None,
            load_for_training=False,
            )
        
        config_path = "pretrained_models/configs/config.json"
        print(f"Creating VLA model from config: {config_path}")
        config = AutoConfig.from_pretrained(config_path)
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
        
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
    
        print("Loading VLM weights into VLA model...")
        missing_keys, unexpected_keys = vla.load_state_dict(converted_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys: {list(missing_keys)[:10]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {list(unexpected_keys)[:10]}...")
    else:
        # Load Model using HF AutoClasses
        print(f"Loading base model: {cfg.base_checkpoint}")
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.base_checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    # Load LoRA weights and merge into base model, then save final checkpoint
    adapter_dir = os.path.join(cfg.lora_finetuned_checkpoint_dir, "lora_adapter")
    # Convert to absolute path to avoid HuggingFace Hub validation issues
    adapter_dir = os.path.abspath(adapter_dir)
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    
    # Check if adapter_config.json exists, if not create it
    if not os.path.exists(adapter_config_path):
        print(f"Warning: adapter_config.json not found, creating one with default LoRA settings...")
        
        # Read base model path from config.json if available
        config_json_path = os.path.join(adapter_dir, "config.json")
        base_model_path = "pretrained_models/configs/config.json"
        if os.path.exists(config_json_path):
            import json
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
        import json
        with open(adapter_config_path, "w") as f:
            json.dump(adapter_config, f, indent=2)
        print(f"Created adapter_config.json at {adapter_config_path}")
    
    print("Merging LoRA weights into base model...")
    print(f"Loading LoRA adapter from: {adapter_dir}")
    start_time = time.time()
    
    # Try to use PeftModel.from_pretrained with explicit local_files_only
    # But first, ensure adapter_config.json exists
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(
            f"adapter_config.json not found at {adapter_config_path}. "
            f"Please ensure the LoRA adapter was saved correctly."
        )
    
    # Use a workaround: temporarily change to the adapter directory to avoid path validation
    original_cwd = os.getcwd()
    try:
        # Change to adapter directory parent to use relative path
        adapter_parent = os.path.dirname(adapter_dir)
        adapter_name = os.path.basename(adapter_dir)
        os.chdir(adapter_parent)
        
        # Now use relative path which should work better
        merged_vla = PeftModel.from_pretrained(vla, adapter_name, local_files_only=True).to("cuda")
    merged_vla = merged_vla.merge_and_unload()
    finally:
        os.chdir(original_cwd)
    
    # Convert checkpoint dir to absolute path for saving
    checkpoint_dir = os.path.abspath(cfg.lora_finetuned_checkpoint_dir)
    merged_vla.save_pretrained(checkpoint_dir)
    print(f"\nMerging complete! Time elapsed (sec): {time.time() - start_time}")
    print(f"\nSaved merged model checkpoint at:\n{checkpoint_dir}")


if __name__ == "__main__":
    main()
