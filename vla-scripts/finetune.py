"""
finetune.py

Fine-tunes Qwen2.5-0.5B via LoRA.
(PATCHED) Fixes:
  - grad_accum "no effect" when using accelerate flags: now prints effective cfg and uses true DDP accumulation with no_sync()
  - device/dtype bugs: labels/proprio moved to GPU; actions kept float32 for stable losses
  - token accuracy weird: metrics computed via shifted logits/labels with -100 masking (no fragile slicing)
  - flow loss instability: optional grad clipping, safer dtype, robust action-token gathering, optional last-K-layer feature
  - validation crash: missing use_flow_matching fixed
  - safer barriers and step accounting; logging/checkpointing tied to optimizer-step (global_step)

Local patch (this version):
  - Fix interface mismatch: run_forward_pass uses num_patches; all callers pass num_patches consistently
  - Fix FiLM wrapping attribute path and make it work with/without LoRA (PeftModel)
  - Fix freeze_vlm to keep LoRA trainable when use_lora=True (and always keep action_queries trainable)
  - Fix NUM_PATCHES accounting: DO NOT +1 for proprio (your VLM forward does not insert a proprio token)
  - modeling_prismatic.py already returns logits (per user), so token-mode does not require extra patch there
"""

import json
import os
import sys
import time
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

# 娣囨繆鐦夋い鍦窗閺嶇懓婀?sys.path 闁插矉绱濇潻娆愮壉閺冪姾顔戞禒搴℃憿娑擃亞娲拌ぐ鏇＄箥鐞涘矂鍏橀懗鑺ュ閸?experiments / prismatic
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import draccus

from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.optimization import get_cosine_schedule_with_warmup
import wandb
from wandb.errors import CommError as WandBCommError

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead, FlowMatchingMLPActionHead
from prismatic.models.adaptive_depth_interface import SkillAdaptiveDepthInterface
from prismatic.models.flow_matching_head import FlowMatchingActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import (
    compute_actions_l1_loss,
    compute_token_accuracy,
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NUM_TOKENS,
    STOP_INDEX,
)
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform, RealRobotDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load, load_vla
from checkpoint_protocol import resolve_checkpoint_step_plan


# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Speed knobs (safe defaults)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# Workaround for broken/mismatched system cuDNN (e.g. libcudnn_cnn_train.so.8 undefined symbol).
# Set VLA_DISABLE_CUDNN=1 to use non-cuDNN CUDA kernels (slower but avoids "FIND was unable to find an engine").
if os.environ.get("VLA_DISABLE_CUDNN", "").lower() in ("1", "true", "yes"):
    torch.backends.cudnn.enabled = False


@dataclass
class FinetuneConfig:
    # fmt: off
    # 姒涙顓婚悽銊︽拱閸?config + 閺堫剙婀?Qwen 閸╁搫楠囬敍鍫滅瑢 train.sh 娑撯偓閼疯揪绱氶敍宀€顬囩痪鍨讲鐠烘埊绱遍悽?7B 閺冩湹绱?--config_file_path openvla/openvla-7b --vlm_path openvla/openvla-7b --use_minivlm False
    config_file_path: str = "pretrained_models/configs"
    vlm_path: str = "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
    use_minivlm: bool = True
    # 鐠侇厾绮岄崗銉ュ經娴滃矂鈧绔撮敍姘矤 checkpoint 缂侇叀顔?vs 娴?VLM 閸╁搫楠囩拋?
    train_from: str = "vlm_base"   # "vlm_base" | "checkpoint" | "checkpoint_init"
    vlm_training: str = "freeze_lora"  # 娴犲懎缍?train_from=="vlm_base" 閺冭埖婀侀弫? "lora" | "full" | "freeze" | "freeze_lora"
    resum_vla_path: str = ""  # required explicitly for checkpoint/checkpoint_init

    # Dataset
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "aloha_scoop_x_into_bowl"
    run_root_dir: Path = Path("runs")
    shuffle_buffer_size: int = 100_000
    real_robot_target_hz: float = 10.0
    real_robot_max_camera_error_ms: float = 40.0
    real_robot_max_camera_skew_ms: float = 30.0
    real_robot_max_state_gap_ms: float = 50.0

    # Algorithm and architecture
    action_head_type: str = "MLP"  # "MLP" (L1 regression) | "DIT" (DiT-X flow) | "FlowMLP" (MLP flow matching)
    use_diffusion: bool = False
    flow_ratio: float = 1.0
    num_diffusion_steps: int = 50
    use_film: bool = False
    freeze_vlm: bool = False       # 閻?vlm_training 濞插墽鏁?
    num_images_in_input: int = 1
    use_proprio: bool = False
    phase1_path: str = "None"

    # Training configuration
    seed: int = 7
    batch_size: int = 8
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 500
    num_steps_before_decay: int = 100000
    use_cosine_schedule: bool = True
    use_constant_lr: bool = False
    grad_accumulation_steps: int = 1
    max_steps: int = 200000
    use_val_set: bool = False
    val_freq: int = 10_000
    val_time_limit: int = 180
    save_freq: int = 10_000
    eval_on_checkpoint: bool = False
    eval_num_trials_per_task: int = 10
    save_latest_checkpoint_only: bool = False
    resume: bool = False            # 閻?train_from 濞插墽鏁撻敍灞藉瑏閻╁瓨甯撮弨?
    resume_step: Optional[int] = None  # train_from=="checkpoint" 閺冭泛绻€婵?
    resume_load_training_state: bool = True
    load_action_head_from_checkpoint: bool = True
    train_action_head_only: bool = False
    freeze_proprio_projector_for_action_head_only: bool = True
    image_aug: bool = True
    diffusion_sample_freq: int = 50

    # LoRA閿涘牏鏁?vlm_training 濞插墽鏁撻敍灞惧灗閸楁洜瀚顔跨殶閻㈩煉绱?
    use_lora: bool = False
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = True

    # Full Finetune閿涘牏鏁?vlm_training 濞插墽鏁撻敍?
    use_fz: bool = False

    # Stability / DDP knobs (NEW)
    ddp_find_unused_params: bool = False
    gradient_clipping_norm: float = 1.0  # set 0 to disable
    action_head_use_last_k_vla_layers: int = -1  # -1 = use all hidden layers; else use last K
    # Latent skill token switches for FlowMLP (default OFF to preserve existing behavior)
    flowmlp_use_latent_skill_token: bool = False
    flowmlp_use_continuous_context: bool = False
    flowmlp_skill_use_layer_routing: bool = True
    flowmlp_skill_use_direct_conditioning: bool = True
    flowmlp_continuous_context_use_direct_conditioning: bool = False
    flowmlp_num_skill_tokens: int = 16
    flowmlp_skill_token_dim: int = 128
    flowmlp_skill_temperature: float = 1.0
    flowmlp_skill_entropy_weight: float = 0.0
    flowmlp_skill_assignment_mode: str = "hard_gumbel"
    flowmlp_skill_routing_mode: str = "legacy"
    flowmlp_skill_layer_temperature: float = 1.0
    flowmlp_skill_temperature_start: float = -1.0
    flowmlp_skill_temperature_anneal_steps: int = 0
    flowmlp_skill_balance_weight: float = 0.0
    flowmlp_skill_z_loss_weight: float = 0.0
    flowmlp_skill_mi_weight: float = 0.0
    flowmlp_skill_layer_mi_weight: float = 0.0
    flowmlp_skill_template_diversity_weight: float = 0.0
    flowmlp_router_lr_scale: float = 1.0
    flowmlp_routing_anchor_layer: int = -1
    flowmlp_routing_adaptive_mix: float = 1.0
    flowmlp_routing_curriculum_warmup_steps: int = 0
    flowmlp_routing_curriculum_teacher_steps: int = 0
    flowmlp_routing_curriculum_num_buckets: int = 5
    flowmlp_routing_teacher_temperature: float = 0.2
    flowmlp_routing_teacher_kl_weight: float = 1.0
    flowmlp_adaptive_layer_alignment: bool = False
    flowmlp_adaptive_num_layers: int = 25
    flowmlp_adaptive_alignment_bottleneck: int = 64
    flowmlp_time_embedding_mode: str = "legacy"
    flowmlp_time_sampling_mode: str = "uniform"
    flowmlp_float32_path: bool = False
    flowmlp_zero_init_output: bool = True
    flowmlp_num_inference_steps: int = 5
    flowmlp_num_inference_samples: int = 8
    flowmlp_supervised_anchor_weight: float = 0.0
    flowmlp_anchor_blend: float = 0.0
    flowmlp_anchor_gripper_weight: float = 1.0
    flowmlp_anchor_rotation_weight: float = 1.0
    flowmlp_anchor_gripper_bce_weight: float = 0.0
    flowmlp_anchor_num_layers: int = 0
    flowmlp_anchor_hidden_dim: int = 1024
    flowmlp_detach_flow_conditioning: bool = False
    flowmlp_flow_curriculum_start_step: int = 0
    flowmlp_flow_curriculum_ramp_steps: int = 0
    flowmlp_include_prompt_tokens: bool = False
    flowmlp_task_token_mode: str = "vision_prompt"
    flowmlp_prompt_direct_conditioning: bool = False
    flowmlp_dense_film_enabled: bool = False
    flowmlp_dense_film_max_layers: int = 64
    flowmlp_dense_film_first_layer_index: int = 1
    flowmlp_dense_film_bottleneck_dim: int = 64
    flowmlp_dense_film_state_dim: int = 128
    flowmlp_dense_film_adapter_only: bool = False
    offline_depth_diagnostic_batches: int = 0
    offline_prompt_swap_diagnostic_batches: int = 0
    dit_num_blocks: int = 12
    dit_num_inference_steps: int = 5
    dit_num_inference_samples: int = 8
    dit_supervised_anchor_weight: float = 0.0
    dit_anchor_blend: float = 0.0
    dit_inference_residual_scale: float = 1.0
    dit_anchor_gripper_weight: float = 1.0
    dit_anchor_gripper_bce_weight: float = 0.0
    dit_flow_xyz_loss_weight: float = 1.0
    dit_flow_rot_loss_weight: float = 1.0
    dit_flow_gripper_loss_weight: float = 1.0
    dit_flow_gripper_bce_weight: float = 0.0
    dit_flow_gripper_bce_logit_scale: float = 1.0
    dit_flow_gripper_bce_balanced: bool = False
    dit_gripper_head_weight: float = 0.0
    dit_gripper_head_override: bool = False
    dit_clip_normalized_actions: bool = False
    dit_anchor_init_checkpoint: str = ""
    dit_freeze_anchor_head: bool = False
    dit_sample_t_mode_flow: str = "beta"
    dit_sample_t_mode_consistency: str = "discrete"
    dit_sample_dt_mode_consistency: str = "uniform"
    dit_sample_target_t_mode: str = "relative"
    dit_detach_flow_conditioning: bool = False
    dit_use_state_conditioning: bool = False
    dit_state_scale_mode: str = "none"
    dit_state_proprio_mode: str = "concat"
    dit_state_use_chunk_pos: bool = False
    dit_state_include_task_tokens: bool = False
    dit_condition_mode: str = "full"
    dit_condition_injection_mode: str = "cross_attn"  # cross_attn | joint_prefix
    dit_include_prompt_tokens: bool = False
    dit_task_token_mode: str = "vision_prompt"
    dit_use_latent_skill_token: bool = False
    dit_num_skill_tokens: int = 16
    dit_skill_token_dim: int = 128
    dit_skill_temperature: float = 1.0
    debug_dit_group_action_tokens_to_chunk: bool = False
    dit_zero_init_adaln: bool = True
    dit_zero_init_output: bool = True
    # State-conditioned dense depth residuals.  The final VLM layer remains an
    # identity baseline; only these new modules are trained in phase two.
    dit_dense_film_enabled: bool = False
    dit_dense_film_max_layers: int = 64
    dit_dense_film_first_layer_index: int = 1
    dit_dense_film_bottleneck_dim: int = 64
    dit_dense_film_state_dim: int = 128
    dit_dense_film_adapter_only: bool = False
    # Shared depth interface placed before the action head. "none" preserves legacy behavior.
    use_depth_interface: bool = False
    depth_interface_mode: str = "none"  # none | fixed | best_fixed | final | uniform | static_learned | adaptive | skill_adaptive
    depth_interface_max_layers: int = 64
    depth_interface_add_proprio: bool = True
    # Adaptive bridge control for representation-to-action alignment experiments
    use_adaptive_bridge: bool = True  # False = fixed layer, True = adaptive aggregation
    bridge_mode: str = "adaptive"  # "fixed" | "uniform" | "static_learned" | "adaptive" | "adaptive_gated"
    fixed_layer_index: int = -1  # >= 0 = use this specific layer index; -1 = auto (middle for DiT, last for MLP)

    # Logging
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10

    # revision version
    use_pro_version: bool = True
    phase: str = "Training"
    # fmt: on


# -------------------------
# Utils
# -------------------------
def dist_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def remove_ddp_in_checkpoint(state_dict) -> dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    if cfg.run_id_override is not None:
        run_id = cfg.run_id_override
    elif cfg.resume:
        run_id = cfg.config_file_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.config_file_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
        )
        if cfg.use_fz:
            run_id += f"+frozen+dropout-{cfg.lora_dropout}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    # DDP works even with world_size=1 if process group is initialized.
    return DDP(
        module,
        device_ids=[device_id],
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=True,
    )


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.
    
    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames
    
    Returns:
        str: Path to the matching checkpoint file, or None if not found
    """
    import os
    if not os.path.isdir(pretrained_checkpoint):
        return None
    
    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)
    
    if len(checkpoint_files) == 0:
        return None
    elif len(checkpoint_files) == 1:
        return checkpoint_files[0]
    else:
        # If multiple files, return the one with the highest step number
        import re
        def get_step(filename):
            match = re.search(r'(\d+)_checkpoint', filename)
            return int(match.group(1)) if match else 0
        return max(checkpoint_files, key=get_step)


def load_component_state_dict(checkpoint_path: str) -> dict:
    """
    Load state dict from checkpoint file, handling DDP wrapper if present.
    """
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    # Remove DDP wrapper if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict


def extract_action_queries_state_dict(vla_module: nn.Module) -> dict:
    """Persist action_queries explicitly because PEFT LoRA checkpoints do not save them."""
    state_dict = vla_module.state_dict()
    for key in (
        "base_model.model.action_queries.weight",
        "module.base_model.model.action_queries.weight",
        "action_queries.weight",
        "module.action_queries.weight",
    ):
        if key in state_dict:
            return {"weight": state_dict[key].detach().cpu()}
    raise KeyError("Could not find action_queries.weight in model state_dict")


def summarize_action_queries_tensor(weight: torch.Tensor) -> str:
    weight = weight.detach().float().cpu()
    return (
        f"shape={tuple(weight.shape)} "
        f"abs_mean={float(weight.abs().mean()):.6f} "
        f"norm={float(weight.norm()):.6f} "
        f"min={float(weight.min()):.6f} "
        f"max={float(weight.max()):.6f}"
    )


def load_action_queries_into_model(model: nn.Module, state_dict: dict, context: str = "") -> None:
    """Restore action_queries on either a base VLA model or a PEFT-wrapped model."""
    target = model
    if hasattr(target, "module"):
        target = target.module
    if hasattr(target, "base_model") and hasattr(target.base_model, "model"):
        target = target.base_model.model
    weight = state_dict["weight"].to(device=target.action_queries.weight.device, dtype=target.action_queries.weight.dtype)
    target.action_queries.weight.data.copy_(weight)
    prefix = f"{context} " if context else ""
    print(f"{prefix}loaded action_queries {summarize_action_queries_tensor(weight)}")


def load_action_queries_from_model_safetensors(checkpoint_dir: str) -> Optional[dict]:
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


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    # Try to load from checkpoint: first check resume, then check config_file_path
    loaded = False
    if cfg.resume:
        assert cfg.resume_step is not None, "resume=True requires resume_step"
        skip_for_explicit_init = (
            module_name == "action_head"
            and not bool(getattr(cfg, "load_action_head_from_checkpoint", True))
        )
        skip_for_dit_anchor = (
            module_name == "action_head"
            and cfg.action_head_type.upper() == "DIT"
            and bool(getattr(cfg, "dit_anchor_init_checkpoint", ""))
        )
        if skip_for_explicit_init:
            print(
                "[checkpoint] action_head load explicitly disabled; "
                f"initializing {cfg.action_head_type} action_head from scratch"
            )
        elif skip_for_dit_anchor:
            print(
                "[resume] skipping action_head resume load because DIT anchor "
                "will be initialized from dit_anchor_init_checkpoint"
            )
        else:
            state_dict = load_checkpoint(module_name, cfg.resum_vla_path, cfg.resume_step)
            looks_like_mlp_action_head = (
                module_name == "action_head"
                and cfg.action_head_type.upper() in {"DIT", "FLOWMLP"}
                and any(str(k).startswith("model.fc1.") for k in state_dict.keys())
            )
            if looks_like_mlp_action_head:
                raise RuntimeError(
                    "Refusing to load an MLP action-head checkpoint into "
                    f"{cfg.action_head_type}. For checkpoint initialization across "
                    "head families, pass --train_from checkpoint_init "
                    "--load_action_head_from_checkpoint False explicitly."
                )
            try:
                module.load_state_dict(state_dict)
                print(f"[resume] loaded {module_name} from step={cfg.resume_step}")
                loaded = True
            except RuntimeError as strict_error:
                allow_flowmlp_extension = (
                    module_name == "action_head"
                    and cfg.action_head_type.upper() == "FLOWMLP"
                )
                allow_dit_dense_film_extension = (
                    module_name == "action_head"
                    and cfg.action_head_type.upper() == "DIT"
                    and bool(getattr(cfg, "dit_dense_film_enabled", False))
                )
                if allow_flowmlp_extension or allow_dit_dense_film_extension:
                    incompatible = module.load_state_dict(state_dict, strict=False)
                    allowed_missing_prefixes = (
                        "skill_selector.",
                        "skill_embedding.",
                        "skill_layer_scorer.",
                        "skill_condition_proj.",
                        "continuous_context_selector.",
                        "continuous_context_projection.",
                        "continuous_context_layer_scorer.",
                        "continuous_context_condition_proj.",
                    )
                    if allow_dit_dense_film_extension:
                        allowed_missing_prefixes = allowed_missing_prefixes + (
                            "velocity_network.dense_depth_film.",
                        )
                    if bool(getattr(cfg, "flowmlp_dense_film_enabled", False)):
                        allowed_missing_prefixes = allowed_missing_prefixes + (
                            "dense_depth_film.",
                        )
                    disallowed_missing = [
                        key
                        for key in incompatible.missing_keys
                        if not key.startswith(allowed_missing_prefixes)
                    ]
                    if disallowed_missing or incompatible.unexpected_keys:
                        raise RuntimeError(
                            "Incompatible action-head checkpoint. Only explicitly "
                            "added dense-FiLM/skill/continuous-context modules may be absent. "
                            f"disallowed_missing={disallowed_missing} "
                            f"unexpected={list(incompatible.unexpected_keys)}"
                        ) from strict_error
                    print(
                        f"[resume] loaded {module_name}; newly introduced optional "
                        f"modules remain initialized: missing={list(incompatible.missing_keys)}"
                    )
                    loaded = True
                else:
                    raise
    else:
        print(
            f"[checkpoint_protocol] train_from=vlm_base: hidden {module_name} "
            "checkpoint discovery is disabled"
        )

    if module_name == "action_head":
        cfg._action_head_checkpoint_loaded = bool(loaded)

    if not loaded:
        print(f'[init_module] Initializing {module_name} from scratch')

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)

    return wrap_ddp(module, device_id, find_unused_params)


def compute_smoothened_metrics(metrics_deques) -> dict:
    smoothened_metrics = {}
    for name, dq in metrics_deques.items():
        if dq and len(dq) > 0:
            smoothened_metrics[name] = sum(dq) / len(dq)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    # WANDB_MODE=disabled intentionally skips wandb.init().  The imported
    # module still exposes ``log`` through a pre-init proxy, which raises if
    # called, so all shared train/validation logging must be a no-op here.
    if getattr(wandb_entity, "run", None) is None:
        return

    allowed_metric_names = {
        "loss_value",
        "curr_action_l1_loss",
        "next_actions_l1_loss",
        "loss_flow",
        "loss_ct",
        "flow_matching_loss",
        "flow_matching_total_loss",
        "dit_anchor_l1_loss",
        "flowmlp_anchor_l1_loss",
        "v_flow_pred_magnitude",
        "v_ct_pred_magnitude",
        "skill_entropy",
        "skill_entropy_loss",
        "skill_max_prob",
        "skill_expected_id",
        "depth_skill_entropy",
        "depth_skill_entropy_loss",
        "depth_skill_max_prob",
        "depth_skill_expected_id",
        "depth_layer_weight_max",
        "depth_selected_layer",
        "token_acc_curr",
        "token_acc_next",
    }

    log_dict = {}
    for name, value in metrics.items():
        if name not in allowed_metric_names:
            continue

        if name == "loss_value":
            key = f"{prefix}/Loss"
        elif name == "loss_flow":
            key = f"{prefix}/Loss Flow"
        elif name == "loss_ct":
            key = f"{prefix}/Loss Ct"
        elif name == "curr_action_l1_loss":
            key = f"{prefix}/Curr Action L1 Loss"
        elif name == "next_actions_l1_loss":
            key = f"{prefix}/Next Actions L1 Loss"
        elif name == "token_acc_curr":
            key = f"{prefix}/Curr Action Token Acc"
        elif name == "token_acc_next":
            key = f"{prefix}/Next Actions Token Acc"
        else:
            key = f"{prefix}/{name.replace('_', ' ').title()}"

        log_dict[key] = float(value)
    if log_dict:
        wandb_entity.log(log_dict, step=step)


def _gather_masked_tokens_per_batch(x: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    Robustly gather exactly k tokens per batch from x using mask.

    x:   (B, T, D)
    mask:(B, T) bool
    out: (B, k, D)
    Behavior:
      - if >k tokens: take first k
      - if <k tokens: pad by repeating last selected (or 0 if none)
    """
    assert x.dim() == 3
    assert mask.dim() == 2
    B, T, D = x.shape
    out = x.new_zeros((B, k, D))
    for b in range(B):
        idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        if idx.numel() >= k:
            idx = idx[:k]
        else:
            if idx.numel() == 0:
                idx = torch.zeros((1,), device=x.device, dtype=torch.long)
            pad = idx[-1].repeat(k - idx.numel())
            idx = torch.cat([idx, pad], dim=0)
        out[b] = x[b, idx]
    return out


def _to_float(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().mean().item())
    return float(v)


def run_forward_pass(
    vla,
    action_head,
    depth_interface,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_flow_matching,
    use_proprio,
    use_film,
    num_patches,  # <-- interface: number of vision patch tokens (per sample, after accounting num_images)
    compute_diffusion_l1=False,
    use_pro_version=True,
    cfg=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Correct alignment for Prismatic/OpenVLA:
      - multimodal sequence is built as: [BOS] + [PATCHES] + [rest of text tokens]
        (see _build_multimodal_attention in modeling_prismatic.py)
      - patches start at position 1, not 0
      - action token positions in original text sequence shift by +num_patches in multimodal seq (for positions >= 1)
      - proprio is NOT inserted into VLM token sequence in current forward(); it is only passed to action_head
    """
    metrics: Dict[str, float] = {}

    # --- Move tensors to device (CRITICAL: labels/masks must be on same device as embeddings) ---
    input_ids = batch["input_ids"].to(device_id)
    attention_mask = batch["attention_mask"].to(device_id)
    pixel_values = batch["pixel_values"].to(device_id, dtype=torch.bfloat16)

    labels = batch["labels"].to(device_id)  # must be on GPU for mask indexing inside forward()
    ground_truth_actions = batch["actions"].to(device_id, dtype=torch.float32)  # keep float32 for stable losses

    proprio = None
    if use_proprio:
        proprio = batch["proprio"].to(device_id, dtype=torch.bfloat16)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)

    # --- Forward VLA ---
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
            output_projector_features=True,  # so we can sanity-check patch token count
            proprio=proprio if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
        )

    # --- Resolve actual number of patch tokens V ---
    V = int(num_patches)
    if hasattr(output, "projector_features") and output.projector_features is not None:
        V = int(output.projector_features.shape[1])

    # --- Build action masks on the *teacher-forcing* label stream (consistent with your original finetune) ---
    gt_token_ids = labels[:, 1:]  # (B, T-1)
    current_action_mask = get_current_action_mask(gt_token_ids)  # (B, T-1)
    next_actions_mask = get_next_actions_mask(gt_token_ids)      # (B, T-1)
    all_actions_mask = current_action_mask | next_actions_mask   # (B, T-1)

    # -------------------------
    # (A) DISCRETE TOKEN MODE
    # -------------------------
    if not (use_l1_regression or use_flow_matching):
        # modeling_prismatic.py already passes logits (per user). Keep a guard anyway.
        if (not hasattr(output, "logits")) or (output.logits is None):
            raise RuntimeError("output.logits is None; cannot compute token metrics in discrete mode.")

        logits = output.logits  # (B, S_mm, vocab)

        # Align multimodal logits back to original text token positions:
        aligned_logits = torch.cat([logits[:, :1, :], logits[:, 1 + V :, :]], dim=1)  # (B, T, vocab)

        predicted_token_ids = aligned_logits[:, :-1, :].argmax(dim=-1)  # (B, T-1)

        curr_action_accuracy = compute_token_accuracy(
            predicted_token_ids, gt_token_ids, mask=current_action_mask
        )
        curr_action_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, gt_token_ids, mask=current_action_mask
        )
        next_actions_accuracy = compute_token_accuracy(
            predicted_token_ids, gt_token_ids, mask=next_actions_mask
        )
        next_actions_l1_loss = compute_actions_l1_loss(
            action_tokenizer, predicted_token_ids, gt_token_ids, mask=next_actions_mask
        )

        loss = output.loss if output.loss is not None else torch.tensor(0.0, device=device_id)

        metrics.update(
            {
                "loss_value": float(loss.detach().item()) if hasattr(loss, "detach") else float(loss),
                "curr_action_accuracy": float(curr_action_accuracy.item()),
                "curr_action_l1_loss": float(curr_action_l1_loss.item()),
                "next_actions_accuracy": float(next_actions_accuracy.item()),
                "next_actions_l1_loss": float(next_actions_l1_loss.item()),
            }
        )
        return loss, metrics

    # -------------------------
    # (B) CONTINUOUS ACTION MODE (L1 or FLOW)
    # -------------------------
    with torch.no_grad():
        action_indices_list = [torch.where(all_actions_mask[b])[0] for b in range(all_actions_mask.shape[0])]
        expected_action_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK
        raw_K = action_indices_list[0].numel()
        if raw_K < expected_action_tokens:
            raise RuntimeError(
                f"Expected at least {expected_action_tokens} action tokens per sample, got {raw_K}. "
                "Check action chunk tokenization and action-mask construction."
            )
        # Some LIBERO no-noop records include an extra trailing action-token
        # segment in the teacher-forcing mask. Continuous heads supervise the
        # fixed 8x7 action chunk, so discard only those trailing tokens.
        K = expected_action_tokens
        action_indices_list = [idx[:K] for idx in action_indices_list]
        for t in action_indices_list[1:]:
            if t.numel() != K:
                raise RuntimeError(
                    f"Non-constant number of action tokens per sample: {K} vs {t.numel()}. "
                    "Your mask logic or batching is inconsistent."
                )
        action_idx = torch.stack(action_indices_list, dim=0)  # (B, K), indices in [0..T-2]

    orig_pos = action_idx + 1  # (B, K), in [1..T-1]
    mm_pos = orig_pos + V      # (B, K)

    prompt_mm_pos = None
    prompt_valid = None
    prompt_lengths = None
    action_head_type = str(getattr(cfg, "action_head_type", "")).upper() if cfg is not None else ""
    if action_head_type == "FLOWMLP":
        task_token_mode = str(getattr(cfg, "flowmlp_task_token_mode", "vision_prompt"))
        include_prompt_tokens = bool(getattr(cfg, "flowmlp_include_prompt_tokens", False))
    else:
        task_token_mode = str(getattr(cfg, "dit_task_token_mode", "vision_prompt"))
        include_prompt_tokens = action_head_type == "DIT" and bool(
            getattr(cfg, "dit_include_prompt_tokens", False)
        )
    if task_token_mode not in {"vision_prompt", "vision_only", "prompt_only", "last_prompt"}:
        raise ValueError(f"Unsupported task_token_mode for {action_head_type}: {task_token_mode}")
    needs_prompt_tokens = include_prompt_tokens or task_token_mode in {"prompt_only", "last_prompt"}
    if needs_prompt_tokens:
        prompt_candidate_mask = (
            (~all_actions_mask)
            & (gt_token_ids != STOP_INDEX)
            & attention_mask[:, 1:].bool()
        )
        prompt_indices_list = [torch.where(prompt_candidate_mask[b])[0] for b in range(prompt_candidate_mask.shape[0])]
        prompt_lengths = torch.tensor(
            [int(token_indices.numel()) for token_indices in prompt_indices_list],
            device=gt_token_ids.device,
            dtype=torch.long,
        )
        P = max(int(token_indices.numel()) for token_indices in prompt_indices_list)
        if P <= 0:
            raise RuntimeError("No valid prompt tokens found for DIT prompt conditioning.")
        prompt_idx_rows = []
        prompt_valid_rows = []
        for token_indices in prompt_indices_list:
            num_valid = int(token_indices.numel())
            if num_valid < P:
                pad = token_indices.new_zeros(P - num_valid)
                prompt_idx_rows.append(torch.cat([token_indices, pad], dim=0))
                prompt_valid_rows.append(
                    torch.cat(
                        [
                            torch.ones(num_valid, device=token_indices.device, dtype=torch.bool),
                            torch.zeros(P - num_valid, device=token_indices.device, dtype=torch.bool),
                        ],
                        dim=0,
                    )
                )
            else:
                prompt_idx_rows.append(token_indices)
                prompt_valid_rows.append(torch.ones(P, device=token_indices.device, dtype=torch.bool))
        prompt_idx = torch.stack(prompt_idx_rows, dim=0)
        prompt_orig_pos = prompt_idx + 1
        prompt_mm_pos = prompt_orig_pos + V
        prompt_valid = torch.stack(prompt_valid_rows, dim=0)

    # Build multi-layer features expected by your ActionHead: (B, L, V+K, D)
    multi_layer_hidden_states = []
    task_token_count = V
    for layer_h in output.hidden_states:
        B, S_mm, D = layer_h.shape

        vision_latents = layer_h[:, 1 : 1 + V, :]  # (B, V, D)
        task_latents = vision_latents
        if needs_prompt_tokens:
            prompt_gather_index = prompt_mm_pos.unsqueeze(-1).expand(B, prompt_mm_pos.shape[1], D)
            prompt_latents = layer_h.gather(dim=1, index=prompt_gather_index)
            prompt_latents = prompt_latents * prompt_valid.unsqueeze(-1).to(dtype=prompt_latents.dtype)
        if task_token_mode == "vision_only":
            task_latents = vision_latents
        elif task_token_mode == "prompt_only":
            task_latents = prompt_latents
        elif task_token_mode == "last_prompt":
            last_prompt_idx = (prompt_lengths - 1).view(B, 1, 1).expand(B, 1, D)
            task_latents = prompt_latents.gather(dim=1, index=last_prompt_idx)
        elif include_prompt_tokens:
            task_latents = torch.cat([task_latents, prompt_latents], dim=1)
        task_token_count = int(task_latents.shape[1])

        gather_index = mm_pos.unsqueeze(-1).expand(B, K, D)  # (B, K, D)
        action_latents = layer_h.gather(dim=1, index=gather_index)  # (B, K, D)

        packed = torch.cat([task_latents.unsqueeze(1), action_latents.unsqueeze(1)], dim=2).to(torch.bfloat16)
        multi_layer_hidden_states.append(packed)

    multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)  # (B, L, V+K, D)
    metrics["task_token_count"] = float(task_token_count)
    metrics["prompt_token_count"] = float(task_token_count - V if include_prompt_tokens else 0)
    if hasattr(action_head.module, "task_token_mode"):
        action_head.module.task_token_mode = task_token_mode
    if (needs_prompt_tokens or task_token_mode != "vision_prompt") and hasattr(action_head.module, "set_num_task_tokens"):
        action_head.module.set_num_task_tokens(task_token_count, V)

    if use_flow_matching:
        if depth_interface is not None:
            depth_out = depth_interface.module(
                multi_layer_hidden_states,
                proprio=proprio if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
            )
            if not hasattr(action_head.module, "flow_matching_loss_from_state"):
                raise RuntimeError(
                    f"{type(action_head.module).__name__} does not support flow_matching_loss_from_state"
                )
            loss, loss_dict = action_head.module.flow_matching_loss_from_state(
                state_emb=depth_out.state_emb,
                target_actions=ground_truth_actions,
                proprio=proprio if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                aux_loss=depth_out.aux_loss,
                aux_metrics=depth_out.metrics,
            )
        else:
            loss, loss_dict = action_head.module.flow_matching_loss(
                actions_hidden_states=multi_layer_hidden_states,
                target_actions=ground_truth_actions,
                proprio=proprio if use_proprio else None,
                proprio_projector=proprio_projector if use_proprio else None,
                mode="flow_matching",
                ema_model=None,
            )
        metrics.update({k: _to_float(v) for k, v in loss_dict.items()})
        return loss, metrics

    if depth_interface is not None and hasattr(action_head.module, "predict_action_from_state"):
        depth_out = depth_interface.module(
            multi_layer_hidden_states,
            proprio=proprio if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
        )
        predicted_actions = action_head.module.predict_action_from_state(depth_out.state_emb).float()
        metrics.update({k: _to_float(v) for k, v in depth_out.metrics.items()})
    else:
        predicted_actions = action_head.module.predict_action(
            multi_layer_hidden_states,
            proprio=proprio if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            phase=cfg.phase if cfg is not None and hasattr(cfg, "phase") else "Training",
        ).float()

    loss = torch.nn.L1Loss()(predicted_actions, ground_truth_actions)

    metrics["loss_value"] = float(loss.detach().item())

    ground_truth_curr_action = ground_truth_actions[:, 0]
    predicted_curr_action = predicted_actions[:, 0]
    ground_truth_next_actions = ground_truth_actions[:, 1:]
    predicted_next_actions = predicted_actions[:, 1:]
    metrics["curr_action_l1_loss"] = float(torch.nn.L1Loss()(ground_truth_curr_action, predicted_curr_action).item())
    metrics["next_actions_l1_loss"] = float(torch.nn.L1Loss()(ground_truth_next_actions, predicted_next_actions).item())

    return loss, metrics


# -------------------------
# Checkpointing (mostly original)
# -------------------------
def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    depth_interface,
    train_dataset,
    distributed_state,
    new_state_dict,
    optimizer,
    scheduler,
) -> None:
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"

    adapter_dir = checkpoint_dir / "lora_adapter"

    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if cfg.use_lora:
            os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")

    dist_barrier()

    if distributed_state.is_main_process:
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        processor.save_pretrained(checkpoint_dir)

        if cfg.use_fz:
            vla.module.save_pretrained(checkpoint_dir)
        elif cfg.use_lora:
            vla.module.save_pretrained(adapter_dir)
            import shutil
            root_config = checkpoint_dir / "config.json"
            if not root_config.exists():
                adapter_config = adapter_dir / "config.json"
                if adapter_config.exists():
                    shutil.copy2(adapter_config, root_config)
                    print(f"Copied config.json from {adapter_dir} to {checkpoint_dir} for evaluation compatibility")
                else:
                    # PEFT does not save config.json in adapter_dir; copy from training config path
                    src_config = Path(cfg.config_file_path) / "config.json"
                    if src_config.exists():
                        shutil.copy2(src_config, root_config)
                        print(f"Copied config.json from {src_config} to {checkpoint_dir} for evaluation compatibility")
                    else:
                        raise FileNotFoundError(f"Need config.json for eval: not in {adapter_dir} and not at {src_config}")
        else:
            vla.module.save_pretrained(checkpoint_dir, max_shard_size="7GB")

        torch.cuda.empty_cache()

        if cfg.use_proprio and proprio_projector is not None:
            torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}")

        if cfg.use_diffusion and noisy_action_projector is not None:
            torch.save(
                noisy_action_projector.state_dict(),
                checkpoint_dir / f"noisy_action_projector--{checkpoint_name_suffix}",
            )

        if (cfg.use_l1_regression or cfg.use_flow_matching) and action_head is not None:
            torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{checkpoint_name_suffix}")

        if depth_interface is not None:
            torch.save(depth_interface.state_dict(), checkpoint_dir / f"depth_interface--{checkpoint_name_suffix}")

        torch.save(
            extract_action_queries_state_dict(vla.module),
            checkpoint_dir / f"action_queries--{checkpoint_name_suffix}",
        )
        print(
            "[save] saved action_queries "
            f"{summarize_action_queries_tensor(extract_action_queries_state_dict(vla.module)['weight'])}"
        )

        if cfg.use_film:
            torch.save(
                vla.module.vision_backbone.state_dict(),
                checkpoint_dir / f"vision_backbone--{checkpoint_name_suffix}",
            )

        torch.save(
            {
                "global_step": int(log_step),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            checkpoint_dir / f"training_state--{checkpoint_name_suffix}",
        )
        training_start_step = int(getattr(cfg, "_training_start_step", 0))
        provenance = {
            "train_from": str(cfg.train_from),
            "source_checkpoint": (
                str(cfg.resum_vla_path) if bool(getattr(cfg, "resume", False)) else None
            ),
            "source_checkpoint_step": (
                int(cfg.resume_step) if cfg.resume_step is not None else None
            ),
            "training_start_step": training_start_step,
            "checkpoint_global_step": int(log_step),
            "optimizer_updates_this_run": int(log_step) - training_start_step,
            "planned_optimizer_updates": int(
                getattr(cfg, "_planned_optimizer_updates", cfg.max_steps - training_start_step)
            ),
            "action_head_type": str(cfg.action_head_type),
            "action_head_checkpoint_load_requested": bool(
                getattr(cfg, "load_action_head_from_checkpoint", True)
            ),
            "action_head_checkpoint_loaded": bool(
                getattr(cfg, "_action_head_checkpoint_loaded", False)
            ),
            "optimizer_state_loaded": bool(
                getattr(cfg, "_optimizer_state_loaded", False)
            ),
        }
        (checkpoint_dir / "training_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    dist_barrier()

    # Optional: merge LoRA and save merged model
    if cfg.use_lora and cfg.merge_lora_during_training:
        if distributed_state.is_main_process:
            print(f"Merging LoRA weights and saving merged model for Step {log_step}...")

        if cfg.use_minivlm:
            config = AutoConfig.from_pretrained(cfg.config_file_path)
            base_vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
            from prismatic.models import load
            vlm = load("pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b", hf_token=None, load_for_training=False)

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

            converted_state_dict = rename_state_dict_keys(vlm.state_dict(), replace_map)
            base_vla.load_state_dict(converted_state_dict, strict=False)

            if "module.base_model.model.action_queries.weight" in vla.state_dict():
                base_vla.action_queries.weight.data.copy_(
                    vla.state_dict()["module.base_model.model.action_queries.weight"].cpu()
                )
        else:
            base_vla = AutoModelForVision2Seq.from_pretrained(
                cfg.config_file_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
                trust_remote_code=False,
            )

        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()

        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")

        dist_barrier()


# -------------------------
# Validation (PATCHED)
# -------------------------
def run_validation(
    vla,
    action_head,
    depth_interface,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,  # <-- interface fixed
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    val_start_time = time.time()

    vla.eval()
    if action_head is not None:
        action_head.eval()
    if depth_interface is not None:
        depth_interface.eval()
    if proprio_projector is not None:
        proprio_projector.eval()

    val_batches_count = 0
    all_val_metrics = []

    with torch.no_grad():
        for batch in val_dataloader:
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                depth_interface=depth_interface,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_flow_matching=cfg.use_flow_matching,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,  # <-- pass consistently
                compute_diffusion_l1=True,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
            )
            all_val_metrics.append(metrics)
            val_batches_count += 1
            if time.time() - val_start_time > val_time_limit:
                break

    if len(all_val_metrics) == 0:
        return

    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [m[metric_name] for m in all_val_metrics if metric_name in m]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)

    avg_val_metrics["val_batches_count"] = val_batches_count

    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)

    vla.train()
    if action_head is not None:
        action_head.train()
    if depth_interface is not None:
        depth_interface.train()
    if proprio_projector is not None:
        proprio_projector.train()


# -------------------------
# Evaluation helper (unchanged)
# -------------------------
def run_libero_evaluation_on_checkpoint(
    cfg: FinetuneConfig,
    checkpoint_dir: Path,
    log_step: int,
    dataset_name: str,
) -> None:
    dataset_to_suite = {
        "libero_10_no_noops": "libero_10",
        "libero_spatial_no_noops": "libero_spatial",
        "libero_object_no_noops": "libero_object",
        "libero_goal_no_noops": "libero_goal",
    }

    task_suite_name = dataset_to_suite.get(dataset_name, "libero_10")

    eval_script = "experiments/robot/libero/run_libero_eval.py"
    checkpoint_path = str(checkpoint_dir.absolute())

    cmd = [
        "python", eval_script,
        "--pretrained_checkpoint", checkpoint_path,
        "--task_suite_name", task_suite_name,
        "--num_trials_per_task", str(cfg.eval_num_trials_per_task),
        "--use_proprio", str(cfg.use_proprio),
        "--num_images_in_input", str(cfg.num_images_in_input),
        "--use_film", str(cfg.use_film),
        "--use_minivlm", str(cfg.use_minivlm),
        "--use_pro_version", str(cfg.use_pro_version),
        "--action_head_type", str(cfg.action_head_type),
        "--dit_num_blocks", str(cfg.dit_num_blocks),
        "--dit_num_inference_steps", str(cfg.dit_num_inference_steps),
        "--dit_num_inference_samples", str(cfg.dit_num_inference_samples),
        "--dit_supervised_anchor_weight", str(cfg.dit_supervised_anchor_weight),
        "--dit_anchor_blend", str(cfg.dit_anchor_blend),
        "--dit_anchor_blend_was_set", "True",
        "--dit_anchor_gripper_weight", str(cfg.dit_anchor_gripper_weight),
        "--dit_anchor_gripper_bce_weight", str(cfg.dit_anchor_gripper_bce_weight),
        "--dit_flow_xyz_loss_weight", str(cfg.dit_flow_xyz_loss_weight),
        "--dit_flow_rot_loss_weight", str(cfg.dit_flow_rot_loss_weight),
        "--dit_flow_gripper_loss_weight", str(cfg.dit_flow_gripper_loss_weight),
        "--dit_flow_gripper_bce_weight", str(cfg.dit_flow_gripper_bce_weight),
        "--dit_flow_gripper_bce_logit_scale", str(cfg.dit_flow_gripper_bce_logit_scale),
        "--dit_flow_gripper_bce_balanced", str(cfg.dit_flow_gripper_bce_balanced),
        "--dit_gripper_head_weight", str(cfg.dit_gripper_head_weight),
        "--dit_gripper_head_override", str(cfg.dit_gripper_head_override),
        "--dit_clip_normalized_actions", str(cfg.dit_clip_normalized_actions),
        "--dit_sample_t_mode_flow", str(cfg.dit_sample_t_mode_flow),
        "--dit_sample_t_mode_consistency", str(cfg.dit_sample_t_mode_consistency),
        "--dit_sample_dt_mode_consistency", str(cfg.dit_sample_dt_mode_consistency),
        "--dit_sample_target_t_mode", str(cfg.dit_sample_target_t_mode),
        "--dit_detach_flow_conditioning", str(cfg.dit_detach_flow_conditioning),
        "--dit_use_state_conditioning", str(cfg.dit_use_state_conditioning),
        "--dit_state_scale_mode", str(cfg.dit_state_scale_mode),
        "--dit_state_proprio_mode", str(cfg.dit_state_proprio_mode),
        "--dit_state_use_chunk_pos", str(cfg.dit_state_use_chunk_pos),
        "--dit_state_include_task_tokens", str(cfg.dit_state_include_task_tokens),
        "--dit_condition_mode", str(cfg.dit_condition_mode),
        "--dit_condition_injection_mode", str(cfg.dit_condition_injection_mode),
        "--dit_include_prompt_tokens", str(cfg.dit_include_prompt_tokens),
        "--dit_task_token_mode", str(cfg.dit_task_token_mode),
        "--debug_dit_group_action_tokens_to_chunk", str(cfg.debug_dit_group_action_tokens_to_chunk),
        "--dit_zero_init_adaln", str(cfg.dit_zero_init_adaln),
        "--dit_zero_init_output", str(cfg.dit_zero_init_output),
        "--dit_dense_film_enabled", str(cfg.dit_dense_film_enabled),
        "--dit_dense_film_max_layers", str(cfg.dit_dense_film_max_layers),
        "--dit_dense_film_first_layer_index", str(cfg.dit_dense_film_first_layer_index),
        "--dit_dense_film_bottleneck_dim", str(cfg.dit_dense_film_bottleneck_dim),
        "--dit_dense_film_state_dim", str(cfg.dit_dense_film_state_dim),
        "--use_depth_interface", str(cfg.use_depth_interface),
        "--depth_interface_mode", str(cfg.depth_interface_mode),
        "--depth_interface_max_layers", str(cfg.depth_interface_max_layers),
        "--depth_interface_add_proprio", str(cfg.depth_interface_add_proprio),
        "--use_adaptive_bridge", str(cfg.use_adaptive_bridge),
        "--bridge_mode", str(cfg.bridge_mode),
        "--fixed_layer_index", str(cfg.fixed_layer_index),
        "--flow_ratio", str(cfg.flow_ratio),
        "--use_wandb", "False",
        "--center_crop", str(cfg.image_aug),
    ]

    print(f"\n{'='*80}")
    print(f"[Step {log_step}] Running LIBERO evaluation on checkpoint: {checkpoint_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eval_script_path = os.path.join(project_root, eval_script)

        if not os.path.exists(eval_script_path):
            print(f"[Step {log_step}] 閴?ERROR: Evaluation script not found!")
            return

        env = os.environ.copy()
        env.pop("MASTER_ADDR", None)
        env.pop("MASTER_PORT", None)
        env.pop("RANK", None)
        env.pop("LOCAL_RANK", None)
        env.pop("WORLD_SIZE", None)
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=project_root,
            env=env,
        )

        print(f"[Step {log_step}] Evaluation return code: {result.returncode}")

        if result.stdout:
            print(f"[Step {log_step}] Evaluation stdout (last 2000 chars):\n{result.stdout[-2000:]}")
        if result.stderr:
            print(f"[Step {log_step}] Evaluation stderr (last 1000 chars):\n{result.stderr[-1000:]}")

        if result.returncode == 0:
            import re
            output_text = result.stdout
            success_rate = None

            patterns = [
                r"Overall success rate:\s*(\d+\.?\d*)\s*\(",
                r"Overall success rate:\s*(\d+\.?\d*)\s*%",
                r"success rate[:\s]+(\d+\.?\d*)\s*%",
                r"Overall success rate:\s*(\d+\.?\d*)",
            ]

            for pattern in patterns:
                match = re.search(pattern, output_text, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    success_rate = value / 100.0 if value > 1.0 else value
                    break

            if success_rate is not None:
                try:
                    if wandb.run is not None:
                        wandb.log(
                            {
                                "VLA Eval/Success Rate": success_rate,
                                "VLA Eval/Checkpoint Step": log_step,
                            },
                            step=log_step,
                        )
                        print(f"[Step {log_step}] 閴?Evaluation success rate logged to W&B: {success_rate:.2%}")
                    else:
                        print(f"[Step {log_step}] 閳?W&B not initialized, skipping log. Success rate: {success_rate:.2%}")
                except Exception as e:
                    print(f"[Step {log_step}] 閴?Failed to log to W&B: {e}")
                    print(f"[Step {log_step}] Evaluation success rate: {success_rate:.2%}")
            else:
                print(f"[Step {log_step}] 閴?Evaluation completed but couldn't parse success rate")
        else:
            print(f"[Step {log_step}] 閴?Evaluation failed with return code {result.returncode}")
            print(result.stderr)

            try:
                if wandb.run is not None:
                    wandb.log(
                        {
                            "VLA Eval/Evaluation Failed": 1,
                            "VLA Eval/Failure Step": log_step,
                            "VLA Eval/Failure Return Code": result.returncode,
                        },
                        step=log_step,
                    )
            except Exception as e:
                print(f"[Step {log_step}] Failed to log evaluation failure to W&B: {e}")

    except subprocess.TimeoutExpired:
        print(f"[Step {log_step}] 閴?Evaluation timed out after 1 hour")
    except FileNotFoundError:
        print(f"[Step {log_step}] 閴?Evaluation script not found: {eval_script}")
    except Exception as e:
        print(f"[Step {log_step}] 閴?Evaluation error: {type(e).__name__}: {e}")


# -------------------------
# Main finetune (PATCHED)
# -------------------------
def _derive_action_head_flags(cfg: FinetuneConfig) -> None:
    """Set use_l1_regression, use_flow_matching, flow_matching_head_type from action_head_type."""
    t = getattr(cfg, "action_head_type", "MLP").upper()
    if t not in ("MLP", "DIT", "FLOWMLP"):
        raise ValueError(f"action_head_type must be one of MLP, DIT, FlowMLP; got {cfg.action_head_type}")
    cfg.use_l1_regression = (t == "MLP")
    cfg.use_flow_matching = (t in ("DIT", "FLOWMLP"))
    cfg.flow_matching_head_type = "ditx" if t == "DIT" else "mlp"
    if getattr(cfg, "use_depth_interface", False) and getattr(cfg, "depth_interface_mode", "none") == "none":
        cfg.depth_interface_mode = "skill_adaptive"


def _derive_train_from_flags(cfg: FinetuneConfig) -> None:
    """
    婢堆勬煙閸氭垳琚辩粔宥忕窗娴?checkpoint 缂侇叀顔?vs 娴?VLM 閸╁搫楠囩拋顓溾偓?
    娴?VLM 閸╁搫楠囬弮鍓佹暠 vlm_training 閸愬啿鐣鹃敍姝璷ra / full / freeze / freeze_lora閵?
    """
    train_from = getattr(cfg, "train_from", "vlm_base").lower()
    if train_from not in ("vlm_base", "checkpoint", "checkpoint_init"):
        raise ValueError(
            "train_from must be 'vlm_base', 'checkpoint', or "
            f"'checkpoint_init'; got {cfg.train_from}"
        )

    if train_from in ("checkpoint", "checkpoint_init"):
        cfg.resume = True
        if not getattr(cfg, "resum_vla_path", "").strip() or cfg.resume_step is None:
            raise ValueError(
                f"train_from={train_from!r} requires resum_vla_path and resume_step"
            )
        if train_from == "checkpoint_init":
            cfg.resume_load_training_state = False
        else:
            if not bool(cfg.resume_load_training_state):
                raise ValueError(
                    "train_from='checkpoint' requires resume_load_training_state=True; "
                    "use train_from='checkpoint_init' for a fresh optimizer timeline"
                )
            if not bool(cfg.load_action_head_from_checkpoint):
                raise ValueError(
                    "train_from='checkpoint' requires action-head restoration; use "
                    "train_from='checkpoint_init' for cross-head initialization"
                )
        # use_lora / use_fz / freeze_vlm 閻?checkpoint 閸愬懐濮搁幀浣稿枀鐎规熬绱濇稉宥呮躬鏉╂瑩鍣风憰鍡欐磰
        return

    cfg.resume = False
    vlm_training = getattr(cfg, "vlm_training", "freeze_lora").lower()
    if vlm_training not in ("lora", "full", "freeze", "freeze_lora"):
        raise ValueError(
            f"vlm_training must be one of lora, full, freeze, freeze_lora; got {cfg.vlm_training}"
        )
    if vlm_training == "lora":
        cfg.use_lora, cfg.use_fz, cfg.freeze_vlm = True, False, False
    elif vlm_training == "full":
        cfg.use_lora, cfg.use_fz, cfg.freeze_vlm = False, True, False
    elif vlm_training == "freeze":
        cfg.use_lora, cfg.use_fz, cfg.freeze_vlm = False, False, True
    else:  # freeze_lora
        cfg.use_lora, cfg.use_fz, cfg.freeze_vlm = True, False, True


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    global RAW_STATE_DICT

    _derive_action_head_flags(cfg)
    _derive_train_from_flags(cfg)
    step_plan = resolve_checkpoint_step_plan(
        train_from=cfg.train_from,
        resume_step=cfg.resume_step,
        max_steps=cfg.max_steps,
        resume_load_training_state=cfg.resume_load_training_state,
    )
    cfg._training_start_step = step_plan.training_start_step
    cfg._planned_optimizer_updates = step_plan.planned_optimizer_updates
    set_seed(cfg.seed)
    assert not (cfg.use_l1_regression and cfg.use_diffusion), "Cannot do both L1 regression and diffusion."

    cfg.config_file_path = cfg.config_file_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.config_file_path}` on `{cfg.dataset_name}`")

    # Show effective grad-accum to avoid 閳ユ竷hanged accelerate flag but no effect閳?confusion
    mode_str = f"train_from={cfg.train_from}"
    if not cfg.resume:
        mode_str += f", vlm_training={cfg.vlm_training} (use_lora={cfg.use_lora}, use_fz={cfg.use_fz}, freeze_vlm={cfg.freeze_vlm})"
    print(
        f"[config] batch_size(per-device)={cfg.batch_size}, grad_accumulation_steps={cfg.grad_accumulation_steps}, "
        f"action_head_type={cfg.action_head_type}, {mode_str}"
    )
    print(
        "[checkpoint_protocol] "
        f"mode={step_plan.mode} source_step={step_plan.source_step} "
        f"training_start_step={step_plan.training_start_step} "
        f"target_step={step_plan.target_step} "
        f"planned_optimizer_updates={step_plan.planned_optimizer_updates} "
        f"load_action_head={cfg.load_action_head_from_checkpoint} "
        f"load_training_state={step_plan.load_training_state}"
    )

    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    # GPU / distributed
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    print(f"[cuda-debug] visible={os.environ.get('CUDA_VISIBLE_DEVICES')} available={torch.cuda.is_available()} count={torch.cuda.device_count()}", flush=True)
    print("[cuda-debug] libs=" + " | ".join(x.strip() for x in open('/proc/self/maps', errors='ignore') if 'libcuda' in x), flush=True)
    torch.cuda.set_device(device_id)
    # No CUDA allocations exist yet.  Calling empty_cache() here is redundant
    # and can segfault when torchvision and torch have loaded different bundled
    # CUDA runtimes before the allocator has been initialized.

    wandb_mode = os.environ.get("WANDB_MODE", "disabled").lower()
    if distributed_state.is_main_process and wandb_mode != "disabled":
        try:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=f"ft+{run_id}",
                mode=os.environ.get("WANDB_MODE", "disabled"),
                sync_tensorboard=False,
            )
        except WandBCommError as e:
            # 403 缁涘绱癳ntity 娑撳秵妲告担鐘垫畱鐠愶箑褰块弮鑸垫￥濞夋洖鍟撻崗銉礉閺€閫涜礋缁傝崵鍤庣拋鏉跨秿閿涘矁顔勭紒鍐х瑝娑擃厽鏌?
            print(f"[WandB] online logging failed ({e}); switching to offline mode. Pass your own entity with --wandb_entity to enable online logging.")
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=f"ft+{run_id}",
                mode="offline",
                sync_tensorboard=False,
            )
    elif distributed_state.is_main_process:
        print("[WandB] disabled; skipping wandb.init()", flush=True)

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}\n"
        f"\tNUM_TOKENS (action tokens expected): {NUM_TOKENS}"
    )

    # Load base model / register
    if model_is_on_hf_hub(cfg.config_file_path):
        vla_download_path = snapshot_download(repo_id=cfg.config_file_path)
        cfg.config_file_path = vla_download_path
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        update_auto_map(cfg.config_file_path)
        check_model_logic_mismatch(cfg.config_file_path)
    dist_barrier()

    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    # When config_file_path is pretrained_models/configs (local dir with only config.json),
    # it may lack processing_prismatic.py; then load tokenizer + image_processor separately and build processor.
    processor_path = cfg.config_file_path
    if cfg.use_minivlm and not model_is_on_hf_hub(cfg.config_file_path):
        preproc_cfg = os.path.join(cfg.config_file_path, "preprocessor_config.json")
        if not os.path.exists(preproc_cfg):
            # Offline experiment nodes commonly keep the lightweight architecture
            # config separate from the complete local VLM snapshot.  Prefer that
            # snapshot's tokenizer/image processor instead of silently requiring
            # a Hugging Face download.
            local_vlm_processor = os.path.join(str(cfg.vlm_path), "preprocessor_config.json")
            if os.path.exists(local_vlm_processor):
                processor_path = str(cfg.vlm_path)
                print(
                    "[processor] using complete local VLM processor snapshot: "
                    f"{processor_path}"
                )
            else:
                processor_path = "VLA-Adapter/LIBERO-Spatial"
    try:
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    except OSError as e:
        if "processing_prismatic.py" in str(e) and os.path.isdir(processor_path):
            # 閻╊喖缍嶉柌灞剧梾閺?custom processor 閻?.py閿涘瞼鏁ら張顑跨波鎼存挾娈戠猾璁崇矤閸氬瞼娲拌ぐ鏇炲鏉?tokenizer + image_processor 閸愬秶绮嶇憗?
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(processor_path)
            image_processor = PrismaticImageProcessor.from_pretrained(processor_path)
            processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
        else:
            raise

    # Load VLA
    if cfg.use_minivlm:
        hf_token = ""
        if "prism-qwen25-extra-dinosiglip-224px-0_5b" in cfg.vlm_path or "huggingface/download" in cfg.vlm_path:
            vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
        else:
            vlm = load_vla(cfg.vlm_path, hf_token=hf_token, load_for_training=True)

        config = AutoConfig.from_pretrained(cfg.config_file_path)
        vla = AutoModelForVision2Seq.from_config(
            config, torch_dtype=torch.bfloat16
        ).to(device_id)

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

        RAW_STATE_DICT = rename_state_dict_keys(vlm.state_dict(), replace_map)
        _missing, _unexpected = vla.load_state_dict(RAW_STATE_DICT, strict=False)
        del vlm
    else:
        RAW_STATE_DICT = {}
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.config_file_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
        ).to(device_id)

    # Set number of images in input (keep same behavior)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # LoRA / trainable params
    resume_adapter_dir = os.path.join(cfg.resum_vla_path, "lora_adapter") if cfg.resume else ""
    if cfg.resume and os.path.isdir(resume_adapter_dir):
        vla = PeftModel.from_pretrained(vla, resume_adapter_dir, is_trainable=True)
        cfg.use_lora = True
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        print(f"[resume] loaded trainable LoRA adapter from {resume_adapter_dir}")
        action_queries_path = os.path.join(cfg.resum_vla_path, f"action_queries--{cfg.resume_step}_checkpoint.pt")
        if os.path.exists(action_queries_path):
            action_queries_state = load_checkpoint("action_queries", cfg.resum_vla_path, cfg.resume_step)
            load_action_queries_into_model(vla, action_queries_state, context="[resume]")
        else:
            action_queries_state = load_action_queries_from_model_safetensors(cfg.resum_vla_path)
            if action_queries_state is not None:
                load_action_queries_into_model(vla, action_queries_state, context="[resume:model.safetensors]")
            elif str(cfg.train_from).lower() == "checkpoint":
                raise FileNotFoundError(
                    "True checkpoint continuation requires trained action queries, "
                    f"but neither {action_queries_path} nor model.safetensors "
                    "contains them"
                )
            else:
                current_weight = extract_action_queries_state_dict(
                    vla.base_model.model if hasattr(vla, "base_model") else vla
                )["weight"]
                print(
                    f"[resume] warning: no explicit action_queries checkpoint found at {action_queries_path} "
                    f"and no action_queries.weight in model.safetensors. "
                    f"Current in-memory action_queries {summarize_action_queries_tensor(current_weight)}"
                )
        vla.print_trainable_parameters()
    elif str(cfg.train_from).lower() == "checkpoint" and bool(cfg.use_lora):
        raise FileNotFoundError(
            "True checkpoint continuation requires the saved LoRA adapter, "
            f"but {resume_adapter_dir} does not exist"
        )
    elif cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()
    else:
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True

    # FiLM (PATCHED: correct attribute target; supports PeftModel)
    if cfg.use_film:
        # unwrap for attribute set: PeftModel keeps real modules in base_model.model
        core = vla.base_model.model if isinstance(vla, PeftModel) else vla

        count_parameters(core.vision_backbone, "vla.vision_backbone (original)")
        core.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=core.vision_backbone,
            llm_dim=core.llm_dim,
        )
        count_parameters(core.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            assert cfg.resume_step is not None, "resume=True requires resume_step"
            state_dict = load_checkpoint("vision_backbone", cfg.resum_vla_path, cfg.resume_step)
            core.vision_backbone.load_state_dict(state_dict)
        core.vision_backbone = core.vision_backbone.to(device_id)

    # Freeze VLM (PATCHED: keep LoRA trainable if enabled; always keep action_queries)
    if cfg.freeze_vlm:
        print("Freezing VLM parameters (keep action_queries + LoRA if enabled)...")
        for name, param in vla.named_parameters():
            keep = ("action_queries" in name)
            if cfg.use_lora:
                keep = keep or ("lora" in name.lower())
            param.requires_grad = keep
        print("Freeze done.")

    # Wrap VLA with DDP
    vla = wrap_ddp(vla, device_id, find_unused=cfg.ddp_find_unused_params)

    # Compute prefix tokens: vision patches * images (PATCHED: DO NOT +1 for proprio; VLM forward doesn't insert it)
    NUM_PATCHES = (
        vla.module.vision_backbone.get_num_patches() * vla.module.vision_backbone.get_num_images_in_input()
    )
    print(f"[model] NUM_PATCHES(prefix)={NUM_PATCHES} (vision patches * num_images)")

    # Proprio projector
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
            to_bf16=True,
        )

    # Action head: MLP (L1) | DIT (DiT-X flow) | FlowMLP (MLP flow)
    action_head = None
    if cfg.action_head_type.upper() == "MLP":
        action_head = init_module(
            L1RegressionActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "use_pro_version": cfg.use_pro_version,
            },
            to_bf16=True,
        )
    elif cfg.action_head_type.upper() == "DIT":
        action_head = init_module(
            FlowMatchingActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": vla.module.llm_dim,
                "action_dim": ACTION_DIM,
                "num_task_tokens": NUM_PATCHES,
                "num_blocks": cfg.dit_num_blocks,
                "num_heads": 8,
                "mlp_ratio": 4.0,
                "use_pro_version": cfg.use_pro_version,
                "flow_ratio": cfg.flow_ratio,
                "time_dist": ("lognorm", -0.4, 1.0),
                "inference_default_mode": "ode",
                "num_inference_steps": cfg.dit_num_inference_steps,
                "num_inference_samples": cfg.dit_num_inference_samples,
                "supervised_anchor_weight": cfg.dit_supervised_anchor_weight,
                "anchor_blend": cfg.dit_anchor_blend,
                "inference_residual_scale": cfg.dit_inference_residual_scale,
                "anchor_gripper_weight": cfg.dit_anchor_gripper_weight,
                "anchor_gripper_bce_weight": cfg.dit_anchor_gripper_bce_weight,
                "flow_xyz_loss_weight": cfg.dit_flow_xyz_loss_weight,
                "flow_rot_loss_weight": cfg.dit_flow_rot_loss_weight,
                "flow_gripper_loss_weight": cfg.dit_flow_gripper_loss_weight,
                "flow_gripper_bce_weight": cfg.dit_flow_gripper_bce_weight,
                "flow_gripper_bce_logit_scale": cfg.dit_flow_gripper_bce_logit_scale,
                "flow_gripper_bce_balanced": cfg.dit_flow_gripper_bce_balanced,
                "gripper_head_weight": cfg.dit_gripper_head_weight,
                "gripper_head_override": cfg.dit_gripper_head_override,
                "clip_normalized_actions": cfg.dit_clip_normalized_actions,
                "sample_t_mode_flow": cfg.dit_sample_t_mode_flow,
                "sample_t_mode_consistency": cfg.dit_sample_t_mode_consistency,
                "sample_dt_mode_consistency": cfg.dit_sample_dt_mode_consistency,
                "sample_target_t_mode": cfg.dit_sample_target_t_mode,
                "detach_flow_conditioning": cfg.dit_detach_flow_conditioning,
                "use_state_conditioning": cfg.dit_use_state_conditioning,
                "state_scale_mode": cfg.dit_state_scale_mode,
                "state_proprio_mode": cfg.dit_state_proprio_mode,
                "state_use_chunk_pos": cfg.dit_state_use_chunk_pos,
                "state_include_task_tokens": cfg.dit_state_include_task_tokens,
                "condition_mode": cfg.dit_condition_mode,
                "condition_injection_mode": cfg.dit_condition_injection_mode,
                "include_prompt_tokens": cfg.dit_include_prompt_tokens,
                "task_token_mode": cfg.dit_task_token_mode,
                "use_latent_skill_token": cfg.dit_use_latent_skill_token,
                "num_skill_tokens": cfg.dit_num_skill_tokens,
                "skill_token_dim": cfg.dit_skill_token_dim,
                "skill_temperature": cfg.dit_skill_temperature,
                "dit_zero_init_adaln": cfg.dit_zero_init_adaln,
                "dit_zero_init_output": cfg.dit_zero_init_output,
                "dense_film_enabled": cfg.dit_dense_film_enabled,
                "dense_film_max_layers": cfg.dit_dense_film_max_layers,
                "dense_film_first_layer_index": cfg.dit_dense_film_first_layer_index,
                "dense_film_bottleneck_dim": cfg.dit_dense_film_bottleneck_dim,
                "dense_film_state_dim": cfg.dit_dense_film_state_dim,
                "use_adaptive_bridge": cfg.use_adaptive_bridge,
                "bridge_mode": cfg.bridge_mode,
                "fixed_layer_index": cfg.fixed_layer_index,
            },
            to_bf16=True,
        )
        if hasattr(action_head.module.velocity_network, "debug_group_action_tokens_to_chunk"):
            action_head.module.velocity_network.debug_group_action_tokens_to_chunk = bool(
                cfg.debug_dit_group_action_tokens_to_chunk
            )
            if cfg.debug_dit_group_action_tokens_to_chunk:
                print("[debug] grouping DIT action-token context into chunk tokens")
    elif cfg.action_head_type.upper() == "FLOWMLP":
        action_head = init_module(
            FlowMatchingMLPActionHead,
            "action_head",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "hidden_dim": getattr(cfg, "flow_matching_mlp_hidden_dim", 1024),
                "output_dim": ACTION_DIM,
                "num_task_tokens": NUM_PATCHES,
                "num_layers": getattr(cfg, "flow_matching_mlp_num_layers", 3),
                "time_dim": 256,
                "dropout": 0.1,
                "use_adaptive_bridge": cfg.use_adaptive_bridge,
                "bridge_mode": cfg.bridge_mode,
                "fixed_layer_index": cfg.fixed_layer_index,
                "use_latent_skill_token": (
                    getattr(cfg, "flowmlp_use_latent_skill_token", False)
                ),
                "use_continuous_context": getattr(cfg, "flowmlp_use_continuous_context", False),
                "skill_use_layer_routing": getattr(cfg, "flowmlp_skill_use_layer_routing", True),
                "skill_use_direct_conditioning": getattr(
                    cfg, "flowmlp_skill_use_direct_conditioning", True
                ),
                "continuous_context_use_direct_conditioning": getattr(
                    cfg, "flowmlp_continuous_context_use_direct_conditioning", False
                ),
                "num_skill_tokens": getattr(cfg, "flowmlp_num_skill_tokens", 16),
                "skill_token_dim": getattr(cfg, "flowmlp_skill_token_dim", 128),
                "skill_temperature": getattr(cfg, "flowmlp_skill_temperature", 1.0),
                "skill_entropy_weight": getattr(cfg, "flowmlp_skill_entropy_weight", 0.0),
                "skill_assignment_mode": getattr(cfg, "flowmlp_skill_assignment_mode", "hard_gumbel"),
                "skill_routing_mode": getattr(cfg, "flowmlp_skill_routing_mode", "legacy"),
                "skill_layer_temperature": getattr(cfg, "flowmlp_skill_layer_temperature", 1.0),
                "skill_temperature_start": getattr(cfg, "flowmlp_skill_temperature_start", -1.0),
                "skill_temperature_anneal_steps": getattr(
                    cfg, "flowmlp_skill_temperature_anneal_steps", 0
                ),
                "skill_balance_weight": getattr(cfg, "flowmlp_skill_balance_weight", 0.0),
                "skill_z_loss_weight": getattr(cfg, "flowmlp_skill_z_loss_weight", 0.0),
                "skill_mi_weight": getattr(cfg, "flowmlp_skill_mi_weight", 0.0),
                "skill_layer_mi_weight": getattr(cfg, "flowmlp_skill_layer_mi_weight", 0.0),
                "skill_template_diversity_weight": getattr(
                    cfg, "flowmlp_skill_template_diversity_weight", 0.0
                ),
                "routing_anchor_layer": getattr(cfg, "flowmlp_routing_anchor_layer", -1),
                "routing_adaptive_mix": getattr(cfg, "flowmlp_routing_adaptive_mix", 1.0),
                "routing_curriculum_warmup_steps": getattr(
                    cfg, "flowmlp_routing_curriculum_warmup_steps", 0
                ),
                "routing_curriculum_teacher_steps": getattr(
                    cfg, "flowmlp_routing_curriculum_teacher_steps", 0
                ),
                "routing_curriculum_num_buckets": getattr(
                    cfg, "flowmlp_routing_curriculum_num_buckets", 5
                ),
                "routing_teacher_temperature": getattr(
                    cfg, "flowmlp_routing_teacher_temperature", 0.2
                ),
                "routing_teacher_kl_weight": getattr(
                    cfg, "flowmlp_routing_teacher_kl_weight", 1.0
                ),
                "adaptive_layer_alignment": getattr(
                    cfg, "flowmlp_adaptive_layer_alignment", False
                ),
                "adaptive_num_layers": getattr(cfg, "flowmlp_adaptive_num_layers", 25),
                "adaptive_alignment_bottleneck": getattr(
                    cfg, "flowmlp_adaptive_alignment_bottleneck", 64
                ),
                "flow_time_embedding_mode": getattr(
                    cfg, "flowmlp_time_embedding_mode", "legacy"
                ),
                "flow_time_sampling_mode": getattr(
                    cfg, "flowmlp_time_sampling_mode", "uniform"
                ),
                "flow_float32_path": getattr(cfg, "flowmlp_float32_path", False),
                "flow_zero_init_output": getattr(cfg, "flowmlp_zero_init_output", True),
                "num_inference_steps": getattr(cfg, "flowmlp_num_inference_steps", 5),
                "num_inference_samples": getattr(cfg, "flowmlp_num_inference_samples", 8),
                "supervised_anchor_weight": getattr(cfg, "flowmlp_supervised_anchor_weight", 0.0),
                "anchor_blend": getattr(cfg, "flowmlp_anchor_blend", 0.0),
                "anchor_gripper_weight": getattr(cfg, "flowmlp_anchor_gripper_weight", 1.0),
                "anchor_gripper_bce_weight": getattr(cfg, "flowmlp_anchor_gripper_bce_weight", 0.0),
                "anchor_num_layers": getattr(cfg, "flowmlp_anchor_num_layers", 0),
                "anchor_hidden_dim": getattr(cfg, "flowmlp_anchor_hidden_dim", 1024),
                "detach_flow_conditioning": getattr(cfg, "flowmlp_detach_flow_conditioning", False),
                "flow_curriculum_start_step": getattr(
                    cfg, "flowmlp_flow_curriculum_start_step", 0
                ),
                "flow_curriculum_ramp_steps": getattr(
                    cfg, "flowmlp_flow_curriculum_ramp_steps", 0
                ),
                "include_prompt_tokens": getattr(cfg, "flowmlp_include_prompt_tokens", False),
                "task_token_mode": getattr(cfg, "flowmlp_task_token_mode", "vision_prompt"),
                "prompt_direct_conditioning": getattr(
                    cfg, "flowmlp_prompt_direct_conditioning", False
                ),
                "dense_film_enabled": getattr(cfg, "flowmlp_dense_film_enabled", False),
                "dense_film_max_layers": getattr(cfg, "flowmlp_dense_film_max_layers", 64),
                "dense_film_first_layer_index": getattr(
                    cfg, "flowmlp_dense_film_first_layer_index", 1
                ),
                "dense_film_bottleneck_dim": getattr(
                    cfg, "flowmlp_dense_film_bottleneck_dim", 64
                ),
                "dense_film_state_dim": getattr(cfg, "flowmlp_dense_film_state_dim", 128),
            },
            to_bf16=True,
        )
    else:
        raise ValueError(f"action_head_type must be MLP, DIT, or FlowMLP; got {cfg.action_head_type}")

    depth_interface = None
    if cfg.use_depth_interface or cfg.depth_interface_mode != "none":
        depth_interface = init_module(
            SkillAdaptiveDepthInterface,
            "depth_interface",
            cfg,
            device_id,
            {
                "input_dim": vla.module.llm_dim,
                "num_task_tokens": NUM_PATCHES,
                "mode": cfg.depth_interface_mode,
                "fixed_layer_index": cfg.fixed_layer_index,
                "max_vlm_layers": cfg.depth_interface_max_layers,
                "num_skill_tokens": getattr(cfg, "flowmlp_num_skill_tokens", 16),
                "skill_token_dim": getattr(cfg, "flowmlp_skill_token_dim", 128),
                "skill_temperature": getattr(cfg, "flowmlp_skill_temperature", 1.0),
                "skill_entropy_weight": getattr(cfg, "flowmlp_skill_entropy_weight", 0.0),
                "add_proprio_to_output": cfg.depth_interface_add_proprio,
            },
            to_bf16=True,
        )

    if (
        cfg.action_head_type.upper() == "DIT"
        and action_head is not None
        and getattr(cfg, "dit_anchor_init_checkpoint", "")
    ):
        anchor_head = getattr(action_head.module, "anchor_head", None)
        if anchor_head is None:
            raise ValueError(
                "dit_anchor_init_checkpoint requires a DIT action head with supervised anchor enabled"
            )
        anchor_init_path = Path(cfg.dit_anchor_init_checkpoint)
        if anchor_init_path.is_dir():
            anchor_init_path = Path(find_checkpoint_file(str(anchor_init_path), "action_head"))
        if not anchor_init_path.exists():
            raise FileNotFoundError(f"DIT anchor init checkpoint not found: {anchor_init_path}")
        anchor_state = load_component_state_dict(str(anchor_init_path))
        missing, unexpected = anchor_head.load_state_dict(anchor_state, strict=False)
        print(
            "[dit_anchor_init] loaded anchor_head from "
            f"{anchor_init_path} missing={list(missing)} unexpected={list(unexpected)}"
        )
        if cfg.dit_freeze_anchor_head:
            for param in anchor_head.parameters():
                param.requires_grad = False
            print("[dit_anchor_init] froze DIT anchor_head parameters")

    if cfg.train_action_head_only:
        print("[train_action_head_only] freezing VLA/LoRA/action_queries parameters")
        for param in vla.parameters():
            param.requires_grad = False
        if cfg.freeze_proprio_projector_for_action_head_only and proprio_projector is not None:
            print("[train_action_head_only] freezing proprio_projector parameters")
            for param in proprio_projector.parameters():
                param.requires_grad = False
        if depth_interface is not None:
            print("[train_action_head_only] freezing depth_interface parameters")
            for param in depth_interface.parameters():
                param.requires_grad = False
        if action_head is None:
            raise ValueError("train_action_head_only=True requires an action head")

    # Phase-two dense-FiLM adaptation: keep the mature VLM/LoRA and original
    # DiT action expert fixed, and train only the newly attached residual module.
    if cfg.dit_dense_film_adapter_only:
        if action_head is None or cfg.action_head_type.upper() != "DIT":
            raise ValueError("dit_dense_film_adapter_only requires action_head_type=DIT")
        ah_module = action_head.module if hasattr(action_head, "module") else action_head
        dense_module = getattr(getattr(ah_module, "velocity_network", None), "dense_depth_film", None)
        if dense_module is None:
            raise ValueError(
                "dit_dense_film_adapter_only=True but dense_depth_film is not enabled"
            )
        for parameter in ah_module.parameters():
            parameter.requires_grad = False
        for parameter in dense_module.parameters():
            parameter.requires_grad = True
        print("[dense_film] froze mature DiT action head; only dense_depth_film is trainable")

    if cfg.flowmlp_dense_film_adapter_only:
        if action_head is None or cfg.action_head_type.upper() != "FLOWMLP":
            raise ValueError("flowmlp_dense_film_adapter_only requires action_head_type=FLOWMLP")
        ah_module = action_head.module if hasattr(action_head, "module") else action_head
        dense_module = getattr(ah_module, "dense_depth_film", None)
        if dense_module is None:
            raise ValueError(
                "flowmlp_dense_film_adapter_only=True but dense_depth_film is not enabled"
            )
        for parameter in ah_module.parameters():
            parameter.requires_grad = False
        for parameter in dense_module.parameters():
            parameter.requires_grad = True
        print("[dense_film] froze mature FlowMLP head; only dense_depth_film is trainable")

    # Optimizer params
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    if action_head is not None:
        trainable_params += [p for p in action_head.parameters() if p.requires_grad]
    if depth_interface is not None:
        trainable_params += [p for p in depth_interface.parameters() if p.requires_grad]
    if proprio_projector is not None:
        trainable_params += [p for p in proprio_projector.parameters() if p.requires_grad]

    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"# total trainable params: {total_trainable}")

    router_lr_scale = float(getattr(cfg, "flowmlp_router_lr_scale", 1.0))
    router_params = []
    if (
        action_head is not None
        and getattr(cfg, "flowmlp_skill_routing_mode", "legacy") == "prototype_soft"
        and router_lr_scale != 1.0
    ):
        for name, parameter in action_head.named_parameters():
            if parameter.requires_grad and (
                "skill_selector." in name or name.endswith("skill_layer_logits")
            ):
                router_params.append(parameter)
    if router_params:
        router_param_ids = {id(parameter) for parameter in router_params}
        base_params = [parameter for parameter in trainable_params if id(parameter) not in router_param_ids]
        optimizer = AdamW(
            [
                {"params": base_params, "lr": cfg.learning_rate},
                {"params": router_params, "lr": cfg.learning_rate * router_lr_scale},
            ],
            lr=cfg.learning_rate,
        )
        print(
            "[optimizer] prototype router parameter group: "
            f"params={sum(p.numel() for p in router_params)} "
            f"lr_scale={router_lr_scale}"
        )
    else:
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]

    # Scheduler
    if cfg.use_constant_lr:
        print("Using Constant LR scheduler")
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    elif cfg.use_cosine_schedule:
        num_warmup_steps = cfg.lr_warmup_steps if isinstance(cfg.lr_warmup_steps, int) else int(cfg.max_steps * cfg.lr_warmup_steps)
        num_training_steps = cfg.max_steps
        print("Using Linear Warmup + Cosine Decay scheduler:")
        print(f"  Warmup steps: {num_warmup_steps} | Total steps: {num_training_steps} | Peak LR: {cfg.learning_rate}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif cfg.use_flow_matching:
        scheduler = MultiStepLR(optimizer, milestones=[10000, 20000, 30000], gamma=0.5)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    cfg._optimizer_state_loaded = False
    if cfg.resume:
        training_state_path = Path(cfg.resum_vla_path) / f"training_state--{cfg.resume_step}_checkpoint.pt"
        if not step_plan.load_training_state:
            print(
                "[checkpoint_protocol] optimizer and scheduler reload disabled; "
                "using a new optimizer timeline"
            )
        elif training_state_path.exists():
            training_state = torch.load(training_state_path, weights_only=True, map_location="cpu")
            saved_step = int(training_state.get("global_step", cfg.resume_step))
            if saved_step != int(cfg.resume_step):
                raise ValueError(
                    f"Resume step mismatch: requested {cfg.resume_step}, training state contains {saved_step}"
                )
            optimizer.load_state_dict(training_state["optimizer"])
            scheduler.load_state_dict(training_state["scheduler"])
            cfg._optimizer_state_loaded = True
            print(f"[resume] loaded optimizer and scheduler from step={saved_step}")
        else:
            raise FileNotFoundError(
                "True checkpoint continuation requires optimizer/scheduler state, "
                f"but it was not found at {training_state_path}. Use "
                "--train_from checkpoint_init for parameter initialization with "
                "a fresh optimizer and step counter."
            )

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Dataset
    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
        use_minivlm=cfg.use_minivlm,
    )

    # 婵″倹鐏?dataset_name 娑撹櫣澹掑▓濠傗偓?"rm_real_npz"閿涘苯鍨担璺ㄦ暏閺堫剙婀撮惇鐔告簚閺佺増宓侀梿鍡氣偓灞肩瑝閺?RLDS/TFDS 缁狅紕鍤?
    if str(cfg.dataset_name) == "rm_real_npz":
        train_dataset = RealRobotDataset(
            cfg.data_root_dir,
            str(cfg.dataset_name),
            batch_transform,
            target_hz=cfg.real_robot_target_hz,
            max_camera_error_s=cfg.real_robot_max_camera_error_ms / 1000.0,
            max_camera_skew_s=cfg.real_robot_max_camera_skew_ms / 1000.0,
            max_state_gap_s=cfg.real_robot_max_state_gap_ms / 1000.0,
        )
    else:
        train_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            image_aug=cfg.image_aug,
        )

    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    print("Len of dataloader: ", len(dataloader))

    diagnostic_batches = max(int(getattr(cfg, "offline_depth_diagnostic_batches", 0)), 0)
    prompt_swap_batches = max(int(getattr(cfg, "offline_prompt_swap_diagnostic_batches", 0)), 0)
    if diagnostic_batches > 0 or prompt_swap_batches > 0:
        if action_head is None or not cfg.use_flow_matching:
            raise ValueError("offline_depth_diagnostic_batches requires the FlowMLP action head")
        unwrapped_action_head = action_head.module if hasattr(action_head, "module") else action_head
        vla.eval()
        action_head.train()
        if proprio_projector is not None:
            proprio_projector.eval()
        diagnostic_rows = []
        prompt_swap_rows = []
        diagnostic_iter = iter(dataloader)
        teacher_step = int(getattr(cfg, "flowmlp_routing_curriculum_warmup_steps", 0))
        unwrapped_action_head.set_routing_step(teacher_step)
        with torch.no_grad():
            for diagnostic_index in range(diagnostic_batches):
                batch = next(diagnostic_iter)
                _, diagnostic_metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    depth_interface=depth_interface,
                    proprio_projector=proprio_projector,
                    batch=batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_flow_matching=cfg.use_flow_matching,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film,
                    num_patches=NUM_PATCHES,
                    cfg=cfg,
                )
                action_error = getattr(unwrapped_action_head, "_last_teacher_action_error", None)
                teacher_probs = getattr(unwrapped_action_head, "_last_teacher_bucket_probs", None)
                row = {
                    "batch": diagnostic_index,
                    "metrics": diagnostic_metrics,
                    "bucket_action_l1": None if action_error is None else action_error.tolist(),
                    "teacher_probs": None if teacher_probs is None else teacher_probs.tolist(),
                }
                diagnostic_rows.append(row)
                print("[offline_depth_diagnostic] " + json.dumps(row, sort_keys=True))
            for diagnostic_index in range(prompt_swap_batches):
                batch = next(diagnostic_iter)
                _, matched_metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    depth_interface=depth_interface,
                    proprio_projector=proprio_projector,
                    batch=batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_flow_matching=cfg.use_flow_matching,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film,
                    num_patches=NUM_PATCHES,
                    cfg=cfg,
                )
                # Keep visual observations, labels, actions, and proprio fixed, but
                # cyclically exchange every non-action text token between samples.
                # This is a semantic-condition ablation, not a new training batch.
                swapped_batch = dict(batch)
                swapped_input_ids = batch["input_ids"].clone()
                gt_token_ids = batch["labels"][:, 1:]
                action_mask = (
                    get_current_action_mask(gt_token_ids)
                    | get_next_actions_mask(gt_token_ids)
                )
                prompt_mask = (
                    (~action_mask)
                    & (gt_token_ids != STOP_INDEX)
                    & batch["attention_mask"][:, 1:].bool()
                )
                token_tail = swapped_input_ids[:, 1:]
                rolled_tail = batch["input_ids"][:, 1:].roll(shifts=1, dims=0)
                token_tail[prompt_mask] = rolled_tail[prompt_mask]
                swapped_batch["input_ids"] = swapped_input_ids
                _, swapped_metrics = run_forward_pass(
                    vla=vla,
                    action_head=action_head,
                    depth_interface=depth_interface,
                    proprio_projector=proprio_projector,
                    batch=swapped_batch,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    use_l1_regression=cfg.use_l1_regression,
                    use_flow_matching=cfg.use_flow_matching,
                    use_proprio=cfg.use_proprio,
                    use_film=cfg.use_film,
                    num_patches=NUM_PATCHES,
                    cfg=cfg,
                )
                row = {
                    "batch": diagnostic_index,
                    "matched": matched_metrics,
                    "prompt_swapped": swapped_metrics,
                    "anchor_l1_delta": float(
                        swapped_metrics.get("flowmlp_anchor_l1_loss", float("nan"))
                        - matched_metrics.get("flowmlp_anchor_l1_loss", float("nan"))
                    ),
                }
                prompt_swap_rows.append(row)
                print("[offline_prompt_swap_diagnostic] " + json.dumps(row, sort_keys=True))
        if distributed_state.is_main_process:
            if diagnostic_batches > 0:
                diagnostic_path = Path(run_dir) / "offline_depth_diagnostic.json"
                diagnostic_path.write_text(json.dumps(diagnostic_rows, indent=2, sort_keys=True))
                print(f"[offline_depth_diagnostic] wrote {diagnostic_path}")
            if prompt_swap_batches > 0:
                prompt_swap_path = Path(run_dir) / "offline_prompt_swap_diagnostic.json"
                prompt_swap_path.write_text(json.dumps(prompt_swap_rows, indent=2, sort_keys=True))
                print(f"[offline_prompt_swap_diagnostic] wrote {prompt_swap_path}")
        return

    if cfg.use_val_set:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )

    # Metric smoothing for micro-steps inside one accumulation window
    recent_metrics = defaultdict(lambda: deque(maxlen=cfg.grad_accumulation_steps))

    # Step accounting (optimizer-step == global_step)
    start_step = int(cfg._training_start_step)
    global_step = start_step
    last_saved_step = start_step if start_step > 0 else -1

    if distributed_state.is_main_process:
        world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        eff_batch = cfg.batch_size * cfg.grad_accumulation_steps * world
        print(f"[effective] world_size={world}, effective_global_batch={eff_batch}")

    vla.train()
    if action_head is not None:
        action_head.train()
    if depth_interface is not None:
        depth_interface.train()
    if proprio_projector is not None:
        proprio_projector.train()

    optimizer.zero_grad(set_to_none=True)

    # Train loop: iterate micro-batches until global_step reaches max_steps
    dataloader_iter = iter(dataloader)

    with tqdm.tqdm(total=max(cfg.max_steps - start_step, 0), leave=False) as progress:
        micro_step = 0
        while global_step < cfg.max_steps:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                continue

            micro_step += 1
            accum_idx = (micro_step - 1) % cfg.grad_accumulation_steps
            is_last_micro_in_accum = (accum_idx == cfg.grad_accumulation_steps - 1)

            compute_monitor = (
                (cfg.use_l1_regression and (global_step % cfg.diffusion_sample_freq == 0) and is_last_micro_in_accum)
                or (cfg.use_diffusion and (global_step % cfg.diffusion_sample_freq == 0) and is_last_micro_in_accum)
            )

            if action_head is not None:
                unwrapped_action_head = action_head.module if hasattr(action_head, "module") else action_head
                if hasattr(unwrapped_action_head, "set_routing_step"):
                    unwrapped_action_head.set_routing_step(global_step)

            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                depth_interface=depth_interface,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=cfg.use_l1_regression,
                use_flow_matching=cfg.use_flow_matching,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=NUM_PATCHES,  # <-- PATCHED: interface consistent
                compute_diffusion_l1=compute_monitor,
                use_pro_version=cfg.use_pro_version,
                cfg=cfg,
            )

            normalized_loss = loss / cfg.grad_accumulation_steps

            # DDP no_sync for true gradient accumulation speed
            if is_distributed() and (not is_last_micro_in_accum):
                vla_ctx = vla.no_sync()
                ah_ctx = action_head.no_sync() if action_head is not None else None
                di_ctx = depth_interface.no_sync() if depth_interface is not None else None
                if ah_ctx is not None and di_ctx is not None:
                    with vla_ctx, ah_ctx, di_ctx:
                        normalized_loss.backward()
                elif ah_ctx is not None:
                    with vla_ctx, ah_ctx:
                        normalized_loss.backward()
                elif di_ctx is not None:
                    with vla_ctx, di_ctx:
                        normalized_loss.backward()
                else:
                    with vla_ctx:
                        normalized_loss.backward()
            else:
                normalized_loss.backward()

            # collect micro-metrics
            for k, v in metrics.items():
                recent_metrics[k].append(float(v))

            # optimizer step on last micro-batch
            if is_last_micro_in_accum:
                # Optional grad clipping
                if cfg.gradient_clipping_norm and cfg.gradient_clipping_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.gradient_clipping_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)

                # micro-metrics averaged over the accumulation window
                smoothened_metrics = compute_smoothened_metrics(recent_metrics)

                # Log to W&B every wandb_log_freq optimizer steps
                if distributed_state.is_main_process and (global_step % cfg.wandb_log_freq == 0):
                    print(
                        "[train_metrics] "
                        + json.dumps(
                            {"global_step": global_step, **smoothened_metrics},
                            sort_keys=True,
                            allow_nan=False,
                        ),
                        flush=True,
                    )
                    log_metrics_to_wandb(smoothened_metrics, "VLA Train", global_step, wandb)
                    if wandb.run is not None:
                        wandb.log(
                            {"VLA Train/Learning Rate": optimizer.param_groups[0]["lr"]},
                            step=global_step,
                        )

                # Clear accumulation deques
                for dq in recent_metrics.values():
                    dq.clear()

                # Save checkpoint
                if global_step > 0 and (global_step % cfg.save_freq == 0):
                    save_training_checkpoint(
                        cfg=cfg,
                        run_dir=run_dir,
                        log_step=global_step,
                        vla=vla,
                        processor=processor,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        noisy_action_projector=None,
                        action_head=action_head,
                        depth_interface=depth_interface,
                        train_dataset=train_dataset,
                        distributed_state=distributed_state,
                        new_state_dict=RAW_STATE_DICT,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )
                    last_saved_step = global_step
                    if distributed_state.is_main_process:
                        print(f"\n[Step {global_step}] Checkpoint saved. Evaluation disabled during training.\n")

                # Validation
                if cfg.use_val_set and global_step > 0 and (global_step % cfg.val_freq == 0):
                    run_validation(
                        vla=vla,
                        action_head=action_head,
                        depth_interface=depth_interface,
                        noisy_action_projector=None,
                        proprio_projector=proprio_projector if cfg.use_proprio else None,
                        val_dataloader=val_dataloader,
                        action_tokenizer=action_tokenizer,
                        device_id=device_id,
                        cfg=cfg,
                        num_patches=NUM_PATCHES,  # <-- PATCHED: interface consistent
                        log_step=global_step,
                        distributed_state=distributed_state,
                        val_time_limit=cfg.val_time_limit,
                    )

        if global_step > 0 and global_step != last_saved_step:
            save_training_checkpoint(
                cfg=cfg,
                run_dir=run_dir,
                log_step=global_step,
                vla=vla,
                processor=processor,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                noisy_action_projector=None,
                action_head=action_head,
                depth_interface=depth_interface,
                train_dataset=train_dataset,
                distributed_state=distributed_state,
                new_state_dict=RAW_STATE_DICT,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            last_saved_step = global_step
            if distributed_state.is_main_process:
                print(f"\n[Step {global_step}] Final checkpoint saved.\n")

        if distributed_state.is_main_process:
            print(f"Training finished at global_step={global_step} (target={cfg.max_steps}).")


if __name__ == "__main__":
    finetune()
