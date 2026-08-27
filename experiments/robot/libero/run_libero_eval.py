"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

# Set matplotlib to use non-interactive backend for headless servers
# This only affects plotting/visualization, not the evaluation logic or results
import matplotlib
matplotlib.use("Agg")

# Set MuJoCo/robosuite renderer backend for headless servers
# IMPORTANT: robosuite has MUJOCO_GPU_RENDERING=True by default, which forces EGL
# We need to set MUJOCO_GL before importing robosuite to override this behavior
# 
# Strategy for headless servers:
# 1. Try to use Xvfb virtual display (if available) - most reliable without admin access
# 2. If Xvfb not available, try OSMesa (requires libosmesa6-dev, needs admin)
# 3. If both fail, try EGL directly (may fail if EGL drivers not properly configured)
if "MUJOCO_GL" not in os.environ:
    has_display = os.environ.get("DISPLAY") is not None
    
    # Force OSMesa for headless rendering to avoid EGL PLATFORM_DEVICE extension issues
    # EGL often fails with "PLATFORM_DEVICE extension not supported" even with Xvfb
    # OSMesa uses software rendering and is more reliable on headless servers
    # Note: OSMesa may be slower but works without GPU/EGL driver support
    os.environ["MUJOCO_GL"] = "osmesa"
    if not has_display:
        print("Headless server detected: using OSMesa for software rendering")
    else:
        print("Using OSMesa for rendering (more reliable than EGL on headless servers)")
    print("Note: OSMesa uses CPU rendering and may be slower than GPU rendering")

import draccus
import numpy as np
import tqdm

import wandb

# Add LIBERO directory to Python path so that libero module can be imported
# The libero module is located at LIBERO/libero/ relative to project root
# Also add project root to sys.path so that experiments.robot can be imported
project_root = Path(__file__).resolve().parents[3]  # Go up from experiments/robot/libero/run_libero_eval.py
sys.path.insert(0, str(project_root))  # Add project root to Python path

# Try to find robosuite - check common locations
# robosuite might be in starVLA project or installed as a package
robosuite_paths = [
    project_root.parent / "starVLA" / "robosuite",  # Common location if starVLA is in same parent dir
    Path("/home/xiaguanxiao/code/starVLA/robosuite"),  # Absolute path to starVLA robosuite
]
for robosuite_path in robosuite_paths:
    if robosuite_path.exists():
        sys.path.insert(0, str(robosuite_path))
        break

libero_path = project_root / "LIBERO" / "libero"
if libero_path.exists():
    sys.path.insert(0, str(libero_path.parent))  # Add LIBERO directory to path
else:
    # Fallback: try to find LIBERO directory relative to current working directory
    cwd_libero_path = Path.cwd() / "LIBERO" / "libero"
    if cwd_libero_path.exists():
        sys.path.insert(0, str(cwd_libero_path.parent))
    else:
        print(f"Warning: LIBERO directory not found at {libero_path} or {cwd_libero_path}. "
              "Please ensure LIBERO is in the project root directory.", file=sys.stderr)

# Initialize LIBERO config non-interactively before importing benchmark
# This prevents the interactive prompt "Do you want to specify a custom path for the dataset folder?"
libero_config_path = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
config_file = os.path.join(libero_config_path, "config.yaml")

if not os.path.exists(config_file):
    # Create config directory if it doesn't exist
    os.makedirs(libero_config_path, exist_ok=True)
    
    # Calculate default paths manually (matching libero.libero.get_default_path_dict logic)
    # Note: benchmark_root should be LIBERO/libero/libero (not LIBERO/libero)
    # because the actual libero package is at LIBERO/libero/libero/
    if libero_path.exists():
        # libero_path is LIBERO/libero, but benchmark_root should be LIBERO/libero/libero
        benchmark_root_path = str(libero_path / "libero")
    else:
        benchmark_root_path = str(cwd_libero_path / "libero") if (cwd_libero_path / "libero").exists() else str(cwd_libero_path)
    
    # Verify benchmark_root_path exists, if not try alternative
    if not os.path.exists(benchmark_root_path):
        # Fallback: try LIBERO/libero/libero directly
        alt_path = project_root / "LIBERO" / "libero" / "libero"
        if alt_path.exists():
            benchmark_root_path = str(alt_path)
        else:
            # Last resort: use libero_path itself
            benchmark_root_path = str(libero_path) if libero_path.exists() else str(cwd_libero_path)
    
    # Try to find the actual dataset location in the project
    # Check common locations: data/libero, LIBERO/datasets, etc.
    dataset_path = None
    possible_dataset_paths = [
        project_root / "data" / "libero",  # Most common location in VLA projects
        project_root / "LIBERO" / "datasets",
        Path.cwd() / "data" / "libero",
        Path.cwd() / "LIBERO" / "datasets",
    ]
    
    for path in possible_dataset_paths:
        if path.exists() and path.is_dir():
            dataset_path = str(path)
            break
    
    # If no dataset found, use default location
    if dataset_path is None:
        dataset_path = os.path.join(benchmark_root_path, "..", "datasets")
    
    default_path_dict = {
        "benchmark_root": benchmark_root_path,
        "bddl_files": os.path.join(benchmark_root_path, "bddl_files"),
        "init_states": os.path.join(benchmark_root_path, "init_files"),
        "datasets": dataset_path,
        "assets": os.path.join(benchmark_root_path, "assets"),
    }
    
    # Write config file without interactive prompts
    try:
        import yaml
    except ImportError:
        # Fallback: use simple YAML-like format if yaml not available
        with open(config_file, "w") as f:
            f.write("# LIBERO Configuration\n")
            for key, value in default_path_dict.items():
                f.write(f"{key}: {value}\n")
    else:
        with open(config_file, "w") as f:
            yaml.dump(default_path_dict, f)
    
    print(f"Initialized LIBERO config file at {config_file}")
    print(f"Using dataset path: {default_path_dict.get('datasets', 'N/A')}")

# Import robosuite and disable GPU rendering enforcement if using OSMesa
# This must be done before importing libero.libero which imports robosuite
if os.environ.get("MUJOCO_GL", "").lower() == "osmesa":
    # Import robosuite macros and disable GPU rendering to allow OSMesa
    try:
        import robosuite.macros as robosuite_macros
        robosuite_macros.MUJOCO_GPU_RENDERING = False
        print("Disabled robosuite GPU rendering enforcement to allow OSMesa")
    except (ImportError, AttributeError):
        # If we can't modify macros, OSMesa may still work if properly configured
        pass

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_depth_interface,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    action_head_type: str = "MLP"                    # "MLP" (L1) | "DIT" (DiT-X flow) | "FlowMLP" (MLP flow)
    use_l1_regression: bool = True                   # (derived from action_head_type if not set)
    use_diffusion: bool = False                      # If True, uses diffusion-based action head
    use_flow_matching: bool = False                  # (derived from action_head_type)
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
    flowmlp_dense_film_state_mode: str = "full"
    dit_num_blocks: int = 12
    dit_num_inference_steps: int = 5
    dit_num_inference_samples: int = 8
    dit_supervised_anchor_weight: float = 0.0
    dit_anchor_blend: float = 0.0
    dit_anchor_blend_was_set: bool = False
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
    dit_detach_flow_conditioning: bool = False
    dit_disable_inference_anchor: bool = False
    dit_pure_inference: bool = False              # Diagnostic: bypass DIT inference-time gates/anchor reconstruction
    dit_zero_init_adaln: bool = True
    dit_zero_init_output: bool = True
    dit_dense_film_enabled: bool = False
    dit_dense_film_max_layers: int = 64
    dit_dense_film_first_layer_index: int = 1
    dit_dense_film_bottleneck_dim: int = 64
    dit_dense_film_state_dim: int = 128
    use_depth_interface: bool = False
    depth_interface_mode: str = "none"
    depth_interface_max_layers: int = 64
    depth_interface_add_proprio: bool = True
    use_adaptive_bridge: bool = True
    bridge_mode: str = "adaptive"
    fixed_layer_index: int = -1
    flow_ratio: float = 1.0
    dit_use_state_conditioning: bool = False         # Use chunk-aligned action state tokens as the sole DIT conditioning sequence
    dit_state_scale_mode: str = "none"               # Optional scaling for chunk-aligned DIT state conditioning
    dit_state_proprio_mode: str = "concat"           # How proprio enters DIT state-conditioned context: concat | add
    dit_state_use_chunk_pos: bool = False            # Add explicit chunk-position embeddings to DIT state-conditioned context
    dit_state_include_task_tokens: bool = False      # Prepend aggregated visual/language task tokens to state-conditioned DIT context
    dit_condition_mode: str = "full"                 # DIT conditioning context: full | task_only
    dit_condition_injection_mode: str = "cross_attn" # DIT condition injection: cross_attn | joint_prefix | action_expert_prefix
    dit_include_prompt_tokens: bool = False          # Include language prompt hidden states in DIT task conditioning
    dit_task_token_mode: str = "vision_prompt"       # DIT task tokens: vision_prompt | vision_only | prompt_only | last_prompt
    use_minivlm: bool = True                         # If True, uses minivlm
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    task_ids: str = ""                               # Optional comma-separated task ids to evaluate
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    # E00/E01 read-only routing trace; disabled unless explicitly requested.
    routing_trace_enabled: bool = False
    routing_trace_path: str = ""
    routing_action_capture_path: str = ""

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    debug_flip_raw_gripper_output: bool = False      # Diagnostic: flip raw gripper x -> 1 - x before env post-processing
    debug_raw_gripper_threshold: float = 0.5         # Diagnostic: threshold raw gripper x in [0,1] before env post-processing
    debug_action_scale: float = 1.0                  # Diagnostic: multiply all raw action dimensions before env post-processing
    debug_non_gripper_action_scale: float = 1.0      # Diagnostic: multiply the first 6 raw action dims before env post-processing
    debug_gripper_scale: float = 1.0                 # Diagnostic: multiply raw gripper before env post-processing
    debug_gripper_bias: float = 0.0                  # Diagnostic: add bias to raw gripper before env post-processing
    debug_dit_group_action_tokens_to_chunk: bool = False  # Diagnostic: average DIT action-token context into 8 chunk tokens before velocity prediction
    debug_dit_use_state_conditioning: bool = False        # Diagnostic: drive DIT from chunk-aligned action state tokens instead of full token context
    debug_dit_state_scale_mode: str = "none"              # Diagnostic: optional scaling for chunk-aligned DIT state conditioning

    # fmt: on
    save_version: str = "vla-adapter"                # version of 
    use_pro_version: bool = True                     # encourage to use the pro models we released.
    phase: str = "Inference"



def _derive_action_head_flags(cfg: GenerateConfig) -> None:
    """Set use_l1_regression, use_flow_matching, flow_matching_head_type from action_head_type."""
    t = getattr(cfg, "action_head_type", "MLP").upper()
    if t not in ("MLP", "DIT", "FLOWMLP"):
        raise ValueError(f"action_head_type must be one of MLP, DIT, FlowMLP; got {cfg.action_head_type}")
    cfg.use_l1_regression = (t == "MLP")
    cfg.use_flow_matching = (t in ("DIT", "FLOWMLP"))
    cfg.flow_matching_head_type = "ditx" if t == "DIT" else "mlp"
    if getattr(cfg, "use_depth_interface", False) and getattr(cfg, "depth_interface_mode", "none") == "none":
        cfg.depth_interface_mode = "skill_adaptive"


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    _derive_action_head_flags(cfg)
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"



def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion or cfg.use_flow_matching:
        action_head = get_action_head(cfg, model.llm_dim, model=model)

    depth_interface = get_depth_interface(cfg, model.llm_dim, model=model)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, depth_interface, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    if not getattr(model, "norm_stats", None):
        raise AssertionError(
            "VLA model has no `norm_stats` (e.g. checkpoint missing dataset_statistics.json). "
            "Cannot un-normalize actions."
        )
    # Initialize unnorm_key from task suite name
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    # If still not found (e.g. checkpoint trained on different dataset like libero_spatial_no_noops),
    # fall back to the first key in norm_stats that has "action" stats so pretrained checkpoints can be evaluated.
    if unnorm_key not in model.norm_stats:
        fallback = None
        for k, v in model.norm_stats.items():
            if isinstance(v, dict) and "action" in v:
                fallback = k
                break
        if fallback is not None:
            logger.warning(
                f"Action un-norm key '{cfg.task_suite_name}' (and '{cfg.task_suite_name}_no_noops') not in VLA `norm_stats`. "
                f"Using checkpoint key '{fallback}' for action/proprio un-normalization. "
                f"Available keys: {list(model.norm_stats.keys())}"
            )
            unnorm_key = fallback
        else:
            raise AssertionError(
                f"Action un-norm key {unnorm_key} not found in VLA `norm_stats` and no fallback key with 'action' stats. "
                f"Available keys: {list(model.norm_stats.keys())}"
            )

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key



def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id



def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def log_eval_configuration(cfg: GenerateConfig, log_file=None) -> None:
    """Persist the key eval configuration so success-rate logs are reproducible."""
    effective_bridge_behavior = cfg.bridge_mode if cfg.use_adaptive_bridge else "fixed"
    log_message("Evaluation configuration:", log_file)
    log_message(f"  pretrained_checkpoint={cfg.pretrained_checkpoint}", log_file)
    log_message(f"  task_suite={cfg.task_suite_name}  task_ids={cfg.task_ids}  num_trials={cfg.num_trials_per_task}", log_file)
    log_message(f"  action_head_type={cfg.action_head_type}", log_file)
    log_message(
        f"  use_depth_interface={cfg.use_depth_interface}  depth_interface_mode={cfg.depth_interface_mode}  "
        f"depth_interface_max_layers={cfg.depth_interface_max_layers}  depth_interface_add_proprio={cfg.depth_interface_add_proprio}",
        log_file,
    )
    log_message(
        f"  use_adaptive_bridge={cfg.use_adaptive_bridge}  bridge_mode={cfg.bridge_mode}  "
        f"fixed_layer_index={cfg.fixed_layer_index}  effective_bridge_behavior={effective_bridge_behavior}  "
        f"flow_ratio={cfg.flow_ratio}",
        log_file,
    )
    if (not cfg.use_adaptive_bridge) and cfg.bridge_mode != "fixed":
        log_message(
            "  WARNING: use_adaptive_bridge=False forces fixed-layer bridging inside the action head; "
            f"bridge_mode={cfg.bridge_mode} does not preserve adaptive aggregation.",
            log_file,
        )
    log_message(
        f"  dit_num_blocks={cfg.dit_num_blocks}  dit_num_inference_steps={cfg.dit_num_inference_steps}  "
        f"dit_num_inference_samples={cfg.dit_num_inference_samples}",
        log_file,
    )
    log_message(
        f"  dit_supervised_anchor_weight={cfg.dit_supervised_anchor_weight}  "
        f"dit_anchor_blend={cfg.dit_anchor_blend}  dit_anchor_blend_was_set={cfg.dit_anchor_blend_was_set}  "
        f"dit_inference_residual_scale={cfg.dit_inference_residual_scale}",
        log_file,
    )
    log_message(
        f"  dit_anchor_gripper_weight={cfg.dit_anchor_gripper_weight}  "
        f"dit_anchor_gripper_bce_weight={cfg.dit_anchor_gripper_bce_weight}  "
        f"dit_flow_xyz_loss_weight={cfg.dit_flow_xyz_loss_weight}  "
        f"dit_flow_rot_loss_weight={cfg.dit_flow_rot_loss_weight}  "
        f"dit_flow_gripper_loss_weight={cfg.dit_flow_gripper_loss_weight}  "
        f"dit_flow_gripper_bce_weight={cfg.dit_flow_gripper_bce_weight}  "
        f"dit_flow_gripper_bce_logit_scale={cfg.dit_flow_gripper_bce_logit_scale}  "
        f"dit_flow_gripper_bce_balanced={cfg.dit_flow_gripper_bce_balanced}  "
        f"dit_gripper_head_weight={cfg.dit_gripper_head_weight}  "
        f"dit_gripper_head_override={cfg.dit_gripper_head_override}  "
        f"dit_detach_flow_conditioning={cfg.dit_detach_flow_conditioning}  "
        f"dit_disable_inference_anchor={cfg.dit_disable_inference_anchor}  "
        f"dit_pure_inference={cfg.dit_pure_inference}",
        log_file,
    )
    log_message(f"  debug_flip_raw_gripper_output={cfg.debug_flip_raw_gripper_output}", log_file)
    log_message(f"  debug_raw_gripper_threshold={cfg.debug_raw_gripper_threshold}", log_file)
    log_message(f"  debug_action_scale={cfg.debug_action_scale}", log_file)
    log_message(f"  debug_non_gripper_action_scale={cfg.debug_non_gripper_action_scale}", log_file)
    log_message(f"  debug_gripper_scale={cfg.debug_gripper_scale}", log_file)
    log_message(f"  debug_gripper_bias={cfg.debug_gripper_bias}", log_file)
    log_message(f"  debug_dit_group_action_tokens_to_chunk={cfg.debug_dit_group_action_tokens_to_chunk}", log_file)
    log_message(f"  dit_use_state_conditioning={cfg.dit_use_state_conditioning}", log_file)
    log_message(f"  dit_state_scale_mode={cfg.dit_state_scale_mode}", log_file)
    log_message(f"  dit_state_proprio_mode={cfg.dit_state_proprio_mode}", log_file)
    log_message(f"  dit_state_use_chunk_pos={cfg.dit_state_use_chunk_pos}", log_file)
    log_message(f"  dit_state_include_task_tokens={cfg.dit_state_include_task_tokens}", log_file)
    log_message(f"  dit_condition_mode={cfg.dit_condition_mode}", log_file)
    log_message(f"  dit_condition_injection_mode={cfg.dit_condition_injection_mode}", log_file)
    log_message(f"  dit_include_prompt_tokens={cfg.dit_include_prompt_tokens}", log_file)
    log_message(f"  dit_task_token_mode={cfg.dit_task_token_mode}", log_file)
    log_message(f"  dit_clip_normalized_actions={cfg.dit_clip_normalized_actions}", log_file)
    log_message(f"  dit_zero_init_adaln={cfg.dit_zero_init_adaln}", log_file)
    log_message(f"  dit_zero_init_output={cfg.dit_zero_init_output}", log_file)
    log_message(f"  dit_dense_film_enabled={cfg.dit_dense_film_enabled}", log_file)
    log_message(f"  dit_dense_film_max_layers={cfg.dit_dense_film_max_layers}", log_file)
    log_message(f"  dit_dense_film_first_layer_index={cfg.dit_dense_film_first_layer_index}", log_file)
    log_message(f"  dit_dense_film_bottleneck_dim={cfg.dit_dense_film_bottleneck_dim}", log_file)
    log_message(f"  dit_dense_film_state_dim={cfg.dit_dense_film_state_dim}", log_file)
    log_message(f"  debug_dit_use_state_conditioning={cfg.debug_dit_use_state_conditioning}", log_file)
    log_message(f"  debug_dit_state_scale_mode={cfg.debug_dit_state_scale_mode}", log_file)
    log_message(f"  seed={cfg.seed}", log_file)



def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None



def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def _unwrap_libero_env(env):
    """Unwrap OffScreenRenderEnv without assuming a particular wrapper depth."""
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        next_env = getattr(current, "env", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    return current


def _collect_pose_snapshot(env):
    """Collect simulator-ground-truth object and goal-region poses for Figure 5 traces."""
    raw_env = _unwrap_libero_env(env)
    sim = getattr(raw_env, "sim", None)
    if sim is None:
        return {}, {}

    def pose_for_name(name):
        try:
            if name in getattr(raw_env, "obj_body_id", {}):
                body_id = raw_env.obj_body_id[name]
                return {
                    "position": np.asarray(sim.data.body_xpos[body_id]).astype(float).tolist(),
                    "quaternion": np.asarray(sim.data.body_xquat[body_id]).astype(float).tolist(),
                    "source": "sim.body_xpos/body_xquat",
                }
            site_id = sim.model.site_name2id(name)
            return {
                "position": np.asarray(sim.data.site_xpos[site_id]).astype(float).tolist(),
                "quaternion": None,
                "source": "sim.site_xpos",
            }
        except Exception:
            return None

    object_pose = {}
    for name in getattr(raw_env, "obj_of_interest", []) or []:
        pose = pose_for_name(name)
        if pose is not None:
            object_pose[name] = pose

    goal_names = set()
    parsed_problem = getattr(raw_env, "parsed_problem", {}) or {}

    def collect_goal_names(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                collect_goal_names(item)
        elif isinstance(value, str):
            goal_names.add(value)

    collect_goal_names(parsed_problem.get("goal_state", []))
    goal_names.discard("And")
    goal_pose = {}
    for name in sorted(goal_names):
        pose = pose_for_name(name)
        if pose is not None:
            goal_pose[name] = pose
    return object_pose, goal_pose



def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def process_action_with_gripper_threshold(action, model_family, raw_gripper_threshold: float):
    """Diagnostic variant of process_action that binarizes raw gripper values with a custom threshold."""
    processed = action.copy()
    processed[..., -1] = 1.0 if processed[..., -1] >= raw_gripper_threshold else 0.0
    return process_action(processed, model_family)



def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    depth_interface=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    episode_id: str = "",
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
               "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    trace_records = []
    action_capture_records = []
    query_step = 0
    object_pose, goal_pose = _collect_pose_snapshot(env)
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                query_env_step_start = t
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    depth_interface=depth_interface,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm
                )

                if cfg.routing_action_capture_path:
                    action_capture_records.append({
                        "episode_id": episode_id,
                        "query_step": query_step,
                        "env_step_start": int(query_env_step_start),
                        "actions": np.asarray(actions).tolist(),
                    })

                if cfg.routing_trace_enabled:
                    diag = getattr(action_head, "last_routing_diagnostics", None)
                    if diag is None and hasattr(action_head, "velocity_network"):
                        diag = getattr(action_head.velocity_network, "last_routing_diagnostics", None)
                    if diag is None:
                        raise RuntimeError("routing_trace_enabled but no routing diagnostics were exposed")

                    def _json_value(value):
                        if value is None:
                            return None
                        if hasattr(value, "detach"):
                            value = value.detach().cpu().numpy()
                        return value.tolist() if hasattr(value, "tolist") else value

                    layer_weights_value = _json_value(diag.get("layer_weights"))
                    expected_depth = None
                    if layer_weights_value is not None:
                        weights = np.asarray(layer_weights_value, dtype=np.float64).reshape(-1)
                        expected_depth = float(np.dot(weights, np.arange(len(weights), dtype=np.float64)))

                    trace_records.append({
                        "episode_id": episode_id,
                        "query_step": query_step,
                        "env_step_start": int(query_env_step_start),
                        "env_step_end": int(query_env_step_start + max(len(actions), 1) - 1),
                        "task": task_description,
                        "seed": int(cfg.seed),
                        "success": None,
                        "skill_id": _json_value(diag.get("skill_ids")),
                        "skill_probs": _json_value(diag.get("skill_probs")),
                        "layer_weights": layer_weights_value,
                        "expected_depth": expected_depth,
                        "gripper": _json_value(np.asarray(actions)[0, -1]),
                        "eef_pose": _json_value(observation.get("state")),
                        "object_pose": object_pose,
                        "goal_pose": goal_pose,
                        "phase": None,
                        "label_source": "not_collected_in_E01_pilot",
                        "actions": _json_value(np.asarray(actions)),
                    })
                query_step += 1
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()
            # action = actions[0]

            if abs(cfg.debug_action_scale - 1.0) > 1e-8:
                action = action.copy()
                action = action * cfg.debug_action_scale

            if abs(cfg.debug_non_gripper_action_scale - 1.0) > 1e-8:
                action = action.copy()
                action[:-1] = action[:-1] * cfg.debug_non_gripper_action_scale

            if abs(cfg.debug_gripper_scale - 1.0) > 1e-8 or abs(cfg.debug_gripper_bias) > 1e-8:
                action = action.copy()
                action[-1] = action[-1] * cfg.debug_gripper_scale + cfg.debug_gripper_bias

            if cfg.debug_flip_raw_gripper_output:
                action = action.copy()
                action[-1] = 1.0 - action[-1]

            # Process action
            if abs(cfg.debug_raw_gripper_threshold - 0.5) > 1e-8:
                action = process_action_with_gripper_threshold(
                    action,
                    cfg.model_family,
                    raw_gripper_threshold=cfg.debug_raw_gripper_threshold,
                )
            else:
                action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    if cfg.routing_trace_enabled and cfg.routing_trace_path:
        os.makedirs(os.path.dirname(cfg.routing_trace_path) or ".", exist_ok=True)
        with open(cfg.routing_trace_path, "a", encoding="utf-8") as trace_file:
            for record in trace_records:
                record["success"] = bool(success)
                trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    if cfg.routing_action_capture_path:
        os.makedirs(os.path.dirname(cfg.routing_action_capture_path) or ".", exist_ok=True)
        with open(cfg.routing_action_capture_path, "a", encoding="utf-8") as action_file:
            for record in action_capture_records:
                action_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return success, replay_images




def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    depth_interface=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    save_version=None
):
    """Run evaluation for a single task."""
    # Get task
    # task_id = 8
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            depth_interface,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            episode_id=f"task{task_id}_episode{task_episodes}",
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file, save_version=save_version
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)
    
    # close env
    env.close()
    del env

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes



@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, depth_interface, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # for name, param in model.named_parameters():
    #     if 'action_queries' in name: 
    #         print(f"{name}: {param}")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_eval_configuration(cfg, log_file)
    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_ids_arg = "" if cfg.task_ids is None else str(cfg.task_ids).strip()
    if task_ids_arg.lower() not in ("", "none"):
        task_ids = [int(task_id.strip()) for task_id in task_ids_arg.split(",") if task_id.strip()]
        invalid_task_ids = [task_id for task_id in task_ids if task_id < 0 or task_id >= num_tasks]
        if invalid_task_ids:
            raise ValueError(f"Invalid task_ids {invalid_task_ids}; valid range is 0..{num_tasks - 1}")
    else:
        task_ids = list(range(num_tasks))
    print(f"[info] using task orders {task_ids}")

    for task_id in tqdm.tqdm(task_ids):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            depth_interface,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            cfg.save_version
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate



if __name__ == "__main__":
    eval_libero()
