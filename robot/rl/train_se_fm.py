"""Self-Evolving Flow Matching (SE-FM) unified trainer.

This script combines offline flow-matching / BC learning and online GRPO fine-tuning
into a single training loop.  It is intended as the entry point for the upcoming,
fully unified SE-FM pipeline.

Current status:
    - Skeleton structure with clearly separated components for
      * offline batch loading
      * online rollout collection
      * unified loss computation (lambda_off / lambda_on)
      * logging and checkpointing
    - Placeholder functions marked with TODOs for the actual flow matching loss,
      offline dataloader, advantage estimation, etc.

Follow-up work:
    1. Implement `build_offline_dataloader` to yield offline batches.
    2. Implement flow matching / BC loss in `compute_offline_loss`.
    3. Implement online rollout + advantage pipeline inside `collect_online_rollouts`
       and `compute_online_loss`.
    4. Plug in adaptive weighting `lambda_off`, `lambda_on`.
    5. Integrate completion head / Flow/MeanFlow sampling options if desired.
"""


import argparse
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import DATE_TIME, get_libero_env
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    TaskSuite,
    TASK_MAX_STEPS,
    initialize_model,
    validate_config,
)
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
from experiments.robot.rl.flow_grpo_trainer import (
    FlowGRPORolloutBatch,
    FlowGRPOTrainer,
    FlowGRPOTrainerConfig,
)
from prismatic.models.action_heads import DiffusionActionHead, FlowGRPOSample
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

if hasattr(sys.stdout, "reconfigure"):
    # Ensure streaming logs show up immediately instead of buffering until the end.
    sys.stdout.reconfigure(line_buffering=True)

@dataclass
class SEFMConfig:
    """Configuration for Self-Evolving Flow Matching training."""

    # Common model settings
    pretrained_checkpoint: str
    output_dir: str = "outputs/se_fm"
    task_suite: str = "libero_spatial"
    num_iterations: int = 500
    device: str = "cuda"
    seed: int = 42
    save_interval: int = 250
    log_interval: int = 20

    # Offline data
    offline_dataset_path: Optional[str] = None
    offline_dataset_name: Optional[str] = None
    offline_batch_size: int = 32
    offline_shuffle_buffer_size: int = 256_000
    offline_use_minivlm: bool = True
    offline_use_proprio: bool = True
    offline_use_wrist: bool = True

    # Online rollout
    max_env_steps: Optional[int] = None
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Loss weighting
    lambda_off: float = 1.0
    lambda_on: float = 0.05
    min_lambda_on: float = 0.01
    max_lambda_on: float = 1.0

    # Optimisation
    # Note: DiffusionActionHead is randomly initialized, so higher learning rate may be needed
    # For random initialization, we typically need 1e-4 to 5e-4 learning rate
    learning_rate: float = 2e-4  # Increased for random initialization (was 1e-4)
    grad_clip: float = 1.0  # Increased from 0.5 to allow larger gradients
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    normalize_advantage: bool = True
    reward_baseline: str = "mean"

    # WandB
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Utility helpers (placeholders)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_generate_config(cfg: SEFMConfig) -> GenerateConfig:
    gen = GenerateConfig()
    gen.model_family = "openvla"
    gen.pretrained_checkpoint = cfg.pretrained_checkpoint
    gen.use_l1_regression = False
    gen.use_diffusion = True
    gen.num_diffusion_steps = 50
    gen.use_minivlm = True
    gen.use_proprio = True
    gen.task_suite_name = TaskSuite(cfg.task_suite).value
    gen.use_pro_version = True
    gen.phase = "RL-Training"
    gen.num_open_loop_steps = NUM_ACTIONS_CHUNK
    # DiffusionActionHead has no pretrained weights, so we train from scratch
    gen.skip_action_head_loading = True  # Random initialization for DiffusionActionHead
    gen.allow_partial_action_head_load = True  # Allow partial loading if needed in future
    validate_config(gen)
    return gen


# ---------------------------------------------------------------------------
# Offline data pipeline (placeholder)
# ---------------------------------------------------------------------------

class OfflineBatch:
    """Container for offline supervised data."""

    def __init__(
        self,
        actions_hidden_states: torch.Tensor,
        target_actions: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        noise_targets: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        action_stats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self.actions_hidden_states = actions_hidden_states
        self.target_actions = target_actions
        self.timesteps = timesteps
        self.noise_targets = noise_targets
        self.proprio = proprio
        self.action_stats = action_stats


class OfflineIterable:
    """Simple iterable yielding `OfflineBatch` objects."""

    def __iter__(self) -> Iterator[OfflineBatch]:
        return self

    def __next__(self) -> OfflineBatch:
        raise StopIteration


class RLDSOfflineIterable(OfflineIterable):
    """Offline iterable backed by an RLDS DataLoader."""

    def __init__(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        proprio_projector: Optional[nn.Module],
        gen_cfg: GenerateConfig,
        device: torch.device,
        use_proprio: bool,
    ) -> None:
        self._dataloader = dataloader
        self._iterator = iter(dataloader)
        self._model = model
        self._proprio_projector = proprio_projector
        self._gen_cfg = gen_cfg
        self._device = device
        self._use_proprio = use_proprio
        self._action_stats = self._extract_action_stats(getattr(dataloader, "dataset", None))

    def __next__(self) -> OfflineBatch:
        # Time the data loading and model inference separately
        data_load_start = time.time()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataloader)
            batch = next(self._iterator)
        data_load_time = time.time() - data_load_start

        device = self._device
        pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        proprio = batch["proprio"].to(device) if (self._use_proprio and batch.get("proprio") is not None) else None

        # This is the bottleneck: VLM forward pass
        model_infer_start = time.time()
        with torch.no_grad():
            _, actions_hidden_states = self._model.predict_action(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                proprio=proprio,
                proprio_projector=self._proprio_projector if proprio is not None else None,
                noisy_actions=None,
                noisy_action_projector=None,
                action_head=None,
                do_sample=False,
                unnorm_key=self._gen_cfg.unnorm_key,
                use_film=self._gen_cfg.use_film,
            )
        model_infer_time = time.time() - model_infer_start
        
        # Log timing for first few batches to identify bottlenecks
        if not hasattr(self, '_batch_count'):
            self._batch_count = 0
        self._batch_count += 1
        if self._batch_count <= 5:
            print(
                f"[RLDSOfflineIterable] Batch #{self._batch_count}: "
                f"data_load={data_load_time:.2f}s, model_infer={model_infer_time:.2f}s, "
                f"total={data_load_time + model_infer_time:.2f}s",
                flush=True,
            )

        target_actions = batch["actions"].to(device=device, dtype=torch.bfloat16)

        proprio_tensor: Optional[torch.Tensor] = None
        if proprio is not None:
            proprio_tensor = torch.as_tensor(proprio, device=device, dtype=torch.bfloat16).view(1, -1)

        return OfflineBatch(
            actions_hidden_states=actions_hidden_states.detach(),
            target_actions=target_actions.detach(),
            timesteps=None,
            noise_targets=None,
            proprio=proprio_tensor.detach() if proprio_tensor is not None else None,
            action_stats=self._action_stats,
        )

    def _extract_action_stats(self, dataset_obj: Optional[Any]) -> Optional[Dict[str, torch.Tensor]]:
        if dataset_obj is None or not hasattr(dataset_obj, "dataset_statistics"):
            print("[RLDSOfflineIterable] WARNING: dataset_obj is None or has no dataset_statistics", flush=True)
            return None
        stats = dataset_obj.dataset_statistics
        action_stats: Optional[Dict[str, Any]] = None
        if isinstance(stats, dict):
            if "action" in stats:
                action_stats = stats["action"]
            else:
                for value in stats.values():
                    if isinstance(value, dict) and "action" in value:
                        action_stats = value["action"]
                        break
        if not action_stats:
            print("[RLDSOfflineIterable] WARNING: Could not find 'action' key in dataset_statistics", flush=True)
            return None

        lows = action_stats.get("q01")
        if lows is None:
            lows = action_stats.get("min")
        highs = action_stats.get("q99")
        if highs is None:
            highs = action_stats.get("max")
        if lows is None or highs is None:
            print(
                f"[RLDSOfflineIterable] WARNING: Could not extract low/high from action_stats. "
                f"Available keys: {list(action_stats.keys())}",
                flush=True,
            )
            return None

        low_tensor = torch.as_tensor(lows, dtype=torch.float32)
        high_tensor = torch.as_tensor(highs, dtype=torch.float32)
        
        # Debug: Print stats shape and range
        print(
            f"[RLDSOfflineIterable] Extracted action_stats: low shape={low_tensor.shape}, "
            f"high shape={high_tensor.shape}, low range=[{low_tensor.min():.3f}, {low_tensor.max():.3f}], "
            f"high range=[{high_tensor.min():.3f}, {high_tensor.max():.3f}]",
            flush=True,
        )
        
        return {"low": low_tensor, "high": high_tensor}


def build_offline_dataloader(
    cfg: SEFMConfig,
    gen_cfg: GenerateConfig,
    model: nn.Module,
    proprio_projector: Optional[nn.Module],
    processor: Any,
) -> OfflineIterable:
    """Return an iterable yielding offline batches (actions_hidden_states + GT actions).

    TODO:
        - Hook into RLDS / TFRecords or any custom dataset to populate:
            * actions_hidden_states: torch.Tensor of shape (B, num_layers, num_tokens, hidden_dim)
            * target_actions: torch.Tensor of shape (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
            * optional timesteps/noise for flow matching objectives
    """
    if cfg.offline_dataset_path is None:
        return OfflineIterable()

    dataset_path = Path(cfg.offline_dataset_path)

    if dataset_path.is_dir():
        # Detect whether `dataset_path` already points to the dataset leaf
        # (i.e. contains versioned subfolders such as 1.0.0).  TFDS expects
        # `data_dir` to be the parent directory that holds `<dataset_name>/X.Y.Z`.
        has_version_subdir = any(
            child.is_dir() and child.name[:1].isdigit() for child in dataset_path.iterdir()
        )
        if has_version_subdir:
            data_root_dir = dataset_path.parent
            inferred_name = dataset_path.name
        else:
            data_root_dir = dataset_path
            inferred_name = None
    else:
        data_root_dir = dataset_path.parent
        inferred_name = dataset_path.stem

    dataset_name = cfg.offline_dataset_name or inferred_name
    if dataset_name is None:
        raise ValueError(
            "Could not infer `offline_dataset_name`. Please provide it explicitly."
        )

    model_config = getattr(getattr(model, "module", model), "config", None)
    if model_config is not None and hasattr(model_config, "image_sizes"):
        resize_resolution = tuple(model_config.image_sizes)
    else:
        resize_resolution = (224, 224)

    print(f"[SE-FM] [Init]    Dataset path: {data_root_dir}, name: {dataset_name}", flush=True)
    print(f"[SE-FM] [Init]    Creating action tokenizer and batch transform...", flush=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=cfg.offline_use_wrist,
        use_proprio=cfg.offline_use_proprio,
        use_minivlm=cfg.offline_use_minivlm,
    )

    print(f"[SE-FM] [Init]    Creating RLDSDataset (this may take a while for large datasets)...", flush=True)
    dataset_start = time.time()
    dataset = RLDSDataset(
        Path(data_root_dir),
        dataset_name,
        batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=cfg.offline_shuffle_buffer_size,
        image_aug=False,
    )
    dataset_time = time.time() - dataset_start
    print(f"[SE-FM] [Init]    RLDSDataset created in {dataset_time:.2f}s", flush=True)
    
    print(f"[SE-FM] [Init]    Creating DataLoader...", flush=True)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side=processor.tokenizer.padding_side if hasattr(processor.tokenizer, "padding_side") else "right",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.offline_batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    print(f"[SE-FM] [Init]    DataLoader created", flush=True)

    device = next(model.parameters()).device
    return RLDSOfflineIterable(dataloader, model, proprio_projector, gen_cfg, device, cfg.offline_use_proprio)


def compute_offline_loss(
    action_head: DiffusionActionHead,
    proprio_projector: nn.Module,
    batch: OfflineBatch,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute offline loss with standard DDPM training.
    
    Standard DDPM: predict noise at random timesteps.
    Loss = MSE(predicted_noise, actual_noise)
    
    Args:
        action_head: DiffusionActionHead model
        proprio_projector: Proprioception projector
        batch: OfflineBatch with actions_hidden_states, target_actions, etc.
    
    Returns:
        total_loss: BC Loss (standard DDPM loss)
        loss_dict: Dictionary with loss components
    """
    device = next(action_head.parameters()).device
    actions_hidden_states = batch.actions_hidden_states.to(device)
    target_actions = batch.target_actions.to(device)  # Already normalized to [-1, 1] from RLDS
    proprio_batch = batch.proprio.to(device) if batch.proprio is not None else None
    
    batch_size = actions_hidden_states.shape[0]
    scheduler = action_head.noise_scheduler
    T = scheduler.config.num_train_timesteps
    
    # ========== Data Validation ==========
    # Check if target_actions are properly normalized
    target_min, target_max = target_actions.min().item(), target_actions.max().item()
    if not hasattr(compute_offline_loss, '_data_check_count'):
        compute_offline_loss._data_check_count = 0
    if compute_offline_loss._data_check_count < 3:
        compute_offline_loss._data_check_count += 1
        if target_min < -1.1 or target_max > 1.1:
            print(
                f"[compute_offline_loss] [WARN] target_actions may not be normalized: "
                f"range=[{target_min:.3f}, {target_max:.3f}], expected [-1, 1]",
                flush=True,
            )
        else:
            print(
                f"[compute_offline_loss] [OK] target_actions normalized: "
                f"range=[{target_min:.3f}, {target_max:.3f}]",
                flush=True,
            )
    
    action_stats = getattr(batch, "action_stats", None)
    if action_stats is None:
        import warnings
        warnings.warn(
            "[compute_offline_loss] action_stats is None! "
            "Cannot properly denormalize/normalize actions. Loss may not converge."
        )
    
    # ========== BC Loss (DDPM Training) ==========
    # Sample random timesteps
    timesteps = torch.randint(
        0, T,
        (batch_size,), device=device
    ).long()
    
    # Add noise to target actions using scheduler.add_noise() (matches reference implementation)
    # Note: target_actions are already normalized to [-1, 1] from RLDS dataset
    # scheduler.add_noise() works in the same space as the input, so we use normalized actions directly
    noise = torch.randn_like(target_actions, dtype=actions_hidden_states.dtype)
    noisy_actions_norm = scheduler.add_noise(target_actions, noise, timesteps)
    
    # Predict noise
    pred_noise_list = []
    for idx in range(batch_size):
        hidden = actions_hidden_states[idx : idx + 1]
        noisy_action = noisy_actions_norm[idx : idx + 1]
        timestep = timesteps[idx : idx + 1]
        if proprio_batch is not None:
            proprio = proprio_batch[idx : idx + 1]
        else:
            proprio = torch.zeros((1, PROPRIO_DIM), device=device, dtype=hidden.dtype)
        
        pred_noise = action_head.predict_noise(
            noisy_action, timestep, hidden, proprio, proprio_projector
        )
        pred_noise_list.append(pred_noise)
    
    pred_noise = torch.cat(pred_noise_list, dim=0)
    
    # BC Loss: MSE between predicted noise and actual noise (standard DDPM)
    # Ensure noise has the same dtype as pred_noise
    noise = noise.to(dtype=pred_noise.dtype)
    bc_loss = F.mse_loss(pred_noise, noise)
    
    # ========== Noise Scale Validation ==========
    # Check if predicted noise scale matches actual noise scale
    noise_std = noise.std().item()
    pred_noise_std = pred_noise.std().item()
    noise_mean = noise.mean().item()
    pred_noise_mean = pred_noise.mean().item()
    
    if not hasattr(compute_offline_loss, '_noise_check_count'):
        compute_offline_loss._noise_check_count = 0
    if compute_offline_loss._noise_check_count < 5:
        compute_offline_loss._noise_check_count += 1
        scale_diff = abs(noise_std - pred_noise_std)
        mean_diff = abs(noise_mean - pred_noise_mean)
        
        if scale_diff > 0.5 or mean_diff > 0.5:
            print(
                f"[compute_offline_loss] [WARN] Noise scale mismatch: "
                f"noise(mean={noise_mean:.4f}, std={noise_std:.4f}) vs "
                f"pred_noise(mean={pred_noise_mean:.4f}, std={pred_noise_std:.4f})",
                flush=True,
            )
        else:
            print(
                f"[compute_offline_loss] [OK] Noise scale match: "
                f"noise(mean={noise_mean:.4f}, std={noise_std:.4f}), "
                f"pred_noise(mean={pred_noise_mean:.4f}, std={pred_noise_std:.4f})",
                flush=True,
            )
    
    # Additional diagnostic: Check if loss is in expected range
    # For normalized actions in [-1, 1], noise should be roughly N(0, 1) scaled by scheduler
    # Expected loss range: [0, ~10] for well-trained model, [0, ~100] for random initialization
    if not hasattr(compute_offline_loss, '_loss_history'):
        compute_offline_loss._loss_history = []
    compute_offline_loss._loss_history.append(bc_loss.item())
    if len(compute_offline_loss._loss_history) > 100:
        compute_offline_loss._loss_history.pop(0)
    
    # Check if loss is decreasing
    if len(compute_offline_loss._loss_history) >= 20:
        recent_avg = np.mean(compute_offline_loss._loss_history[-10:])
        earlier_avg = np.mean(compute_offline_loss._loss_history[-20:-10])
        if recent_avg > earlier_avg * 1.1 and len(compute_offline_loss._loss_history) % 50 == 0:
            print(
                f"[compute_offline_loss] [WARN] Loss may not be decreasing: "
                f"recent_avg={recent_avg:.4f}, earlier_avg={earlier_avg:.4f}. "
                f"Consider increasing learning rate or checking data normalization.",
                flush=True,
            )
    
    # Debug: Log noise statistics for first few iterations
    if not hasattr(compute_offline_loss, '_debug_count'):
        compute_offline_loss._debug_count = 0
    if compute_offline_loss._debug_count < 5:
        compute_offline_loss._debug_count += 1
        print(
            f"[compute_offline_loss] Debug #{compute_offline_loss._debug_count}: "
            f"bc_loss={bc_loss.item():.6f}, "
            f"pred_noise: mean={pred_noise.mean().item():.4f}, std={pred_noise.std().item():.4f}, "
            f"range=[{pred_noise.min().item():.4f}, {pred_noise.max().item():.4f}], "
            f"noise: mean={noise.mean().item():.4f}, std={noise.std().item():.4f}, "
            f"range=[{noise.min().item():.4f}, {noise.max().item():.4f}]",
            flush=True,
        )
    
    # Check for NaN/Inf loss
    if not torch.isfinite(bc_loss):
        import warnings
        warnings.warn(
            f"[compute_offline_loss] BC loss is not finite: {bc_loss.item()}. "
            f"pred_noise range: [{pred_noise.min().item():.3f}, {pred_noise.max().item():.3f}], "
            f"noise range: [{noise.min().item():.3f}, {noise.max().item():.3f}]"
        )
        # Replace with a small finite value to prevent training crash
        bc_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # ========== Total Loss ==========
    # Standard DDPM: only BC Loss (noise prediction)
    total_loss = bc_loss
    
    loss_dict = {
        "bc_loss": bc_loss.item(),
        "total_loss": total_loss.item(),
    }
    
    return total_loss, loss_dict


def _denormalize_actions_with_stats(
    actions: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Denormalize actions from [-1, 1] back to original space using action_stats."""
    if stats is None:
        return actions
    
    # Get normalization bounds
    lows = stats.get("q01")
    if lows is None:
        lows = stats.get("min")
    highs = stats.get("q99")
    if highs is None:
        highs = stats.get("max")
    
    if lows is None or highs is None:
        return actions
    
    # Convert to tensors if needed
    if not isinstance(lows, torch.Tensor):
        lows = torch.tensor(lows, device=device, dtype=actions.dtype)
    if not isinstance(highs, torch.Tensor):
        highs = torch.tensor(highs, device=device, dtype=actions.dtype)
    
    # Ensure proper shape
    while lows.dim() < actions.dim():
        lows = lows.unsqueeze(0)
    while highs.dim() < actions.dim():
        highs = highs.unsqueeze(0)
    
    # Denormalize: x_orig = (x_norm + 1) * (high - low) / 2 + low
    # Inverse of: x_norm = 2 * (x - low) / (high - low) - 1
    denorm_actions = (actions + 1.0) * (highs - lows) / 2.0 + lows
    return denorm_actions


def _normalize_actions_with_stats(
    actions: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Normalize actions to [-1, 1] range using dataset statistics.
    
    This matches VLA-Adapter's normalization scheme (see `prismatic/vla/datasets/rlds/utils/data_utils.py`).
    
    Args:
        actions: Tensor of shape (batch, num_chunks, action_dim) or (num_chunks, action_dim)
        stats: Dict with 'low' and 'high' tensors of shape (action_dim,)
        device: Target device
    
    Returns:
        Normalized actions in [-1, 1] range (clipped)
    """
    low = stats.get("low")
    high = stats.get("high")
    if low is None or high is None:
        return actions
    
    low = low.to(device=device, dtype=actions.dtype)
    high = high.to(device=device, dtype=actions.dtype)
    
    # Ensure low and high are 1D (action_dim,)
    if low.dim() > 1:
        low = low.flatten()
    if high.dim() > 1:
        high = high.flatten()
    
    # Ensure they match action_dim
    action_dim = actions.shape[-1]
    if low.shape[0] != action_dim:
        # If stats have wrong dimension, try to broadcast or truncate
        if low.shape[0] > action_dim:
            low = low[:action_dim]
            high = high[:action_dim]
        else:
            # Pad with zeros (shouldn't happen, but handle gracefully)
            pad_size = action_dim - low.shape[0]
            low = torch.cat([low, torch.zeros(pad_size, device=low.device, dtype=low.dtype)])
            high = torch.cat([high, torch.ones(pad_size, device=high.device, dtype=high.dtype)])
    
    denom = (high - low).clamp(min=1e-8)
    
    # Expand to match actions shape: (batch, num_chunks, action_dim)
    # low/high should be (1, 1, action_dim) or (1, action_dim)
    while low.dim() < actions.dim():
        low = low.unsqueeze(0)
        high = high.unsqueeze(0)
        denom = denom.unsqueeze(0)
    
    # Normalize to [-1, 1] (matches VLA-Adapter: 2 * (x - low) / (high - low) - 1)
    normalized = 2.0 * (actions - low) / denom - 1.0
    # Clip to [-1, 1] as in VLA-Adapter
    normalized = torch.clamp(normalized, min=-1.0, max=1.0)
    return normalized


# ---------------------------------------------------------------------------
# Online rollout helpers (Flow-GRPO reuse)
# ---------------------------------------------------------------------------


def prepare_policy_inputs(
    gen_cfg: GenerateConfig,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Optional[np.ndarray]]:
    """Prepare transformer inputs and optional proprio features from an observation."""

    primary_candidates = ["full_image", "agentview_image", "rgb_static"]
    primary_image = None
    for key in primary_candidates:
        if key in obs:
            primary_image = obs[key]
            break
    if primary_image is None:
        raise KeyError(
            "Observation missing expected primary image keys. "
            f"Available keys: {list(obs.keys())}"
        )

    images: List[np.ndarray] = [primary_image]
    if gen_cfg.num_images_in_input > 1:
        wrist_tokens = ["wrist", "eye_in_hand", "hand_image", "handcamera", "robot0_eye"]
        wrist_keys = [
            key
            for key in obs.keys()
            if key != primary_candidates[0] and any(token in key.lower() for token in wrist_tokens)
        ]
        for key in wrist_keys:
            images.append(obs[key])
        while len(images) < gen_cfg.num_images_in_input:
            images.append(primary_image)

    processed_images = prepare_images_for_vla(images, gen_cfg)
    primary_processed = processed_images.pop(0)

    if gen_cfg.use_minivlm:
        prompt = (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\nWhat action should the robot take to {task_label.lower()}?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    inputs = processor(prompt, primary_processed)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)

    if processed_images:
        wrist_tensors = [
            processor(prompt, wrist_image)["pixel_values"].to(device=device, dtype=torch.bfloat16)
            for wrist_image in processed_images
        ]
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"]] + wrist_tensors, dim=1)

    proprio = None
    if gen_cfg.use_proprio:
        if "robot0_proprio-state" in obs:
            proprio_array = np.asarray(obs["robot0_proprio-state"]).reshape(-1).astype(np.float32)
            if proprio_array.shape[0] == PROPRIO_DIM:
                proprio = proprio_array
            elif proprio_array.shape[0] > PROPRIO_DIM:
                if not getattr(prepare_policy_inputs, "_warned_proprio_shape", False):
                    print(
                        "[SE-FM][WARN] robot0_proprio-state has shape",
                        proprio_array.shape,
                        f"; truncating to first {PROPRIO_DIM} elements.",
                    )
                    prepare_policy_inputs._warned_proprio_shape = True
                proprio = proprio_array[:PROPRIO_DIM]
            else:
                proprio = None
        else:
            components: List[np.ndarray] = []
            if "robot0_eef_pos" in obs:
                components.append(np.asarray(obs["robot0_eef_pos"]).reshape(-1))
            if "robot0_eef_quat" in obs:
                components.append(np.asarray(obs["robot0_eef_quat"]).reshape(-1))
            if "robot0_gripper_qpos" in obs:
                components.append(np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)[:1])
            if components and sum(comp.shape[0] for comp in components) == PROPRIO_DIM:
                proprio = np.concatenate(components, axis=0)

        if proprio is None:
            if not getattr(prepare_policy_inputs, "_warned_missing_proprio", False):
                print(
                    "[SE-FM][WARN] Unable to assemble proprio features from observation keys; "
                    "falling back to zeros."
                )
                prepare_policy_inputs._warned_missing_proprio = True
            proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)

        stats = getattr(gen_cfg, "proprio_stats", None)
        if stats is not None:
            if isinstance(stats, dict) and "q99" in stats:
                stats_q99 = np.asarray(stats["q99"])
                if stats_q99.shape[0] == proprio.shape[0]:
                    proprio = normalize_proprio(proprio, stats)
                else:
                    if not getattr(prepare_policy_inputs, "_warned_proprio_stats_mismatch", False):
                        print(
                            "[SE-FM][WARN] Proprio dimension mismatch with normalization stats:",
                            proprio.shape,
                            "vs",
                            stats_q99.shape,
                            "; skipping normalization.",
                        )
                        prepare_policy_inputs._warned_proprio_stats_mismatch = True
            else:
                if not getattr(prepare_policy_inputs, "_warned_proprio_stats_type", False):
                    print("[SE-FM][WARN] Unsupported proprio_stats format; skipping normalization.")
                    prepare_policy_inputs._warned_proprio_stats_type = True

    return inputs, proprio


def collect_online_rollouts(
    cfg: SEFMConfig,
    gen_cfg: GenerateConfig,
    model: nn.Module,
    action_head: DiffusionActionHead,
    proprio_projector: nn.Module,
    processor: Any,
    suite_tasks: List[Any],
    iteration: int,
    device: torch.device,
) -> Tuple[List[FlowGRPORolloutBatch], Dict[str, float]]:
    """Collect online rollouts and build FlowGRPO batches."""

    batches: List[FlowGRPORolloutBatch] = []
    stats = {"episode_success": 0.0, "chunk_success_rate": 0.0, "episode_return": 0.0}

    task = suite_tasks[iteration % len(suite_tasks)]
    env, task_desc = get_libero_env(task, model_family=gen_cfg.model_family, resolution=gen_cfg.env_img_res)
    obs = env.reset()

    done = False
    step_count = 0
    chunk_success_flags: List[float] = []
    chunk_rewards: List[float] = []

    chunk_count = 0
    while not done and (cfg.max_env_steps is None or step_count < cfg.max_env_steps):
        chunk_start = time.time()
        inputs, proprio_norm = prepare_policy_inputs(gen_cfg, processor, obs, task_desc, device)

        proprio_tensor: Optional[torch.Tensor] = None
        if proprio_norm is not None:
            proprio_tensor = torch.as_tensor(proprio_norm, device=device, dtype=torch.bfloat16).view(1, -1)

        model_infer_start = time.time()
        with torch.no_grad():
            _, actions_hidden_states = model.predict_action(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                proprio=proprio_tensor if proprio_tensor is not None else None,
                proprio_projector=proprio_projector if proprio_tensor is not None else None,
                noisy_actions=None,
                noisy_action_projector=None,
                action_head=None,
                do_sample=False,
                unnorm_key=gen_cfg.unnorm_key,
                use_film=gen_cfg.use_film,
            )
        model_infer_time = time.time() - model_infer_start

        proprio_for_head = (
            proprio_tensor
            if proprio_tensor is not None
            else torch.zeros((1, PROPRIO_DIM), device=device, dtype=actions_hidden_states.dtype)
        )

        action_sample_start = time.time()
        flow_sample: FlowGRPOSample = action_head.sample_with_logprob(
            actions_hidden_states=actions_hidden_states,
            proprio=proprio_for_head,
            proprio_projector=proprio_projector,
        )
        action_sample_time = time.time() - action_sample_start

        chunk_success_local: List[float] = []
        chunk_reward_local: List[float] = []
        env_step_start = time.time()
        for chunk_idx in range(NUM_ACTIONS_CHUNK):
            if done:
                break
            action_tensor = flow_sample.actions[chunk_idx]
            action_np = (
                action_tensor.detach()
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
                .tolist()
            )
            try:
                env_step_result = env.step(action_np)
            except ValueError as exc:
                if "terminated episode" in str(exc).lower():
                    if not getattr(collect_online_rollouts, "_warned_terminated_step", False):
                        print(
                            "[SE-FM][WARN] Attempted to step terminated episode; "
                            "terminating rollout early."
                        )
                        collect_online_rollouts._warned_terminated_step = True
                    done = True
                    break
                raise
            if len(env_step_result) == 5:
                obs, reward, terminated, truncated, info = env_step_result
            else:
                obs, reward, done_flag, info = env_step_result
                terminated = done_flag
                truncated = False
            chunk_reward_local.append(reward)
            chunk_success_local.append(float(info.get("success", False)) if isinstance(info, dict) else 0.0)
            step_count += 1
            if terminated or truncated or (cfg.max_env_steps is not None and step_count >= cfg.max_env_steps):
                done = True
                break

        env_step_time = time.time() - env_step_start
        chunk_total_time = time.time() - chunk_start
        
        chunk_rewards.extend(chunk_reward_local)
        chunk_success_flags.extend(chunk_success_local)
        chunk_count += 1

        # Print progress every 10 chunks or at the end
        if chunk_count % 10 == 0 or done or (cfg.max_env_steps is not None and step_count >= cfg.max_env_steps):
            print(
                f"[SE-FM]    Rollout progress: {chunk_count} chunks, {step_count} steps, "
                f"success_rate={np.mean(chunk_success_flags) if chunk_success_flags else 0.0:.3f}, "
                f"chunk_time={chunk_total_time:.2f}s (model={model_infer_time:.2f}s, "
                f"action={action_sample_time:.2f}s, env={env_step_time:.2f}s)",
                flush=True,
            )

        reward_tensor = torch.tensor([float(any(chunk_success_local))], device=device, dtype=torch.float32)
        batch = FlowGRPORolloutBatch(
            actions_hidden_states=actions_hidden_states.detach(),
            proprio=proprio_for_head.detach(),
            rewards=reward_tensor,
            flow_sample=flow_sample,
        )
        batches.append(batch)

    env.close()

    if chunk_success_flags:
        stats["chunk_success_rate"] = float(np.mean(chunk_success_flags))
        stats["episode_success"] = float(any(flag > 0.0 for flag in chunk_success_flags))
    stats["episode_return"] = float(sum(chunk_rewards))

    return batches, stats


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def run_training(cfg: SEFMConfig) -> None:
    print("[SE-FM] ========================================", flush=True)
    print("[SE-FM] Starting SE-FM training initialization...", flush=True)
    print("[SE-FM] ========================================", flush=True)
    
    init_start = time.time()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    print(f"[SE-FM] [Init] Device: {device}, seed: {cfg.seed}", flush=True)

    print("[SE-FM] [Init] Building generate config and loading model...", flush=True)
    gen_cfg = build_generate_config(cfg)
    model_load_start = time.time()
    model, action_head, proprio_projector, _, processor = initialize_model(gen_cfg)
    model_load_time = time.time() - model_load_start
    print(f"[SE-FM] [Init] Model loaded in {model_load_time:.2f}s", flush=True)
    
    # Check action_head initialization status
    if action_head is not None:
        if gen_cfg.skip_action_head_loading:
            print(
                f"[SE-FM] [Init] Action head: Randomly initialized (DiffusionActionHead from scratch). "
                f"Using learning_rate={cfg.learning_rate} (may need adjustment if loss doesn't converge).",
                flush=True,
            )
        else:
            # Check if pretrained weights were actually loaded
            total_params = sum(p.numel() for p in action_head.parameters())
            non_zero_params = sum((p != 0).sum().item() for p in action_head.parameters())
            zero_ratio = 1.0 - (non_zero_params / total_params) if total_params > 0 else 0.0
            if zero_ratio > 0.5:
                print(
                    f"[SE-FM] [WARN] Action head appears to be randomly initialized "
                    f"({zero_ratio*100:.1f}% parameters are zero) despite skip_action_head_loading=False. "
                    f"This may require more training steps or higher learning rate.",
                    flush=True,
                )
            else:
                print(
                    f"[SE-FM] [Init] Action head loaded with pretrained weights "
                    f"({(1-zero_ratio)*100:.1f}% parameters are non-zero)",
                    flush=True,
                )
    
    if hasattr(model, "norm_stats") and gen_cfg.unnorm_key in model.norm_stats:
        gen_cfg.proprio_stats = model.norm_stats[gen_cfg.unnorm_key].get("proprio", None)
    else:
        gen_cfg.proprio_stats = None
    model = model.to(device)
    action_head = action_head.to(device)
    proprio_projector = proprio_projector.to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    proprio_projector.eval()
    for p in proprio_projector.parameters():
        p.requires_grad = False

    action_head.train()

    print("[SE-FM] [Init] Setting up optimizer and trainer...", flush=True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, action_head.parameters()),
        lr=cfg.learning_rate,
    )

    trainer_config = FlowGRPOTrainerConfig(
        clip_epsilon=cfg.clip_epsilon,
        normalize_advantage=cfg.normalize_advantage,
        entropy_coef=cfg.entropy_coef,
        reward_baseline=cfg.reward_baseline,
        max_grad_norm=cfg.grad_clip,
        lambda_off=cfg.lambda_off,
        lambda_on=cfg.lambda_on,
    )
    trainer = FlowGRPOTrainer(
        action_head=action_head,
        proprio_projector=proprio_projector,
        optimizer=optimizer,
        config=trainer_config,
        device=device,
    )

    print("[SE-FM] [Init] Building offline dataloader (this may take a while)...", flush=True)
    dataloader_start = time.time()
    offline_loader = build_offline_dataloader(cfg, gen_cfg, model, proprio_projector, processor)
    dataloader_time = time.time() - dataloader_start
    print(f"[SE-FM] [Init] Offline dataloader built in {dataloader_time:.2f}s", flush=True)

    print("[SE-FM] [Init] Loading task suite...", flush=True)
    suite_cls = benchmark.get_benchmark(TaskSuite(cfg.task_suite).value)
    suite = suite_cls()
    tasks = suite.tasks
    print(f"[SE-FM] [Init] Loaded {len(tasks)} tasks", flush=True)

    use_wandb = cfg.use_wandb and cfg.wandb_entity and cfg.wandb_project
    if use_wandb:
        print("[SE-FM] [Init] Initializing WandB...", flush=True)
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or f"se-fm-{cfg.task_suite}-{DATE_TIME}",
            config=vars(cfg),
        )
        print("[SE-FM] [Init] WandB initialized", flush=True)

    print("[SE-FM] [Init] Creating offline data iterator...", flush=True)
    offline_iter = iter(offline_loader) if offline_loader else None
    
    init_time = time.time() - init_start
    print("[SE-FM] ========================================", flush=True)
    print(f"[SE-FM] Initialization complete in {init_time:.2f}s", flush=True)
    print(f"[SE-FM] Starting training loop for {cfg.num_iterations} iterations...", flush=True)
    print("[SE-FM] ========================================", flush=True)
    print("", flush=True)  # Empty line for readability

    for iteration in range(1, cfg.num_iterations + 1):
        iter_start = time.time()
        print(f"[SE-FM] Iteration {iteration}/{cfg.num_iterations} starting...", flush=True)
        # ------------------------------------------------------------------
        # Offline loss
        # ------------------------------------------------------------------
        offline_loss = torch.zeros([], device=device)
        offline_fetch_time: Optional[float] = None
        offline_batch: Optional[OfflineBatch] = None
        if offline_iter is not None:
            if iteration == 1:
                print("[SE-FM]    Fetching first offline batch (this may take a while due to dataset warm-up)...", flush=True)
            fetch_start = time.time()
            try:
                offline_batch = next(offline_iter)
            except StopIteration:
                offline_iter = iter(offline_loader)
                try:
                    offline_batch = next(offline_iter)
                except StopIteration:
                    offline_iter = None
                    offline_batch = None
            if offline_batch is not None:
                offline_loss, offline_loss_dict = compute_offline_loss(
                    action_head, proprio_projector, offline_batch
                )
                offline_fetch_time = time.time() - fetch_start
                if iteration == 1:
                    print(f"[SE-FM]    First offline batch fetched in {offline_fetch_time:.2f}s", flush=True)
                
                # Debug: Check if normalization is working
                action_stats = getattr(offline_batch, "action_stats", None)
                if action_stats is None and iteration <= 5:
                    print(
                        f"[SE-FM]  ⚠️  WARNING: action_stats is None! "
                        f"Loss will be computed on unnormalized actions (loss={float(offline_loss):.2f}). "
                        f"This may prevent convergence.",
                        flush=True,
                    )
                elif action_stats is not None and iteration <= 5:
                    # Check if normalized loss is in reasonable range [0, 4]
                    # MSE loss in [-1, 1] normalized actions has max value of 4 (when pred=-1, target=1 or vice versa)
                    if float(offline_loss) > 4.0:
                        print(
                            f"[SE-FM]  ⚠️  WARNING: Normalized loss is very large ({float(offline_loss):.2f}). "
                            f"Expected range is [0, 4] for normalized actions in [-1, 1] range. "
                            f"Check if normalization is working correctly.",
                            flush=True,
                        )

        if offline_batch is None:
            print("[SE-FM]  └─ Offline stage: skipped (no batch available).", flush=True)
            offline_loss = torch.tensor(0.0, device=device)
            offline_loss_dict = {}
        else:
            fetch_msg = (
                f"[SE-FM]  └─ Offline stage: batch fetched in {offline_fetch_time:.2f}s, "
                f"bc_loss={float(offline_loss):.6f}"
            )
            print(fetch_msg, flush=True)

        # ------------------------------------------------------------------
        # Online loss (rollouts) + optimizer step via FlowGRPOTrainer
        # ------------------------------------------------------------------
        rollout_start = time.time()
        if cfg.lambda_on > 0:
            online_batches, rollout_stats = collect_online_rollouts(
                cfg, gen_cfg, model, action_head, proprio_projector, processor, tasks, iteration, device
            )
            rollout_time = time.time() - rollout_start
            print(
                f"[SE-FM]  └─ Online stage: collected {len(online_batches)} batches in {rollout_time:.2f}s "
                f"(episode_success={rollout_stats['episode_success']:.3f}, "
                f"chunk_success_rate={rollout_stats['chunk_success_rate']:.3f}, "
                f"episode_return={rollout_stats['episode_return']:.3f})",
                flush=True,
            )
        else:
            # Skip online rollouts when lambda_on=0
            online_batches = []
            rollout_stats = {"episode_success": 0.0, "chunk_success_rate": 0.0, "episode_return": 0.0}
            rollout_time = 0.0
            if iteration <= 3:
                print(f"[SE-FM]  └─ Online stage: skipped (lambda_on=0)", flush=True)

        # ------------------------------------------------------------------
        # Optimizer step
        # ------------------------------------------------------------------
        opt_start = time.time()
        if not online_batches:
            total_loss = cfg.lambda_off * offline_loss
            
            # Check loss before backward
            if not torch.isfinite(total_loss):
                print(
                    f"[SE-FM] [WARN] Total loss is not finite: {total_loss.item()}, "
                    f"skipping optimizer step",
                    flush=True,
                )
                trainer_metrics = {
                    "loss": float('nan'),
                    "offline_loss": offline_loss.item() if torch.isfinite(offline_loss) else float('nan'),
                    "offline_bc_loss": offline_loss_dict.get("bc_loss", 0.0),
                    "online_loss": 0.0,
                    "total_loss": float('nan'),
                }
                continue
            
            optimizer.zero_grad()
            backward_start = time.time()
            total_loss.backward()
            backward_time = time.time() - backward_start
            
            # Check gradients before clipping
            grad_norm_before = torch.nn.utils.clip_grad_norm_(action_head.parameters(), float('inf'))
            
            # Check for NaN/Inf gradients
            has_nan_grad = False
            for name, param in action_head.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        has_nan_grad = True
                        if iteration <= 5:
                            print(
                                f"[SE-FM] [WARN] NaN/Inf gradient detected in {name}, "
                                f"grad_norm_before={grad_norm_before:.4f}",
                                flush=True,
                            )
                        break
            
            clip_start = time.time()
            if has_nan_grad:
                # Skip clipping and step if gradients are invalid
                print(
                    f"[SE-FM] [WARN] Skipping gradient clipping and optimizer step due to NaN/Inf gradients",
                    flush=True,
                )
                optimizer.zero_grad()  # Clear invalid gradients
                clip_time = 0.0
                step_time = 0.0
                grad_norm_after = float('inf')
            else:
                # Adaptive gradient clipping: if grad_norm is very small, use larger clip value
                # This prevents over-clipping when gradients are already small
                effective_clip = cfg.grad_clip
                if grad_norm_before < cfg.grad_clip * 0.1:
                    # If gradient is already very small, use a larger clip to avoid over-clipping
                    effective_clip = cfg.grad_clip * 10.0
                    if iteration <= 5:
                        print(
                            f"[SE-FM] [DEBUG] Grad norm is very small ({grad_norm_before:.6f}), "
                            f"using larger clip value ({effective_clip:.2f})",
                            flush=True,
                        )
                
                grad_norm_after = torch.nn.utils.clip_grad_norm_(action_head.parameters(), effective_clip)
                clip_time = time.time() - clip_start
                
                # Warn if gradient is being clipped too aggressively
                if grad_norm_after < grad_norm_before * 0.1 and iteration <= 10:
                    print(
                        f"[SE-FM] [WARN] Gradient was clipped aggressively: "
                        f"{grad_norm_before:.6f} -> {grad_norm_after:.6f} "
                        f"(clip={effective_clip:.2f}). This may slow down training.",
                        flush=True,
                    )
                
                step_start = time.time()
                optimizer.step()
                step_time = time.time() - step_start
            opt_time = time.time() - opt_start
            
            # Print gradient info for debugging (only first few iterations)
            if iteration <= 10 or iteration % 50 == 0:
                print(
                    f"[SE-FM]    Optimizer step: {opt_time:.2f}s "
                    f"(backward={backward_time:.2f}s, clip={clip_time:.2f}s, step={step_time:.2f}s) | "
                    f"grad_norm: {grad_norm_before:.4f} -> {grad_norm_after:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"[SE-FM]    Optimizer step: {opt_time:.2f}s",
                    flush=True,
                )
            trainer_metrics = {
                "loss": total_loss.item(),
                "offline_loss": offline_loss.item(),
                "offline_bc_loss": offline_loss_dict.get("bc_loss", 0.0),
                "online_loss": 0.0,
                "approx_kl": 0.0,
                "advantages_mean": 0.0,
                "advantages_std": 0.0,
                "lambda_off": cfg.lambda_off,
                "lambda_on": 0.0,
                "total_loss": total_loss.item(),
            }
        else:
            train_step_start = time.time()
            primary_metrics = trainer.train_step(
                online_batches[0],
                offline_loss=offline_loss,
                lambda_off=cfg.lambda_off,
                lambda_on=cfg.lambda_on,
            )
            trainer_metrics = primary_metrics
            for extra_batch in online_batches[1:]:
                trainer_metrics = trainer.train_step(
                    extra_batch,
                    offline_loss=None,
                    lambda_off=0.0,
                    lambda_on=cfg.lambda_on,
                )
            opt_time = time.time() - opt_start
            print(
                f"[SE-FM]    Optimizer step: {opt_time:.2f}s "
                f"({len(online_batches)} batches)",
                flush=True,
            )
            if "offline_loss" not in trainer_metrics and "offline_loss" in primary_metrics:
                trainer_metrics["offline_loss"] = primary_metrics["offline_loss"]
            trainer_metrics["lambda_off"] = primary_metrics.get("lambda_off", cfg.lambda_off)
            trainer_metrics["total_loss"] = (
                trainer_metrics["lambda_on"] * trainer_metrics["online_loss"]
                + trainer_metrics["lambda_off"] * trainer_metrics["offline_loss"]
            )

        iter_time = time.time() - iter_start
        offline_time = offline_fetch_time if offline_fetch_time is not None else 0.0
        other_time = iter_time - offline_time - rollout_time - opt_time
        
        # Warn if iteration is taking too long
        if iter_time > 120:  # More than 2 minutes per iteration
            print(
                f"[SE-FM] [WARN] Iteration {iteration} took {iter_time:.1f}s (>2min). "
                f"Breakdown: offline={offline_time:.1f}s, rollout={rollout_time:.1f}s, "
                f"opt={opt_time:.1f}s, other={other_time:.1f}s",
                flush=True,
            )
        
        # Calculate progress and ETA
        progress = (iteration / cfg.num_iterations) * 100
        
        # Use a simple moving average of last few iterations for ETA
        if not hasattr(run_training, '_iter_times'):
            run_training._iter_times = []
        run_training._iter_times.append(iter_time)
        if len(run_training._iter_times) > 10:
            run_training._iter_times.pop(0)
        
        if len(run_training._iter_times) >= 2:
            avg_iter_time = sum(run_training._iter_times) / len(run_training._iter_times)
            remaining_iters = cfg.num_iterations - iteration
            eta_seconds = avg_iter_time * remaining_iters
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)
        else:
            eta_hours = 0
            eta_mins = 0
        
        # Progress bar
        bar_length = 30
        filled = int(bar_length * iteration / cfg.num_iterations)
        bar = "=" * filled + "-" * (bar_length - filled)
        
        # Always print iteration summary with progress bar
        print(
            f"\n[Iter {iteration:4d}/{cfg.num_iterations}] [{bar}] {progress:5.1f}% | "
            f"offline_loss={trainer_metrics.get('offline_loss', offline_loss.item()):.4f} | "
            f"online_loss={trainer_metrics.get('online_loss', 0.0):.4f} | "
            f"total_loss={trainer_metrics.get('total_loss', trainer_metrics.get('loss', 0.0)):6.4f} | "
            f"time={iter_time:5.1f}s",
            end="",
            flush=True,
        )
        
        if len(run_training._iter_times) >= 2:
            print(f" | ETA: {eta_hours}h{eta_mins}m", end="", flush=True)
        print("", flush=True)  # New line
        
        if iteration % cfg.log_interval == 0:
            # Print detailed summary at log_interval
            print(
                f"  └─ Details: "
                f"bc_loss={offline_loss_dict.get('bc_loss', 0.0):.4f} | "
                f"lambda_off={trainer_metrics.get('lambda_off', cfg.lambda_off):.3f} | "
                f"lambda_on={trainer_metrics.get('lambda_on', cfg.lambda_on):.3f} | "
                f"success={rollout_stats['episode_success']:.3f} | "
                f"chunk_success={rollout_stats['chunk_success_rate']:.3f} | "
                f"return={rollout_stats['episode_return']:.3f} | "
                f"breakdown: offline={offline_time:.1f}s, rollout={rollout_time:.1f}s, "
                f"opt={opt_time:.1f}s, other={other_time:.1f}s",
                flush=True,
            )
            
            if use_wandb:
                wandb.log(
                    {
                        "iteration": iteration,
                        **trainer_metrics,
                        **rollout_stats,
                    },
                    step=iteration,
                )

        if iteration % cfg.save_interval == 0 or iteration == cfg.num_iterations:
            save_path = Path(cfg.output_dir) / f"se_fm_action_head_step{iteration:06d}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": action_head.state_dict(),
                    "iteration": iteration,
                    "timestamp": DATE_TIME,
                    "lambda_off": cfg.lambda_off,
                    "lambda_on": cfg.lambda_on,
                },
                save_path,
            )
            print(f"[SE-FM] Checkpoint saved: {save_path.name}", flush=True)

    if use_wandb and wandb.run is not None:
        wandb.finish()


def parse_args() -> SEFMConfig:
    parser = argparse.ArgumentParser(description="Self-Evolving Flow Matching trainer")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/se_fm")
    parser.add_argument("--task_suite", type=str, default="libero_spatial", choices=[s.value for s in TaskSuite])
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument(
        "--offline_dataset_path",
        type=str,
        default="data/libero/libero_spatial",
        help="Root directory containing the RLDS-formatted LIBERO spatial dataset.",
    )
    parser.add_argument(
        "--offline_dataset_name",
        type=str,
        default="libero_spatial",
        help="RLDS dataset name to load for offline demonstrations.",
    )
    parser.add_argument("--offline_batch_size", type=int, default=32)
    parser.add_argument("--offline_shuffle_buffer_size", type=int, default=256_000)
    parser.add_argument("--offline_use_minivlm", action="store_true")
    parser.add_argument("--offline_no_minivlm", dest="offline_use_minivlm", action="store_false")
    parser.add_argument("--offline_use_proprio", action="store_true")
    parser.add_argument("--offline_no_proprio", dest="offline_use_proprio", action="store_false")
    parser.add_argument("--offline_use_wrist", action="store_true")
    parser.add_argument("--offline_no_wrist", dest="offline_use_wrist", action="store_false")
    parser.add_argument("--max_env_steps", type=int, default=None)
    parser.add_argument("--lambda_off", type=float, default=1.0)
    parser.add_argument("--lambda_on", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--normalize_advantage", action="store_true")
    parser.add_argument("--no_normalize_advantage", dest="normalize_advantage", action="store_false")
    parser.add_argument("--reward_baseline", type=str, default="mean", choices=["mean", "none"])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    parser.set_defaults(
        offline_use_minivlm=True,
        offline_use_proprio=True,
        offline_use_wrist=True,
        normalize_advantage=True,
    )

    args = parser.parse_args()

    return SEFMConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        output_dir=args.output_dir,
        task_suite=args.task_suite,
        num_iterations=args.num_iterations,
        device=args.device,
        seed=args.seed,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        offline_dataset_path=args.offline_dataset_path,
        offline_dataset_name=args.offline_dataset_name,
        offline_batch_size=args.offline_batch_size,
        offline_shuffle_buffer_size=args.offline_shuffle_buffer_size,
        max_env_steps=args.max_env_steps,
        lambda_off=args.lambda_off,
        lambda_on=args.lambda_on,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        normalize_advantage=args.normalize_advantage,
        reward_baseline=args.reward_baseline,
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        offline_use_minivlm=args.offline_use_minivlm,
        offline_use_proprio=args.offline_use_proprio,
        offline_use_wrist=args.offline_use_wrist,
    )


def main() -> None:
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()

