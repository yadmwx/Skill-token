"""
Flow Matching action head using DiT-X (canonical DiT) architecture.

This implementation uses a complete, canonical DiT (Diffusion Transformer) architecture,
not a simple transformer imitation. The DiT-X blocks include:

**Canonical DiT Features**:
- Proper AdaLN (Adaptive Layer Normalization) with time embedding modulation
- Time-dependent processing at every layer
- Diffusion-specific design patterns

**VLA-Adapter Enhancements**:
- Self-attention for action sequence modeling
- Cross-attention for VLA conditioning (vision + language + proprio)
- Extended AdaLN modulation (9 parameters for 3 components)
- Dual time encoding (t and r) for Flow Matching
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prismatic.models.action_heads import L1RegressionActionHead, VelocityNetwork
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM


@dataclass
class FlowMatchingConfig:
    """Configuration switches for Flow/MeanFlow behaviour."""

    flow_ratio: float = 1.00  # Ratio of batch for flow loss
    consistency_batch_ratio: float = 0.00  # Ratio of batch for consistency loss
    time_dist: Tuple[str, float, float] = ("lognorm", -0.4, 1.0)
    sample_t_mode_flow: str = "beta"  # Sampling mode for flow: "uniform", "lognorm", "beta", etc.
    sample_t_mode_consistency: str = "discrete"  # Sampling mode for consistency: "discrete", "uniform", etc.
    sample_dt_mode_consistency: str = "uniform"  # Sampling mode for dt in consistency training
    sample_target_t_mode: str = "relative"  # "relative" or "absolute"
    denoise_timesteps: int = 10  # Number of discrete timesteps for consistency training
    cfg_ratio: float = 0.1
    cfg_scale: float = 2.0
    inference_default_mode: str = "ode"  # options: ode / sde / consistency


class FlowMatchingActionHead(nn.Module):
    """
    Flow Matching action head using DiT-X (canonical DiT) architecture.
    
    This is a complete, canonical DiT implementation, not a simple transformer imitation.
    It includes proper AdaLN modulation, time embedding injection, and diffusion-specific
    design patterns that distinguish it from naive transformer approaches.
    
    **Canonical DiT Features**:
    - Proper AdaLN with time-dependent modulation at every layer
    - Time embedding integration following DiT design patterns
    - Diffusion-specific initialization and normalization
    
    **VLA-Adapter Enhancements**:
    - Cross-attention for multi-modal conditioning (vision + language + proprio)
    - Dual time encoding (t and r) for Flow Matching
    - Extended AdaLN modulation (9 parameters)
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        num_task_tokens: int = 512,
        num_blocks: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_pro_version: bool = False,
        flow_ratio: float = 1.0,  # Default to 1.0 (only flow loss, no consistency) to match FlowMatchingConfig
        time_dist: Tuple[str, float, float] = ("lognorm", -0.4, 1.0),
        sample_t_mode_flow: str = "beta",
        sample_t_mode_consistency: str = "discrete",
        sample_dt_mode_consistency: str = "uniform",
        sample_target_t_mode: str = "relative",
        inference_default_mode: str = "ode",
        num_inference_steps: int = 5,
        num_inference_samples: int = 1,
        supervised_anchor_weight: float = 0.0,
        anchor_blend: float = 0.0,
        inference_residual_scale: float = 1.0,
        anchor_gripper_weight: float = 1.0,
        anchor_gripper_bce_weight: float = 0.0,
        flow_xyz_loss_weight: float = 1.0,
        flow_rot_loss_weight: float = 1.0,
        flow_gripper_loss_weight: float = 1.0,
        flow_gripper_bce_weight: float = 0.0,
        flow_gripper_bce_logit_scale: float = 1.0,
        flow_gripper_bce_balanced: bool = False,
        gripper_head_weight: float = 0.0,
        gripper_head_override: bool = False,
        clip_normalized_actions: bool = False,
        detach_flow_conditioning: bool = False,
        use_adaptive_bridge: bool = True,
        bridge_mode: str = "adaptive",
        fixed_layer_index: int = -1,
        use_state_conditioning: bool = False,
        state_scale_mode: str = "none",
        state_proprio_mode: str = "concat",
        state_use_chunk_pos: bool = False,
        state_include_task_tokens: bool = False,
        condition_mode: str = "full",
        condition_injection_mode: str = "cross_attn",
        include_prompt_tokens: bool = False,
        task_token_mode: str = "vision_prompt",
        use_latent_skill_token: bool = False,
        num_skill_tokens: int = 16,
        skill_token_dim: int = 128,
        skill_temperature: float = 1.0,
        dit_zero_init_adaln: bool = True,
        dit_zero_init_output: bool = True,
        dense_film_enabled: bool = False,
        dense_film_max_layers: int = 64,
        dense_film_first_layer_index: int = 1,
        dense_film_bottleneck_dim: int = 64,
        dense_film_state_dim: int = 128,
    ) -> None:
        super().__init__()

        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_inference_steps = num_inference_steps
        self.num_inference_samples = max(int(num_inference_samples), 1)
        self.supervised_anchor_weight = float(supervised_anchor_weight)
        self.anchor_blend = float(anchor_blend)
        self.inference_residual_scale = float(inference_residual_scale)
        self.anchor_gripper_weight = float(anchor_gripper_weight)
        self.anchor_gripper_bce_weight = float(anchor_gripper_bce_weight)
        self.flow_xyz_loss_weight = float(flow_xyz_loss_weight)
        self.flow_rot_loss_weight = float(flow_rot_loss_weight)
        self.flow_gripper_loss_weight = float(flow_gripper_loss_weight)
        self.flow_gripper_bce_weight = float(flow_gripper_bce_weight)
        self.flow_gripper_bce_logit_scale = float(flow_gripper_bce_logit_scale)
        self.flow_gripper_bce_balanced = bool(flow_gripper_bce_balanced)
        self.gripper_head_weight = float(gripper_head_weight)
        self.gripper_head_override = bool(gripper_head_override)
        self.clip_normalized_actions = bool(clip_normalized_actions)
        self.detach_flow_conditioning = bool(detach_flow_conditioning)
        self.disable_inference_anchor = False
        self.pure_inference = False
        self.use_state_conditioning = bool(use_state_conditioning)
        self.state_scale_mode = str(state_scale_mode)
        self.state_proprio_mode = str(state_proprio_mode)
        self.state_use_chunk_pos = bool(state_use_chunk_pos)
        self.state_include_task_tokens = bool(state_include_task_tokens)
        self.condition_mode = str(condition_mode)
        if self.condition_mode not in {"full", "task_only"}:
            raise ValueError(f"Unsupported DIT condition_mode: {self.condition_mode}")
        self.condition_injection_mode = str(condition_injection_mode)
        if self.condition_injection_mode not in {"cross_attn", "joint_prefix", "action_expert_prefix"}:
            raise ValueError(f"Unsupported DIT condition_injection_mode: {self.condition_injection_mode}")
        self.include_prompt_tokens = bool(include_prompt_tokens)
        self.task_token_mode = str(task_token_mode)
        if self.task_token_mode not in {"vision_prompt", "vision_only", "prompt_only", "last_prompt"}:
            raise ValueError(f"Unsupported DIT task_token_mode: {self.task_token_mode}")
        self.debug_use_state_conditioning = False
        self.debug_state_scale_mode = "none"
        if self.state_use_chunk_pos:
            self.state_chunk_pos_emb = nn.Parameter(torch.zeros(1, NUM_ACTIONS_CHUNK, hidden_dim))
            nn.init.normal_(self.state_chunk_pos_emb, std=0.02)
        else:
            self.state_chunk_pos_emb = None

        # Velocity network (DiT-X, canonical DiT) - includes time encoders
        self.velocity_network = VelocityNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_blocks=num_blocks,
            num_task_tokens=num_task_tokens,
            use_adaptive_bridge=use_adaptive_bridge,
            bridge_mode=bridge_mode,
            fixed_layer_index=fixed_layer_index,
            condition_mode=self.condition_mode,
            condition_injection_mode=self.condition_injection_mode,
            use_latent_skill_token=use_latent_skill_token,
            num_skill_tokens=num_skill_tokens,
            skill_token_dim=skill_token_dim,
            skill_temperature=skill_temperature,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dit_zero_init_adaln=dit_zero_init_adaln,
            dit_zero_init_output=dit_zero_init_output,
            dense_film_enabled=dense_film_enabled,
            dense_film_max_layers=dense_film_max_layers,
            dense_film_first_layer_index=dense_film_first_layer_index,
            dense_film_bottleneck_dim=dense_film_bottleneck_dim,
            dense_film_state_dim=dense_film_state_dim,
        )
        
        # Access time encoders from DiT-X model
        self.time_encoder = self.velocity_network.ditx.flow_timestep_encoder
        self.target_t_encoder = self.velocity_network.ditx.flow_target_t_encoder

        anchor_enabled = self.supervised_anchor_weight > 0.0 or self.anchor_blend > 0.0
        self.anchor_head = (
            L1RegressionActionHead(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                num_task_tokens=num_task_tokens,
                use_pro_version=use_pro_version,
            )
            if anchor_enabled
            else None
        )
        self.state_anchor_proj = (
            nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, action_dim))
            if anchor_enabled
            else None
        )
        gripper_head_enabled = self.gripper_head_override or self.gripper_head_weight > 0.0
        self.gripper_head = (
            nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 1))
            if gripper_head_enabled
            else None
        )
        
        self.flow_cfg = FlowMatchingConfig(
            flow_ratio=flow_ratio,
            consistency_batch_ratio=1.0 - flow_ratio,  # Ensure they sum to 1.0
            time_dist=time_dist,
            sample_t_mode_flow=sample_t_mode_flow,
            sample_t_mode_consistency=sample_t_mode_consistency,
            sample_dt_mode_consistency=sample_dt_mode_consistency,
            sample_target_t_mode=sample_target_t_mode,
            inference_default_mode=inference_default_mode,
        )

    def _select_task_and_action_tokens(
        self,
        actions_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        action_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]
        task_valid_mask = task_hidden_states.float().abs().sum(dim=-1) > 0
        velocity_network = self.velocity_network
        num_layers = task_hidden_states.shape[1]
        layer_idx = velocity_network.fixed_layer_index if velocity_network.fixed_layer_index >= 0 else num_layers // 2

        if not velocity_network.use_adaptive_bridge:
            task_tokens = task_hidden_states[:, layer_idx, :, :]
            action_tokens = action_hidden_states[:, layer_idx, :, :]
        elif velocity_network.bridge_mode == "uniform":
            task_tokens = task_hidden_states.mean(dim=1)
            action_tokens = action_hidden_states.mean(dim=1)
        elif velocity_network.bridge_mode in {"adaptive", "adaptive_gated"}:
            task_tokens, layer_weights = velocity_network.task_layer_selector(
                task_hidden_states,
                task_mask=task_valid_mask,
            )
            weights = layer_weights.view(task_hidden_states.shape[0], num_layers, 1, 1)
            action_tokens = (weights * action_hidden_states).sum(dim=1)
        elif velocity_network.bridge_mode == "dense_film_residual":
            # Dense-FiLM fuses the residual inside VelocityNetwork.forward.
            # Debug extraction uses the stable final-layer baseline only.
            task_tokens = task_hidden_states[:, -1, :, :]
            action_tokens = action_hidden_states[:, -1, :, :]
        else:
            raise ValueError(f"Unsupported bridge_mode for debug state extraction: {velocity_network.bridge_mode}")

        return task_tokens, action_tokens

    def set_num_task_tokens(self, num_task_tokens: int, *_unused_legacy_args) -> None:
        """Update the runtime task/action split for packed hidden-state tensors."""
        self.num_task_tokens = int(num_task_tokens)
        if hasattr(self.velocity_network, "num_task_tokens"):
            self.velocity_network.num_task_tokens = int(num_task_tokens)
        if hasattr(self.velocity_network, "ditx") and hasattr(self.velocity_network.ditx, "num_task_tokens"):
            self.velocity_network.ditx.num_task_tokens = int(num_task_tokens)
        if self.anchor_head is not None and hasattr(self.anchor_head, "num_task_tokens"):
            self.anchor_head.num_task_tokens = int(num_task_tokens)

    def _extract_state_emb(
        self,
        actions_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _, action_tokens = self._select_task_and_action_tokens(actions_hidden_states)
        bsz, num_tokens, hidden_dim = action_tokens.shape
        if num_tokens % NUM_ACTIONS_CHUNK != 0:
            raise ValueError(
                f"Cannot align {num_tokens} action tokens into NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}"
            )
        state_emb = action_tokens.reshape(
            bsz,
            NUM_ACTIONS_CHUNK,
            num_tokens // NUM_ACTIONS_CHUNK,
            hidden_dim,
        ).mean(dim=2)

        scale_mode = getattr(self, "debug_state_scale_mode", "none")
        if scale_mode == "none":
            scale_mode = getattr(self, "state_scale_mode", "none")
        if scale_mode == "sqrt_group":
            group_size = num_tokens // NUM_ACTIONS_CHUNK
            state_emb = state_emb * math.sqrt(group_size)

        return state_emb

    def _extract_task_and_state_emb(
        self,
        actions_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        task_tokens, action_tokens = self._select_task_and_action_tokens(actions_hidden_states)
        bsz, num_tokens, hidden_dim = action_tokens.shape
        if num_tokens % NUM_ACTIONS_CHUNK != 0:
            raise ValueError(
                f"Cannot align {num_tokens} action tokens into NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}"
            )
        state_emb = action_tokens.reshape(
            bsz,
            NUM_ACTIONS_CHUNK,
            num_tokens // NUM_ACTIONS_CHUNK,
            hidden_dim,
        ).mean(dim=2)

        scale_mode = getattr(self, "debug_state_scale_mode", "none")
        if scale_mode == "none":
            scale_mode = getattr(self, "state_scale_mode", "none")
        if scale_mode == "sqrt_group":
            state_emb = state_emb * math.sqrt(num_tokens // NUM_ACTIONS_CHUNK)

        return task_tokens, state_emb

    # ------------------------------------------------------------------
    # Flow Matching helper utilities (adapted from meanflow.py)
    # ------------------------------------------------------------------

    def _anchor_l1_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = (predicted.float() - target.float()).abs()
        error[..., -1] *= self.anchor_gripper_weight
        loss = error.mean()
        if self.anchor_gripper_bce_weight > 0.0:
            gripper_target = (target[..., -1] > 0.0).to(dtype=torch.float32)
            gripper_logits = predicted[..., -1].float()
            gripper_bce = F.binary_cross_entropy_with_logits(gripper_logits, gripper_target)
            loss = loss + self.anchor_gripper_bce_weight * gripper_bce
        return loss

    def _flow_mse_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = (predicted.float() - target.float()).pow(2)
        if abs(self.flow_xyz_loss_weight - 1.0) > 1e-8:
            error[..., :3] *= self.flow_xyz_loss_weight
        if abs(self.flow_rot_loss_weight - 1.0) > 1e-8:
            error[..., 3:6] *= self.flow_rot_loss_weight
        if abs(self.flow_gripper_loss_weight - 1.0) > 1e-8:
            error[..., -1] *= self.flow_gripper_loss_weight
        return error.mean()

    def _flow_dim_metrics(self, predicted: torch.Tensor, target: torch.Tensor, prefix: str) -> dict:
        error = (predicted.float() - target.float()).pow(2)
        return {
            f"{prefix}_xyz_mse": error[..., :3].mean().detach(),
            f"{prefix}_rot_mse": error[..., 3:6].mean().detach(),
            f"{prefix}_gripper_mse": error[..., -1].mean().detach(),
        }

    def _flow_gripper_bce_loss(
        self,
        pred_velocity: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.flow_gripper_bce_weight <= 0.0:
            return pred_velocity.new_zeros(())
        batch_size = pred_velocity.shape[0]
        t_view = t.to(device=pred_velocity.device, dtype=pred_velocity.dtype).view(batch_size, 1, 1)
        pred_x1 = x_t.to(dtype=pred_velocity.dtype) + (1.0 - t_view) * pred_velocity
        gripper_target = (target_actions[..., -1] > 0.0).to(dtype=torch.float32)
        gripper_logits = pred_x1[..., -1].float() * self.flow_gripper_bce_logit_scale
        if not self.flow_gripper_bce_balanced:
            return F.binary_cross_entropy_with_logits(gripper_logits, gripper_target)

        # LIBERO gripper labels can be heavily imbalanced in a short action chunk.
        # Use per-batch inverse-frequency weights so BCE cannot be minimized by
        # collapsing to the majority open/close state.
        positive = gripper_target
        negative = 1.0 - gripper_target
        pos_frac = positive.mean().clamp_min(1e-3)
        neg_frac = negative.mean().clamp_min(1e-3)
        sample_weight = positive * (0.5 / pos_frac) + negative * (0.5 / neg_frac)
        return F.binary_cross_entropy_with_logits(
            gripper_logits,
            gripper_target,
            weight=sample_weight,
        )

    def _gripper_head_features(
        self,
        state_emb: torch.Tensor,
        proprio: Optional[torch.Tensor],
        proprio_projector: Optional[nn.Module],
    ) -> torch.Tensor:
        if self.gripper_head is None:
            raise RuntimeError("gripper_head is not initialized")
        dtype = next(self.gripper_head.parameters()).dtype
        state_emb = state_emb.to(dtype=dtype)
        if proprio is not None and proprio_projector is not None:
            batch_size = state_emb.shape[0]
            proprio = proprio.reshape(batch_size, -1).to(device=state_emb.device, dtype=dtype)
            proprio_features = proprio_projector(proprio).unsqueeze(1).to(dtype=dtype)
            state_emb = state_emb + proprio_features
        return state_emb

    def _gripper_head_logits_from_state(
        self,
        state_emb: torch.Tensor,
        proprio: Optional[torch.Tensor],
        proprio_projector: Optional[nn.Module],
    ) -> torch.Tensor:
        features = self._gripper_head_features(state_emb, proprio, proprio_projector)
        return self.gripper_head(features).squeeze(-1)

    def _gripper_head_loss_from_state(
        self,
        state_emb: torch.Tensor,
        target_actions: torch.Tensor,
        proprio: Optional[torch.Tensor],
        proprio_projector: Optional[nn.Module],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self._gripper_head_logits_from_state(state_emb, proprio, proprio_projector)
        target = (target_actions[..., -1] > 0.0).to(dtype=torch.float32, device=logits.device)
        loss = F.binary_cross_entropy_with_logits(logits.float(), target)
        return loss, logits.detach()

    def _sample_t(self, batch_size: int, device: torch.device, mode: str = "uniform") -> torch.Tensor:
        """
        Sample timestep t for flow or consistency training.
        
        Trick 1: Specialized time sampling
        - Flow uses Beta distribution (focuses on mid-to-late timesteps, more stable)
        - Consistency uses discrete timesteps (for consistency training)
        """
        if mode == "uniform":
            t = torch.rand((batch_size,), device=device, dtype=torch.float32)
        elif mode == "lognorm":
            mu, sigma = self.flow_cfg.time_dist[-2], self.flow_cfg.time_dist[-1]
            normal_samples = torch.randn(batch_size, device=device) * sigma + mu
            t = torch.sigmoid(normal_samples)
        elif mode == "beta":
            # Beta distribution sampling (alpha=2, beta=5) focuses on mid-to-late timesteps
            # This helps stabilize flow matching training
            alpha, beta = 2.0, 5.0
            t = torch.distributions.Beta(alpha, beta).sample((batch_size,)).to(device)
        elif mode == "discrete":
            # Discrete timesteps for consistency training
            t = torch.randint(low=0, high=self.flow_cfg.denoise_timesteps, size=(batch_size,), device=device).float()
            t = t / self.flow_cfg.denoise_timesteps
        else:
            raise ValueError(f"Unsupported sample_t_mode: {mode}")
        return t

    def _sample_dt(self, batch_size: int, device: torch.device, mode: str = "uniform") -> torch.Tensor:
        """Sample dt for consistency training."""
        if mode == "uniform":
            dt = torch.rand((batch_size,), device=device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported sample_dt_mode: {mode}")
        return dt

    def _linear_interpolate(self, noise: torch.Tensor, target: torch.Tensor, timestep: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """
        Linear interpolation between noise and target (Rectified Flow path).
        
        When epsilon=0.0, this implements the standard Rectified Flow linear path:
        x_t = (1-t) * x_0 + t * x_1
        
        This is Trick 2: Linear Rectified Flow path for stable training.
        """
        noise_coeff = 1.0 - (1.0 - epsilon) * timestep
        interpolated = noise_coeff * noise + timestep * target
        return interpolated

    def _sample_t_r(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample time pairs (t, r) following the MeanFlow strategy.

        Returns:
            t, r tensors of shape (batch_size,)
        """
        dist_type = self.flow_cfg.time_dist[0]
        if dist_type == "uniform":
            samples = np.random.rand(batch_size, 2).astype(np.float32)
        elif dist_type == "lognorm":
            mu, sigma = self.flow_cfg.time_dist[-2], self.flow_cfg.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # sigmoid
        else:
            raise ValueError(f"Unsupported time distribution: {dist_type}")

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_cfg.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    # ------------------------------------------------------------------
    # Flow Matching / MeanFlow losses (placeholders)
    # ------------------------------------------------------------------

    def predict_velocity(
        self,
        z: torch.Tensor,  # (batch, num_chunks, action_dim)
        t: torch.Tensor,  # (batch,)
        r: torch.Tensor,  # (batch,)
        actions_hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        proprio_projector: nn.Module,
        detach_conditioning: bool = False,
    ) -> torch.Tensor:
        """
        Predict velocity field v_θ(z_t, t, r, conditioning).
        
        Trick 3: Unified network interface compatible with both training and ODE sampling.
        This same function is used during:
        - Training: for flow loss and consistency loss
        - Inference: for ODE/SDE sampling
        """
        batch_size = actions_hidden_states.shape[0]
        device = z.device
        
        # Ensure t and r are tensors
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device, dtype=torch.float32)
        if not torch.is_tensor(r):
            r = torch.tensor([r], device=device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if r.dim() == 0:
            r = r.unsqueeze(0)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size)
        if r.shape[0] == 1 and batch_size > 1:
            r = r.expand(batch_size)
        
        # Encode time steps (t and r are in [0, 1])
        # DiT-X uses SinusoidalPosEmb which accepts continuous values directly
        # Ensure dtype matches model dtype (bfloat16 for mixed precision)
        model_dtype = next(self.time_encoder.parameters()).dtype
        t_emb = self.time_encoder(t.to(dtype=model_dtype))  # (batch, hidden_dim)
        r_emb = self.target_t_encoder(r.to(dtype=model_dtype))  # (batch, hidden_dim)
        
        # Project proprio (same behavior as L1RegressionActionHead)
        if proprio is not None:
            proprio = proprio.reshape(batch_size, -1).to(dtype=actions_hidden_states.dtype)  # (bsz, proprio_dim)
        else:
            proprio_dim = getattr(proprio_projector, "proprio_dim", self.hidden_dim)
            proprio = torch.zeros(
                batch_size,
                proprio_dim,
                device=z.device,
                dtype=actions_hidden_states.dtype,
            )

        proprio_features = proprio_projector(proprio).unsqueeze(1)  # (batch, 1, hidden_dim)
        if detach_conditioning:
            actions_hidden_states = actions_hidden_states.detach()
            proprio_features = proprio_features.detach()
        
        # Extract task and action hidden states
        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        action_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]
        
        # Ensure z has the correct dtype to match model weights
        z = z.to(dtype=model_dtype)
        
        # Predict velocity
        velocity = self.velocity_network(
            z,
            t_emb,
            r_emb,
            task_hidden_states,
            action_hidden_states,
            proprio_features,
        )
        
        return velocity

    def predict_velocity_from_state(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        state_emb: torch.Tensor,
        proprio: Optional[torch.Tensor],
        proprio_projector: Optional[nn.Module],
        detach_conditioning: bool = False,
        task_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity from a pre-aggregated depth-interface state."""
        batch_size = state_emb.shape[0]
        device = z.device
        model_dtype = next(self.time_encoder.parameters()).dtype

        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device, dtype=torch.float32)
        if not torch.is_tensor(r):
            r = torch.tensor([r], device=device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if r.dim() == 0:
            r = r.unsqueeze(0)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size)
        if r.shape[0] == 1 and batch_size > 1:
            r = r.expand(batch_size)

        t_emb = self.time_encoder(t.to(dtype=model_dtype))
        r_emb = self.target_t_encoder(r.to(dtype=model_dtype))

        state_emb = state_emb.to(dtype=model_dtype)
        if self.state_use_chunk_pos:
            if self.state_chunk_pos_emb is None:
                raise ValueError("state_use_chunk_pos=True requires state_chunk_pos_emb to be initialized")
            if state_emb.shape[1] != self.state_chunk_pos_emb.shape[1]:
                raise ValueError(
                    "state_use_chunk_pos expects chunk-aligned state tokens with "
                    f"length={self.state_chunk_pos_emb.shape[1]}, got {state_emb.shape[1]}"
                )
            state_emb = state_emb + self.state_chunk_pos_emb.to(
                device=state_emb.device, dtype=state_emb.dtype
            )
        if proprio is not None and proprio_projector is not None:
            proprio = proprio.reshape(batch_size, -1).to(device=state_emb.device, dtype=state_emb.dtype)
            proprio_features = proprio_projector(proprio).unsqueeze(1).to(dtype=state_emb.dtype)
            if self.state_proprio_mode == "add":
                context = state_emb + proprio_features
                context_mask = state_emb.float().abs().sum(dim=-1) > 0
            elif self.state_proprio_mode == "concat":
                context = torch.cat([state_emb, proprio_features], dim=1)
                state_mask = state_emb.float().abs().sum(dim=-1) > 0
                proprio_mask = torch.ones(
                    proprio_features.shape[:2],
                    device=proprio_features.device,
                    dtype=torch.bool,
                )
                context_mask = torch.cat([state_mask, proprio_mask], dim=1)
            else:
                raise ValueError(f"Unsupported state_proprio_mode: {self.state_proprio_mode}")
        else:
            context = state_emb
            context_mask = state_emb.float().abs().sum(dim=-1) > 0
        if task_tokens is not None:
            task_tokens = task_tokens.to(device=context.device, dtype=context.dtype)
            context = torch.cat([task_tokens, context], dim=1)
            task_mask = task_tokens.float().abs().sum(dim=-1) > 0
            context_mask = torch.cat([task_mask, context_mask], dim=1)
        if detach_conditioning:
            context = context.detach()

        timestep = torch.zeros(batch_size, device=z.device)
        target_t = torch.zeros(batch_size, device=z.device)
        return self.velocity_network.ditx(
            sample=z.to(dtype=model_dtype),
            timestep=timestep,
            target_t=target_t,
            vis_cond=context,
            vis_cond_mask=context_mask,
            timestep_emb=t_emb,
            target_t_emb=r_emb,
        )
    
    def _get_flow_velocity(
        self,
        actions: torch.Tensor,
        actions_hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor],
        proprio_projector: torch.nn.Module,
    ) -> dict:
        """Get flow velocity targets for training."""
        flow_batchsize = actions.shape[0]
        device = actions.device
        model_dtype = next(self.time_encoder.parameters()).dtype
        
        # Sample t for flow (dt is zero for flow)
        t_flow = self._sample_t(flow_batchsize, device, mode=self.flow_cfg.sample_t_mode_flow)
        t_flow = t_flow.view(-1, 1, 1).to(dtype=model_dtype)
        dt_flow = torch.zeros((flow_batchsize,), device=device, dtype=model_dtype)
        
        # Get target timestep
        if self.flow_cfg.sample_target_t_mode == "absolute":
            target_t_flow = t_flow.squeeze() + dt_flow
        else:  # relative
            target_t_flow = dt_flow
        
        # Compute interpolated data points
        x_0_flow = torch.randn_like(actions).to(dtype=model_dtype)
        x_1_flow = actions.to(dtype=model_dtype)
        x_t_flow = self._linear_interpolate(x_0_flow, x_1_flow, t_flow, epsilon=0.0)
        v_t_flow = x_1_flow - x_0_flow
        
        return {
            'x_t': x_t_flow,
            't': t_flow.squeeze(),
            'target_t': target_t_flow,
            'v_target': v_t_flow,
        }
    
    def _get_consistency_velocity(
        self,
        actions: torch.Tensor,
        actions_hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor],
        proprio_projector: torch.nn.Module,
        ema_model: Optional[nn.Module] = None,
    ) -> dict:
        """Get consistency velocity targets for training."""
        consistency_batchsize = actions.shape[0]
        device = actions.device
        model_dtype = next(self.time_encoder.parameters()).dtype
        
        # Sample t and dt for consistency training
        t_ct = self._sample_t(consistency_batchsize, device, mode=self.flow_cfg.sample_t_mode_consistency)
        t_ct = t_ct.view(-1, 1, 1).to(dtype=model_dtype)
        delta_t1 = self._sample_dt(consistency_batchsize, device, mode=self.flow_cfg.sample_dt_mode_consistency).to(dtype=model_dtype)
        delta_t2 = delta_t1.clone()  # Use same delta_t
        
        # Compute next timestep
        t_next = t_ct.squeeze() + delta_t1
        t_next = torch.clamp(t_next, max=1.0)
        t_next = t_next.view(-1, 1, 1).to(dtype=model_dtype)
        
        # Compute target timestep
        t_next_flat = t_next.squeeze(-1).squeeze(-1)  # (batch_size,)
        if self.flow_cfg.sample_target_t_mode == "absolute":
            target_t_next = t_next_flat + delta_t2
        else:  # relative
            target_t_next = delta_t2
        
        # Compute interpolated data points
        x0_ct = torch.randn_like(actions).to(dtype=model_dtype)
        x1_ct = actions.to(dtype=model_dtype)
        x_t_ct = self._linear_interpolate(x0_ct, x1_ct, t_ct, epsilon=0.0)
        x_t_next = self._linear_interpolate(x0_ct, x1_ct, t_next, epsilon=0.0)
        
        # Predict average velocity from t_next toward target using EMA model or current model
        # Always use no_grad for consistency target generation (this is just for generating target, no gradients needed)
        model_to_use = ema_model if ema_model is not None else self
        with torch.no_grad():
            # Extract hidden states for consistency batch
            num_layers = actions_hidden_states.shape[1]
            layer_idx = num_layers // 2
            task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
            action_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]
            
            # Prepare proprio features
            if proprio is not None:
                proprio_reshaped = proprio.reshape(consistency_batchsize, -1).to(dtype=actions_hidden_states.dtype)
                proprio_features = proprio_projector(proprio_reshaped).unsqueeze(1)
            else:
                proprio_features = torch.zeros(consistency_batchsize, 1, self.hidden_dim, device=device, dtype=model_dtype)
            
            # Encode time steps (ensure at least 1D: (batch_size,))
            t_next_for_encoder = t_next_flat.to(dtype=model_dtype)
            target_t_next_for_encoder = target_t_next.to(dtype=model_dtype)
            if t_next_for_encoder.dim() == 0:
                t_next_for_encoder = t_next_for_encoder.unsqueeze(0)
            if target_t_next_for_encoder.dim() == 0:
                target_t_next_for_encoder = target_t_next_for_encoder.unsqueeze(0)
            
            t_emb = model_to_use.time_encoder(t_next_for_encoder)
            r_emb = model_to_use.target_t_encoder(target_t_next_for_encoder)
            
            # Predict velocity using velocity_network (no gradients needed for target generation)
            v_avg_to_next_target = model_to_use.velocity_network(
                x_t_next, t_emb, r_emb,
                task_hidden_states, action_hidden_states,
                proprio_features,
            )
        
        # Predict target data point using average velocity
        pred_x1_ct = x_t_next + (1 - t_next) * v_avg_to_next_target
        # Estimate velocity at t using predicted endpoint
        v_ct = (pred_x1_ct - x_t_ct) / (1 - t_ct + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Target timestep for current timestep t
        target_t_ct = delta_t1 if self.flow_cfg.sample_target_t_mode == "relative" else t_next_flat
        
        return {
            'x_t': x_t_ct,
            't': t_ct.squeeze(),
            'target_t': target_t_ct,
            'v_target': v_ct,
        }

    def flow_matching_loss(
        self,
        actions_hidden_states: torch.Tensor,
        target_actions: torch.Tensor,
        proprio: Optional[torch.Tensor],
        proprio_projector: torch.nn.Module,
        mode: str = "flow_matching",
        ema_model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Flow Matching loss with both flow loss and consistency loss.
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses and metrics
        """
        if self.use_state_conditioning:
            task_tokens = None
            if self.state_include_task_tokens:
                task_tokens, state_emb = self._extract_task_and_state_emb(actions_hidden_states)
            else:
                state_emb = self._extract_state_emb(actions_hidden_states)
            return self.flow_matching_loss_from_state(
                state_emb,
                target_actions,
                proprio=proprio,
                proprio_projector=proprio_projector,
                task_tokens=task_tokens,
            )

        batch_size = target_actions.shape[0]
        device = target_actions.device
        model_dtype = next(self.time_encoder.parameters()).dtype
        
        # Split batch for flow and consistency training
        # NOTE: keep the original simple split to avoid complicated corner cases
        # with very small batch sizes. For small batches, this may result in
        # either the flow or consistency branch getting 0 elements, which is
        # acceptable and was the behavior in the original implementation.
        flow_batchsize = int(batch_size * self.flow_cfg.flow_ratio)
        consistency_batchsize = batch_size - flow_batchsize
        
        loss = 0.0
        loss_dict = {}
        anchor_actions = None
        flow_target_actions = target_actions
        if self.anchor_head is not None and self.supervised_anchor_weight > 0.0:
            anchor_actions = self.anchor_head.predict_action(
                actions_hidden_states,
                proprio,
                proprio_projector,
                phase="Inference",
            )
            flow_target_actions = target_actions - anchor_actions.detach().to(dtype=target_actions.dtype)
        
        # Flow loss
        if flow_batchsize > 0:
            flow_target_dict = self._get_flow_velocity(
                flow_target_actions[:flow_batchsize],
                actions_hidden_states[:flow_batchsize],
                proprio[:flow_batchsize] if proprio is not None else None,
                proprio_projector,
            )
            
            v_flow_pred = self.predict_velocity(
                flow_target_dict['x_t'],
                flow_target_dict['t'],
                flow_target_dict['target_t'],
                actions_hidden_states[:flow_batchsize],
                proprio[:flow_batchsize] if proprio is not None else None,
                proprio_projector,
                detach_conditioning=self.detach_flow_conditioning,
            )
            
            v_flow_target = flow_target_dict['v_target']
            loss_flow = self._flow_mse_loss(v_flow_pred, v_flow_target)
            if self.flow_gripper_bce_weight > 0.0:
                gripper_bce = self._flow_gripper_bce_loss(
                    v_flow_pred,
                    flow_target_dict['x_t'],
                    flow_target_dict['t'],
                    flow_target_actions[:flow_batchsize],
                )
                loss_flow = loss_flow + self.flow_gripper_bce_weight * gripper_bce.to(dtype=loss_flow.dtype)
                loss_dict['dit_flow_gripper_bce'] = gripper_bce.detach()
            loss += loss_flow
            loss_dict['loss_flow'] = loss_flow.item()
            loss_dict['v_flow_pred_magnitude'] = torch.sqrt(torch.mean(v_flow_pred ** 2)).item()
            loss_dict.update(self._flow_dim_metrics(v_flow_pred, v_flow_target, "dit_flow_velocity"))
        
        # Consistency loss
        if consistency_batchsize > 0:
            consistency_target_dict = self._get_consistency_velocity(
                flow_target_actions[flow_batchsize:],
                actions_hidden_states[flow_batchsize:],
                proprio[flow_batchsize:] if proprio is not None else None,
                proprio_projector,
                ema_model=ema_model,
            )
            
            v_ct_pred = self.predict_velocity(
                consistency_target_dict['x_t'],
                consistency_target_dict['t'],
                consistency_target_dict['target_t'],
                actions_hidden_states[flow_batchsize:],
                proprio[flow_batchsize:] if proprio is not None else None,
                proprio_projector,
                detach_conditioning=self.detach_flow_conditioning,
            )
            
            v_ct_target = consistency_target_dict['v_target']
            loss_ct = F.mse_loss(v_ct_pred, v_ct_target, reduction='mean')
            loss += loss_ct
            loss_dict['loss_ct'] = loss_ct.item()
            loss_dict['v_ct_pred_magnitude'] = torch.sqrt(torch.mean(v_ct_pred ** 2)).item()
        
        loss_dict['flow_matching_loss'] = loss.detach()
        if anchor_actions is not None:
            anchor_loss = self._anchor_l1_loss(anchor_actions, target_actions)
            loss = loss + self.supervised_anchor_weight * anchor_loss.to(dtype=loss.dtype)
            loss_dict['dit_anchor_l1_loss'] = anchor_loss.detach()
            loss_dict['flow_matching_total_loss'] = loss.detach()
        if self.gripper_head is not None and self.gripper_head_weight > 0.0:
            gripper_state_emb = self._extract_state_emb(actions_hidden_states)
            gripper_loss, gripper_logits = self._gripper_head_loss_from_state(
                gripper_state_emb,
                target_actions,
                proprio,
                proprio_projector,
            )
            loss = loss + self.gripper_head_weight * gripper_loss.to(dtype=loss.dtype)
            loss_dict["dit_gripper_head_bce"] = gripper_loss.detach()
            loss_dict["dit_gripper_head_logit_mean"] = gripper_logits.float().mean().detach()
            loss_dict["flow_matching_total_loss"] = loss.detach()
        loss_dict['loss_value'] = loss.detach()

        return loss, loss_dict

    def flow_matching_loss_from_state(
        self,
        state_emb: torch.Tensor,
        target_actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
        aux_loss: Optional[torch.Tensor] = None,
        aux_metrics: Optional[dict] = None,
        task_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute rectified-flow loss from a pre-aggregated state sequence."""
        batch_size = target_actions.shape[0]
        device = target_actions.device
        model_dtype = next(self.time_encoder.parameters()).dtype
        if proprio is not None and proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio is not None and proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)

        t = torch.rand(batch_size, device=device, dtype=model_dtype)
        t_expanded = t.view(batch_size, 1, 1)
        x0 = torch.randn_like(target_actions, device=device, dtype=model_dtype)
        anchor_actions = None
        flow_target_actions = target_actions
        if self.state_anchor_proj is not None and self.supervised_anchor_weight > 0.0:
            anchor_actions = self.state_anchor_proj(state_emb)
            flow_target_actions = target_actions - anchor_actions.detach().to(dtype=target_actions.dtype)
        x1 = flow_target_actions.to(dtype=model_dtype)
        x_t = t_expanded * x1 + (1 - t_expanded) * x0
        v_target = x1 - x0

        if self.flow_cfg.sample_target_t_mode == "relative":
            r = torch.zeros_like(t)
        else:
            r = torch.ones_like(t)
        v_pred = self.predict_velocity_from_state(
            x_t,
            t,
            r,
            state_emb,
            proprio,
            proprio_projector,
            detach_conditioning=self.detach_flow_conditioning,
            task_tokens=task_tokens,
        )
        loss = self._flow_mse_loss(v_pred, v_target)
        if self.flow_gripper_bce_weight > 0.0:
            gripper_bce = self._flow_gripper_bce_loss(v_pred, x_t, t, flow_target_actions)
            loss = loss + self.flow_gripper_bce_weight * gripper_bce.to(dtype=loss.dtype)
        else:
            gripper_bce = None
        metrics = {
            "loss_flow": loss.detach(),
            "loss_value": loss.detach(),
            "flow_matching_loss": loss.detach(),
            "v_flow_pred_magnitude": torch.sqrt(torch.mean(v_pred ** 2)).detach(),
        }
        metrics.update(self._flow_dim_metrics(v_pred, v_target, "dit_flow_velocity"))
        if gripper_bce is not None:
            metrics["dit_flow_gripper_bce"] = gripper_bce.detach()
        if aux_metrics:
            metrics.update(aux_metrics)
        if anchor_actions is not None:
            anchor_loss = self._anchor_l1_loss(anchor_actions, target_actions)
            loss = loss + self.supervised_anchor_weight * anchor_loss.to(dtype=loss.dtype)
            metrics["dit_anchor_l1_loss"] = anchor_loss.detach()
            metrics["flow_matching_total_loss"] = loss.detach()
        if self.gripper_head is not None and self.gripper_head_weight > 0.0:
            gripper_loss, gripper_logits = self._gripper_head_loss_from_state(
                state_emb,
                target_actions,
                proprio,
                proprio_projector,
            )
            loss = loss + self.gripper_head_weight * gripper_loss.to(dtype=loss.dtype)
            metrics["dit_gripper_head_bce"] = gripper_loss.detach()
            metrics["dit_gripper_head_logit_mean"] = gripper_logits.float().mean().detach()
            metrics["flow_matching_total_loss"] = loss.detach()
        if aux_loss is not None:
            loss = loss + aux_loss.to(dtype=loss.dtype)
            metrics["flow_matching_total_loss"] = loss.detach()
        metrics["loss_value"] = loss.detach()
        return loss, metrics

    # ------------------------------------------------------------------
    # Sampling interfaces (ODE / SDE / Consistency)
    # ------------------------------------------------------------------

    def predict_action(
        self,
        actions_hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        proprio_projector: nn.Module,
        phase: str = "Inference",
    ) -> torch.Tensor:
        """Generate actions using ODE sampling."""
        return self.sample_action_with_mode(
            actions_hidden_states, proprio, proprio_projector, mode="ode"
        )
    
    def sample_action_with_mode(
        self,
        actions_hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        proprio_projector: torch.nn.Module,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """ODE/SDE sampling."""
        if mode is None:
            mode = self.flow_cfg.inference_default_mode

        if self.use_state_conditioning or getattr(self, "debug_use_state_conditioning", False):
            task_tokens = None
            if self.state_include_task_tokens:
                task_tokens, state_emb = self._extract_task_and_state_emb(actions_hidden_states)
            else:
                state_emb = self._extract_state_emb(actions_hidden_states)
            return self.predict_action_from_state(
                state_emb,
                proprio=proprio,
                proprio_projector=proprio_projector,
                task_tokens=task_tokens,
            )

        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        base_hidden_states = actions_hidden_states
        base_proprio = proprio
        num_samples = max(int(getattr(self, "num_inference_samples", 1)), 1)
        if num_samples > 1:
            actions_hidden_states = actions_hidden_states.repeat_interleave(num_samples, dim=0)
            if proprio is not None:
                proprio = proprio.repeat_interleave(num_samples, dim=0)
            batch_size *= num_samples
        
        # Start from random noise
        z = torch.randn(
            (batch_size, NUM_ACTIONS_CHUNK, self.action_dim),
            device=device,
            dtype=actions_hidden_states.dtype
        )
        
        # Training interpolates from noise at t=0 to actions at t=1.
        dt = 1.0 / self.num_inference_steps
        t_vals = torch.linspace(0.0, 1.0, self.num_inference_steps + 1, device=device)
        
        if mode == "ode":
            # ODE sampling (Euler method)
            for i in range(self.num_inference_steps):
                t = torch.full((batch_size,), t_vals[i], device=device)
                
                if self.flow_cfg.sample_target_t_mode == "relative":
                    # Pure flow training uses a zero relative target offset.
                    r = torch.zeros((batch_size,), device=device)
                else:
                    r = torch.full((batch_size,), t_vals[i + 1], device=device)
                
                # Predict velocity
                v = self.predict_velocity(
                    z, t, r,
                    actions_hidden_states,
                    proprio,
                    proprio_projector,
                )
                
                z = z + dt * v
        
        elif mode == "sde":
            # SDE sampling (add noise)
            for i in range(self.num_inference_steps):
                t = torch.full((batch_size,), t_vals[i], device=device)
                
                # Match training semantics
                if self.flow_cfg.sample_target_t_mode == "relative":
                    r = torch.zeros((batch_size,), device=device)
                else:
                    r = torch.full((batch_size,), t_vals[i + 1], device=device)
                
                v = self.predict_velocity(
                    z, t, r,
                    actions_hidden_states,
                    proprio,
                    proprio_projector,
                )
                
                z = z + dt * v
                # Add noise
                noise_scale = torch.sqrt(2 * dt)
                z = z + noise_scale * torch.randn_like(z)
        
        else:
            raise ValueError(f"Unknown sampling mode: {mode}")
        
        if num_samples > 1:
            z = z.reshape(-1, num_samples, NUM_ACTIONS_CHUNK, self.action_dim).mean(dim=1)
        if (
            self.anchor_head is not None
            and not getattr(self, "disable_inference_anchor", False)
            and not getattr(self, "pure_inference", False)
        ):
            anchor_actions = self.anchor_head.predict_action(
                base_hidden_states,
                base_proprio,
                proprio_projector,
                phase="Inference",
            )
            blend = max(0.0, min(self.anchor_blend, 1.0))
            residual_scale = max(0.0, float(self.inference_residual_scale))
            # The flow branch models an action residual around the supervised anchor.
            # Even when blend=0 we must reconstruct the full action as anchor + residual.
            z = anchor_actions + residual_scale * (1.0 - blend) * z

        if self.gripper_head is not None and self.gripper_head_override:
            gripper_state_emb = self._extract_state_emb(base_hidden_states)
            gripper_logits = self._gripper_head_logits_from_state(
                gripper_state_emb,
                base_proprio,
                proprio_projector,
            )
            z[..., -1] = torch.sigmoid(gripper_logits).to(dtype=z.dtype)

        if self.clip_normalized_actions:
            z = z.clamp(-1.0, 1.0)

        # Return final actions (always keep batch dimension)
        return z

    def predict_action_from_state(
        self,
        state_emb: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
        phase: str = "Inference",
        task_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = state_emb.shape[0]
        device = state_emb.device
        model_dtype = next(self.time_encoder.parameters()).dtype
        base_state_emb = state_emb
        base_proprio = proprio
        num_samples = max(int(getattr(self, "num_inference_samples", 1)), 1)
        if num_samples > 1:
            state_emb = state_emb.repeat_interleave(num_samples, dim=0)
            if task_tokens is not None:
                task_tokens = task_tokens.repeat_interleave(num_samples, dim=0)
            if proprio is not None:
                proprio = proprio.repeat_interleave(num_samples, dim=0)
            batch_size *= num_samples
        z = torch.randn(
            (batch_size, NUM_ACTIONS_CHUNK, self.action_dim),
            device=device,
            dtype=model_dtype,
        )
        dt = 1.0 / self.num_inference_steps
        t_vals = torch.linspace(0.0, 1.0, self.num_inference_steps + 1, device=device)
        for i in range(self.num_inference_steps):
            t = torch.full((batch_size,), t_vals[i], device=device)
            if self.flow_cfg.sample_target_t_mode == "relative":
                r = torch.zeros((batch_size,), device=device)
            else:
                r = torch.full((batch_size,), t_vals[i + 1], device=device)
            v = self.predict_velocity_from_state(
                z,
                t,
                r,
                state_emb,
                proprio,
                proprio_projector,
                task_tokens=task_tokens,
            )
            z = z + dt * v
        if num_samples > 1:
            z = z.reshape(-1, num_samples, NUM_ACTIONS_CHUNK, self.action_dim).mean(dim=1)
        if (
            self.state_anchor_proj is not None
            and not getattr(self, "disable_inference_anchor", False)
            and not getattr(self, "pure_inference", False)
        ):
            anchor_actions = self.state_anchor_proj(base_state_emb)
            blend = max(0.0, min(self.anchor_blend, 1.0))
            residual_scale = max(0.0, float(self.inference_residual_scale))
            z = anchor_actions + residual_scale * (1.0 - blend) * z
        if self.gripper_head is not None and self.gripper_head_override:
            gripper_logits = self._gripper_head_logits_from_state(
                base_state_emb,
                base_proprio,
                proprio_projector,
            )
            z[..., -1] = torch.sigmoid(gripper_logits).to(dtype=z.dtype)
        if self.clip_normalized_actions:
            z = z.clamp(-1.0, 1.0)
        return z


__all__ = ["FlowMatchingActionHead", "FlowMatchingConfig"]
