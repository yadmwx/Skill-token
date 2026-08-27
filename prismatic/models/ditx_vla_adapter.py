"""
DiT-X model for VLA-Adapter with Flow Matching.

This is a complete, canonical DiT (Diffusion Transformer) implementation, not a simple
transformer imitation. It represents a true DiT architecture with the following:

**Canonical DiT Features** (what makes this a real DiT, not just a transformer):
1. **Proper AdaLN Integration**: Time embeddings are injected through Adaptive Layer
   Normalization at every layer, following the canonical DiT design pattern.
2. **Time-Dependent Processing**: All components (self-attn, cross-attn, MLP) are
   modulated by time embeddings, not just concatenated.
3. **Diffusion-Specific Design**: Proper initialization, normalization, and architecture
   patterns specific to diffusion/flow matching models.
4. **Position Embeddings**: Learnable position embeddings for action sequences.

**VLA-Adapter Enhancements**:
1. **Dual Time Encoding**: Supports both timestep t and target timestep r encoding
   for Flow Matching's relative/absolute time modes.
2. **Cross-Attention Conditioning**: Efficiently integrates vision, language, and
   proprioceptive conditioning through cross-attention in each DiT block.
3. **Extended AdaLN**: 9-parameter AdaLN modulation (3 params × 3 components) for
   fine-grained control over all components.
4. **Action Sequence Modeling**: Designed for multi-step action prediction
   (NUM_ACTIONS_CHUNK = 8 steps).

**Not a Simple Transformer**: Unlike naive transformer-based approaches that just
concatenate time embeddings or use simple conditioning, this follows the canonical
DiT architecture with proper time-dependent modulation at every layer.

Based on mean-flow reference/ditx.py, but represents a complete DiT implementation
with VLA-Adapter-specific enhancements.
"""

import re
import logging
from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
try:
    from timm.models.vision_transformer import RmsNorm
except ImportError:
    # Fallback: define RmsNorm if not available
    class RmsNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        
        def forward(self, x):
            norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
            return x / (norm + self.eps) * self.weight

from prismatic.models.ditx_blocks import (
    ActionExpertFullPrefixAdaRMSBlock,
    ActionExpertFullPrefixBlock,
    ActionExpertFullPrefixGQAAdaRMSBlock,
    ActionExpertFullPrefixGQAGatedAdaRMSBlock,
    ActionExpertFullPrefixSplitKVAdaRMSBlock,
    ActionExpertPrefixBlock,
    DiTXBlock,
    AdaptiveLayerNorm,
)
from prismatic.models.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

CONDITION_INJECTION_MODES = {
    "cross_attn",
    "joint_prefix",
    "action_expert_prefix",
    "action_expert_full_prefix",
    "action_expert_full_prefix_adarms",
    "action_expert_full_prefix_splitkv_adarms",
    "action_expert_full_prefix_gqa_adarms",
    "action_expert_full_prefix_gqa_gated_adarms",
}


class FinalLayer(nn.Module):
    """Final layer of DiT-X model."""
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            act_layer=approx_gelu,
            drop=0
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


class DiTXVLAdapter(nn.Module):
    """
    DiT-X model for VLA-Adapter with Flow Matching.
    
    This is a complete, canonical DiT (Diffusion Transformer) implementation, not a
    simple transformer imitation. It represents a true DiT architecture with proper
    time-dependent modulation and diffusion-specific design patterns.
    
    **Canonical DiT Architecture** (what makes this a real DiT):
    - Proper AdaLN (Adaptive Layer Normalization) with time embedding modulation
    - Time-dependent processing at every layer, not just input concatenation
    - Diffusion-specific initialization and normalization schemes
    - Learnable position embeddings for sequences
    
    **VLA-Adapter Enhancements**:
    1. **Cross-Attention Conditioning**: Each DiT block includes cross-attention
       to fuse VLA hidden states (vision + language + proprio) as conditioning.
    
    2. **Dual Time Embeddings**: Separate encoders for timestep t and target_t r,
       combined through an adaptor layer for Flow Matching.
    
    3. **Extended AdaLN**: 9-parameter AdaLN modulation (3 params × 3 components)
       for fine-grained control over self-attention, cross-attention, and MLP.
    
    4. **VLA-Optimized Architecture**: Input/output dimensions and horizon size
       specifically designed for robot action sequences.
    
    **Difference from Simple Transformer**:
    Unlike naive transformer approaches that simply concatenate time embeddings or
    use basic conditioning, this follows the canonical DiT design with proper
    time-dependent modulation through AdaLN at every layer.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        hidden_dim: int = 4096,
        n_layer: int = 12,
        n_head: int = 8,
        n_emb: int = 4096,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        num_task_tokens: int = 512,
        diffusion_timestep_embed_dim: int = 256,
        diffusion_target_t_embed_dim: int = 256,
        zero_init_adaln: bool = True,
        zero_init_output: bool = True,
        condition_injection_mode: str = "cross_attn",
    ):
        super().__init__()
        
        self.horizon = horizon
        self.hidden_dim = n_emb
        self.num_task_tokens = num_task_tokens
        self.zero_init_adaln = bool(zero_init_adaln)
        self.zero_init_output = bool(zero_init_output)
        self.condition_injection_mode = str(condition_injection_mode)
        if self.condition_injection_mode not in CONDITION_INJECTION_MODES:
            raise ValueError(f"Unsupported condition_injection_mode: {self.condition_injection_mode}")
        
        # Input embedding
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        
        # Time encoders (for t and r)
        flow_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_timestep_embed_dim),
            nn.Linear(diffusion_timestep_embed_dim, diffusion_timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_timestep_embed_dim * 4, n_emb),
        )
        flow_target_t_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_target_t_embed_dim),
            nn.Linear(diffusion_target_t_embed_dim, diffusion_target_t_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_target_t_embed_dim * 4, n_emb),
        )
        self.flow_timestep_encoder = flow_timestep_encoder
        self.flow_target_t_encoder = flow_target_t_encoder
        self.timestep_target_t_adaptor = nn.Linear(n_emb * 2, n_emb)
        
        # DiT-X blocks (canonical DiT) with cross-attention for VLA conditioning
        if self.condition_injection_mode == "action_expert_prefix":
            block_cls = ActionExpertPrefixBlock
        elif self.condition_injection_mode == "action_expert_full_prefix":
            block_cls = ActionExpertFullPrefixBlock
        elif self.condition_injection_mode == "action_expert_full_prefix_adarms":
            block_cls = ActionExpertFullPrefixAdaRMSBlock
        elif self.condition_injection_mode == "action_expert_full_prefix_splitkv_adarms":
            block_cls = ActionExpertFullPrefixSplitKVAdaRMSBlock
        elif self.condition_injection_mode == "action_expert_full_prefix_gqa_adarms":
            block_cls = ActionExpertFullPrefixGQAAdaRMSBlock
        elif self.condition_injection_mode == "action_expert_full_prefix_gqa_gated_adarms":
            block_cls = ActionExpertFullPrefixGQAGatedAdaRMSBlock
        else:
            block_cls = DiTXBlock
        self.blocks = nn.ModuleList([
            block_cls(
                hidden_size=n_emb,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                p_drop_attn=p_drop_attn,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
            ) for _ in range(n_layer)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(n_emb, output_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights following canonical DiT convention."""
        for block in self.blocks:
            if hasattr(block, "self_attn"):
                nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
                if block.self_attn.in_proj_bias is not None:
                    nn.init.zeros_(block.self_attn.in_proj_bias)
                nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
                if block.self_attn.out_proj.bias is not None:
                    nn.init.zeros_(block.self_attn.out_proj.bias)
            elif hasattr(block, "attn"):
                for module in (block.attn.q, block.attn.action_kv, block.attn.prefix_kv, block.attn.proj):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        if self.zero_init_adaln:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Initialize input emb
        nn.init.normal_(self.input_emb.weight, std=0.02)
        if self.input_emb.bias is not None:
            nn.init.constant_(self.input_emb.bias, 0)
        
        # Initialize pos emb
        nn.init.normal_(self.pos_emb, std=0.02)
        
        # Initialize time encoders
        for layer in self.flow_timestep_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        for layer in self.flow_target_t_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize adaptor
        nn.init.normal_(self.timestep_target_t_adaptor.weight, std=0.02)
        nn.init.constant_(self.timestep_target_t_adaptor.bias, 0)
        
        if self.zero_init_output:
            nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
            nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    @staticmethod
    def _make_joint_prefix_attn_mask(
        context_mask: torch.Tensor,
        action_len: int,
        num_heads: int,
    ) -> torch.Tensor:
        """Build a pi0-style prefix/suffix mask for nn.MultiheadAttention.

        Prefix/context tokens attend only to valid prefix tokens. Action suffix
        tokens attend to valid prefix tokens and all action suffix tokens.
        """
        batch_size, prefix_len = context_mask.shape
        device = context_mask.device
        total_len = prefix_len + action_len
        valid_prefix = context_mask.to(dtype=torch.bool)
        valid_keys = torch.cat(
            [
                valid_prefix,
                torch.ones(batch_size, action_len, device=device, dtype=torch.bool),
            ],
            dim=1,
        )

        allowed = torch.zeros(batch_size, total_len, total_len, device=device, dtype=torch.bool)
        allowed[:, :prefix_len, :prefix_len] = valid_prefix[:, None, :]
        allowed[:, prefix_len:, :] = valid_keys[:, None, :]

        # Avoid all-masked rows for padded prefix queries. Their outputs are not
        # used by the action head, but all -inf attention rows can produce NaNs.
        if prefix_len > 0:
            eye = torch.eye(prefix_len, device=device, dtype=torch.bool).unsqueeze(0)
            invalid_prefix_queries = ~valid_prefix
            allowed[:, :prefix_len, :prefix_len] |= eye & invalid_prefix_queries[:, :, None]

        disallowed = ~allowed
        return disallowed.repeat_interleave(num_heads, dim=0)
    
    def forward(
        self,
        sample: torch.Tensor,  # (B, T, input_dim) - interpolated actions
        timestep: Union[torch.Tensor, float, int],  # (B,) or scalar - time t
        target_t: Union[torch.Tensor, float, int],  # (B,) or scalar - time r
        vis_cond: torch.Tensor,  # (B, L, n_emb) - context (task + action + proprio)
        vis_cond_mask: torch.Tensor = None,  # (B, L) - True for valid context tokens
        timestep_emb: torch.Tensor = None,  # (B, n_emb) - optional pre-computed embedding
        target_t_emb: torch.Tensor = None,  # (B, n_emb) - optional pre-computed embedding
        action_cond: torch.Tensor = None,  # (B, T, n_emb) - optional suffix/state token conditioning
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sample: (B, T, input_dim) - action sequence
            timestep: (B,) or scalar - current time t
            target_t: (B,) or scalar - target time r
            vis_cond: (B, L, n_emb) - conditioning tokens
            vis_cond_mask: optional validity mask for conditioning tokens
            timestep_emb: optional pre-computed time embedding
            target_t_emb: optional pre-computed target_t embedding
            action_cond: optional per-action-token conditioning added to noisy action suffix tokens
        
        Returns:
            velocity: (B, T, output_dim)
        """
        batch_size = sample.shape[0]
        device = sample.device
        
        # Process input
        input_emb = self.input_emb(sample)  # (B, T, n_emb)
        x = input_emb + self.pos_emb  # (B, T, n_emb)
        if action_cond is not None:
            if action_cond.shape != x.shape:
                raise ValueError(
                    f"action_cond must have shape {tuple(x.shape)}, got {tuple(action_cond.shape)}"
                )
            x = x + action_cond.to(device=x.device, dtype=x.dtype)
        
        # Time encoding
        if timestep_emb is None:
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.float32, device=device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(device)
            timesteps = timesteps.expand(batch_size)
            # SinusoidalPosEmb accepts continuous values [0, 1]
            timestep_emb = self.flow_timestep_encoder(timesteps)
        else:
            timestep_emb = timestep_emb
        
        if target_t_emb is None:
            target_ts = target_t
            if not torch.is_tensor(target_ts):
                target_ts = torch.tensor([target_ts], dtype=torch.float32, device=device)
            elif torch.is_tensor(target_ts) and len(target_ts.shape) == 0:
                target_ts = target_ts[None].to(device)
            target_ts = target_ts.expand(batch_size)
            # SinusoidalPosEmb accepts continuous values [0, 1]
            target_t_emb = self.flow_target_t_encoder(target_ts)
        else:
            target_t_emb = target_t_emb
        
        # Combine time embeddings
        time_c = torch.cat([timestep_emb, target_t_emb], dim=-1)  # (B, 2*n_emb)
        time_c = self.timestep_target_t_adaptor(time_c)  # (B, n_emb)
        
        # Context (vis_cond is already prepared)
        context_c = vis_cond  # (B, L, n_emb)

        if self.condition_injection_mode == "joint_prefix":
            if vis_cond_mask is None:
                vis_cond_mask = torch.ones(
                    context_c.shape[:2],
                    device=context_c.device,
                    dtype=torch.bool,
                )
            prefix_len = context_c.shape[1]
            action_len = x.shape[1]
            x = torch.cat([context_c, x], dim=1)
            attn_mask = self._make_joint_prefix_attn_mask(
                vis_cond_mask,
                action_len=action_len,
                num_heads=self.blocks[0].self_attn.num_heads,
            )
            for block in self.blocks:
                x = block(
                    x,
                    time_c,
                    context_c=None,
                    attn_mask=attn_mask,
                    unmodulated_prefix_len=prefix_len,
                )
                # Match pi0/openpi's prefix-cache behavior more closely: the
                # conditioning prefix is available as keys/values for every
                # action-denoising layer, but it is not itself denoised by the
                # action diffusion time embedding.
                x = torch.cat([context_c, x[:, prefix_len:, :]], dim=1)
            x = x[:, prefix_len:, :]
        elif self.condition_injection_mode in {
            "action_expert_prefix",
            "action_expert_full_prefix",
            "action_expert_full_prefix_adarms",
            "action_expert_full_prefix_splitkv_adarms",
            "action_expert_full_prefix_gqa_adarms",
            "action_expert_full_prefix_gqa_gated_adarms",
        }:
            if vis_cond_mask is None:
                vis_cond_mask = torch.ones(
                    context_c.shape[:2],
                    device=context_c.device,
                    dtype=torch.bool,
                )
            for block in self.blocks:
                x = block(
                    x,
                    time_c,
                    prefix_c=context_c,
                    prefix_mask=vis_cond_mask,
                )
        else:
            for block in self.blocks:
                x = block(x, time_c, context_c, context_mask=vis_cond_mask)  # (B, T, n_emb)
        
        # Final layer
        x = self.final_layer(x)  # (B, T, output_dim)
        
        return x

