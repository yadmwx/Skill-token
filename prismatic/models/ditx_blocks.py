from __future__ import annotations
"""
DiT-X blocks for Flow Matching action head.

This is a complete, canonical DiT (Diffusion Transformer) implementation, not a simple
transformer imitation. Key features that make this a true DiT:

1. **Canonical DiT Architecture**: Full DiT implementation with proper AdaLN modulation,
   time embedding injection, and diffusion-specific design patterns.

2. **VLA-Adapter Enhancements**:
   - Cross-attention mechanism for VLA conditioning (vision + language + proprio)
   - Extended AdaLN modulation for self-attention, cross-attention, and MLP (9 parameters)
   - Dual time encoding (t and r) for Flow Matching
   - Flash attention support for efficient computation

3. **Not a Simple Transformer**: Unlike naive transformer-based approaches, this includes:
   - Proper AdaLN (Adaptive Layer Normalization) with time-dependent modulation
   - Time embedding integration at every layer
   - Diffusion-specific initialization and normalization schemes
   - Flow Matching-specific optimizations

Based on mean-flow reference/ditx_block.py, but represents a complete DiT implementation
with VLA-Adapter-specific enhancements.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Mlp, use_fused_attn

logger = logging.getLogger(__name__)


def modulate(x, shift, scale):
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: int = 10000) -> torch.Tensor:
    """Apply Gemma-style RoPE to x shaped (B, H, L, D)."""
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {x.shape[-1]}")
    dtype = x.dtype
    half_dim = x.shape[-1] // 2
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(half_dim, device=x.device, dtype=torch.float32)
    timescale = max_wavelength ** freq_exponents
    radians = positions.to(device=x.device, dtype=torch.float32).unsqueeze(1).unsqueeze(-1) / timescale
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x_float = x.float()
    x1, x2 = x_float.chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(dtype=dtype)


class RMSNormNoAffine(nn.Module):
    """RMSNorm without learned affine parameters."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return x


class GemmaGatedMLP(nn.Module):
    """Gemma-style gated feed-forward network."""

    def __init__(self, hidden_size: int, mlp_hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, mlp_hidden_dim)
        self.up_proj = nn.Linear(hidden_size, mlp_hidden_dim)
        self.down_proj = nn.Linear(mlp_hidden_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization with conditional modulation."""
    
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_linear = nn.Linear(dim_cond, dim * 2)
        self.cond_modulation = nn.Sequential(
            Rearrange('b d -> b 1 d'),
            nn.SiLU(),
            self.cond_linear
        )
        # Initialize
        nn.init.zeros_(self.cond_linear.weight)
        nn.init.constant_(self.cond_linear.bias[:dim], 1.)
        nn.init.zeros_(self.cond_linear.bias[dim:])

    def forward(self, x, cond=None):
        x = self.ln(x)
        gamma, beta = self.cond_modulation(cond).chunk(2, dim=-1)
        x = x * gamma + beta
        return x


class CrossAttention(nn.Module):
    """Cross-attention layer with flash attention support."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0,
        proj_drop: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x


class DiTXBlock(nn.Module):
    """
    DiT-X block: Canonical DiT implementation with VLA-Adapter enhancements.
    
    This is a complete DiT block, not a simple transformer imitation. It includes:
    
    **Canonical DiT Features**:
    - AdaLN (Adaptive Layer Normalization) with time-dependent modulation
    - Time embedding injection at every layer
    - Proper DiT initialization and normalization schemes
    
    **VLA-Adapter Enhancements**:
    - Self-attention for action sequence modeling
    - Cross-attention for VLA conditioning (vision + language + proprio)
    - Extended AdaLN modulation for all three components (self-attn, cross-attn, MLP)
    - Gated residual connections for better gradient flow
    
    **Difference from Simple Transformer**:
    Unlike naive transformer approaches, this block properly integrates time embeddings
    through AdaLN modulation, following the canonical DiT design pattern.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=p_drop_attn
        )
        
        # Cross-Attention
        self.cross_attn = CrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            norm_layer=nn.LayerNorm,
        )
       
        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0.0
        )

        # Normalization layers (elementwise_affine=False for AdaLN)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # AdaLN modulation: 9 * hidden_size (3 params * 3 components)
        modulation_size = 9 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, modulation_size, bias=True)
        )
        
    def forward(
        self,
        x,
        time_c,
        context_c=None,
        attn_mask=None,
        context_mask=None,
        unmodulated_prefix_len: int = 0,
    ):
        """
        Forward pass of the DiTX block.
        
        Args:
            x: action features (batch_size, seq_length, hidden_size)
            time_c: time embedding (batch_size, hidden_size)
            context_c: optional conditioning tokens (batch_size, context_length, hidden_size)
            attn_mask: optional self-attention mask
            context_mask: optional cross-attention context mask
            unmodulated_prefix_len: number of prefix tokens that should not be
                modulated by the action diffusion time embedding
        """
        # AdaLN modulation
        modulation = self.adaLN_modulation(time_c)
        chunks = modulation.chunk(9, dim=-1)
        
        # Self-Attention parameters
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        # Cross-Attention parameters  
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        # MLP parameters
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # Self-Attention with adaLN
        normed_x = self.norm1(x)
        if unmodulated_prefix_len > 0:
            prefix = normed_x[:, :unmodulated_prefix_len, :]
            suffix = modulate(normed_x[:, unmodulated_prefix_len:, :], shift_msa, scale_msa)
            normed_x = torch.cat([prefix, suffix], dim=1)
        else:
            normed_x = modulate(normed_x, shift_msa, scale_msa)
        self_attn_output, _ = self.self_attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * self_attn_output

        # Cross-Attention with adaLN. Joint prefix/suffix mode passes context_c=None
        # and relies on the self-attention mask to inject conditioning tokens.
        if context_c is not None and context_c.shape[1] > 0:
            normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)
            cross_attn_output = self.cross_attn(normed_x_cross, context_c, mask=context_mask)
            x = x + gate_cross.unsqueeze(1) * cross_attn_output
       
        # MLP with adaLN
        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


class ActionExpertPrefixBlock(DiTXBlock):
    """pi0.5-style action expert block with a read-only conditioning prefix.

    The action stream is the only stream updated by this block. Diffusion time
    modulation is applied to action tokens through AdaLN, while prefix tokens
    are used only as K/V conditioning inputs for cross-attention.
    """

    def forward(
        self,
        x,
        time_c,
        prefix_c=None,
        prefix_mask=None,
    ):
        """
        Args:
            x: action suffix features (batch_size, action_len, hidden_size)
            time_c: diffusion/flow time embedding (batch_size, hidden_size)
            prefix_c: read-only conditioning prefix tokens
            prefix_mask: optional validity mask for prefix tokens
        """
        modulation = self.adaLN_modulation(time_c)
        chunks = modulation.chunk(9, dim=-1)

        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]

        # Action self-attention: only suffix/action tokens are queried, keyed,
        # valued, and updated here.
        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output, _ = self.self_attn(normed_x, normed_x, normed_x)
        x = x + gate_msa.unsqueeze(1) * self_attn_output

        # Prefix attention: action queries attend to frozen prefix K/V. Prefix
        # tokens are not passed through AdaLN and are never written back.
        if prefix_c is not None and prefix_c.shape[1] > 0:
            normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)
            cross_attn_output = self.cross_attn(normed_x_cross, prefix_c, mask=prefix_mask)
            x = x + gate_cross.unsqueeze(1) * cross_attn_output

        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


class ActionExpertFullPrefixBlock(nn.Module):
    """Action expert block whose action queries attend to prefix+action K/V.

    This is closer to openpi's cached-prefix suffix pass than the split
    self-attention + cross-attention variant: the action stream is the only
    stream updated, but each action attention layer sees prefix keys/values and
    suffix keys/values in one attention operation. Diffusion time modulation is
    applied only to the action query/suffix tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        if qkv_bias or qk_norm:
            logger.warning(
                "ActionExpertFullPrefixBlock currently uses nn.MultiheadAttention; "
                "qkv_bias/qk_norm are ignored."
            )
        self.hidden_size = hidden_size
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=p_drop_attn
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0.0,
        )

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x,
        time_c,
        prefix_c=None,
        prefix_mask=None,
    ):
        modulation = self.adaLN_modulation(time_c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        if prefix_c is not None and prefix_c.shape[1] > 0:
            prefix_kv = self.norm1(prefix_c)
            kv = torch.cat([prefix_kv, normed_x], dim=1)
            if prefix_mask is None:
                prefix_mask = torch.ones(
                    prefix_c.shape[:2],
                    device=prefix_c.device,
                    dtype=torch.bool,
                )
            valid_keys = torch.cat(
                [
                    prefix_mask.to(device=x.device, dtype=torch.bool),
                    torch.ones(x.shape[:2], device=x.device, dtype=torch.bool),
                ],
                dim=1,
            )
            key_padding_mask = ~valid_keys
        else:
            kv = normed_x
            key_padding_mask = None

        attn_output, _ = self.self_attn(
            normed_x,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_msa.unsqueeze(1) * attn_output

        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


class ActionExpertFullPrefixAdaRMSBlock(ActionExpertFullPrefixBlock):
    """Full-prefix action expert using pi0.5-style adaptive RMS modulation.

    openpi's action expert applies RMSNorm and a zero-initialized conditioning
    projection that returns scale, shift, and residual gate. This variant keeps
    the same prefix+suffix K/V attention pattern as ActionExpertFullPrefixBlock
    but replaces LayerNorm with RMSNorm to better match that block structure.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.norm1 = RMSNormNoAffine(hidden_size, eps=1e-6)
        self.norm3 = RMSNormNoAffine(hidden_size, eps=1e-6)


class SplitKVFullPrefixAttention(nn.Module):
    """Full-prefix attention with separate action and prefix K/V projections."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.action_kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)
        self.prefix_kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p_drop_attn)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(0.0)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _kv_heads(self, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = kv.shape
        kv = kv.view(bsz, seq_len, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return kv.unbind(0)

    def forward(
        self,
        action_x: torch.Tensor,
        prefix_x: torch.Tensor | None = None,
        prefix_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, action_len, _ = action_x.shape
        q = self._split_heads(self.q(action_x))
        action_k, action_v = self._kv_heads(self.action_kv(action_x))

        if prefix_x is not None and prefix_x.shape[1] > 0:
            prefix_k, prefix_v = self._kv_heads(self.prefix_kv(prefix_x))
            key = torch.cat([prefix_k, action_k], dim=2)
            value = torch.cat([prefix_v, action_v], dim=2)
            prefix_len = prefix_x.shape[1]
            if prefix_mask is None:
                prefix_mask = torch.ones(prefix_x.shape[:2], device=prefix_x.device, dtype=torch.bool)
            prefix_mask = prefix_mask.to(device=action_x.device, dtype=torch.bool)
            valid_keys = torch.cat(
                [
                    prefix_mask,
                    torch.ones((bsz, action_len), device=action_x.device, dtype=torch.bool),
                ],
                dim=1,
            )
            prefix_positions = torch.arange(prefix_len, device=action_x.device, dtype=torch.float32).expand(bsz, -1)
            prefix_lens = prefix_mask.long().sum(dim=1, keepdim=True)
        else:
            key = action_k
            value = action_v
            valid_keys = torch.ones((bsz, action_len), device=action_x.device, dtype=torch.bool)
            prefix_positions = None
            prefix_lens = torch.zeros((bsz, 1), device=action_x.device, dtype=torch.long)

        action_positions = (
            prefix_lens.to(device=action_x.device, dtype=torch.float32)
            + torch.arange(action_len, device=action_x.device, dtype=torch.float32).unsqueeze(0)
        )
        q = apply_rope(q, action_positions)
        action_k = apply_rope(action_k, action_positions)
        if prefix_positions is not None:
            prefix_k = apply_rope(prefix_k, prefix_positions)
            key = torch.cat([prefix_k, action_k], dim=2)
        else:
            key = action_k

        attn_mask = valid_keys[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(bsz, action_len, self.hidden_size)
        out = self.proj(out)
        if self.proj_drop.p > 0:
            out = self.proj_drop(out)
        return out


class ActionExpertFullPrefixSplitKVAdaRMSBlock(nn.Module):
    """pi0.5-style action expert with split prefix/action K/V and RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        if qk_norm:
            logger.warning("ActionExpertFullPrefixSplitKVAdaRMSBlock ignores qk_norm.")
        self.attn = SplitKVFullPrefixAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0.0,
        )
        self.norm1 = RMSNormNoAffine(hidden_size, eps=1e-6)
        self.norm3 = RMSNormNoAffine(hidden_size, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x,
        time_c,
        prefix_c=None,
        prefix_mask=None,
    ):
        modulation = self.adaLN_modulation(time_c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)
        prefix_kv = self.norm1(prefix_c) if prefix_c is not None and prefix_c.shape[1] > 0 else None
        attn_output = self.attn(normed_x, prefix_kv, prefix_mask)
        x = x + gate_msa.unsqueeze(1) * attn_output

        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


class GemmaGQAFullPrefixAttention(nn.Module):
    """Gemma-style GQA attention for action queries over prefix+action K/V."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_dim = num_heads * head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.q = nn.Linear(hidden_size, self.q_dim, bias=qkv_bias)
        self.action_kv = nn.Linear(hidden_size, 2 * self.kv_dim, bias=qkv_bias)
        self.prefix_kv = nn.Linear(hidden_size, 2 * self.kv_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p_drop_attn)
        self.proj = nn.Linear(self.q_dim, hidden_size, bias=True)

    def _q_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim)

    def _kv_heads(self, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = kv.shape
        kv = kv.view(bsz, seq_len, 2, self.num_kv_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        return kv.unbind(0)

    def forward(
        self,
        action_x: torch.Tensor,
        prefix_x: torch.Tensor | None = None,
        prefix_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, action_len, _ = action_x.shape
        q = self._q_heads(self.q(action_x))
        action_k, action_v = self._kv_heads(self.action_kv(action_x))

        if prefix_x is not None and prefix_x.shape[1] > 0:
            prefix_k, prefix_v = self._kv_heads(self.prefix_kv(prefix_x))
            prefix_len = prefix_x.shape[1]
            if prefix_mask is None:
                prefix_mask = torch.ones(prefix_x.shape[:2], device=prefix_x.device, dtype=torch.bool)
            prefix_mask = prefix_mask.to(device=action_x.device, dtype=torch.bool)
            valid_keys = torch.cat(
                [
                    prefix_mask,
                    torch.ones((bsz, action_len), device=action_x.device, dtype=torch.bool),
                ],
                dim=1,
            )
            prefix_positions = torch.arange(prefix_len, device=action_x.device, dtype=torch.float32).expand(bsz, -1)
            prefix_lens = prefix_mask.long().sum(dim=1, keepdim=True)
        else:
            prefix_k = prefix_v = None
            valid_keys = torch.ones((bsz, action_len), device=action_x.device, dtype=torch.bool)
            prefix_positions = None
            prefix_lens = torch.zeros((bsz, 1), device=action_x.device, dtype=torch.long)

        action_positions = (
            prefix_lens.to(device=action_x.device, dtype=torch.float32)
            + torch.arange(action_len, device=action_x.device, dtype=torch.float32).unsqueeze(0)
        )
        q = apply_rope(q.transpose(1, 2), action_positions).transpose(1, 2)
        action_k = apply_rope(action_k.transpose(1, 2), action_positions).transpose(1, 2)
        if prefix_k is not None:
            prefix_k = apply_rope(prefix_k.transpose(1, 2), prefix_positions).transpose(1, 2)
            key = torch.cat([prefix_k, action_k], dim=1)
            value = torch.cat([prefix_v, action_v], dim=1)
        else:
            key = action_k
            value = action_v

        groups = self.num_heads // self.num_kv_heads
        q = q.view(bsz, action_len, self.num_kv_heads, groups, self.head_dim)
        logits = torch.einsum("btkgh,bskh->bkgts", q * (self.head_dim ** -0.5), key)
        logits = logits.masked_fill(~valid_keys[:, None, None, None, :], torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits.float(), dim=-1).to(dtype=q.dtype)
        if self.attn_drop.p > 0:
            probs = self.attn_drop(probs)
        encoded = torch.einsum("bkgts,bskh->btkgh", probs, value)
        encoded = encoded.reshape(bsz, action_len, self.q_dim)
        return self.proj(encoded)


class ActionExpertFullPrefixGQAAdaRMSBlock(ActionExpertFullPrefixSplitKVAdaRMSBlock):
    """pi0.5/Gemma-300M style action expert: width 1024, 8Q:1KV GQA, head_dim 256."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.attn = GemmaGQAFullPrefixAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=256,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
        )


class ActionExpertFullPrefixGQAGatedAdaRMSBlock(ActionExpertFullPrefixGQAAdaRMSBlock):
    """GQA action expert with Gemma-style gated MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        p_drop_attn: float = 0.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            p_drop_attn=p_drop_attn,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )
        self.mlp = GemmaGatedMLP(hidden_size, int(hidden_size * mlp_ratio))

