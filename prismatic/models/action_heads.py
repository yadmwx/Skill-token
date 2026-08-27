"""
action_heads.py

Implementations of various action heads, which serve as alternatives to VLM sequential token prediction.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_ACTIONS_CHUNK, PROPRIO_DIM, STOP_INDEX, NUM_TOKENS
from prismatic.models.ditx_blocks import DiTXBlock
from prismatic.models.dense_depth_film import StateConditionedDenseDepthFiLM

try:
    from diffusers import DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not available. DiffusionActionHead will not work.")

LOG_2PI = math.log(2 * math.pi)



def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations



class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
        use_pro_version=False,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = MLPResNet(
            num_blocks=24, 
            input_dim=input_dim*ACTION_DIM, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim,
            use_pro_version=use_pro_version
            )

    def predict_action(
            self, 
            actions_hidden_states, 
            proprio=None, 
            proprio_projector=None,
            phase="Inference"
            ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()  

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, chunk_len, action_dim * hidden_dim)

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)

        task_valid_mask = task_hidden_states.float().abs().sum(dim=-1) > 0
        action = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states,
            h_t_mask=task_valid_mask,
            )

        return action
    

class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            use_pro_version=False
            ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            if use_pro_version:
                self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
            else:
                self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p=None, h_t_mask=None):
 
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            block_h_t_mask = h_t_mask[:, i + 1, :] if h_t_mask is not None else None
            x = block(x, h_t=h_t[:, i + 1, :], h_a=h_a[:, i + 1, :], p=p, h_t_mask=block_h_t_mask)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x   



def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None, h_t_mask=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)
        if h_t_mask is not None:
            attn_scores_adapter = attn_scores_adapter.masked_fill(
                ~h_t_mask[:, None, None, :],
                torch.finfo(attn_scores_adapter.dtype).min,
            )

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now with FiLM modulation."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            )

        # Q (from x only)
        self.q_proj = nn.Linear(dim, dim)

        # Self-Attention: K, V
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)

        # Adapter cross-attention: K, V
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

        # ---- FiLM ----
        # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now.
        self.film_gen = nn.Sequential(
            nn.Linear(dim, dim * 2),  # output gamma and beta
            )


    def apply_film(self, x, gamma, beta):
        """FiLM: per-channel modulation"""
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


    def forward(self, x, h_a=None, h_t=None, p=None, h_t_mask=None):
        """
        h_a: adapter tokens
        h_t: task tokens
        p:   possible conditioning vector (for FiLM)
        """
        g = self.gating_factor
        ratio_g = torch.tanh(g)

        # concat h_a and p
        h_adapter = torch.cat((h_a, p),dim=1)

        h_task = h_t
        B, T, C = x.shape
        K_a = h_adapter.size(1) if h_a is not None else 0
        K_t = h_task.size(1) if h_task is not None else 0

        # Q
        q_1 = self.q_proj(x)

        # self tokens
        k_tokens = self.k_self(x)
        v_tokens = self.v_self(x)

        # adapter tokens
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)


        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        q_1 = reshape_heads(q_1, B, T)
        k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)     
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores
        attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))]
        attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1)))
        task_scores = torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g
        if h_t_mask is not None:
            task_scores = task_scores.masked_fill(
                ~h_t_mask[:, None, None, :],
                torch.finfo(task_scores.dtype).min,
            )
        attn_scores.append(task_scores)
        attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V
        v_list = [v_tokens,v_adapter,v_task]
        v_combined = torch.cat(v_list, dim=2)

        output = torch.matmul(attn_weights, v_combined)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # # ---- FiLM ---- 
        # gamma_beta = self.film_gen(p)  # [B, 2C]
        # gamma, beta = gamma_beta.chunk(2, dim=-1)  # [B, C], [B, C]
        # output = self.apply_film(output, gamma, beta)

        # residual + FFN
        x = self.ffn(output + x)
        return x


# ============================================================================
# Diffusion Action Head Implementation
# ============================================================================

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Similar to the positional encoding in transformers, but for diffusion timesteps.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: 1-D Tensor of N indices, one per batch element
            dim: Dimension of the output
            max_period: Controls the minimum frequency of the embeddings
            
        Returns:
            Tensor of shape (N, dim)
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of shape (batch_size,) containing timestep values
            
        Returns:
            Tensor of shape (batch_size, hidden_size)
        """
        t_freq = self.timestep_embedding(timesteps, self.frequency_embedding_size)
        target_dtype = self.mlp[0].weight.dtype
        if t_freq.dtype != target_dtype:
            t_freq = t_freq.to(target_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class DenoisingNetwork(nn.Module):
    """
    Denoising network that predicts noise given noisy actions and conditioning.
    Uses cross-attention to incorporate visual and language features.
    """
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        num_blocks: int = 12,
        num_task_tokens: int = 512,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input projection: noisy actions -> hidden_dim
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Timestep embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Proprio projection
        self.proprio_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Denoising blocks (similar to MLPResNetBlock but adapted for diffusion)
        self.denoising_blocks = nn.ModuleList([
            DenoisingBlock(dim=hidden_dim) for _ in range(num_blocks)
        ])
        
        # Output projection: hidden -> noise prediction
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(
        self,
        noisy_actions: torch.Tensor,  # (batch_size, num_actions_chunk, action_dim)
        timestep_emb: torch.Tensor,   # (batch_size, hidden_dim)
        task_hidden_states: torch.Tensor,  # (batch_size, num_layers, num_task_tokens, hidden_dim)
        actions_hidden_states: torch.Tensor,  # (batch_size, num_layers, num_actions, hidden_dim)
        proprio_features: torch.Tensor,  # (batch_size, 1, hidden_dim)
    ) -> torch.Tensor:
        """
        Predict noise from noisy actions.
        
        Returns:
            noise_pred: Tensor of shape (batch_size, num_actions_chunk, action_dim)
        """
        batch_size, num_chunks, _ = noisy_actions.shape
        
        # Project noisy actions
        x = self.action_proj(noisy_actions)  # (batch, num_chunks, hidden_dim)
        
        # Add timestep conditioning (broadcast across chunks)
        t_emb = self.time_proj(timestep_emb).unsqueeze(1)  # (batch, 1, hidden_dim)
        x = x + t_emb
        
        # Add proprio conditioning
        p_emb = self.proprio_proj(proprio_features)  # (batch, 1, hidden_dim)
        
        # Pass through denoising blocks with cross-attention
        num_layers = task_hidden_states.shape[1]
        for i, block in enumerate(self.denoising_blocks):
            # Use layer i+1 if available, otherwise use the last layer
            layer_idx = min(i + 1, num_layers - 1)
            x = block(
                x,
                h_t=task_hidden_states[:, layer_idx, :, :],
                h_a=actions_hidden_states[:, layer_idx, :, :],
                p=p_emb,
            )
        
        # Predict noise
        noise_pred = self.output_proj(x)  # (batch, num_chunks, action_dim)
        
        return noise_pred


class DenoisingBlock(nn.Module):
    """
    Single denoising block with cross-attention to task and action features.
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Self-attention for noisy actions
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        # Cross-attention to conditioning
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        Args:
            x: noisy action features (batch, num_chunks, dim)
            h_t: task features (batch, num_task_tokens, dim)
            h_a: action features (batch, num_action_tokens, dim)
            p: proprio features (batch, 1, dim)
        """
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # Cross-attention to conditioning (concatenate all conditioning)
        if h_t is not None and h_a is not None and p is not None:
            conditioning = torch.cat([h_t, h_a, p], dim=1)  # (batch, total_tokens, dim)
            x = x + self.cross_attn(self.norm2(x), conditioning, conditioning)[0]
        
        # Feedforward
        x = x + self.ffn(self.norm3(x))
        
        return x


class TokenBandwidthAllocator(nn.Module):
    """
    Learnable bandwidth allocator:
      - Produces per-token gates in [0, 1] for VLM task tokens.
      - Produces a global alpha in [0, 1] that scales the DiT velocity residual.
    This module is trained end-to-end from the task loss without an explicit uncertainty input.
    """

    def __init__(self, hidden_dim: int, num_task_tokens: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_task_tokens = num_task_tokens

        # Token-level gate: each task token maps to one scalar gate.
        self.token_gate_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Global alpha produced from the concatenated t / r time embeddings.
        self.alpha_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        task_tokens: torch.Tensor,   # (B, T_task, H)
        t_emb: torch.Tensor,         # (B, H)
        r_emb: torch.Tensor,         # (B, H)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            token_gate: (B, T_task) gating weights in [0, 1] for task tokens.
            alpha: (B, 1, 1) residual blend factor in [0, 1] for DiT velocity.
        """
        B, T_task, H = task_tokens.shape

        # ---- token gate ----
        gate_logits = self.token_gate_mlp(task_tokens)      # (B, T_task, 1)
        token_gate = torch.sigmoid(gate_logits).squeeze(-1) # (B, T_task)

        # ---- global alpha ----
        global_ctx = torch.cat([t_emb, r_emb], dim=-1)      # (B, 2H)
        alpha_logits = self.alpha_mlp(global_ctx)           # (B, 1)
        alpha = torch.sigmoid(alpha_logits).view(B, 1, 1)   # (B, 1, 1)

        # Keep dtypes aligned to avoid mixed-precision surprises.
        token_gate = token_gate.to(dtype=task_tokens.dtype)
        alpha = alpha.to(dtype=task_tokens.dtype)

        return token_gate, alpha


class TaskLayerSelector(nn.Module):
    """
    Select or aggregate task-token representations across transformer depth.
    Input shape: (B, L, N_t, H).
    Returns aggregated task tokens of shape (B, N_t, H) plus layer weights.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        task_hidden_states: torch.Tensor,  # (B, L, N_t, H)
        task_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, N_t, H = task_hidden_states.shape

        # Pool task tokens within each layer to obtain per-layer summaries.
        # Prompt/task tokens can be padded with zeros for batching; openpi/pi0.5
        # keeps those padding tokens out of attention with explicit masks. Do the
        # same for layer scoring so padding does not alter depth aggregation.
        if task_mask is None:
            pooled = task_hidden_states.mean(dim=2)
            output_mask = None
        else:
            if task_mask.ndim == 2:
                task_mask = task_mask[:, None, :].expand(B, L, N_t)
            if task_mask.shape != (B, L, N_t):
                raise ValueError(
                    f"TaskLayerSelector mask shape must be {(B, L, N_t)}, got {tuple(task_mask.shape)}"
                )
            mask_f = task_mask.to(dtype=task_hidden_states.dtype).unsqueeze(-1)
            pooled = (task_hidden_states * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp_min(1.0)
            output_mask = task_mask[:, 0, :]

        # Score each layer and normalize with softmax.
        scores = self.score_mlp(pooled).squeeze(-1)  # (B, L)
        layer_weights = torch.softmax(scores, dim=1)  # (B, L)

        # Aggregate task tokens across layers with the learned weights.
        weights_expanded = layer_weights.view(B, L, 1, 1)  # (B, L, 1, 1)
        aggregated_task_tokens = (weights_expanded * task_hidden_states).sum(dim=1)
        if output_mask is not None:
            aggregated_task_tokens = aggregated_task_tokens * output_mask.unsqueeze(-1).to(
                dtype=aggregated_task_tokens.dtype
            )

        return aggregated_task_tokens, layer_weights


class VelocityNetwork(nn.Module):
    """
    Velocity network using DiT-X (canonical DiT) for Flow Matching.

    This variant augments DiT-X with two extra mechanisms:
      - TokenBandwidthAllocator for task-token gating.
      - A learned alpha factor for residual blending on DiT velocity outputs.
    """
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        num_blocks: int = 12,
        num_task_tokens: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        use_adaptive_bridge: bool = True,
        bridge_mode: str = "adaptive",
        fixed_layer_index: int = -1,
        condition_mode: str = "full",
        condition_injection_mode: str = "cross_attn",
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
        dense_film_state_mode: str = "full",
    ):
        super().__init__()
        from prismatic.models.ditx_vla_adapter import DiTXVLAdapter

        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_adaptive_bridge = use_adaptive_bridge
        self.bridge_mode = bridge_mode
        self.fixed_layer_index = fixed_layer_index
        self.condition_mode = str(condition_mode)
        if self.condition_mode not in {"full", "task_only"}:
            raise ValueError(f"Unsupported DIT condition_mode: {self.condition_mode}")
        self.condition_injection_mode = str(condition_injection_mode)
        if self.condition_injection_mode not in {"cross_attn", "joint_prefix", "action_expert_prefix"}:
            raise ValueError(f"Unsupported DIT condition_injection_mode: {self.condition_injection_mode}")
        self.debug_group_action_tokens_to_chunk = False
        self.pure_inference = False
        self.use_latent_skill_token = bool(use_latent_skill_token)
        self.num_skill_tokens = int(num_skill_tokens)
        self.skill_token_dim = int(skill_token_dim)
        self.skill_temperature = float(skill_temperature)
        self.dense_film_enabled = bool(dense_film_enabled)

        # Core DiT-X backbone used to predict velocities.
        self.ditx = DiTXVLAdapter(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=NUM_ACTIONS_CHUNK,
            hidden_dim=hidden_dim,
            n_layer=num_blocks,
            n_head=num_heads,
            n_emb=hidden_dim,
            mlp_ratio=mlp_ratio,
            num_task_tokens=num_task_tokens,
            zero_init_adaln=dit_zero_init_adaln,
            zero_init_output=dit_zero_init_output,
            condition_injection_mode=self.condition_injection_mode,
        )

        # Layer selector for depth-aware task-token aggregation.
        self.task_layer_selector = TaskLayerSelector(hidden_dim=hidden_dim)
        self.dense_depth_film = (
            StateConditionedDenseDepthFiLM(
                hidden_dim=hidden_dim,
                num_task_tokens=num_task_tokens,
                max_layers=dense_film_max_layers,
                first_layer_index=dense_film_first_layer_index,
                bottleneck_dim=dense_film_bottleneck_dim,
                state_dim=dense_film_state_dim,
                state_mode=self.dense_film_state_mode,
            )
            if self.dense_film_enabled
            else None
        )

        if self.use_latent_skill_token:
            self.skill_selector = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, self.num_skill_tokens),
            )
            self.skill_embedding = nn.Embedding(self.num_skill_tokens, self.skill_token_dim)
            self.skill_layer_scorer = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2 + self.skill_token_dim),
                nn.Linear(hidden_dim * 2 + self.skill_token_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.skill_condition_proj = nn.Sequential(
                nn.LayerNorm(self.skill_token_dim),
                nn.Linear(self.skill_token_dim, hidden_dim),
            )
            nn.init.zeros_(self.skill_condition_proj[-1].weight)
            nn.init.zeros_(self.skill_condition_proj[-1].bias)
        else:
            self.skill_selector = None
            self.skill_embedding = None
            self.skill_layer_scorer = None
            self.skill_condition_proj = None

        # Bandwidth allocator used only in adaptive_gated mode.
        self.bandwidth_allocator = TokenBandwidthAllocator(
            hidden_dim=hidden_dim,
            num_task_tokens=num_task_tokens,
        )

        # Learned static weights (for "static_learned" mode)
        self._static_layer_weights = nn.Parameter(torch.ones(1) / 16)  # will be resized in forward

    @staticmethod
    def _masked_pool_task_tokens(
        task_hidden_states: torch.Tensor,
        task_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Average only valid task tokens so zero padding cannot alter skills."""
        mask = task_valid_mask.to(dtype=task_hidden_states.dtype).unsqueeze(-1)
        return (task_hidden_states * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)

    def _skill_layer_weights(
        self,
        pooled_layers: torch.Tensor,
        proprio_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer a discrete latent skill and use it to select representation depth."""
        dtype = pooled_layers.dtype
        bsz, num_layers, _ = pooled_layers.shape
        proprio = proprio_features.reshape(bsz, -1).to(dtype=dtype)
        global_context = pooled_layers.mean(dim=1)
        logits = self.skill_selector(torch.cat([global_context, proprio], dim=-1))
        if self.training:
            probs = F.gumbel_softmax(
                logits.float(), tau=max(self.skill_temperature, 1e-6), hard=True, dim=-1
            ).to(dtype)
        else:
            ids = logits.argmax(dim=-1)
            probs = F.one_hot(ids, num_classes=self.num_skill_tokens).to(dtype)
        skill_emb = probs @ self.skill_embedding.weight.to(dtype)
        scorer_input = torch.cat(
            [
                pooled_layers,
                skill_emb.unsqueeze(1).expand(-1, num_layers, -1),
                proprio.unsqueeze(1).expand(-1, num_layers, -1),
            ],
            dim=-1,
        )
        weights = torch.softmax(self.skill_layer_scorer(scorer_input).squeeze(-1), dim=1)
        return weights, skill_emb

    def forward(
        self,
        z: torch.Tensor,                    # (batch, num_chunks, action_dim)
        timestep_emb: torch.Tensor,         # (batch, hidden_dim) - time t embedding
        target_t_emb: torch.Tensor,         # (batch, hidden_dim) - time r embedding
        task_hidden_states: torch.Tensor,   # (batch, num_layers, num_task_tokens, hidden_dim)
        action_hidden_states: torch.Tensor, # (batch, num_layers, num_actions, hidden_dim)
        proprio_features: torch.Tensor,     # (batch, 1, hidden_dim)
    ) -> torch.Tensor:
        """
        Predict velocity field using DiTX, with learnable bandwidth allocation.

        Returns:
            velocity: (batch, num_chunks, action_dim)
        """
        num_layers = task_hidden_states.shape[1]
        layer_idx = self.fixed_layer_index if self.fixed_layer_index >= 0 else num_layers // 2
        task_valid_mask = task_hidden_states.float().abs().sum(dim=-1) > 0

        skill_emb = None
        dense_film_metrics = None
        if self.dense_film_enabled:
            if not self.use_adaptive_bridge or self.bridge_mode != "dense_film_residual":
                raise ValueError(
                    "dense_film_enabled requires use_adaptive_bridge=True and "
                    "bridge_mode='dense_film_residual'"
                )
            if self.dense_depth_film is None:
                raise RuntimeError("dense depth FiLM module was not initialized")
            task_tokens, action_tokens, dense_film_metrics = self.dense_depth_film(
                task_hidden_states,
                action_hidden_states,
                proprio_features,
            )
        elif not self.use_adaptive_bridge:
            # Fixed layer mode: use the same single layer for both task and action tokens.
            task_tokens = task_hidden_states[:, layer_idx, :, :]  # (B, T_task, H)
            action_tokens = action_hidden_states[:, layer_idx, :, :]  # (B, T_act, H)
        elif self.bridge_mode == "uniform":
            # Uniform average: 1/L weights for both task and action tokens.
            task_tokens = task_hidden_states.mean(dim=1)  # (B, T_task, H)
            action_tokens = action_hidden_states.mean(dim=1)  # (B, T_act, H)
        elif self.bridge_mode == "static_learned":
            # Learned static weights (shared across all samples).
            if self._static_layer_weights.shape[0] != num_layers:
                self._static_layer_weights = nn.Parameter(torch.ones(num_layers, device=task_hidden_states.device) / num_layers)
            weights = torch.softmax(self._static_layer_weights, dim=0).view(1, num_layers, 1, 1)
            task_tokens = (weights * task_hidden_states).sum(dim=1)  # (B, T_task, H)
            action_tokens = (weights * action_hidden_states).sum(dim=1)  # (B, T_act, H)
        elif self.bridge_mode == "adaptive":
            # Reuse the task-adaptive layer weights for action tokens so both branches see aligned depth.
            if self.use_latent_skill_token:
                pooled_layers = self._masked_pool_task_tokens(task_hidden_states, task_valid_mask)
                layer_weights, skill_emb = self._skill_layer_weights(pooled_layers, proprio_features)
                weights = layer_weights.view(task_hidden_states.shape[0], num_layers, 1, 1)
                task_tokens = (weights * task_hidden_states).sum(dim=1)
            else:
                task_tokens, layer_weights = self.task_layer_selector(
                    task_hidden_states,
                    task_mask=task_valid_mask,
                )  # (B, T_task, H), (B, L)
            weights = layer_weights.view(task_hidden_states.shape[0], num_layers, 1, 1)
            action_tokens = (weights * action_hidden_states).sum(dim=1)  # (B, T_act, H)
        elif self.bridge_mode == "adaptive_gated":
            # Full method: task-adaptive + bandwidth gating, with aligned action-token aggregation.
            if self.use_latent_skill_token:
                pooled_layers = self._masked_pool_task_tokens(task_hidden_states, task_valid_mask)
                layer_weights, skill_emb = self._skill_layer_weights(pooled_layers, proprio_features)
                weights = layer_weights.view(task_hidden_states.shape[0], num_layers, 1, 1)
                task_tokens = (weights * task_hidden_states).sum(dim=1)
            else:
                task_tokens, layer_weights = self.task_layer_selector(
                    task_hidden_states,
                    task_mask=task_valid_mask,
                )  # (B, T_task, H), (B, L)
            weights = layer_weights.view(task_hidden_states.shape[0], num_layers, 1, 1)
            action_tokens = (weights * action_hidden_states).sum(dim=1)  # (B, T_act, H)
        else:
            raise ValueError(f"Unknown bridge_mode: {self.bridge_mode}")

        if skill_emb is not None and self.skill_condition_proj is not None:
            skill_condition = self.skill_condition_proj(skill_emb).unsqueeze(1)
            task_tokens = task_tokens + skill_condition
            action_tokens = action_tokens + skill_condition

        if self.debug_group_action_tokens_to_chunk:
            bsz, num_action_tokens, hidden_dim = action_tokens.shape
            if num_action_tokens % NUM_ACTIONS_CHUNK != 0:
                raise ValueError(
                    f"Cannot group {num_action_tokens} action tokens into NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}"
                )
            action_tokens = action_tokens.reshape(
                bsz,
                NUM_ACTIONS_CHUNK,
                num_action_tokens // NUM_ACTIONS_CHUNK,
                hidden_dim,
            ).mean(dim=2)

        # Bandwidth allocation: only in adaptive_gated mode
        if self.use_adaptive_bridge and self.bridge_mode == "adaptive_gated" and not self.pure_inference:
            token_gate, alpha = self.bandwidth_allocator(
                task_tokens=task_tokens,
                t_emb=timestep_emb,
                r_emb=target_t_emb,
            )  # token_gate: (B, T_task), alpha: (B, 1, 1)
            gated_task_tokens = task_tokens * token_gate.unsqueeze(-1)  # (B, T_task, H)
            task_mask = task_tokens.float().abs().sum(dim=-1) > 0
            proprio_mask = torch.ones(
                proprio_features.shape[:2],
                device=proprio_features.device,
                dtype=torch.bool,
            )
            if self.condition_mode == "task_only":
                context = torch.cat(
                    [gated_task_tokens, proprio_features],
                    dim=1,
                )  # (B, T_task + 1, H)
                context_mask = torch.cat([task_mask, proprio_mask], dim=1)
            else:
                action_mask = action_tokens.float().abs().sum(dim=-1) > 0
                context = torch.cat(
                    [gated_task_tokens, action_tokens, proprio_features],
                    dim=1,
                )  # (B, T_task + T_act + 1, H)
                context_mask = torch.cat([task_mask, action_mask, proprio_mask], dim=1)
        else:
            alpha = torch.ones(task_tokens.shape[0], 1, 1, device=task_tokens.device, dtype=task_tokens.dtype)
            task_mask = task_tokens.float().abs().sum(dim=-1) > 0
            proprio_mask = torch.ones(
                proprio_features.shape[:2],
                device=proprio_features.device,
                dtype=torch.bool,
            )
            if self.condition_mode == "task_only":
                context = torch.cat(
                    [task_tokens, proprio_features],
                    dim=1,
                )  # (B, T_task + 1, H)
                context_mask = torch.cat([task_mask, proprio_mask], dim=1)
            else:
                action_mask = action_tokens.float().abs().sum(dim=-1) > 0
                context = torch.cat(
                    [task_tokens, action_tokens, proprio_features],
                    dim=1,
                )  # (B, T_task + T_act + 1, H)
                context_mask = torch.cat([task_mask, action_mask, proprio_mask], dim=1)

        # Call the DiT-X backbone.
        batch_size = z.shape[0]
        timestep = torch.zeros(batch_size, device=z.device)
        target_t = torch.zeros(batch_size, device=z.device)

        velocity = self.ditx(
            sample=z,
            timestep=timestep,
            target_t=target_t,
            vis_cond=context,
            vis_cond_mask=context_mask,
            timestep_emb=timestep_emb,
            target_t_emb=target_t_emb,
        )  # (B, num_chunks, action_dim)

        # Apply the learned global residual scaling factor.
        velocity = alpha * velocity  # alpha is broadcast from shape (B, 1, 1).

        if dense_film_metrics is not None:
            # This is a dense residual diagnostic, not a probability router.
            self.last_routing_diagnostics = {
                "layer_weights": None,
                "depth_film": {
                    key: value.detach().float().cpu()
                    for key, value in dense_film_metrics.items()
                },
            }

        return velocity


@dataclass
class FlowGRPOSample:
    """Trajectory data produced by DiffusionActionHead for Flow-GRPO updates."""

    actions: torch.Tensor
    latents: List[torch.Tensor]
    log_probs: torch.Tensor
    timesteps: torch.Tensor
    timestep_indices: torch.Tensor
    means: List[torch.Tensor]
    stds: List[torch.Tensor]


class DiffusionActionHead(nn.Module):
    """
    Diffusion-based action head for continuous action prediction.
    
    Uses DDPM (Denoising Diffusion Probabilistic Models) to iteratively
    denoise random noise into action predictions.
    
    Args:
        input_dim: Dimension of input features from LLM
        hidden_dim: Hidden dimension for the denoising network
        action_dim: Dimension of action space
        num_task_tokens: Number of task/vision tokens
        num_diffusion_steps: Number of denoising steps during inference
        use_pro_version: Whether to use enhanced architecture
    """
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        action_dim: int = 7,
        num_task_tokens: int = 512,
        num_diffusion_steps: int = 50,
        use_pro_version: bool = False,
    ):
        super().__init__()
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "DiffusionActionHead requires diffusers library. "
                "Install with: pip install diffusers"
            )
        
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps
        
        # Diffusion scheduler (DDPM)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",  # Predict noise
            clip_sample=False,
        )
        
        # Timestep embedder
        self.time_encoder = TimestepEmbedder(hidden_dim)
        
        # Denoising network
        self.denoising_network = DenoisingNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_blocks=12 if not use_pro_version else 24,
            num_task_tokens=num_task_tokens,
        )
        
    def predict_noise(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        actions_hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        proprio_projector: nn.Module,
    ) -> torch.Tensor:
        """
        Predict noise from noisy actions (used during inference).
        
        Args:
            noisy_actions: (batch, num_chunks, action_dim)
            timestep: (batch,) or scalar
            actions_hidden_states: (batch, num_layers, num_tokens, hidden_dim)
            proprio: (batch, proprio_dim)
            proprio_projector: Module to project proprio
            
        Returns:
            noise_pred: (batch, num_chunks, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        
        # Ensure noisy_actions has the same dtype as model weights (bfloat16)
        # This is critical to avoid dtype mismatch errors
        model_dtype = next(self.parameters()).dtype
        noisy_actions = noisy_actions.to(dtype=model_dtype)
        
        # Ensure timestep is a tensor matching batch size and dtype
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep], device=noisy_actions.device, dtype=torch.long)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(device=noisy_actions.device)
        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        elif timestep.shape[0] != batch_size:
            timestep = timestep[:1].expand(batch_size)
        # Timestep should be long for embedding lookup
        timestep = timestep.long()
        
        # Encode timestep (time_encoder expects long dtype)
        timestep_emb = self.time_encoder(timestep)  # (batch, hidden_dim)
        
        # Process proprio (keep original dtype)
        proprio = proprio.reshape(batch_size, -1).to(dtype=actions_hidden_states.dtype)
        proprio_features = proprio_projector(proprio).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Extract task and action hidden states
        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        action_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]
        
        # Predict noise
        noise_pred = self.denoising_network(
            noisy_actions,
            timestep_emb,
            task_hidden_states,
            action_hidden_states,
            proprio_features,
        )
        
        return noise_pred
    
    def predict_action(
        self,
        actions_hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        proprio_projector: nn.Module,
        phase: str = "Inference",
    ) -> torch.Tensor:
        """
        Generate actions using diffusion sampling (reverse process).
        
        This method is called during inference to generate actions from noise.
        
        Args:
            actions_hidden_states: Hidden states from VLM (batch, num_layers, num_tokens, hidden_dim)
            proprio: Proprioception state (batch, proprio_dim)
            proprio_projector: Module to project proprio features
            phase: "Training" or "Inference"
            
        Returns:
            actions: Predicted actions (num_chunks, action_dim)
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # Start from random noise (match the dtype of actions_hidden_states)
        noisy_actions = torch.randn(
            (batch_size, NUM_ACTIONS_CHUNK, self.action_dim),
            device=device,
            dtype=actions_hidden_states.dtype
        )
        
        # Set timesteps for inference
        self.noise_scheduler.set_timesteps(self.num_diffusion_steps, device=device)
        
        # Reverse diffusion process (denoising)
        for t in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.predict_noise(
                noisy_actions=noisy_actions,
                timestep=t,
                actions_hidden_states=actions_hidden_states,
                proprio=proprio,
                proprio_projector=proprio_projector,
            )
            
            # Denoise one step
            noisy_actions = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_actions,
            ).prev_sample
        
        # Final denoised actions
        actions = noisy_actions.squeeze(0)  # (num_chunks, action_dim)
        
        return actions
    
    def _compute_transition_statistics(
        self,
        sample: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute (prev_mean, std, t_int) for the reverse diffusion transition."""

        scheduler = self.noise_scheduler

        sample_f = sample.float()
        noise_pred_f = noise_pred.float()

        if isinstance(timestep, torch.Tensor):
            t_int = int(timestep.flatten()[0].item())
        else:
            t_int = int(timestep)

        prev_t = scheduler.previous_timestep(t_int)

        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample_f.dtype)
        alpha_prod_t = alphas_cumprod[t_int]
        if prev_t >= 0:
            alpha_prod_t_prev = alphas_cumprod[prev_t]
        else:
            alpha_prod_t_prev = torch.tensor(1.0, device=sample.device, dtype=sample_f.dtype)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample_f - beta_prod_t.sqrt() * noise_pred_f) / alpha_prod_t.sqrt()
        elif scheduler.config.prediction_type == "sample":
            pred_original_sample = noise_pred_f
        elif scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t.sqrt() * sample_f - beta_prod_t.sqrt() * noise_pred_f
        else:
            raise ValueError(f"Unsupported prediction_type: {scheduler.config.prediction_type}")

        if scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -scheduler.config.clip_sample_range,
                scheduler.config.clip_sample_range,
            )

        pred_original_sample_coeff = (alpha_prod_t_prev.sqrt() * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t
        prev_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample_f

        if t_int > 0:
            variance = scheduler._get_variance(t_int, predicted_variance=None)
            variance = variance.to(device=sample.device, dtype=sample_f.dtype)
            while variance.ndim < sample_f.ndim:
                variance = variance.unsqueeze(-1)
            std = variance.expand_as(sample_f).sqrt()
        else:
            std = torch.zeros_like(sample_f)

        return prev_mean, std, t_int

    def _step_with_details(
        self,
        sample: torch.Tensor,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single scheduler step returning sample, mean, std, and log-probability."""

        sample_dtype = sample.dtype

        prev_mean_f, std_f, t_int = self._compute_transition_statistics(sample, noise_pred, timestep)

        if t_int > 0:
            if generator is not None:
                noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=std_f.dtype)
            else:
                noise = torch.randn_like(std_f)

            prev_sample_f = prev_mean_f + std_f * noise
            diff = (prev_sample_f - prev_mean_f)
            var = std_f.pow(2) + 1e-12
            log_prob_f = -0.5 * (diff.pow(2) / var + torch.log(var) + LOG_2PI)
            log_prob_f = log_prob_f.view(log_prob_f.shape[0], -1).sum(dim=1)

            std = std_f.to(sample_dtype)
            prev_sample = prev_sample_f.to(sample_dtype)
            prev_mean = prev_mean_f.to(sample_dtype)
            log_prob = log_prob_f.to(sample_dtype)
        else:
            std = torch.zeros_like(prev_mean_f, dtype=sample_dtype)
            prev_sample = prev_mean_f.to(sample_dtype)
            prev_mean = prev_mean_f.to(sample_dtype)
            log_prob = torch.zeros(sample.shape[0], device=sample.device, dtype=sample_dtype)

        return prev_sample, prev_mean, std, log_prob

    def evaluate_logprob_from_latents(
        self,
        actions_hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        proprio_projector: nn.Module,
        latents: List[torch.Tensor],
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Re-evaluate log-probabilities of a stored latent trajectory."""

        device = actions_hidden_states.device
        dtype = actions_hidden_states.dtype

        proprio = proprio.to(device=device, dtype=dtype)
        latents_fp = [lat.to(device=device, dtype=dtype) for lat in latents]
        timestep_indices = timestep_indices.to(device=device)

        self.noise_scheduler.set_timesteps(len(timestep_indices), device=device)

        log_probs: List[torch.Tensor] = []

        for step_idx, t in enumerate(timestep_indices):
            current_latent = latents_fp[step_idx]
            next_latent = latents_fp[step_idx + 1]

            noise_pred = self.predict_noise(
                noisy_actions=current_latent,
                timestep=t,
                actions_hidden_states=actions_hidden_states,
                proprio=proprio,
                proprio_projector=proprio_projector,
            )

            prev_mean, std, t_int = self._compute_transition_statistics(current_latent, noise_pred, t)

            if t_int > 0:
                diff = (next_latent.float() - prev_mean)
                var = std.pow(2) + 1e-12
                log_prob = -0.5 * (diff.pow(2) / var + torch.log(var) + LOG_2PI)
                log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
            else:
                log_prob = torch.zeros(current_latent.shape[0], device=device, dtype=dtype)

            log_probs.append(log_prob.to(dtype))

        return torch.stack(log_probs, dim=0)
    
    def sample_with_logprob(
        self,
        actions_hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        proprio_projector: nn.Module,
        generator: Optional[torch.Generator] = None,
    ) -> FlowGRPOSample:
        """
        Generate actions while recording diffusion statistics for Flow-GRPO.
        
        This method performs the full diffusion reverse process and collects:
        - Latents at each step
        - Means and standard deviations of the transition distributions
        - Log-probabilities of each transition
        - Normalized timesteps
        
        Args:
            actions_hidden_states: Hidden states from VLM (batch, num_layers, num_tokens, hidden_dim)
            proprio: Proprioception state (batch, proprio_dim)
            proprio_projector: Module to project proprio features
            generator: Optional random generator for reproducibility
            
        Returns:
            FlowGRPOSample: Container with actions, latents, log_probs, timesteps, means, and stds
        """
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        
        # Start from random noise
        if generator is not None:
            noisy_actions = torch.randn(
                (batch_size, NUM_ACTIONS_CHUNK, self.action_dim),
                generator=generator,
                device=device,
                dtype=actions_hidden_states.dtype,
            )
        else:
            noisy_actions = torch.randn(
                (batch_size, NUM_ACTIONS_CHUNK, self.action_dim),
                device=device,
                dtype=actions_hidden_states.dtype,
            )
        
        # Set timesteps for inference
        self.noise_scheduler.set_timesteps(self.num_diffusion_steps, device=device)
        
        # Storage for trajectory data
        latents_history: List[torch.Tensor] = [noisy_actions.detach().clone()]
        means_history: List[torch.Tensor] = []
        stds_history: List[torch.Tensor] = []
        log_probs_history: List[torch.Tensor] = []
        timestep_history: List[float] = []
        
        scheduler_timesteps = self.noise_scheduler.timesteps
        timestep_indices = scheduler_timesteps.detach().clone().to(device)
        num_train_timesteps = getattr(
            self.noise_scheduler.config,
            "num_train_timesteps",
            1000,
        )
        
        # Reverse diffusion process (denoising)
        for t in scheduler_timesteps:
            # Predict noise
            noise_pred = self.predict_noise(
                noisy_actions=noisy_actions,
                timestep=t,
                actions_hidden_states=actions_hidden_states,
                proprio=proprio,
                proprio_projector=proprio_projector,
            )
            
            # Step with detailed statistics
            noisy_actions, prev_mean, std, log_prob = self._step_with_details(
                sample=noisy_actions,
                noise_pred=noise_pred,
                timestep=t,
                generator=generator,
            )
            
            # Store statistics
            latents_history.append(noisy_actions.detach().clone())
            means_history.append(prev_mean.detach().clone())
            stds_history.append(std.detach().clone())
            log_probs_history.append(log_prob.detach().clone())
            
            # Normalize timestep to [0, 1] range (1 -> 0)
            timestep_normalized = float(t.item() if isinstance(t, torch.Tensor) else t) / float(num_train_timesteps)
            timestep_history.append(timestep_normalized)
        
        # Final actions (after all denoising steps)
        # If batch_size=1, squeeze to (num_chunks, action_dim), otherwise keep (batch, num_chunks, action_dim)
        if batch_size == 1:
            actions = noisy_actions.squeeze(0)  # (num_chunks, action_dim)
        else:
            actions = noisy_actions  # (batch, num_chunks, action_dim)
        
        # Stack log-probs: (num_steps, batch)
        log_prob_tensor = torch.stack(log_probs_history, dim=0)
        
        # Create timesteps tensor
        timesteps_tensor = torch.tensor(
            timestep_history,
            device=device,
            dtype=torch.float32,
        )
        
        return FlowGRPOSample(
            actions=actions.detach(),
            latents=[lat.detach() for lat in latents_history],
            log_probs=log_prob_tensor,
            timesteps=timesteps_tensor,
            timestep_indices=timestep_indices.detach().cpu(),
            means=[mean.detach() for mean in means_history],
            stds=[std.detach() for std in stds_history],
        )


# ==============================================================================
#  Flow Matching MLP Head (MIP / Rectified Flow Implementation)
#  Based on "Much Ado About Noising" & "OAT"
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding with an explicit continuous-time mode."""

    def __init__(
        self,
        dim: int,
        mode: str = "legacy",
        min_period: float = 4e-3,
        max_period: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.mode = str(mode).lower()
        self.min_period = float(min_period)
        self.max_period = float(max_period)
        if self.mode not in {"legacy", "continuous"}:
            raise ValueError("time embedding mode must be 'legacy' or 'continuous'")
        if self.min_period <= 0 or self.max_period <= self.min_period:
            raise ValueError("continuous time periods must satisfy 0 < min_period < max_period")

    def forward(self, x):
        device, dtype = x.device, x.dtype
        half_dim = self.dim // 2
        if self.mode == "continuous":
            # OpenPI-style encoding: t is in [0, 1], so periods must cover that
            # interval.  The legacy transformer encoding assumes much larger
            # integer timesteps and is nearly constant over [0, 1].
            fraction = torch.linspace(0.0, 1.0, half_dim, device=device, dtype=torch.float32)
            period = self.min_period * (self.max_period / self.min_period) ** fraction
            emb = x.float()[:, None] / period[None, :] * (2.0 * math.pi)
        else:
            freq = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -freq)
            emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLPResBlock(nn.Module):
    """Simple MLP residual block with LayerNorm and SiLU activation."""
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),  # Smooth nonlinearity for the residual MLP.
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class LayerwiseResidualAligner(nn.Module):
    """Independently align each VLM depth while preserving its initial features.

    Each layer owns its LayerNorm and bottleneck projection.  The final
    projection is zero-initialized, so enabling the aligner is an exact identity
    at initialization instead of replacing a useful pretrained representation
    with an arbitrary shared coordinate system.
    """

    def __init__(self, num_layers: int, hidden_dim: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.aligners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(bottleneck_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        for aligner in self.aligners:
            nn.init.zeros_(aligner[-1].weight)
            nn.init.zeros_(aligner[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 4:
            raise ValueError(
                "LayerwiseResidualAligner expects (batch, layers, tokens, hidden), "
                f"got {tuple(hidden_states.shape)}"
            )
        actual_layers = hidden_states.shape[1]
        if actual_layers > self.num_layers:
            raise ValueError(
                f"received {actual_layers} layers but aligner was built for {self.num_layers}"
            )
        return torch.stack(
            [
                hidden_states[:, layer_idx]
                + self.aligners[layer_idx](hidden_states[:, layer_idx])
                for layer_idx in range(actual_layers)
            ],
            dim=1,
        )


class LayerwiseNormalizedProjector(nn.Module):
    """Put every VLM depth in a comparable coordinate and scale before mixing.

    Unlike ``LayerwiseResidualAligner``, the normalized representation is the
    main path.  A zero-initialized bottleneck residual preserves the normalized
    VLM feature at initialization while allowing each depth to adapt
    independently during training.
    """

    def __init__(self, num_layers: int, hidden_dim: int, bottleneck_dim: int = 64) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        self.num_layers = int(num_layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(bottleneck_dim, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )
        for adapter in self.adapters:
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 4:
            raise ValueError(
                "LayerwiseNormalizedProjector expects (batch, layers, tokens, hidden), "
                f"got {tuple(hidden_states.shape)}"
            )
        actual_layers = hidden_states.shape[1]
        if actual_layers > self.num_layers:
            raise ValueError(
                f"received {actual_layers} layers but projector was built for {self.num_layers}"
            )
        outputs = []
        for layer_idx in range(actual_layers):
            normalized = self.norms[layer_idx](hidden_states[:, layer_idx])
            outputs.append(normalized + self.adapters[layer_idx](normalized))
        return torch.stack(outputs, dim=1)


class DeepRoutedActionDecoder(nn.Module):
    """Decode actions from the already-routed representation only.

    Unlike ``L1RegressionActionHead``, this module never receives the original
    per-depth tensor.  Consequently it cannot bypass the router by consuming a
    different VLM layer in each residual block.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [MLPResBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, routed_state: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(routed_state)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_proj(hidden)


class FlowMatchingMLPActionHead(nn.Module):
    """
    MLP-based Flow Matching action head.

    This head remains fully interface-compatible with the existing training/eval code.
    Supports adaptive bridge modes for multi-layer feature aggregation.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 1024,
        output_dim: int = 7,
        num_layers: int = 3,
        time_dim: int = 256,
        dropout: float = 0.1,
        num_task_tokens: int = 512,
        use_adaptive_bridge: bool = True,
        bridge_mode: str = "adaptive",
        fixed_layer_index: int = -1,
        use_latent_skill_token: bool = False,
        use_continuous_context: bool = False,
        skill_use_layer_routing: bool = True,
        skill_use_direct_conditioning: bool = True,
        continuous_context_use_direct_conditioning: bool = False,
        num_skill_tokens: int = 16,
        skill_token_dim: int = 128,
        skill_temperature: float = 1.0,
        skill_entropy_weight: float = 0.0,
        skill_assignment_mode: str = "hard_gumbel",
        skill_routing_mode: str = "legacy",
        skill_layer_temperature: float = 1.0,
        skill_temperature_start: float = -1.0,
        skill_temperature_anneal_steps: int = 0,
        skill_balance_weight: float = 0.0,
        skill_z_loss_weight: float = 0.0,
        skill_mi_weight: float = 0.0,
        skill_layer_mi_weight: float = 0.0,
        skill_template_diversity_weight: float = 0.0,
        routing_anchor_layer: int = -1,
        routing_adaptive_mix: float = 1.0,
        routing_curriculum_warmup_steps: int = 0,
        routing_curriculum_teacher_steps: int = 0,
        routing_curriculum_num_buckets: int = 5,
        routing_teacher_temperature: float = 0.2,
        routing_teacher_kl_weight: float = 1.0,
        adaptive_layer_alignment: bool = False,
        adaptive_num_layers: int = 25,
        adaptive_alignment_bottleneck: int = 64,
        flow_time_embedding_mode: str = "legacy",
        flow_time_sampling_mode: str = "uniform",
        flow_float32_path: bool = False,
        flow_zero_init_output: bool = True,
        num_inference_steps: int = 5,
        num_inference_samples: int = 8,
        supervised_anchor_weight: float = 0.0,
        anchor_blend: float = 0.0,
        anchor_gripper_weight: float = 1.0,
        anchor_rotation_weight: float = 1.0,
        anchor_gripper_bce_weight: float = 0.0,
        anchor_num_layers: int = 0,
        anchor_hidden_dim: int = 1024,
        detach_flow_conditioning: bool = False,
        flow_curriculum_start_step: int = 0,
        flow_curriculum_ramp_steps: int = 0,
        include_prompt_tokens: bool = False,
        task_token_mode: str = "vision_prompt",
        prompt_direct_conditioning: bool = False,
        dense_film_enabled: bool = False,
        dense_film_max_layers: int = 64,
        dense_film_first_layer_index: int = 1,
        dense_film_bottleneck_dim: int = 64,
        dense_film_state_dim: int = 128,
        dense_film_state_mode: str = "full",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_dim = time_dim
        self.num_task_tokens = num_task_tokens
        self.num_vision_tokens = num_task_tokens
        self.use_adaptive_bridge = use_adaptive_bridge
        self.bridge_mode = bridge_mode
        self.fixed_layer_index = fixed_layer_index
        self.use_latent_skill_token = bool(use_latent_skill_token)
        self.use_continuous_context = bool(use_continuous_context)
        if self.use_latent_skill_token and self.use_continuous_context:
            raise ValueError("latent skill tokens and continuous context are mutually exclusive")
        self.skill_use_layer_routing = bool(skill_use_layer_routing)
        self.skill_use_direct_conditioning = bool(skill_use_direct_conditioning)
        self.continuous_context_use_direct_conditioning = bool(continuous_context_use_direct_conditioning)
        self.num_skill_tokens = int(num_skill_tokens)
        self.skill_token_dim = int(skill_token_dim)
        self.skill_temperature = float(skill_temperature)
        self.skill_entropy_weight = float(skill_entropy_weight)
        self.skill_assignment_mode = str(skill_assignment_mode).lower()
        if self.skill_assignment_mode not in {"hard_gumbel", "soft"}:
            raise ValueError(
                "skill_assignment_mode must be 'hard_gumbel' or 'soft'; "
                f"got {skill_assignment_mode!r}"
            )
        self.skill_routing_mode = str(skill_routing_mode).lower()
        if self.skill_routing_mode not in {"legacy", "prototype_soft", "query_key_soft"}:
            raise ValueError(
                "skill_routing_mode must be 'legacy', 'prototype_soft', or "
                "'query_key_soft'; "
                f"got {skill_routing_mode!r}"
            )
        if self.skill_routing_mode in {"prototype_soft", "query_key_soft"} and self.skill_use_direct_conditioning:
            raise ValueError(
                f"{self.skill_routing_mode} is a routing-only interface; "
                "skill_use_direct_conditioning must be False"
            )
        self.skill_layer_temperature = float(skill_layer_temperature)
        self.skill_temperature_start = float(skill_temperature_start)
        self.skill_temperature_anneal_steps = int(skill_temperature_anneal_steps)
        self.skill_balance_weight = float(skill_balance_weight)
        self.skill_z_loss_weight = float(skill_z_loss_weight)
        self.skill_mi_weight = float(skill_mi_weight)
        self.skill_layer_mi_weight = float(skill_layer_mi_weight)
        self.skill_template_diversity_weight = float(skill_template_diversity_weight)
        self._routing_step = 0
        self.routing_anchor_layer = int(routing_anchor_layer)
        self.routing_adaptive_mix = float(routing_adaptive_mix)
        if not 0.0 <= self.routing_adaptive_mix <= 1.0:
            raise ValueError("routing_adaptive_mix must be in [0, 1]")
        self.routing_curriculum_warmup_steps = max(int(routing_curriculum_warmup_steps), 0)
        self.routing_curriculum_teacher_steps = max(int(routing_curriculum_teacher_steps), 0)
        self.routing_curriculum_num_buckets = max(int(routing_curriculum_num_buckets), 1)
        self.routing_teacher_temperature = max(float(routing_teacher_temperature), 1e-6)
        self.routing_teacher_kl_weight = max(float(routing_teacher_kl_weight), 0.0)
        self.adaptive_layer_alignment = bool(adaptive_layer_alignment)
        self.adaptive_num_layers = int(adaptive_num_layers)
        self.adaptive_alignment_bottleneck = int(adaptive_alignment_bottleneck)
        self.flow_time_embedding_mode = str(flow_time_embedding_mode).lower()
        self.flow_time_sampling_mode = str(flow_time_sampling_mode).lower()
        if self.flow_time_embedding_mode not in {"legacy", "continuous"}:
            raise ValueError("flow_time_embedding_mode must be 'legacy' or 'continuous'")
        if self.flow_time_sampling_mode not in {"uniform", "openpi_beta"}:
            raise ValueError("flow_time_sampling_mode must be 'uniform' or 'openpi_beta'")
        self.flow_float32_path = bool(flow_float32_path)
        self.flow_zero_init_output = bool(flow_zero_init_output)
        self.num_inference_steps = int(num_inference_steps)
        self.num_inference_samples = int(num_inference_samples)
        self.supervised_anchor_weight = float(supervised_anchor_weight)
        self.anchor_blend = float(anchor_blend)
        self.anchor_gripper_weight = float(anchor_gripper_weight)
        self.anchor_rotation_weight = float(anchor_rotation_weight)
        self.anchor_gripper_bce_weight = float(anchor_gripper_bce_weight)
        self.anchor_num_layers = int(anchor_num_layers)
        self.anchor_hidden_dim = int(anchor_hidden_dim)
        self.detach_flow_conditioning = bool(detach_flow_conditioning)
        self.flow_curriculum_start_step = max(int(flow_curriculum_start_step), 0)
        self.flow_curriculum_ramp_steps = max(int(flow_curriculum_ramp_steps), 0)
        # Opt-in prompt states allow the router to see the requested object.
        # The old visual-only packing remains the default for compatibility.
        self.include_prompt_tokens = bool(include_prompt_tokens)
        self.task_token_mode = str(task_token_mode)
        self.prompt_direct_conditioning = bool(prompt_direct_conditioning)
        self.dense_film_enabled = bool(dense_film_enabled)
        self.dense_film_state_mode = str(dense_film_state_mode).lower()
        if self.task_token_mode not in {"vision_prompt", "vision_only", "prompt_only", "last_prompt"}:
            raise ValueError(f"Unsupported FlowMLP task_token_mode: {task_token_mode}")
        # A separate prompt path prevents the small instruction tail from being
        # diluted by the much larger visual-token pool used by layer routing.
        self.prompt_condition_proj = (
            nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, input_dim))
            if self.prompt_direct_conditioning
            else None
        )
        self.dense_depth_film = (
            StateConditionedDenseDepthFiLM(
                hidden_dim=input_dim,
                num_task_tokens=num_task_tokens,
                max_layers=dense_film_max_layers,
                first_layer_index=dense_film_first_layer_index,
                bottleneck_dim=dense_film_bottleneck_dim,
                state_dim=dense_film_state_dim,
                state_mode=self.dense_film_state_mode,
            )
            if self.dense_film_enabled
            else None
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim, mode=self.flow_time_embedding_mode),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Linear(input_dim + output_dim + time_dim, hidden_dim)
        self.blocks = nn.ModuleList([MLPResBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        if self.supervised_anchor_weight > 0.0 or self.anchor_blend > 0.0:
            if self.anchor_num_layers > 0:
                self.anchor_proj = DeepRoutedActionDecoder(
                    input_dim=input_dim,
                    hidden_dim=self.anchor_hidden_dim,
                    output_dim=output_dim,
                    num_layers=self.anchor_num_layers,
                    dropout=dropout,
                )
            else:
                self.anchor_proj = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, output_dim),
                )
        else:
            self.anchor_proj = None

        # Task layer selector for adaptive modes
        self.task_layer_selector = TaskLayerSelector(hidden_dim=input_dim)
        self.layerwise_aligner = (
            LayerwiseResidualAligner(
                num_layers=self.adaptive_num_layers,
                hidden_dim=input_dim,
                bottleneck_dim=self.adaptive_alignment_bottleneck,
            )
            if self.adaptive_layer_alignment
            else None
        )
        self.prototype_layer_projector = (
            LayerwiseNormalizedProjector(
                num_layers=self.adaptive_num_layers,
                hidden_dim=input_dim,
                bottleneck_dim=self.adaptive_alignment_bottleneck,
            )
            if self.use_latent_skill_token
            and self.skill_routing_mode in {"prototype_soft", "query_key_soft"}
            else None
        )

        if self.use_latent_skill_token:
            self.skill_selector = nn.Sequential(
                nn.LayerNorm(input_dim * 2),
                nn.Linear(input_dim * 2, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, self.num_skill_tokens),
            )
            if self.skill_routing_mode == "prototype_soft":
                self.skill_embedding = None
                # Each latent control mode owns a distribution over VLM depths.
                # The selector is the only path from (global visual, proprio) to
                # layer routing; no layer scorer can bypass the bottleneck.
                self.skill_layer_logits = nn.Parameter(
                    torch.empty(self.num_skill_tokens, self.adaptive_num_layers)
                )
                nn.init.normal_(self.skill_layer_logits, mean=0.0, std=0.02)
                self.skill_layer_scorer = None
                self.register_buffer(
                    "_skill_usage_ema",
                    torch.full((self.num_skill_tokens,), 1.0 / self.num_skill_tokens),
                    persistent=True,
                )
                for parameter in self.task_layer_selector.parameters():
                    parameter.requires_grad = False
            elif self.skill_routing_mode == "query_key_soft":
                self.skill_embedding = nn.Embedding(self.num_skill_tokens, self.skill_token_dim)
                self.skill_layer_logits = nn.Parameter(
                    torch.empty(1, self.adaptive_num_layers)
                )
                depth = torch.arange(self.adaptive_num_layers, dtype=torch.float32)
                prior_center = min(13, self.adaptive_num_layers - 1)
                with torch.no_grad():
                    self.skill_layer_logits.copy_(
                        -0.05 * ((depth - float(prior_center)) / 4.0).square()
                    )
                self.skill_query_key_log_scale = nn.Parameter(
                    torch.tensor(math.log(10.0), dtype=torch.float32)
                )
                self.skill_query_projection = nn.Sequential(
                    nn.LayerNorm(self.skill_token_dim + input_dim),
                    nn.Linear(self.skill_token_dim + input_dim, self.skill_token_dim),
                    nn.GELU(),
                    nn.Linear(self.skill_token_dim, self.skill_token_dim),
                )
                self.skill_key_projection = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, self.skill_token_dim),
                )
                self.skill_layer_scorer = None
                self.register_buffer(
                    "_skill_usage_ema",
                    torch.full((self.num_skill_tokens,), 1.0 / self.num_skill_tokens),
                    persistent=True,
                )
                for parameter in self.task_layer_selector.parameters():
                    parameter.requires_grad = False
            else:
                self.skill_embedding = nn.Embedding(self.num_skill_tokens, self.skill_token_dim)
                self.skill_layer_logits = None
                self.skill_query_projection = None
                self.skill_key_projection = None
                self.skill_query_key_log_scale = None
                self.skill_layer_scorer = nn.Sequential(
                    nn.LayerNorm(input_dim * 2 + self.skill_token_dim),
                    nn.Linear(input_dim * 2 + self.skill_token_dim, input_dim // 2),
                    nn.GELU(),
                    nn.Linear(input_dim // 2, 1),
                )
            # Keep the selected skill available to the action predictor itself.
            # Zero initialization preserves old checkpoint behavior at step zero.
            if self.skill_routing_mode in {"prototype_soft", "query_key_soft"}:
                self.skill_condition_proj = None
            else:
                self.skill_condition_proj = nn.Sequential(
                    nn.LayerNorm(self.skill_token_dim),
                    nn.Linear(self.skill_token_dim, input_dim),
                )
                nn.init.zeros_(self.skill_condition_proj[-1].weight)
                nn.init.zeros_(self.skill_condition_proj[-1].bias)
        else:
            self.skill_selector = None
            self.skill_embedding = None
            self.skill_layer_scorer = None
            self.skill_layer_logits = None
            self.skill_condition_proj = None
            self.skill_query_projection = None
            self.skill_key_projection = None
            self.skill_query_key_log_scale = None

        # E07 controlled continuous visual--proprioceptive context.  Its selector,
        # projection, scorer, and optional direct-condition projection deliberately
        # mirror the R2 skill modules, while its routing context stays continuous:
        # no Gumbel-Softmax, argmax, one-hot, or discrete skill id is used.
        if self.use_continuous_context:
            self.continuous_context_selector = nn.Sequential(
                nn.LayerNorm(input_dim * 2),
                nn.Linear(input_dim * 2, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, self.num_skill_tokens),
            )
            self.continuous_context_projection = nn.Linear(
                self.num_skill_tokens, self.skill_token_dim, bias=False
            )
            self.continuous_context_layer_scorer = nn.Sequential(
                nn.LayerNorm(input_dim * 2 + self.skill_token_dim),
                nn.Linear(input_dim * 2 + self.skill_token_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, 1),
            )
            self.continuous_context_condition_proj = nn.Sequential(
                nn.LayerNorm(self.skill_token_dim),
                nn.Linear(self.skill_token_dim, input_dim),
            )
            nn.init.zeros_(self.continuous_context_condition_proj[-1].weight)
            nn.init.zeros_(self.continuous_context_condition_proj[-1].bias)
        else:
            self.continuous_context_selector = None
            self.continuous_context_projection = None
            self.continuous_context_layer_scorer = None
            self.continuous_context_condition_proj = None

        if self.flow_zero_init_output:
            nn.init.zeros_(self.output_proj[-1].weight)
            nn.init.zeros_(self.output_proj[-1].bias)

    def set_num_task_tokens(
        self, num_task_tokens: int, num_vision_tokens: Optional[int] = None
    ) -> None:
        """Keep the task/action split aligned with multimodal prompt packing."""
        if int(num_task_tokens) <= 0:
            raise ValueError("num_task_tokens must be positive")
        self.num_task_tokens = int(num_task_tokens)
        self.num_vision_tokens = int(
            num_task_tokens if num_vision_tokens is None else num_vision_tokens
        )
        if not 0 < self.num_vision_tokens <= self.num_task_tokens:
            raise ValueError("num_vision_tokens must be in [1, num_task_tokens]")

    def _anchor_l1_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = (predicted.float() - target.float()).abs()
        error[..., 3:6] *= self.anchor_rotation_weight
        error[..., -1] *= self.anchor_gripper_weight
        loss = error.mean()
        if self.anchor_gripper_bce_weight > 0.0:
            gripper_target = (target[..., -1] > 0).to(dtype=torch.float32)
            gripper_bce = F.binary_cross_entropy_with_logits(predicted[..., -1].float(), gripper_target)
            loss = loss + self.anchor_gripper_bce_weight * gripper_bce
        return loss

    def set_routing_step(self, step: int) -> None:
        """Update the optimizer-step used by the soft-router temperature schedule."""
        self._routing_step = max(int(step), 0)

    def _current_skill_temperature(self) -> float:
        end = max(float(self.skill_temperature), 1e-6)
        start = end if self.skill_temperature_start <= 0.0 else self.skill_temperature_start
        if self.skill_temperature_anneal_steps <= 0:
            return end
        progress = min(self._routing_step / float(self.skill_temperature_anneal_steps), 1.0)
        return max(start + progress * (end - start), 1e-6)

    def _skill_layer_weights(
        self,
        pooled_layers: torch.Tensor,
        proprio_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        dtype = next(self.parameters()).dtype
        pooled_layers = pooled_layers.to(dtype)
        proprio_feat = proprio_feat.to(dtype)
        bsz, num_layers, dim = pooled_layers.shape

        global_context = pooled_layers.mean(dim=1)
        skill_input = torch.cat([global_context, proprio_feat], dim=-1)
        logits = self.skill_selector(skill_input)
        # Always retain the soft posterior for diagnostics.  Inference still uses
        # argmax/one-hot routing below, so this record-only change cannot alter
        # actions or checkpoint behavior.
        soft_probs = torch.softmax(logits.float(), dim=-1)
        self._last_skill_probs = soft_probs.detach().float()
        temperature = self._current_skill_temperature()
        if self.skill_routing_mode in {"prototype_soft", "query_key_soft"}:
            # Training and inference use exactly the same differentiable
            # posterior.  Robot state and global visual context cannot reach
            # layer routing through any other branch in this mode.
            soft_probs = torch.softmax(logits.float() / temperature, dim=-1)
            self._last_skill_probs = soft_probs.detach().float()
            skill_probs = soft_probs.to(dtype)
        elif self.skill_assignment_mode == "soft":
            # Use the same differentiable posterior in training and inference.
            # This removes the hard-Gumbel/argmax distribution shift while
            # preserving the legacy behavior behind the default mode.
            skill_probs = soft_probs.to(dtype)
        elif self.training:
            skill_probs = F.gumbel_softmax(logits.float(), tau=temperature, hard=True, dim=-1).to(dtype)
        else:
            skill_ids = logits.argmax(dim=-1)
            skill_probs = F.one_hot(skill_ids, num_classes=self.num_skill_tokens).to(dtype)

        # Inference diagnostics are intentionally kept out of the model output and
        # detached immediately.  The evaluator can read this snapshot after each
        # action query without changing the action computation or autograd graph.
        self._last_skill_ids = skill_probs.argmax(dim=-1).detach().long()

        skill_emb = (
            None
            if self.skill_embedding is None
            else skill_probs @ self.skill_embedding.weight.to(dtype)
        )
        routing_aux_loss = logits.new_tensor(0.0, dtype=torch.float32)
        layer_mi_loss = logits.new_tensor(0.0, dtype=torch.float32)
        template_max_cosine = logits.new_tensor(0.0, dtype=torch.float32)
        if self.skill_routing_mode == "prototype_soft":
            actual_layers = pooled_layers.shape[1]
            if actual_layers > self.skill_layer_logits.shape[1]:
                raise ValueError(
                    f"received {actual_layers} VLM depths but prototype router supports "
                    f"{self.skill_layer_logits.shape[1]}"
                )
            template_logits = self.skill_layer_logits[:, :actual_layers].float()
            templates = torch.softmax(
                template_logits / max(self.skill_layer_temperature, 1e-6),
                dim=-1,
            )
            layer_weights = skill_probs.float() @ templates
            layer_weights = layer_weights.to(dtype)

            batch_usage = soft_probs.mean(dim=0)
            uniform = torch.full_like(batch_usage, 1.0 / self.num_skill_tokens)
            balance_loss = self.num_skill_tokens * (batch_usage - uniform).square().sum()
            z_loss = torch.logsumexp(logits.float(), dim=-1).square().mean()
            conditional_entropy = -(
                soft_probs * torch.log(soft_probs.clamp_min(1e-8))
            ).sum(dim=-1).mean()
            marginal_entropy = -(
                batch_usage * torch.log(batch_usage.clamp_min(1e-8))
            ).sum()
            mi_loss = conditional_entropy - marginal_entropy

            normalized_templates = F.normalize(templates, dim=-1)
            template_gram = normalized_templates @ normalized_templates.transpose(0, 1)
            off_diagonal = ~torch.eye(
                self.num_skill_tokens,
                device=template_gram.device,
                dtype=torch.bool,
            )
            diversity_loss = template_gram[off_diagonal].square().mean()
            template_max_cosine = template_gram[off_diagonal].max().detach()

            routing_aux_loss = (
                self.skill_balance_weight * balance_loss
                + self.skill_z_loss_weight * z_loss
                + self.skill_mi_weight * mi_loss
                + self.skill_template_diversity_weight * diversity_loss
            )
            with torch.no_grad():
                self._skill_usage_ema.mul_(0.99).add_(batch_usage.detach().to(self._skill_usage_ema), alpha=0.01)
        elif self.skill_routing_mode == "query_key_soft":
            actual_layers = pooled_layers.shape[1]
            if actual_layers > self.skill_layer_logits.shape[1]:
                raise ValueError(
                    f"received {actual_layers} VLM depths but query-key router supports "
                    f"{self.skill_layer_logits.shape[1]}"
                )
            query_input = torch.cat([skill_emb, proprio_feat], dim=-1)
            query = F.normalize(
                self.skill_query_projection(query_input).float(), dim=-1
            )
            keys = F.normalize(
                self.skill_key_projection(pooled_layers).float(), dim=-1
            )
            query_key_scale = self.skill_query_key_log_scale.float().exp().clamp(max=100.0)
            similarity = query_key_scale * torch.einsum("bd,bld->bl", query, keys)
            prior = self.skill_layer_logits[:, :actual_layers].float()
            layer_scores = similarity + prior
            layer_weights = torch.softmax(
                layer_scores / max(self.skill_layer_temperature, 1e-6), dim=-1
            ).to(dtype)
            layer_probs = layer_weights.float()
            layer_batch_usage = layer_probs.mean(dim=0)
            layer_conditional_entropy = -(
                layer_probs * torch.log(layer_probs.clamp_min(1e-8))
            ).sum(dim=-1).mean()
            layer_marginal_entropy = -(
                layer_batch_usage * torch.log(layer_batch_usage.clamp_min(1e-8))
            ).sum()
            layer_mi_loss = layer_conditional_entropy - layer_marginal_entropy

            batch_usage = soft_probs.mean(dim=0)
            uniform = torch.full_like(batch_usage, 1.0 / self.num_skill_tokens)
            balance_loss = self.num_skill_tokens * (batch_usage - uniform).square().sum()
            z_loss = torch.logsumexp(logits.float(), dim=-1).square().mean()
            conditional_entropy = -(
                soft_probs * torch.log(soft_probs.clamp_min(1e-8))
            ).sum(dim=-1).mean()
            marginal_entropy = -(
                batch_usage * torch.log(batch_usage.clamp_min(1e-8))
            ).sum()
            mi_loss = conditional_entropy - marginal_entropy
            diversity_loss = logits.new_tensor(0.0, dtype=torch.float32)
            if actual_layers > 1:
                key_gram = torch.einsum("bld,bmd->blm", keys, keys)
                key_off_diagonal = ~torch.eye(
                    actual_layers, device=keys.device, dtype=torch.bool
                ).unsqueeze(0)
                key_off_diagonal = key_off_diagonal.expand(keys.shape[0], -1, -1)
                template_max_cosine = key_gram[key_off_diagonal].max().detach()
            routing_aux_loss = (
                self.skill_balance_weight * balance_loss
                + self.skill_z_loss_weight * z_loss
                + self.skill_mi_weight * mi_loss
                + self.skill_layer_mi_weight * layer_mi_loss
            )
            with torch.no_grad():
                self._skill_usage_ema.mul_(0.99).add_(
                    batch_usage.detach().to(self._skill_usage_ema), alpha=0.01
                )
        elif self.skill_use_layer_routing:
            expanded_skill = skill_emb.unsqueeze(1).expand(-1, num_layers, -1)
            expanded_proprio = proprio_feat.unsqueeze(1).expand(-1, num_layers, -1)
            scorer_input = torch.cat([pooled_layers, expanded_skill, expanded_proprio], dim=-1)
            layer_scores = self.skill_layer_scorer(scorer_input).squeeze(-1)
            layer_weights = torch.softmax(layer_scores, dim=1)
        else:
            # Keep the non-skill adaptive bridge as the controlled routing baseline.
            task_states = pooled_layers.unsqueeze(2)
            _, layer_weights = self.task_layer_selector(task_states)

        if self.routing_anchor_layer >= 0:
            if self.routing_anchor_layer >= num_layers:
                raise ValueError(
                    f"routing_anchor_layer={self.routing_anchor_layer} is outside "
                    f"the available layer range [0, {num_layers - 1}]"
                )
            anchor = F.one_hot(
                torch.full(
                    (bsz,),
                    self.routing_anchor_layer,
                    device=pooled_layers.device,
                    dtype=torch.long,
                ),
                num_classes=num_layers,
            ).to(dtype=layer_weights.dtype)
            layer_weights = (
                (1.0 - self.routing_adaptive_mix) * anchor
                + self.routing_adaptive_mix * layer_weights
            )

        entropy = -(soft_probs * torch.log(soft_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        max_prob = soft_probs.max(dim=-1).values.mean()
        expected_id = (soft_probs * torch.arange(self.num_skill_tokens, device=pooled_layers.device, dtype=soft_probs.dtype)).sum(dim=-1).mean()
        layer_weights_float = layer_weights.float()
        layer_entropy = -(
            layer_weights_float * torch.log(layer_weights_float.clamp_min(1e-8))
        ).sum(dim=-1).mean()
        layer_indices = torch.arange(
            num_layers,
            device=layer_weights.device,
            dtype=layer_weights_float.dtype,
        )
        expected_depth = (layer_weights_float * layer_indices).sum(dim=-1).mean()
        early_layer_mass = layer_weights_float[:, : min(3, num_layers)].sum(dim=-1).mean()
        layer_batch_variation = layer_weights_float.std(dim=0, unbiased=False).mean()
        metrics = {
            "skill_entropy": entropy.detach(),
            "skill_max_prob": max_prob.detach(),
            "skill_expected_id": expected_id.detach(),
            "routing_adaptive_mix": layer_weights.new_tensor(self.routing_adaptive_mix).detach(),
            "routing_layer_entropy": layer_entropy.detach(),
            "routing_expected_depth": expected_depth.detach(),
            "routing_early3_mass": early_layer_mass.detach(),
            "routing_layer_batch_variation": layer_batch_variation.detach(),
            "skill_temperature": layer_weights.new_tensor(temperature).detach(),
            "skill_effective_count": torch.exp(entropy.detach()),
            "skill_template_max_cosine": template_max_cosine,
        }
        entropy_loss = -self.skill_entropy_weight * entropy + routing_aux_loss
        if self.skill_entropy_weight != 0.0:
            metrics["skill_entropy_loss"] = entropy_loss.detach()
        if self.skill_routing_mode in {"prototype_soft", "query_key_soft"}:
            metrics.update(
                {
                    "skill_balance_loss": balance_loss.detach(),
                    "skill_z_loss": z_loss.detach(),
                    "skill_mi_loss": mi_loss.detach(),
                    "skill_template_diversity_loss": diversity_loss.detach(),
                    "skill_routing_aux_loss": routing_aux_loss.detach(),
                    "routing_layer_mi_loss": layer_mi_loss.detach(),
                    "skill_usage_ema_max": self._skill_usage_ema.max().detach(),
                }
            )
        return layer_weights, metrics, entropy_loss, skill_emb

    def _continuous_context_layer_weights(
        self,
        pooled_layers: torch.Tensor,
        proprio_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Route with a continuous visual--proprioceptive context for E07 R1."""
        dtype = next(self.parameters()).dtype
        pooled_layers = pooled_layers.to(dtype)
        proprio_feat = proprio_feat.to(dtype)
        _, num_layers, _ = pooled_layers.shape

        global_context = pooled_layers.mean(dim=1)
        context_input = torch.cat([global_context, proprio_feat], dim=-1)
        # tanh retains a continuous MLP vector without normalizing it into a
        # categorical posterior.  The projection has the same shape as R2's
        # skill embedding table so the controlled router parameter counts match.
        context = self.continuous_context_projection(
            torch.tanh(self.continuous_context_selector(context_input))
        )
        self._last_continuous_context = context.detach().float()
        expanded_context = context.unsqueeze(1).expand(-1, num_layers, -1)
        expanded_proprio = proprio_feat.unsqueeze(1).expand(-1, num_layers, -1)
        scorer_input = torch.cat([pooled_layers, expanded_context, expanded_proprio], dim=-1)
        layer_scores = self.continuous_context_layer_scorer(scorer_input).squeeze(-1)
        layer_weights = torch.softmax(layer_scores, dim=1)
        metrics = {"continuous_context_l2": context.float().norm(dim=-1).mean().detach()}
        return layer_weights, metrics, context

    @staticmethod
    def _masked_pool_task_tokens(
        task_states: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool valid task tokens without letting zero padding dilute features."""
        mask = task_mask.to(device=task_states.device, dtype=task_states.dtype).unsqueeze(-1)
        valid_counts = mask.sum(dim=2).clamp_min(1.0)
        return (task_states * mask).sum(dim=2) / valid_counts

    def forward(self, state_emb, noisy_actions, timesteps):
        dtype = next(self.parameters()).dtype
        state_emb = state_emb.to(dtype)
        noisy_actions = noisy_actions.to(dtype)

        _, seq_len, _ = state_emb.shape

        # Preserve float32 continuous timesteps through the sinusoidal mapping,
        # then cast once before the learned BF16 layers.
        timestep_input = timesteps.float() if self.flow_float32_path else timesteps.to(dtype)
        t_emb = self.time_mlp[0](timestep_input).to(dtype)
        for layer in self.time_mlp[1:]:
            t_emb = layer(t_emb)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)

        x = torch.cat([state_emb, noisy_actions, t_emb], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)

    def _align_state_emb_to_action_chunk(self, state_emb: torch.Tensor) -> torch.Tensor:
        bsz, k_token, dim = state_emb.shape
        if k_token == NUM_ACTIONS_CHUNK:
            return state_emb
        if k_token % NUM_ACTIONS_CHUNK != 0:
            raise ValueError(
                f"FlowMatchingMLPActionHead: action state token count {k_token} must be divisible by "
                f"NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}"
            )
        return state_emb.reshape(bsz, NUM_ACTIONS_CHUNK, k_token // NUM_ACTIONS_CHUNK, dim).mean(dim=2)

    def _compute_loss_components(
        self,
        state_emb: torch.Tensor,
        gt_actions: torch.Tensor,
        skill_metrics: Optional[Dict[str, torch.Tensor]] = None,
        skill_entropy_loss: Optional[torch.Tensor] = None,
        flow_t: Optional[torch.Tensor] = None,
        flow_x0: Optional[torch.Tensor] = None,
        return_per_sample: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dtype = next(self.parameters()).dtype
        device = state_emb.device
        state_emb = state_emb.to(dtype)
        compute_dtype = torch.float32 if self.flow_float32_path else dtype
        gt_actions = gt_actions.to(compute_dtype)
        state_emb = self._align_state_emb_to_action_chunk(state_emb)

        prompt_token_count = max(self.num_task_tokens - self.num_vision_tokens, 0)
        if (
            self.prompt_condition_proj is not None
            and self.include_prompt_tokens
            and prompt_token_count > 0
        ):
            prompt_states = adaptive_hidden_states[
                :, :, self.num_vision_tokens : self.num_task_tokens, :
            ]
            prompt_mask = prompt_states.float().abs().sum(dim=-1) > 0
            prompt_per_layer = self._masked_pool_task_tokens(prompt_states, prompt_mask)
            if layer_weights is not None:
                prompt_context = (
                    layer_weights.to(prompt_per_layer.dtype).unsqueeze(-1)
                    * prompt_per_layer
                ).sum(dim=1)
            elif not self.use_adaptive_bridge:
                prompt_context = prompt_per_layer[:, layer_idx, :]
            else:
                prompt_context = prompt_per_layer.mean(dim=1)
            prompt_residual = self.prompt_condition_proj(prompt_context.to(dtype))
            state_emb = state_emb + prompt_residual.unsqueeze(1).to(state_emb.dtype)
            skill_metrics["flowmlp_prompt_token_count"] = state_emb.new_tensor(
                float(prompt_token_count)
            )
            skill_metrics["flowmlp_prompt_residual_norm"] = (
                prompt_residual.float().norm(dim=-1).mean().detach()
            )

        batch_size, _, _ = gt_actions.shape

        if flow_t is not None and flow_x0 is not None:
            t = flow_t.to(device=device, dtype=compute_dtype)
            x_0 = flow_x0.to(device=device, dtype=compute_dtype)
        elif self.flow_time_sampling_mode == "openpi_beta":
            concentration1 = torch.tensor(1.5, device=device, dtype=torch.float32)
            concentration0 = torch.tensor(1.0, device=device, dtype=torch.float32)
            noise_fraction = torch.distributions.Beta(concentration1, concentration0).sample((batch_size,))
            noise_fraction = noise_fraction * 0.999 + 0.001
            # This implementation integrates from noise at t=0 to actions at
            # t=1, the reverse convention of OpenPI.
            t = 1.0 - noise_fraction
        else:
            t = torch.rand(batch_size, device=device, dtype=compute_dtype)
        if flow_t is None or flow_x0 is None:
            t = t.to(compute_dtype)
            x_0 = torch.randn_like(gt_actions, device=device, dtype=compute_dtype)
        t = t.to(compute_dtype)
        t_expanded = t.view(batch_size, 1, 1)
        x_0 = x_0.to(device=device, dtype=compute_dtype)

        anchor_actions = None
        flow_target_actions = gt_actions
        if self.anchor_proj is not None and self.supervised_anchor_weight > 0.0:
            anchor_actions = self.anchor_proj(state_emb)
            flow_target_actions = gt_actions - anchor_actions.detach().to(dtype=gt_actions.dtype)

        x_t = t_expanded * flow_target_actions + (1 - t_expanded) * x_0
        target_velocity = flow_target_actions - x_0

        flow_state_emb = state_emb.detach() if self.detach_flow_conditioning else state_emb
        pred_velocity = self.forward(flow_state_emb, x_t, t)
        per_sample_flow_loss = F.mse_loss(
            pred_velocity.float() if self.flow_float32_path else pred_velocity,
            target_velocity.float() if self.flow_float32_path else target_velocity,
            reduction="none",
        )
        per_sample_flow_loss = per_sample_flow_loss.mean(dim=(1, 2))
        flow_loss = per_sample_flow_loss.mean()
        if return_per_sample:
            return per_sample_flow_loss, {"flow_matching_loss": flow_loss.detach()}

        skill_metrics = skill_metrics or {}
        if self.flow_curriculum_start_step <= 0:
            flow_loss_weight = 1.0
        elif self._routing_step < self.flow_curriculum_start_step:
            flow_loss_weight = 0.0
        elif self.flow_curriculum_ramp_steps <= 0:
            flow_loss_weight = 1.0
        else:
            flow_loss_weight = min(
                (self._routing_step - self.flow_curriculum_start_step)
                / float(self.flow_curriculum_ramp_steps),
                1.0,
            )
        metrics: Dict[str, torch.Tensor] = {
            "flow_matching_loss": flow_loss.detach(),
            "flow_curriculum_weight": flow_loss.detach().new_tensor(flow_loss_weight),
            **skill_metrics,
        }
        total_loss = flow_loss * flow_loss_weight
        if anchor_actions is not None:
            anchor_loss = self._anchor_l1_loss(anchor_actions, gt_actions)
            total_loss = total_loss + self.supervised_anchor_weight * anchor_loss.to(dtype=total_loss.dtype)
            metrics["flowmlp_anchor_l1_loss"] = anchor_loss.detach()
            metrics["flow_matching_total_loss"] = total_loss.detach()
        if (
            self.use_latent_skill_token
            and skill_entropy_loss is not None
            and (
                self.skill_entropy_weight != 0.0
                or self.skill_balance_weight != 0.0
                or self.skill_z_loss_weight != 0.0
                or self.skill_mi_weight != 0.0
                or self.skill_layer_mi_weight != 0.0
                or self.skill_template_diversity_weight != 0.0
                or self.routing_curriculum_teacher_steps > 0
            )
        ):
            total_loss = total_loss + skill_entropy_loss.to(dtype=flow_loss.dtype)
            metrics["flow_matching_total_loss"] = total_loss.detach()

        return total_loss, metrics

    def _expected_depth_risk_loss(
        self,
        actions_hidden_states: torch.Tensor,
        target_actions: torch.Tensor,
        proprio: Optional[torch.Tensor],
        proprio_projector: Optional[nn.Module],
    ) -> Tuple[torch.Tensor, dict]:
        """Train routing from candidate action risks without mixing hidden states.

        Every candidate depth is passed independently through the shared FlowMLP.
        The router receives the expected per-depth Flow loss, while inference uses
        the single top-probability depth in ``_extract_state_emb``.
        """
        bsz, num_layers = actions_hidden_states.shape[:2]
        dtype = next(self.parameters()).dtype
        dim = actions_hidden_states.shape[-1]
        if proprio is not None and proprio_projector is not None:
            proprio_flat = proprio.reshape(bsz, -1).to(
                device=actions_hidden_states.device, dtype=dtype
            )
            proprio_feat = proprio_projector(proprio_flat)
        else:
            proprio_feat = actions_hidden_states.new_zeros(bsz, dim).to(dtype)

        task_states = actions_hidden_states[:, :, : self.num_task_tokens, :]
        task_mask = task_states.float().abs().sum(dim=-1) > 0
        _, layer_weights = self.task_layer_selector(task_states, task_mask=task_mask)
        # Expected-risk routing is deliberately delayed until every candidate
        # depth has received a comparable amount of decoder training.  During
        # warmup the selector is not allowed to create a winner-take-all
        # gradient path; all candidates contribute equally and inference
        # remains unchanged (top-1 selection is only used after warmup).
        warmup_steps = int(getattr(self, "routing_curriculum_warmup_steps", 0))
        if self.training and int(self._routing_step) < warmup_steps:
            layer_weights = torch.full_like(layer_weights, 1.0 / float(num_layers))

        candidate = actions_hidden_states[:, :, self.num_task_tokens :, :]
        candidate = candidate.reshape(bsz * num_layers, candidate.shape[2], candidate.shape[3])
        candidate = self._align_state_emb_to_action_chunk(candidate)
        candidate = candidate + proprio_feat.to(candidate.dtype).repeat_interleave(
            num_layers, dim=0
        ).unsqueeze(1)

        target_rep = target_actions.repeat_interleave(num_layers, dim=0)
        compute_dtype = torch.float32 if self.flow_float32_path else dtype
        if self.flow_time_sampling_mode == "openpi_beta":
            concentration1 = torch.tensor(1.5, device=target_actions.device, dtype=torch.float32)
            concentration0 = torch.tensor(1.0, device=target_actions.device, dtype=torch.float32)
            t = 1.0 - torch.distributions.Beta(concentration1, concentration0).sample((bsz,))
            t = t * 0.999 + 0.001
        else:
            t = torch.rand(bsz, device=target_actions.device, dtype=compute_dtype)
        x0 = torch.randn_like(target_actions, dtype=compute_dtype)
        t_rep = t.repeat_interleave(num_layers, dim=0)
        x0_rep = x0.repeat_interleave(num_layers, dim=0)
        per_candidate, _ = self._compute_loss_components(
            candidate,
            target_rep,
            flow_t=t_rep,
            flow_x0=x0_rep,
            return_per_sample=True,
        )
        per_layer = per_candidate.reshape(bsz, num_layers)
        expected_loss = (layer_weights.float() * per_layer.float()).sum(dim=1).mean()
        metrics = self._actual_layer_metrics(layer_weights)
        metrics.update(
            {
                "routing_expected_depth_risk": expected_loss.detach(),
                "routing_selected_depth": layer_weights.argmax(dim=1).float().mean().detach(),
                "flow_matching_loss": expected_loss.detach(),
                "flow_matching_total_loss": expected_loss.detach(),
                "routing_curriculum_phase": expected_loss.new_tensor(
                    1.0 if self.training and int(self._routing_step) < warmup_steps else 3.0
                ),
            }
        )
        return expected_loss.to(dtype=next(self.parameters()).dtype), metrics

    def compute_loss(self, state_emb, gt_actions):
        loss, _ = self._compute_loss_components(state_emb, gt_actions)
        return loss

    def flow_matching_loss_from_state(
        self,
        state_emb: torch.Tensor,
        target_actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
        aux_loss: Optional[torch.Tensor] = None,
        aux_metrics: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        loss, loss_dict = self._compute_loss_components(
            state_emb,
            target_actions,
            skill_metrics=aux_metrics,
            skill_entropy_loss=None,
        )
        if aux_loss is not None:
            loss = loss + aux_loss.to(dtype=loss.dtype)
            loss_dict["flow_matching_total_loss"] = loss.detach()
        return loss, loss_dict

    def predict_action_from_state(
        self,
        state_emb: torch.Tensor,
        num_steps: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        return self._predict_action_internal(state_emb, num_steps=num_steps)

    def _curriculum_bucket_layers(self, num_layers: int, device: torch.device) -> torch.Tensor:
        """Return one representative depth from each approximately equal bucket."""
        num_buckets = min(self.routing_curriculum_num_buckets, num_layers)
        bucket_ids = torch.arange(num_buckets, device=device, dtype=torch.float32)
        centers = torch.floor((bucket_ids + 0.5) * float(num_layers) / float(num_buckets)).long()
        return centers.clamp_(0, num_layers - 1)

    @staticmethod
    def _actual_layer_metrics(layer_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Diagnostics for the weights that actually reach the action decoder."""
        weights = layer_weights.float()
        num_layers = weights.shape[-1]
        indices = torch.arange(num_layers, device=weights.device, dtype=weights.dtype)
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
        return {
            "routing_layer_entropy": entropy.detach(),
            "routing_expected_depth": (weights * indices).sum(dim=-1).mean().detach(),
            "routing_early3_mass": weights[:, : min(3, num_layers)].sum(dim=-1).mean().detach(),
            "routing_layer_batch_variation": weights.std(dim=0, unbiased=False).mean().detach(),
        }

    def _apply_depth_curriculum(
        self,
        adaptive_hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        target_actions: Optional[torch.Tensor],
        proprio_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], bool]:
        """Warm the policy across depth, then teach routing with action utility.

        During the first phase, samples are deterministically stratified across
        coarse depth buckets, so the decoder cannot specialize to one anchor
        layer.  During the second phase, the already-warmed supervised decoder
        scores each bucket representative.  A detached softmin of those action
        errors teaches the router and is annealed into the router's own weights.
        The curriculum is training-only; inference always uses the learned router.
        """
        zero = router_weights.new_tensor(0.0, dtype=torch.float32)
        metrics: Dict[str, torch.Tensor] = {
            "routing_curriculum_phase": zero.detach(),
            "routing_curriculum_teacher_kl": zero.detach(),
            "routing_curriculum_teacher_entropy": zero.detach(),
        }
        if (
            not self.training
            or target_actions is None
            or self.routing_curriculum_warmup_steps + self.routing_curriculum_teacher_steps <= 0
        ):
            metrics.update(self._actual_layer_metrics(router_weights))
            return router_weights, zero, metrics, False

        bsz, num_layers = router_weights.shape
        bucket_layers = self._curriculum_bucket_layers(num_layers, router_weights.device)
        num_buckets = int(bucket_layers.numel())
        step = int(self._routing_step)

        if step < self.routing_curriculum_warmup_steps:
            # Rotate bucket assignment with the optimizer step.  This is
            # deterministic across retries while covering every depth region.
            bucket_choice = (
                torch.arange(bsz, device=router_weights.device) + step
            ) % num_buckets
            selected_layers = bucket_layers[bucket_choice]
            weights = F.one_hot(selected_layers, num_classes=num_layers).to(router_weights.dtype)
            metrics["routing_curriculum_phase"] = zero.new_tensor(1.0)
            metrics.update(self._actual_layer_metrics(weights))
            return weights, zero, metrics, True

        teacher_end = self.routing_curriculum_warmup_steps + self.routing_curriculum_teacher_steps
        if step < teacher_end and self.anchor_proj is not None:
            # Candidate action states are taken from the same aligned tensor and
            # the same shared decoder.  No extra VLM forward pass or action head
            # is introduced.
            candidate_states = adaptive_hidden_states[
                :, bucket_layers, self.num_task_tokens :, :
            ]
            bsz, num_buckets, num_tokens, dim = candidate_states.shape
            candidate_states = candidate_states.reshape(bsz * num_buckets, num_tokens, dim)
            candidate_states = self._align_state_emb_to_action_chunk(candidate_states)
            candidate_states = candidate_states + proprio_feat.to(
                dtype=candidate_states.dtype
            ).repeat_interleave(num_buckets, dim=0).unsqueeze(1)

            anchor_was_training = self.anchor_proj.training
            self.anchor_proj.eval()
            with torch.no_grad():
                candidate_actions = self.anchor_proj(candidate_states)
                target = target_actions.to(candidate_actions.dtype).repeat_interleave(
                    num_buckets, dim=0
                )
                action_error = (candidate_actions - target).abs().mean(dim=(1, 2))
                action_error = action_error.reshape(bsz, num_buckets)
                teacher_bucket_probs = torch.softmax(
                    -action_error.float() / self.routing_teacher_temperature,
                    dim=-1,
                )
                self._last_teacher_action_error = action_error.detach().float().cpu()
                self._last_teacher_bucket_probs = teacher_bucket_probs.detach().float().cpu()
                teacher_weights = torch.zeros_like(router_weights, dtype=torch.float32)
                teacher_weights.scatter_(1, bucket_layers.unsqueeze(0).expand(bsz, -1), teacher_bucket_probs)
            self.anchor_proj.train(anchor_was_training)

            progress = (
                step - self.routing_curriculum_warmup_steps
            ) / float(max(self.routing_curriculum_teacher_steps, 1))
            progress = max(0.0, min(progress, 1.0))
            weights = (
                (1.0 - progress) * teacher_weights
                + progress * router_weights.float()
            ).to(router_weights.dtype)
            teacher_kl = (
                teacher_weights
                * (
                    torch.log(teacher_weights.clamp_min(1e-8))
                    - torch.log(router_weights.float().clamp_min(1e-8))
                )
            ).sum(dim=-1).mean()
            teacher_entropy = -(
                teacher_bucket_probs
                * torch.log(teacher_bucket_probs.clamp_min(1e-8))
            ).sum(dim=-1).mean()
            curriculum_loss = self.routing_teacher_kl_weight * teacher_kl
            metrics.update(
                {
                    "routing_curriculum_phase": zero.new_tensor(2.0),
                    "routing_curriculum_teacher_kl": teacher_kl.detach(),
                    "routing_curriculum_teacher_entropy": teacher_entropy.detach(),
                    "routing_curriculum_router_mix": zero.new_tensor(progress),
                }
            )
            metrics.update(self._actual_layer_metrics(weights))
            return weights, curriculum_loss, metrics, False

        metrics["routing_curriculum_phase"] = zero.new_tensor(3.0)
        metrics.update(self._actual_layer_metrics(router_weights))
        return router_weights, zero, metrics, False

    def _extract_state_emb(
        self,
        actions_hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
        target_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        # actions_hidden_states: (B, L, V+K, D)
        dtype = next(self.parameters()).dtype
        num_layers = actions_hidden_states.shape[1]
        layer_idx = self.fixed_layer_index if self.fixed_layer_index >= 0 else num_layers - 1
        bsz = actions_hidden_states.shape[0]
        dim = actions_hidden_states.shape[-1]

        if proprio is not None and proprio_projector is not None:
            proprio = proprio.reshape(bsz, -1).to(device=actions_hidden_states.device, dtype=dtype)
            proprio_feat = proprio_projector(proprio)
        else:
            proprio_feat = actions_hidden_states.new_zeros(bsz, dim).to(dtype)

        skill_emb = None
        continuous_context = None
        layer_weights = None
        adaptive_hidden_states = actions_hidden_states
        if self.use_adaptive_bridge and self.prototype_layer_projector is not None:
            adaptive_hidden_states = self.prototype_layer_projector(actions_hidden_states.to(dtype))
        elif self.use_adaptive_bridge and self.layerwise_aligner is not None:
            adaptive_hidden_states = self.layerwise_aligner(actions_hidden_states.to(dtype))
        if self.dense_film_enabled:
            if not self.use_adaptive_bridge or self.bridge_mode != "dense_film_residual":
                raise ValueError(
                    "dense_film_enabled requires use_adaptive_bridge=True and "
                    "bridge_mode='dense_film_residual'"
                )
            task_states = actions_hidden_states[:, :, :self.num_task_tokens, :].to(dtype)
            action_states = actions_hidden_states[:, :, self.num_task_tokens:, :].to(dtype)
            _, state_emb, skill_metrics = self.dense_depth_film(
                task_states, action_states, proprio_feat
            )
            skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
        elif not self.use_adaptive_bridge:
            # Fixed layer: just take the specified layer's action tokens
            state_emb = actions_hidden_states[:, layer_idx, self.num_task_tokens:, :]
            skill_metrics: Dict[str, torch.Tensor] = {}
            skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
        elif self.bridge_mode == "uniform":
            # Uniform average across layers
            state_emb = adaptive_hidden_states[:, :, self.num_task_tokens:, :].mean(dim=1)
            skill_metrics = {}
            skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
        elif self.bridge_mode == "static_learned":
            # Learned static weights
            if not hasattr(self, '_static_layer_weights') or self._static_layer_weights.shape[0] != num_layers:
                self._static_layer_weights = nn.Parameter(
                    torch.ones(num_layers, device=actions_hidden_states.device) / num_layers)
            weights = torch.softmax(self._static_layer_weights, dim=0).view(1, num_layers, 1, 1)
            state_emb = (weights * adaptive_hidden_states[:, :, self.num_task_tokens:, :]).sum(dim=1)
            skill_metrics = {}
            skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
        elif self.bridge_mode == "adaptive":
            # Task-adaptive weighted aggregation
            task_states = adaptive_hidden_states[:, :, :self.num_task_tokens, :]  # (B, L, V, D)
            task_mask = task_states.float().abs().sum(dim=-1) > 0
            if self.use_latent_skill_token:
                pooled_layers = self._masked_pool_task_tokens(task_states, task_mask)
                layer_weights, skill_metrics, skill_entropy_loss, skill_emb = self._skill_layer_weights(
                    pooled_layers, proprio_feat
                )
                (
                    layer_weights,
                    curriculum_loss,
                    curriculum_metrics,
                    suppress_router_aux,
                ) = self._apply_depth_curriculum(
                    adaptive_hidden_states,
                    layer_weights,
                    target_actions,
                    proprio_feat,
                )
                if suppress_router_aux:
                    skill_entropy_loss = skill_entropy_loss * 0.0
                skill_entropy_loss = skill_entropy_loss + curriculum_loss.to(
                    dtype=skill_entropy_loss.dtype
                )
                skill_metrics.update(curriculum_metrics)
            elif self.use_continuous_context:
                pooled_layers = self._masked_pool_task_tokens(task_states, task_mask)
                layer_weights, skill_metrics, continuous_context = self._continuous_context_layer_weights(
                    pooled_layers, proprio_feat
                )
                skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
            else:
                _, layer_weights = self.task_layer_selector(task_states, task_mask=task_mask)  # (B, L)
                skill_metrics = {}
                skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
                skill_emb = None
            weights = layer_weights.view(actions_hidden_states.shape[0], num_layers, 1, 1)
            state_emb = (weights * adaptive_hidden_states[:, :, self.num_task_tokens:, :]).sum(dim=1)
        elif self.bridge_mode == "adaptive_gated":
            # Same as adaptive for FlowMLP (gating is a DiT-specific feature)
            task_states = adaptive_hidden_states[:, :, :self.num_task_tokens, :]
            task_mask = task_states.float().abs().sum(dim=-1) > 0
            if self.use_latent_skill_token:
                pooled_layers = self._masked_pool_task_tokens(task_states, task_mask)
                layer_weights, skill_metrics, skill_entropy_loss, skill_emb = self._skill_layer_weights(
                    pooled_layers, proprio_feat
                )
                (
                    layer_weights,
                    curriculum_loss,
                    curriculum_metrics,
                    suppress_router_aux,
                ) = self._apply_depth_curriculum(
                    adaptive_hidden_states,
                    layer_weights,
                    target_actions,
                    proprio_feat,
                )
                if suppress_router_aux:
                    skill_entropy_loss = skill_entropy_loss * 0.0
                skill_entropy_loss = skill_entropy_loss + curriculum_loss.to(
                    dtype=skill_entropy_loss.dtype
                )
                skill_metrics.update(curriculum_metrics)
            elif self.use_continuous_context:
                pooled_layers = self._masked_pool_task_tokens(task_states, task_mask)
                layer_weights, skill_metrics, continuous_context = self._continuous_context_layer_weights(
                    pooled_layers, proprio_feat
                )
                skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
            else:
                _, layer_weights = self.task_layer_selector(task_states, task_mask=task_mask)
                skill_metrics = {}
                skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
                skill_emb = None
            weights = layer_weights.view(actions_hidden_states.shape[0], num_layers, 1, 1)
            state_emb = (weights * adaptive_hidden_states[:, :, self.num_task_tokens:, :]).sum(dim=1)
        elif self.bridge_mode == "expected_risk":
            task_states = adaptive_hidden_states[:, :, :self.num_task_tokens, :]
            task_mask = task_states.float().abs().sum(dim=-1) > 0
            _, layer_weights = self.task_layer_selector(task_states, task_mask=task_mask)
            # Training computes independent per-depth losses.  Inference reads
            # one real hidden layer, avoiding an off-manifold cross-depth blend.
            selected = layer_weights.argmax(dim=1)
            state_emb = adaptive_hidden_states[
                torch.arange(bsz, device=adaptive_hidden_states.device),
                selected,
                self.num_task_tokens :,
                :,
            ]
            skill_metrics = {}
            skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
        else:
            raise ValueError(f"Unknown bridge_mode: {self.bridge_mode}")

        state_emb = self._align_state_emb_to_action_chunk(state_emb)

        if (
            skill_emb is not None
            and self.skill_condition_proj is not None
            and self.skill_use_direct_conditioning
        ):
            state_emb = state_emb + self.skill_condition_proj(skill_emb.to(dtype)).unsqueeze(1)

        if (
            continuous_context is not None
            and self.continuous_context_condition_proj is not None
            and self.continuous_context_use_direct_conditioning
        ):
            state_emb = state_emb + self.continuous_context_condition_proj(
                continuous_context.to(dtype)
            ).unsqueeze(1)

        if proprio is not None and proprio_projector is not None:
            state_emb = state_emb + proprio_feat.to(dtype=state_emb.dtype).unsqueeze(1)

        # Stable, read-only snapshot for E00/E01 action-query trace collection.
        # Keep the values on CPU so the diagnostic path cannot retain GPU tensors.
        self.last_routing_diagnostics = {
            "skill_probs": None
            if not hasattr(self, "_last_skill_probs")
            else self._last_skill_probs.detach().float().cpu(),
            "skill_ids": None
            if not hasattr(self, "_last_skill_ids")
            else self._last_skill_ids.detach().long().cpu(),
            "continuous_context": None
            if not hasattr(self, "_last_continuous_context")
            else self._last_continuous_context.detach().float().cpu(),
            "layer_weights": None
            if layer_weights is None
            else layer_weights.detach().float().cpu(),
            "layer_alignment_enabled": self.adaptive_layer_alignment,
        }
        return state_emb, skill_metrics, skill_entropy_loss

    def flow_matching_loss(
        self,
        actions_hidden_states: torch.Tensor,
        target_actions: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
        mode: str = "flow_matching",
        ema_model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if self.use_adaptive_bridge and self.bridge_mode == "expected_risk":
            return self._expected_depth_risk_loss(
                actions_hidden_states,
                target_actions,
                proprio,
                proprio_projector,
            )
        state_emb, skill_metrics, skill_entropy_loss = self._extract_state_emb(
            actions_hidden_states,
            proprio,
            proprio_projector,
            target_actions=target_actions,
        )
        loss, loss_dict = self._compute_loss_components(state_emb, target_actions, skill_metrics, skill_entropy_loss)
        return loss, loss_dict

    def predict_action(
        self,
        actions_hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
        phase: str = "Inference",
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        state_emb, _, _ = self._extract_state_emb(actions_hidden_states, proprio, proprio_projector)
        return self._predict_action_internal(state_emb, num_steps=num_steps)

    @torch.no_grad()
    def _predict_action_internal(self, state_emb: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        state_emb = self._align_state_emb_to_action_chunk(state_emb)
        batch_size, seq_len, _ = state_emb.shape
        device = state_emb.device
        num_steps = int(self.num_inference_steps if num_steps is None else num_steps)
        num_samples = max(int(self.num_inference_samples), 1)
        base_state_emb = state_emb
        if num_samples > 1:
            state_emb = state_emb.repeat_interleave(num_samples, dim=0)
            batch_size = batch_size * num_samples

        sample_dtype = torch.float32 if self.flow_float32_path else state_emb.dtype
        x_t = torch.randn(batch_size, seq_len, self.output_dim, device=device, dtype=sample_dtype)
        dt = 1.0 / max(num_steps, 1)

        for i in range(max(num_steps, 1)):
            t_value = i / max(num_steps, 1)
            t = torch.full((batch_size,), t_value, device=device, dtype=sample_dtype)
            v_pred = self.forward(state_emb, x_t, t)
            x_t = x_t + v_pred.to(sample_dtype) * dt
        if num_samples > 1:
            x_t = x_t.reshape(-1, num_samples, seq_len, self.output_dim).mean(dim=1)
        if self.anchor_proj is not None:
            anchor_actions = self.anchor_proj(base_state_emb)
            blend = max(0.0, min(float(self.anchor_blend), 1.0))
            # The flow branch models an action residual around the supervised anchor.
            # Even when blend=0 we must reconstruct the full action as anchor + residual.
            x_t = anchor_actions + (1.0 - blend) * x_t
        return x_t
