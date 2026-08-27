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
        if not self.use_adaptive_bridge:
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
    """Standard sinusoidal positional embedding with dtype-safe computation."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device, dtype = x.device, x.dtype
        half_dim = self.dim // 2
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
        num_inference_steps: int = 5,
        num_inference_samples: int = 8,
        supervised_anchor_weight: float = 0.0,
        anchor_blend: float = 0.0,
        anchor_gripper_weight: float = 1.0,
        anchor_gripper_bce_weight: float = 0.0,
        detach_flow_conditioning: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.time_dim = time_dim
        self.num_task_tokens = num_task_tokens
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
        self.num_inference_steps = int(num_inference_steps)
        self.num_inference_samples = int(num_inference_samples)
        self.supervised_anchor_weight = float(supervised_anchor_weight)
        self.anchor_blend = float(anchor_blend)
        self.anchor_gripper_weight = float(anchor_gripper_weight)
        self.anchor_gripper_bce_weight = float(anchor_gripper_bce_weight)
        self.detach_flow_conditioning = bool(detach_flow_conditioning)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
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
            self.anchor_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
            )
        else:
            self.anchor_proj = None

        # Task layer selector for adaptive modes
        self.task_layer_selector = TaskLayerSelector(hidden_dim=input_dim)

        if self.use_latent_skill_token:
            self.skill_selector = nn.Sequential(
                nn.LayerNorm(input_dim * 2),
                nn.Linear(input_dim * 2, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, self.num_skill_tokens),
            )
            self.skill_embedding = nn.Embedding(self.num_skill_tokens, self.skill_token_dim)
            self.skill_layer_scorer = nn.Sequential(
                nn.LayerNorm(input_dim * 2 + self.skill_token_dim),
                nn.Linear(input_dim * 2 + self.skill_token_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, 1),
            )
            # Keep the selected skill available to the action predictor itself.
            # Zero initialization preserves old checkpoint behavior at step zero.
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
            self.skill_condition_proj = None

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

        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def _anchor_l1_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = (predicted.float() - target.float()).abs()
        error[..., -1] *= self.anchor_gripper_weight
        loss = error.mean()
        if self.anchor_gripper_bce_weight > 0.0:
            gripper_target = (target[..., -1] > 0).to(dtype=torch.float32)
            gripper_bce = F.binary_cross_entropy_with_logits(predicted[..., -1].float(), gripper_target)
            loss = loss + self.anchor_gripper_bce_weight * gripper_bce
        return loss

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
        temperature = max(self.skill_temperature, 1e-6)
        if self.training:
            skill_probs = F.gumbel_softmax(logits.float(), tau=temperature, hard=True, dim=-1).to(dtype)
        else:
            skill_ids = logits.argmax(dim=-1)
            skill_probs = F.one_hot(skill_ids, num_classes=self.num_skill_tokens).to(dtype)

        # Inference diagnostics are intentionally kept out of the model output and
        # detached immediately.  The evaluator can read this snapshot after each
        # action query without changing the action computation or autograd graph.
        self._last_skill_ids = skill_probs.argmax(dim=-1).detach().long()

        skill_emb = skill_probs @ self.skill_embedding.weight.to(dtype)
        if self.skill_use_layer_routing:
            expanded_skill = skill_emb.unsqueeze(1).expand(-1, num_layers, -1)
            expanded_proprio = proprio_feat.unsqueeze(1).expand(-1, num_layers, -1)
            scorer_input = torch.cat([pooled_layers, expanded_skill, expanded_proprio], dim=-1)
            layer_scores = self.skill_layer_scorer(scorer_input).squeeze(-1)
            layer_weights = torch.softmax(layer_scores, dim=1)
        else:
            # Keep the non-skill adaptive bridge as the controlled routing baseline.
            task_states = pooled_layers.unsqueeze(2)
            _, layer_weights = self.task_layer_selector(task_states)

        entropy = -(soft_probs * torch.log(soft_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        max_prob = soft_probs.max(dim=-1).values.mean()
        expected_id = (soft_probs * torch.arange(self.num_skill_tokens, device=pooled_layers.device, dtype=soft_probs.dtype)).sum(dim=-1).mean()
        metrics = {
            "skill_entropy": entropy.detach(),
            "skill_max_prob": max_prob.detach(),
            "skill_expected_id": expected_id.detach(),
        }
        entropy_loss = -self.skill_entropy_weight * entropy
        if self.skill_entropy_weight != 0.0:
            metrics["skill_entropy_loss"] = entropy_loss.detach()
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
        timesteps = timesteps.to(dtype)

        _, seq_len, _ = state_emb.shape

        t_emb = self.time_mlp(timesteps)
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dtype = next(self.parameters()).dtype
        device = state_emb.device
        state_emb = state_emb.to(dtype)
        gt_actions = gt_actions.to(dtype)
        state_emb = self._align_state_emb_to_action_chunk(state_emb)

        batch_size, _, _ = gt_actions.shape

        t = torch.rand(batch_size, device=device, dtype=dtype)
        t_expanded = t.view(batch_size, 1, 1)
        x_0 = torch.randn_like(gt_actions, device=device, dtype=dtype)

        anchor_actions = None
        flow_target_actions = gt_actions
        if self.anchor_proj is not None and self.supervised_anchor_weight > 0.0:
            anchor_actions = self.anchor_proj(state_emb)
            flow_target_actions = gt_actions - anchor_actions.detach().to(dtype=gt_actions.dtype)

        x_t = t_expanded * flow_target_actions + (1 - t_expanded) * x_0
        target_velocity = flow_target_actions - x_0

        flow_state_emb = state_emb.detach() if self.detach_flow_conditioning else state_emb
        pred_velocity = self.forward(flow_state_emb, x_t, t)
        flow_loss = F.mse_loss(pred_velocity, target_velocity)

        skill_metrics = skill_metrics or {}
        metrics: Dict[str, torch.Tensor] = {"flow_matching_loss": flow_loss.detach(), **skill_metrics}
        total_loss = flow_loss
        if anchor_actions is not None:
            anchor_loss = self._anchor_l1_loss(anchor_actions, gt_actions)
            total_loss = total_loss + self.supervised_anchor_weight * anchor_loss.to(dtype=total_loss.dtype)
            metrics["flowmlp_anchor_l1_loss"] = anchor_loss.detach()
            metrics["flow_matching_total_loss"] = total_loss.detach()
        if (
            self.use_latent_skill_token
            and self.skill_entropy_weight != 0.0
            and skill_entropy_loss is not None
        ):
            total_loss = total_loss + skill_entropy_loss.to(dtype=flow_loss.dtype)
            metrics["flow_matching_total_loss"] = total_loss.detach()

        return total_loss, metrics

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

    def _extract_state_emb(
        self,
        actions_hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
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
        if not self.use_adaptive_bridge:
            # Fixed layer: just take the specified layer's action tokens
            state_emb = actions_hidden_states[:, layer_idx, self.num_task_tokens:, :]
            skill_metrics: Dict[str, torch.Tensor] = {}
            skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
        elif self.bridge_mode == "uniform":
            # Uniform average across layers
            state_emb = actions_hidden_states[:, :, self.num_task_tokens:, :].mean(dim=1)
            skill_metrics = {}
            skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
        elif self.bridge_mode == "static_learned":
            # Learned static weights
            if not hasattr(self, '_static_layer_weights') or self._static_layer_weights.shape[0] != num_layers:
                self._static_layer_weights = nn.Parameter(
                    torch.ones(num_layers, device=actions_hidden_states.device) / num_layers)
            weights = torch.softmax(self._static_layer_weights, dim=0).view(1, num_layers, 1, 1)
            state_emb = (weights * actions_hidden_states[:, :, self.num_task_tokens:, :]).sum(dim=1)
            skill_metrics = {}
            skill_entropy_loss = actions_hidden_states.new_tensor(0.0)
        elif self.bridge_mode == "adaptive":
            # Task-adaptive weighted aggregation
            task_states = actions_hidden_states[:, :, :self.num_task_tokens, :]  # (B, L, V, D)
            task_mask = task_states.float().abs().sum(dim=-1) > 0
            if self.use_latent_skill_token:
                pooled_layers = self._masked_pool_task_tokens(task_states, task_mask)
                layer_weights, skill_metrics, skill_entropy_loss, skill_emb = self._skill_layer_weights(
                    pooled_layers, proprio_feat
                )
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
            state_emb = (weights * actions_hidden_states[:, :, self.num_task_tokens:, :]).sum(dim=1)
        elif self.bridge_mode == "adaptive_gated":
            # Same as adaptive for FlowMLP (gating is a DiT-specific feature)
            task_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
            task_mask = task_states.float().abs().sum(dim=-1) > 0
            if self.use_latent_skill_token:
                pooled_layers = self._masked_pool_task_tokens(task_states, task_mask)
                layer_weights, skill_metrics, skill_entropy_loss, skill_emb = self._skill_layer_weights(
                    pooled_layers, proprio_feat
                )
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
            state_emb = (weights * actions_hidden_states[:, :, self.num_task_tokens:, :]).sum(dim=1)
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
        state_emb, skill_metrics, skill_entropy_loss = self._extract_state_emb(actions_hidden_states, proprio, proprio_projector)
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

        x_t = torch.randn(batch_size, seq_len, self.output_dim, device=device, dtype=state_emb.dtype)
        dt = 1.0 / max(num_steps, 1)

        for i in range(max(num_steps, 1)):
            t_value = i / max(num_steps, 1)
            t = torch.full((batch_size,), t_value, device=device, dtype=state_emb.dtype)
            v_pred = self.forward(state_emb, x_t, t)
            x_t = x_t + v_pred * dt
        if num_samples > 1:
            x_t = x_t.reshape(-1, num_samples, seq_len, self.output_dim).mean(dim=1)
        if self.anchor_proj is not None:
            anchor_actions = self.anchor_proj(base_state_emb)
            blend = max(0.0, min(float(self.anchor_blend), 1.0))
            # The flow branch models an action residual around the supervised anchor.
            # Even when blend=0 we must reconstruct the full action as anchor + residual.
            x_t = anchor_actions + (1.0 - blend) * x_t
        return x_t
