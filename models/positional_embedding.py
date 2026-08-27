"""
Sinusoidal positional embedding for timesteps.
Adapted from mean-flow reference implementation.
"""

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for continuous timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: timesteps of shape (batch_size,) or (batch_size, 1) or scalar
        Returns:
            embeddings of shape (batch_size, dim)
        """
        device = x.device
        dtype = x.dtype
        
        # Ensure x is at least 1D
        if x.dim() == 0:
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            x = x.squeeze(-1)
        
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, dtype=dtype, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

