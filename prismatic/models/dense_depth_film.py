"""Dense, state-conditioned depth residuals for a frozen VLA checkpoint.

The final VLM layer is an identity path.  Intermediate layers contribute a
zero-initialized residual whose channel-wise FiLM parameters are generated from
the current proprio/task state.  This is deliberately not a softmax router:
layers cooperate and each layer has an independent, state-dependent strength.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class StateConditionedDenseDepthFiLM(nn.Module):
    """Fuse all non-final packed VLM layers into the final-layer condition.

    Input is split as task tokens followed by action tokens, matching the
    existing action-head packing.  The output has the same ``(B, T, D)`` shape
    as the final layer.  At initialization every adapter output is exactly
    zero, so the module is an identity around the final layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_task_tokens: int,
        max_layers: int = 64,
        first_layer_index: int = 1,
        bottleneck_dim: int = 64,
        state_dim: int = 128,
        state_mode: str = "full",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_task_tokens = int(num_task_tokens)
        self.max_layers = int(max_layers)
        self.first_layer_index = max(int(first_layer_index), 0)
        self.bottleneck_dim = int(bottleneck_dim)
        self.state_dim = int(state_dim)
        self.state_mode = str(state_mode).lower()
        if self.state_mode not in {"full", "static", "proprio", "shuffled"}:
            raise ValueError(f"unsupported dense FiLM state_mode: {state_mode}")
        if self.max_layers < 2:
            raise ValueError("dense depth FiLM requires at least two VLM layers")
        if self.first_layer_index >= self.max_layers - 1:
            raise ValueError("first_layer_index must leave at least one residual layer")

        self.state_encoder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.state_dim),
            nn.GELU(),
            nn.Linear(self.state_dim, self.state_dim),
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.max_layers)]
        )
        self.input_projections = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.bottleneck_dim) for _ in range(self.max_layers)]
        )
        self.film_generators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.state_dim, self.bottleneck_dim),
                    nn.GELU(),
                    nn.Linear(self.bottleneck_dim, self.bottleneck_dim * 2),
                )
                for _ in range(self.max_layers)
            ]
        )
        self.output_projections = nn.ModuleList(
            [nn.Linear(self.bottleneck_dim, self.hidden_dim) for _ in range(self.max_layers)]
        )

        # Identity-at-initialization: FiLM starts neutral and each residual
        # output starts exactly at zero.  The output projection receives a
        # non-zero gradient on the first backward pass.
        for generator in self.film_generators:
            nn.init.zeros_(generator[-1].weight)
            nn.init.zeros_(generator[-1].bias)
        for projection in self.output_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

        self.last_metrics: Dict[str, torch.Tensor] = {}

    def set_num_task_tokens(self, num_task_tokens: int, *_unused_legacy_args) -> None:
        """Update token packing metadata.

        Older action-head callers passed a second VLM/context argument.  It
        is not needed by dense-FiLM, but accepting and ignoring legacy
        positional arguments keeps old checkpoints and training scripts
        source-compatible.
        """
        self.num_task_tokens = int(num_task_tokens)

    def _masked_pool(self, task_tokens: torch.Tensor) -> torch.Tensor:
        valid = task_tokens.float().abs().sum(dim=-1) > 0
        mask = valid.to(dtype=task_tokens.dtype).unsqueeze(-1)
        return (task_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        task_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        proprio_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if task_hidden_states.ndim != 4 or action_hidden_states.ndim != 4:
            raise ValueError("dense depth FiLM expects task/action tensors shaped (B,L,T,D)")
        if task_hidden_states.shape[:2] != action_hidden_states.shape[:2]:
            raise ValueError("task and action hidden states must share batch/layer dimensions")
        batch, layers = task_hidden_states.shape[:2]
        if layers > self.max_layers:
            raise ValueError(f"received {layers} layers, max_layers={self.max_layers}")

        final_task = task_hidden_states[:, -1]
        final_action = action_hidden_states[:, -1]
        final_tokens = torch.cat([final_task, final_action], dim=1)
        if proprio_features is None:
            proprio = final_task.new_zeros(batch, self.hidden_dim)
        else:
            proprio = proprio_features.reshape(batch, -1).to(
                device=final_task.device, dtype=final_task.dtype
            )
            if proprio.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"proprio feature dim must be {self.hidden_dim}, got {proprio.shape[-1]}"
                )
        task_pool = self._masked_pool(final_task)
        if self.state_mode == "static":
            task_pool = torch.zeros_like(task_pool)
            proprio = torch.zeros_like(proprio)
        elif self.state_mode == "proprio":
            task_pool = torch.zeros_like(task_pool)
        elif self.state_mode == "shuffled":
            task_pool = torch.flip(task_pool, dims=[-1])
            proprio = torch.roll(proprio, shifts=1, dims=-1)
        state = self.state_encoder(torch.cat([task_pool, proprio], dim=-1))

        residual = torch.zeros_like(final_tokens)
        layer_norms = []
        gamma_rms = []
        beta_rms = []
        # The final layer is intentionally excluded: it is the stable identity
        # baseline.  Layer 0 is normally the embedding state; first_layer_index
        # defaults to 1 to use transformer block outputs only.
        candidate_count = max(layers - self.first_layer_index - 1, 1)
        valid_task = final_task.float().abs().sum(dim=-1) > 0
        valid = torch.cat(
            [valid_task, torch.ones(batch, final_action.shape[1], device=final_action.device, dtype=torch.bool)],
            dim=1,
        ).unsqueeze(-1).to(dtype=final_tokens.dtype)
        for layer_idx in range(self.first_layer_index, layers - 1):
            tokens = torch.cat(
                [task_hidden_states[:, layer_idx], action_hidden_states[:, layer_idx]], dim=1
            )
            u = self.input_projections[layer_idx](self.layer_norms[layer_idx](tokens))
            gamma, beta = self.film_generators[layer_idx](state).chunk(2, dim=-1)
            modulated = (1.0 + gamma.unsqueeze(1)) * u + beta.unsqueeze(1)
            layer_residual = self.output_projections[layer_idx](torch.nn.functional.gelu(modulated))
            layer_residual = layer_residual * valid
            residual = residual + layer_residual / float(candidate_count)
            layer_norms.append(layer_residual.float().norm(dim=-1).mean(dim=-1))
            gamma_rms.append(gamma.float().pow(2).mean(dim=-1).sqrt())
            beta_rms.append(beta.float().pow(2).mean(dim=-1).sqrt())

        fused = final_tokens + residual
        base_norm = final_tokens.float().norm(dim=-1).mean().clamp_min(1e-8)
        residual_norm = residual.float().norm(dim=-1).mean()
        metrics = {
            "depth_film_residual_ratio": (residual_norm / base_norm).detach(),
            "depth_film_residual_norm": residual_norm.detach(),
            "depth_film_state_norm": state.float().norm(dim=-1).mean().detach(),
            "depth_film_gamma_rms": torch.stack(gamma_rms).mean().detach()
            if gamma_rms
            else state.new_zeros(()),
            "depth_film_beta_rms": torch.stack(beta_rms).mean().detach()
            if beta_rms
            else state.new_zeros(()),
            "depth_film_layer_residual_norm": torch.stack(layer_norms, dim=1).detach()
            if layer_norms
            else state.new_zeros((batch, 0)),
        }
        self.last_metrics = metrics
        return fused[:, : final_task.shape[1]], fused[:, final_task.shape[1] :], metrics


__all__ = ["StateConditionedDenseDepthFiLM"]
