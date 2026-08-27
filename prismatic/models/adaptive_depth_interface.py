"""Skill-conditioned adaptive depth interface for VLA action heads.

This module owns the representation routing between frozen VLM hidden states
and continuous action decoders. It consumes packed multi-layer VLM states with
shape ``(B, L, V + K, D)`` and returns an action-query sequence ``(B, K, D)``
that can be passed to MLP, Flow Matching, or DiT-style action heads.
"""


from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DepthInterfaceOutput:
    state_emb: torch.Tensor
    metrics: Dict[str, torch.Tensor]
    aux_loss: torch.Tensor


class SkillAdaptiveDepthInterface(nn.Module):
    """Aggregate VLM action-query states across depth.

    Modes:
      - ``fixed`` / ``best_fixed``: use ``fixed_layer_index``.
      - ``final``: use the final hidden layer.
      - ``uniform``: average action-query states across all layers.
      - ``static_learned``: learn sample-independent layer weights.
      - ``adaptive``: score layers from action-query summaries and proprio.
      - ``skill_adaptive`` / ``adaptive_gated``: infer a latent skill token and
        use it to condition the layer scorer.
    """

    def __init__(
        self,
        input_dim: int,
        num_task_tokens: int,
        mode: str = "skill_adaptive",
        fixed_layer_index: int = -1,
        max_vlm_layers: int = 64,
        num_skill_tokens: int = 16,
        skill_token_dim: int = 128,
        skill_temperature: float = 1.0,
        skill_entropy_weight: float = 0.0,
        add_proprio_to_output: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_task_tokens = int(num_task_tokens)
        self.mode = str(mode)
        self.fixed_layer_index = int(fixed_layer_index)
        self.max_vlm_layers = int(max_vlm_layers)
        self.num_skill_tokens = int(num_skill_tokens)
        self.skill_token_dim = int(skill_token_dim)
        self.skill_temperature = float(skill_temperature)
        self.skill_entropy_weight = float(skill_entropy_weight)
        self.add_proprio_to_output = bool(add_proprio_to_output)

        self.layer_projections = nn.ModuleList(
            [nn.Linear(self.input_dim, self.input_dim) for _ in range(self.max_vlm_layers)]
        )
        self.static_layer_logits = nn.Parameter(torch.zeros(self.max_vlm_layers))

        self.adaptive_scorer = nn.Sequential(
            nn.LayerNorm(self.input_dim * 2),
            nn.Linear(self.input_dim * 2, self.input_dim // 2),
            nn.GELU(),
            nn.Linear(self.input_dim // 2, 1),
        )

        self.skill_selector = nn.Sequential(
            nn.LayerNorm(self.input_dim * 2),
            nn.Linear(self.input_dim * 2, self.input_dim // 2),
            nn.GELU(),
            nn.Linear(self.input_dim // 2, self.num_skill_tokens),
        )
        self.skill_embedding = nn.Embedding(self.num_skill_tokens, self.skill_token_dim)
        self.skill_layer_scorer = nn.Sequential(
            nn.LayerNorm(self.input_dim * 2 + self.skill_token_dim),
            nn.Linear(self.input_dim * 2 + self.skill_token_dim, self.input_dim // 2),
            nn.GELU(),
            nn.Linear(self.input_dim // 2, 1),
        )

    def _proprio_features(
        self,
        actions_hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor],
        proprio_projector: Optional[nn.Module],
    ) -> torch.Tensor:
        bsz = actions_hidden_states.shape[0]
        dtype = actions_hidden_states.dtype
        device = actions_hidden_states.device
        if proprio is None or proprio_projector is None:
            return actions_hidden_states.new_zeros(bsz, self.input_dim)
        proprio = proprio.reshape(bsz, -1).to(device=device, dtype=dtype)
        return proprio_projector(proprio).to(dtype=dtype)

    def _project_action_layers(self, action_states: torch.Tensor) -> torch.Tensor:
        num_layers = action_states.shape[1]
        if num_layers > self.max_vlm_layers:
            raise ValueError(
                f"Received {num_layers} VLM layers, but max_vlm_layers={self.max_vlm_layers}. "
                "Increase depth_interface_max_layers."
            )
        projected = [
            self.layer_projections[idx](action_states[:, idx, :, :])
            for idx in range(num_layers)
        ]
        return torch.stack(projected, dim=1)

    def _fixed_index(self, num_layers: int) -> int:
        if self.mode == "final" or self.fixed_layer_index < 0:
            return num_layers - 1
        return max(0, min(self.fixed_layer_index, num_layers - 1))

    def _skill_weights(
        self,
        pooled_actions: torch.Tensor,
        proprio_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        dtype = pooled_actions.dtype
        bsz, num_layers, _ = pooled_actions.shape
        global_context = pooled_actions.mean(dim=1)

        skill_logits = self.skill_selector(torch.cat([global_context, proprio_feat], dim=-1))
        temperature = max(self.skill_temperature, 1e-6)
        if self.training:
            skill_probs = F.gumbel_softmax(skill_logits.float(), tau=temperature, hard=True, dim=-1).to(dtype)
        else:
            skill_ids = skill_logits.argmax(dim=-1)
            skill_probs = F.one_hot(skill_ids, num_classes=self.num_skill_tokens).to(dtype)

        skill_emb = skill_probs @ self.skill_embedding.weight.to(dtype)
        scorer_input = torch.cat(
            [
                pooled_actions,
                skill_emb.unsqueeze(1).expand(-1, num_layers, -1),
                proprio_feat.unsqueeze(1).expand(-1, num_layers, -1),
            ],
            dim=-1,
        )
        layer_scores = self.skill_layer_scorer(scorer_input).squeeze(-1)
        layer_weights = torch.softmax(layer_scores, dim=1)

        soft_probs = torch.softmax(skill_logits.float(), dim=-1)
        entropy = -(soft_probs * torch.log(soft_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        max_prob = soft_probs.max(dim=-1).values.mean()
        expected_id = (
            soft_probs
            * torch.arange(self.num_skill_tokens, device=pooled_actions.device, dtype=soft_probs.dtype)
        ).sum(dim=-1).mean()
        aux_loss = -self.skill_entropy_weight * entropy
        metrics = {
            "depth_skill_entropy": entropy.detach(),
            "depth_skill_max_prob": max_prob.detach(),
            "depth_skill_expected_id": expected_id.detach(),
            "depth_layer_weight_max": layer_weights.max(dim=1).values.mean().detach(),
        }
        if self.skill_entropy_weight != 0.0:
            metrics["depth_skill_entropy_loss"] = aux_loss.detach()
        return layer_weights, metrics, aux_loss.to(dtype=dtype)

    def _adaptive_weights(
        self,
        pooled_actions: torch.Tensor,
        proprio_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        num_layers = pooled_actions.shape[1]
        scorer_input = torch.cat(
            [pooled_actions, proprio_feat.unsqueeze(1).expand(-1, num_layers, -1)],
            dim=-1,
        )
        layer_scores = self.adaptive_scorer(scorer_input).squeeze(-1)
        layer_weights = torch.softmax(layer_scores, dim=1)
        metrics = {
            "depth_layer_weight_max": layer_weights.max(dim=1).values.mean().detach(),
        }
        return layer_weights, metrics, pooled_actions.new_tensor(0.0)

    def forward(
        self,
        actions_hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[nn.Module] = None,
    ) -> DepthInterfaceOutput:
        if actions_hidden_states.dim() != 4:
            raise ValueError(
                "actions_hidden_states must have shape (B, L, V + K, D); "
                f"got {tuple(actions_hidden_states.shape)}"
            )

        num_layers = actions_hidden_states.shape[1]
        action_states = actions_hidden_states[:, :, self.num_task_tokens :, :]
        if action_states.shape[2] == 0:
            raise ValueError("No action-query tokens found after num_task_tokens split.")

        projected_actions = self._project_action_layers(action_states)
        pooled_actions = action_states.mean(dim=2)
        proprio_feat = self._proprio_features(actions_hidden_states, proprio, proprio_projector)

        mode = self.mode
        metrics: Dict[str, torch.Tensor] = {}
        aux_loss = actions_hidden_states.new_tensor(0.0)

        if mode in {"fixed", "best_fixed", "final"}:
            layer_idx = self._fixed_index(num_layers)
            state_emb = projected_actions[:, layer_idx, :, :]
            metrics["depth_selected_layer"] = actions_hidden_states.new_tensor(float(layer_idx))
        elif mode == "uniform":
            state_emb = projected_actions.mean(dim=1)
        elif mode == "static_learned":
            weights = torch.softmax(self.static_layer_logits[:num_layers], dim=0).view(1, num_layers, 1, 1)
            state_emb = (weights * projected_actions).sum(dim=1)
            metrics["depth_layer_weight_max"] = weights.max().detach()
        elif mode == "adaptive":
            layer_weights, metrics, aux_loss = self._adaptive_weights(pooled_actions, proprio_feat)
            state_emb = (layer_weights.view(-1, num_layers, 1, 1) * projected_actions).sum(dim=1)
        elif mode in {"skill_adaptive", "adaptive_gated"}:
            layer_weights, metrics, aux_loss = self._skill_weights(pooled_actions, proprio_feat)
            state_emb = (layer_weights.view(-1, num_layers, 1, 1) * projected_actions).sum(dim=1)
        else:
            raise ValueError(f"Unknown depth interface mode: {mode}")

        if self.add_proprio_to_output:
            state_emb = state_emb + proprio_feat.to(dtype=state_emb.dtype).unsqueeze(1)

        return DepthInterfaceOutput(state_emb=state_emb, metrics=metrics, aux_loss=aux_loss)
