"""Flow-GRPO trainer for diffusion-based action heads."""


from dataclasses import dataclass
from typing import Optional

import torch

from prismatic.models.action_heads import DiffusionActionHead, FlowGRPOSample


@dataclass
class FlowGRPOTrainerConfig:
    """Configuration for Flow-GRPO trainer."""

    clip_epsilon: float = 0.2
    normalize_advantage: bool = True
    entropy_coef: float = 0.0
    reward_baseline: str = "mean"  # "mean" | "none"
    max_grad_norm: Optional[float] = 1.0
    lambda_off: float = 0.0
    lambda_on: float = 1.0


@dataclass
class FlowGRPORolloutBatch:
    """Container for rollouts used by Flow-GRPO trainer."""

    actions_hidden_states: torch.Tensor
    proprio: torch.Tensor
    rewards: torch.Tensor
    flow_sample: FlowGRPOSample
    advantages: Optional[torch.Tensor] = None


class FlowGRPOTrainer:
    """Trainer implementing GRPO updates for diffusion action heads."""

    def __init__(
        self,
        action_head: DiffusionActionHead,
        proprio_projector: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[FlowGRPOTrainerConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.optimizer = optimizer
        self.config = config or FlowGRPOTrainerConfig()
        self.device = device or next(action_head.parameters()).device

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.config.reward_baseline == "mean":
            baseline = rewards.mean()
            advantages = rewards - baseline
        else:
            advantages = rewards

        if self.config.normalize_advantage:
            std = advantages.std(unbiased=False)
            advantages = advantages / (std + 1e-6)

        return advantages

    def train_step(
        self,
        batch: FlowGRPORolloutBatch,
        offline_loss: Optional[torch.Tensor] = None,
        lambda_off: Optional[float] = None,
        lambda_on: Optional[float] = None,
    ) -> dict:
        self.action_head.train()

        actions_hidden_states = batch.actions_hidden_states.to(self.device)
        proprio = batch.proprio.to(self.device)
        rewards = batch.rewards.to(self.device)

        flow_sample = batch.flow_sample
        old_log_probs_steps = flow_sample.log_probs.to(self.device)
        old_log_probs = old_log_probs_steps.sum(dim=0)

        with torch.no_grad():
            advantages = (
                batch.advantages.to(self.device)
                if batch.advantages is not None
                else self._compute_advantages(rewards)
            )

        new_log_probs_steps = self.action_head.evaluate_logprob_from_latents(
            actions_hidden_states=actions_hidden_states,
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            latents=flow_sample.latents,
            timestep_indices=flow_sample.timestep_indices,
        )
        new_log_probs = new_log_probs_steps.sum(dim=0)

        log_prob_delta = new_log_probs - old_log_probs
        ratios = torch.exp(log_prob_delta)
        clipped_ratios = torch.clamp(
            ratios,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )

        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

        entropy = -new_log_probs.mean()
        online_loss = policy_loss - self.config.entropy_coef * entropy

        if offline_loss is not None:
            offline_loss = offline_loss.to(self.device)

        lambda_off = self.config.lambda_off if lambda_off is None else lambda_off
        lambda_on = self.config.lambda_on if lambda_on is None else lambda_on

        total_loss = lambda_on * online_loss
        if offline_loss is not None:
            total_loss = total_loss + lambda_off * offline_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.action_head.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()

        approx_kl = 0.5 * (log_prob_delta.pow(2)).mean()

        metrics = {
            "loss": total_loss.item(),
            "online_loss": online_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": approx_kl.item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std(unbiased=False).item(),
        }
        if offline_loss is not None:
            metrics["offline_loss"] = offline_loss.item()
            metrics["lambda_off"] = lambda_off
            metrics["lambda_on"] = lambda_on
        else:
            metrics["lambda_off"] = 0.0
            metrics["lambda_on"] = lambda_on

        return metrics

