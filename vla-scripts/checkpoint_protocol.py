"""Checkpoint initialization and step-accounting rules for VLA training.

This module intentionally has no ML dependencies so the protocol can be tested
without importing PyTorch, Transformers, or the training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


VALID_TRAIN_FROM = ("vlm_base", "checkpoint", "checkpoint_init")


@dataclass(frozen=True)
class CheckpointStepPlan:
    """Resolved optimizer-step semantics for one training invocation."""

    mode: str
    source_step: Optional[int]
    training_start_step: int
    target_step: int
    planned_optimizer_updates: int
    load_training_state: bool


def resolve_checkpoint_step_plan(
    train_from: str,
    resume_step: Optional[int],
    max_steps: int,
    resume_load_training_state: bool,
) -> CheckpointStepPlan:
    """Resolve source-checkpoint lookup separately from optimizer-step counting.

    ``checkpoint`` is a true continuation: global step resumes at
    ``resume_step``. ``checkpoint_init`` only imports selected parameters from
    that checkpoint and starts a new optimization run at global step zero.
    """

    mode = str(train_from).lower()
    if mode not in VALID_TRAIN_FROM:
        raise ValueError(
            f"train_from must be one of {VALID_TRAIN_FROM}; got {train_from!r}"
        )
    if int(max_steps) <= 0:
        raise ValueError(f"max_steps must be positive; got {max_steps}")

    if mode == "vlm_base":
        if resume_step is not None:
            raise ValueError("resume_step is invalid when train_from='vlm_base'")
        start_step = 0
        source_step = None
        load_training_state = False
    else:
        if resume_step is None or int(resume_step) < 0:
            raise ValueError(
                f"train_from={mode!r} requires a non-negative resume_step"
            )
        source_step = int(resume_step)
        if mode == "checkpoint_init":
            start_step = 0
            load_training_state = False
        else:
            if not bool(resume_load_training_state):
                raise ValueError(
                    "train_from='checkpoint' is a true continuation and cannot "
                    "disable optimizer/scheduler restore; use 'checkpoint_init' "
                    "for a fresh optimizer timeline"
                )
            start_step = source_step
            load_training_state = True

    planned_updates = int(max_steps) - start_step
    if planned_updates <= 0:
        raise ValueError(
            "No optimizer updates are planned: "
            f"mode={mode} start_step={start_step} max_steps={max_steps}"
        )

    return CheckpointStepPlan(
        mode=mode,
        source_step=source_step,
        training_start_step=start_step,
        target_step=int(max_steps),
        planned_optimizer_updates=planned_updates,
        load_training_state=load_training_state,
    )


def protocol_tag(
    train_from: str,
    resume_step: Optional[int],
    max_steps: int,
    load_action_head_from_checkpoint: Optional[bool] = None,
) -> str:
    """Return a filesystem-safe tag that prevents protocol-overwrite collisions."""

    mode = str(train_from).lower()
    if mode == "vlm_base":
        return f"vlmbase-{int(max_steps)}updates"
    if resume_step is None:
        raise ValueError(f"train_from={mode!r} requires resume_step")
    if mode == "checkpoint_init":
        head_tag = (
            "headload"
            if bool(load_action_head_from_checkpoint)
            else "headfresh"
        )
        return f"ckpt{int(resume_step)}init-{head_tag}-{int(max_steps)}updates"
    if mode == "checkpoint":
        return f"resume{int(resume_step)}to{int(max_steps)}"
    raise ValueError(f"unknown train_from mode: {train_from!r}")
