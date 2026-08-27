from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "vla-scripts" / "checkpoint_protocol.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_protocol_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
checkpoint_protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint_protocol
SPEC.loader.exec_module(checkpoint_protocol)


def test_vlm_base_runs_the_requested_number_of_updates() -> None:
    plan = checkpoint_protocol.resolve_checkpoint_step_plan(
        train_from="vlm_base",
        resume_step=None,
        max_steps=10_000,
        resume_load_training_state=True,
    )
    assert plan.training_start_step == 0
    assert plan.planned_optimizer_updates == 10_000
    assert plan.source_step is None
    assert plan.load_training_state is False


def test_checkpoint_init_does_not_inherit_the_source_global_step() -> None:
    plan = checkpoint_protocol.resolve_checkpoint_step_plan(
        train_from="checkpoint_init",
        resume_step=5_000,
        max_steps=10_000,
        resume_load_training_state=True,
    )
    assert plan.source_step == 5_000
    assert plan.training_start_step == 0
    assert plan.planned_optimizer_updates == 10_000
    assert plan.load_training_state is False


def test_checkpoint_is_a_true_continuation() -> None:
    plan = checkpoint_protocol.resolve_checkpoint_step_plan(
        train_from="checkpoint",
        resume_step=5_000,
        max_steps=10_000,
        resume_load_training_state=True,
    )
    assert plan.source_step == 5_000
    assert plan.training_start_step == 5_000
    assert plan.planned_optimizer_updates == 5_000
    assert plan.load_training_state is True

    with pytest.raises(ValueError, match="true continuation"):
        checkpoint_protocol.resolve_checkpoint_step_plan(
            train_from="checkpoint",
            resume_step=5_000,
            max_steps=10_000,
            resume_load_training_state=False,
        )


@pytest.mark.parametrize(
    ("mode", "resume_step", "max_steps"),
    [
        ("vlm_base", 5_000, 10_000),
        ("checkpoint_init", None, 10_000),
        ("checkpoint", None, 10_000),
        ("checkpoint", 10_000, 10_000),
        ("unknown", None, 10_000),
    ],
)
def test_invalid_step_protocols_fail_fast(
    mode: str, resume_step: int | None, max_steps: int
) -> None:
    with pytest.raises(ValueError):
        checkpoint_protocol.resolve_checkpoint_step_plan(
            train_from=mode,
            resume_step=resume_step,
            max_steps=max_steps,
            resume_load_training_state=True,
        )


def test_protocol_tags_prevent_initialization_collisions() -> None:
    assert (
        checkpoint_protocol.protocol_tag("vlm_base", None, 10_000)
        == "vlmbase-10000updates"
    )
    assert (
        checkpoint_protocol.protocol_tag(
            "checkpoint_init", 5_000, 10_000, False
        )
        == "ckpt5000init-headfresh-10000updates"
    )
    assert (
        checkpoint_protocol.protocol_tag(
            "checkpoint_init", 5_000, 10_000, True
        )
        == "ckpt5000init-headload-10000updates"
    )
    assert (
        checkpoint_protocol.protocol_tag("checkpoint", 5_000, 10_000)
        == "resume5000to10000"
    )


def test_formal_launchers_default_to_vlm_base() -> None:
    for relative_path in (
        "scripts/launch_e04_fixed_layer_v48.sh",
        "scripts/launch_e04_fixed_layer_a100.sh",
        "scripts/launch_e07_context_control_v48.sh",
        "scripts/launch_flowmlp_ablation_v48.sh",
        "scripts/launch_flowmlp_ablation_a100.sh",
        "scripts/queue_v48_dit_prefix_after_flowmlp.sh",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "TRAIN_FROM=${TRAIN_FROM:-vlm_base}" in text or (
            relative_path.endswith("launch_e04_fixed_layer_a100.sh")
            and "export TRAIN_FROM=vlm_base" in text
        )


def test_formal_e04_launchers_do_not_reference_mlp5000() -> None:
    for relative_path in (
        "scripts/launch_e04_fixed_layer_v48.sh",
        "scripts/launch_e04_fixed_layer_a100.sh",
        "scripts/queue_e04_fixed_layer_v48.sh",
        "scripts/queue_e04_fixed_layer_a100.sh",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "MLP-object-5000" not in text


def test_e04_uses_final_index_24_not_formal_index_16() -> None:
    for relative_path in (
        "scripts/queue_e04_fixed_layer_v48.sh",
        "scripts/queue_e04_fixed_layer_a100.sh",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert '"24:7"' in text
        assert '"24:8"' in text
        assert '"24:9"' in text
        assert '"16:7"' not in text
        assert '"16:8"' not in text
        assert '"16:9"' not in text


def test_fresh_protocol_queues_cover_every_seed() -> None:
    e04 = (ROOT / "scripts/queue_e04_fixed_layer_a100.sh").read_text(
        encoding="utf-8"
    )
    for layer in (1, 5, 9, 13, 24):
        for seed in (7, 8, 9):
            assert f'"{layer}:{seed}"' in e04

    e07 = (ROOT / "scripts/queue_e07_context_control_v48.sh").read_text(
        encoding="utf-8"
    )
    for variant in ("no_skill", "continuous_context", "routing_only"):
        for seed in (7, 8, 9):
            assert f"{variant}:{seed}" in e07


def test_flowmlp_wrapper_has_no_hidden_checkpoint_default() -> None:
    text = (ROOT / "scripts/run_flowmlp_ablation.sh").read_text(encoding="utf-8")
    assert "TRAIN_FROM=${TRAIN_FROM:-vlm_base}" in text
    assert ': "${BASE_CKPT:?set BASE_CKPT}"' not in text
    assert "--train_from checkpoint --resum_vla_path" not in text
    assert "--load_action_head_from_checkpoint" in text
    assert "${PROTOCOL_TAG}" in text


def test_true_resume_requires_training_state_and_cross_head_load_fails() -> None:
    text = (ROOT / "vla-scripts" / "finetune.py").read_text(encoding="utf-8")
    assert "True checkpoint continuation requires optimizer/scheduler state" in text
    assert "True checkpoint continuation requires the saved LoRA adapter" in text
    assert "True checkpoint continuation requires trained action queries" in text
    assert "Refusing to load an MLP action-head checkpoint into" in text
    assert "disallowed_missing" in text
    assert "checkpoint discovery is disabled" in text


def test_optimizer_step_is_counted_before_checkpoint_decisions() -> None:
    text = (ROOT / "vla-scripts" / "finetune.py").read_text(encoding="utf-8")
    loop_start = text.index("# optimizer step on last micro-batch")
    increment = text.index("global_step += 1", loop_start)
    save_decision = text.index("# Save checkpoint", loop_start)
    assert increment < save_decision
    assert "start_step = int(cfg._training_start_step)" in text
    assert '"optimizer_updates_this_run"' in text
