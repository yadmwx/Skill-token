import torch
from torch import nn

from prismatic.models.action_heads import FlowMatchingMLPActionHead, NUM_ACTIONS_CHUNK, SinusoidalPosEmb


def _head(**kwargs):
    return FlowMatchingMLPActionHead(
        input_dim=16,
        hidden_dim=32,
        output_dim=7,
        num_layers=1,
        num_task_tokens=4,
        num_skill_tokens=16,
        skill_token_dim=8,
        num_inference_steps=1,
        num_inference_samples=1,
        **kwargs,
    )


def _routing_inputs():
    # Four task tokens and one action token per action in the fixed action chunk.
    hidden = torch.randn(2, 3, 4 + NUM_ACTIONS_CHUNK, 16)
    proprio = torch.randn(2, 8)
    return hidden, proprio, nn.Linear(8, 16)


def _nparams(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_continuous_context_is_continuous_and_matches_r2_parameter_count():
    torch.manual_seed(7)
    r1 = _head(use_continuous_context=True, continuous_context_use_direct_conditioning=False)
    torch.manual_seed(7)
    r2 = _head(use_latent_skill_token=True, skill_use_direct_conditioning=False)
    assert _nparams(r1) == _nparams(r2)

    hidden, proprio, projector = _routing_inputs()
    state, metrics, _ = r1._extract_state_emb(hidden, proprio, projector)
    state.square().mean().backward()

    assert state.shape == (2, NUM_ACTIONS_CHUNK, 16)
    assert "continuous_context_l2" in metrics
    assert r1.last_routing_diagnostics["skill_probs"] is None
    assert r1.last_routing_diagnostics["skill_ids"] is None
    assert r1.last_routing_diagnostics["continuous_context"].shape == (2, 8)
    assert r1.continuous_context_selector[1].weight.grad is not None
    weights = r1.last_routing_diagnostics["layer_weights"]
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))


def test_r0_keeps_proprio_on_the_shared_action_path():
    torch.manual_seed(11)
    r0 = _head()
    hidden, proprio, projector = _routing_inputs()
    state_with_u, _, _ = r0._extract_state_emb(hidden, proprio, projector)
    state_without_u, _, _ = r0._extract_state_emb(hidden, None, None)

    assert not torch.allclose(state_with_u, state_without_u)
    assert r0.last_routing_diagnostics["skill_probs"] is None


def test_soft_skill_assignment_is_identical_in_train_and_eval():
    torch.manual_seed(17)
    head = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_assignment_mode="soft",
    )
    hidden, proprio, projector = _routing_inputs()
    head.train()
    train_state, _, _ = head._extract_state_emb(hidden, proprio, projector)
    train_weights = head.last_routing_diagnostics["layer_weights"].clone()
    head.eval()
    eval_state, _, _ = head._extract_state_emb(hidden, proprio, projector)
    torch.testing.assert_close(train_state, eval_state)
    torch.testing.assert_close(train_weights, head.last_routing_diagnostics["layer_weights"])


def test_zero_adaptive_mix_is_exactly_fixed_anchor_and_backpropagates():
    torch.manual_seed(19)
    head = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_assignment_mode="soft",
        routing_anchor_layer=1,
        routing_adaptive_mix=0.0,
    )
    hidden, proprio, projector = _routing_inputs()
    state, _, _ = head._extract_state_emb(hidden, proprio, projector)
    expected = hidden[:, 1, 4:, :] + projector(proprio).unsqueeze(1)
    torch.testing.assert_close(state, expected)
    weights = head.last_routing_diagnostics["layer_weights"]
    torch.testing.assert_close(weights[:, 1], torch.ones(2))
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    state.square().mean().backward()
    assert projector.weight.grad is not None


def test_layerwise_alignment_starts_as_identity_and_is_independent_per_depth():
    torch.manual_seed(23)
    baseline = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_assignment_mode="soft",
    )
    aligned = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_assignment_mode="soft",
        adaptive_layer_alignment=True,
        adaptive_num_layers=3,
        adaptive_alignment_bottleneck=4,
    )
    aligned.load_state_dict(baseline.state_dict(), strict=False)

    hidden, proprio, projector = _routing_inputs()
    baseline_state, _, _ = baseline._extract_state_emb(hidden, proprio, projector)
    aligned_state, _, _ = aligned._extract_state_emb(hidden, proprio, projector)
    torch.testing.assert_close(aligned_state, baseline_state)

    aligned_state.square().mean().backward()
    grads = [module[-1].weight.grad for module in aligned.layerwise_aligner.aligners]
    assert all(grad is not None for grad in grads)
    assert all(torch.count_nonzero(grad).item() > 0 for grad in grads)
    assert aligned.layerwise_aligner.aligners[0][0] is not aligned.layerwise_aligner.aligners[1][0]


def test_fixed_mode_bypasses_optional_layer_alignment_exactly():
    head = _head(
        use_adaptive_bridge=False,
        fixed_layer_index=1,
        adaptive_layer_alignment=True,
        adaptive_num_layers=3,
        adaptive_alignment_bottleneck=4,
    )
    hidden, proprio, projector = _routing_inputs()
    state, _, _ = head._extract_state_emb(hidden, proprio, projector)
    expected = hidden[:, 1, 4:, :] + projector(proprio).unsqueeze(1)
    torch.testing.assert_close(state, expected)


def test_continuous_time_embedding_resolves_unit_interval():
    timesteps = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    legacy = SinusoidalPosEmb(256, mode="legacy")(timesteps)
    continuous = SinusoidalPosEmb(256, mode="continuous")(timesteps)
    legacy_adjacent = (legacy[1:] - legacy[:-1]).norm(dim=-1).mean()
    continuous_adjacent = (continuous[1:] - continuous[:-1]).norm(dim=-1).mean()
    assert continuous_adjacent > 10 * legacy_adjacent


def test_openpi_style_flow_path_keeps_loss_in_float32_and_backpropagates():
    head = _head(
        flow_time_embedding_mode="continuous",
        flow_time_sampling_mode="openpi_beta",
        flow_float32_path=True,
        flow_zero_init_output=False,
    )
    hidden, proprio, projector = _routing_inputs()
    target = torch.randn(2, NUM_ACTIONS_CHUNK, 7)
    loss, metrics = head.flow_matching_loss(hidden, target, proprio, projector)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert head.time_mlp[1].weight.grad is not None
    assert head.output_proj[-1].weight.grad is not None
    assert torch.count_nonzero(head.output_proj[-1].weight).item() > 0
    assert "flow_matching_loss" in metrics


def test_prototype_soft_router_has_no_context_bypass_and_matches_template_mixture():
    torch.manual_seed(29)
    head = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_assignment_mode="soft",
        skill_routing_mode="prototype_soft",
        adaptive_num_layers=3,
        adaptive_alignment_bottleneck=4,
    )
    hidden, proprio, projector = _routing_inputs()
    head.train()
    state, _, _ = head._extract_state_emb(hidden, proprio, projector)

    assert head.skill_layer_scorer is None
    probs = head.last_routing_diagnostics["skill_probs"]
    templates = torch.softmax(head.skill_layer_logits[:, :3], dim=-1)
    expected_weights = probs @ templates.detach().cpu()
    torch.testing.assert_close(
        head.last_routing_diagnostics["layer_weights"],
        expected_weights,
        rtol=2e-3,
        atol=2e-3,
    )

    state.square().mean().backward()
    assert head.skill_selector[-1].weight.grad is not None
    assert head.skill_layer_logits.grad is not None
    assert torch.count_nonzero(head.skill_layer_logits.grad).item() > 0


def test_prototype_soft_router_is_identical_in_train_and_eval():
    torch.manual_seed(31)
    head = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_routing_mode="prototype_soft",
        skill_temperature_start=2.0,
        skill_temperature=0.7,
        skill_temperature_anneal_steps=1000,
        adaptive_num_layers=3,
        adaptive_alignment_bottleneck=4,
    )
    hidden, proprio, projector = _routing_inputs()
    head.set_routing_step(400)
    head.train()
    train_state, _, _ = head._extract_state_emb(hidden, proprio, projector)
    train_probs = head.last_routing_diagnostics["skill_probs"].clone()
    train_weights = head.last_routing_diagnostics["layer_weights"].clone()
    head.eval()
    eval_state, _, _ = head._extract_state_emb(hidden, proprio, projector)
    torch.testing.assert_close(train_state, eval_state)
    torch.testing.assert_close(train_probs, head.last_routing_diagnostics["skill_probs"])
    torch.testing.assert_close(train_weights, head.last_routing_diagnostics["layer_weights"])


def test_prototype_alignment_normalizes_each_depth_before_mixing():
    torch.manual_seed(37)
    head = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_routing_mode="prototype_soft",
        adaptive_num_layers=3,
        adaptive_alignment_bottleneck=4,
    )
    hidden, _, _ = _routing_inputs()
    hidden[:, 0] = hidden[:, 0] * 100.0 + 50.0
    hidden[:, 1] = hidden[:, 1] * 0.1 - 20.0
    aligned = head.prototype_layer_projector(hidden)
    means = aligned.float().mean(dim=(0, 2, 3))
    variances = aligned.float().var(dim=(0, 2, 3), unbiased=False)
    torch.testing.assert_close(means, torch.zeros_like(means), atol=2e-4, rtol=0)
    torch.testing.assert_close(variances, torch.ones_like(variances), atol=3e-3, rtol=0)


def test_prototype_regularizers_are_finite_and_update_router_and_templates():
    torch.manual_seed(41)
    head = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_routing_mode="prototype_soft",
        skill_balance_weight=0.01,
        skill_z_loss_weight=1e-4,
        skill_mi_weight=0.01,
        skill_template_diversity_weight=0.01,
        adaptive_num_layers=3,
        adaptive_alignment_bottleneck=4,
        flow_time_embedding_mode="continuous",
        flow_time_sampling_mode="openpi_beta",
        flow_float32_path=True,
        flow_zero_init_output=False,
    )
    hidden, proprio, projector = _routing_inputs()
    target = torch.randn(2, NUM_ACTIONS_CHUNK, 7)
    loss, metrics = head.flow_matching_loss(hidden, target, proprio, projector)
    assert torch.isfinite(loss)
    for key in (
        "skill_balance_loss",
        "skill_z_loss",
        "skill_mi_loss",
        "skill_template_diversity_loss",
        "skill_routing_aux_loss",
    ):
        assert key in metrics
        assert torch.isfinite(metrics[key])
    loss.backward()
    assert torch.count_nonzero(head.skill_selector[-1].weight.grad).item() > 0
    assert torch.count_nonzero(head.skill_layer_logits.grad).item() > 0


def test_query_key_soft_router_is_input_dependent_and_backpropagates():
    torch.manual_seed(43)
    head = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_assignment_mode="soft",
        skill_routing_mode="query_key_soft",
        skill_temperature_start=2.0,
        skill_temperature=0.7,
        skill_temperature_anneal_steps=1000,
        skill_layer_temperature=1.0,
        skill_balance_weight=0.01,
        skill_z_loss_weight=1e-4,
        skill_mi_weight=0.01,
        skill_layer_mi_weight=0.1,
        adaptive_num_layers=3,
        adaptive_alignment_bottleneck=4,
        flow_time_embedding_mode="continuous",
        flow_time_sampling_mode="openpi_beta",
        flow_float32_path=True,
        flow_zero_init_output=False,
    )
    hidden, proprio, projector = _routing_inputs()
    target = torch.randn(2, NUM_ACTIONS_CHUNK, 7)
    loss, metrics = head.flow_matching_loss(hidden, target, proprio, projector)
    weights = head.last_routing_diagnostics["layer_weights"]

    assert torch.isfinite(loss)
    assert weights.shape == (2, 3)
    assert not torch.allclose(weights[0], weights[1], atol=1e-6, rtol=0)
    assert metrics["routing_layer_batch_variation"] > 0
    assert torch.isfinite(metrics["routing_layer_mi_loss"])
    loss.backward()
    assert torch.count_nonzero(head.skill_selector[-1].weight.grad).item() > 0
    assert torch.count_nonzero(head.skill_embedding.weight.grad).item() > 0
    assert torch.count_nonzero(head.skill_query_projection[-1].weight.grad).item() > 0
    assert torch.count_nonzero(head.skill_key_projection[-1].weight.grad).item() > 0
    assert torch.count_nonzero(head.skill_layer_logits.grad).item() > 0
    assert head.skill_query_key_log_scale.grad is not None


def test_deep_anchor_decodes_only_the_routed_state_and_backpropagates():
    torch.manual_seed(47)
    head = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_assignment_mode="soft",
        skill_routing_mode="query_key_soft",
        adaptive_num_layers=3,
        adaptive_alignment_bottleneck=4,
        supervised_anchor_weight=1.0,
        anchor_blend=1.0,
        anchor_num_layers=3,
        anchor_hidden_dim=32,
        flow_time_embedding_mode="continuous",
        flow_time_sampling_mode="openpi_beta",
        flow_float32_path=True,
        flow_zero_init_output=False,
    )
    hidden, proprio, projector = _routing_inputs()
    target = torch.randn(2, NUM_ACTIONS_CHUNK, 7)
    loss, metrics = head.flow_matching_loss(hidden, target, proprio, projector)

    assert torch.isfinite(loss)
    assert "flowmlp_anchor_l1_loss" in metrics
    assert len(head.anchor_proj.blocks) == 3
    loss.backward()
    assert torch.count_nonzero(head.anchor_proj.input_proj.weight.grad).item() > 0
    assert torch.count_nonzero(head.anchor_proj.output_proj[-1].weight.grad).item() > 0
    assert torch.count_nonzero(head.skill_query_projection[-1].weight.grad).item() > 0


def test_depth_curriculum_warmup_stratifies_decoder_inputs_and_suppresses_flow():
    torch.manual_seed(53)
    head = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_assignment_mode="soft",
        skill_routing_mode="query_key_soft",
        adaptive_num_layers=3,
        adaptive_alignment_bottleneck=4,
        supervised_anchor_weight=1.0,
        anchor_blend=1.0,
        anchor_num_layers=2,
        anchor_hidden_dim=32,
        routing_curriculum_warmup_steps=20,
        routing_curriculum_teacher_steps=10,
        routing_curriculum_num_buckets=3,
        flow_curriculum_start_step=30,
        flow_curriculum_ramp_steps=10,
        flow_time_embedding_mode="continuous",
        flow_time_sampling_mode="openpi_beta",
        flow_float32_path=True,
        flow_zero_init_output=False,
    )
    hidden, proprio, projector = _routing_inputs()
    target = torch.randn(2, NUM_ACTIONS_CHUNK, 7)
    head.set_routing_step(0)
    loss, metrics = head.flow_matching_loss(hidden, target, proprio, projector)
    weights = head.last_routing_diagnostics["layer_weights"]

    assert torch.isfinite(loss)
    assert metrics["routing_curriculum_phase"] == 1
    assert metrics["flow_curriculum_weight"] == 0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
    assert torch.all((weights == 0) | (weights == 1))
    assert weights.argmax(dim=-1).unique().numel() == 2


def test_action_utility_teacher_trains_router_then_inference_uses_router_only():
    torch.manual_seed(59)
    head = _head(
        use_latent_skill_token=True,
        skill_use_direct_conditioning=False,
        skill_assignment_mode="soft",
        skill_routing_mode="query_key_soft",
        skill_layer_mi_weight=0.005,
        adaptive_num_layers=3,
        adaptive_alignment_bottleneck=4,
        supervised_anchor_weight=1.0,
        anchor_blend=1.0,
        anchor_num_layers=2,
        anchor_hidden_dim=32,
        routing_curriculum_warmup_steps=10,
        routing_curriculum_teacher_steps=10,
        routing_curriculum_num_buckets=3,
        routing_teacher_temperature=0.2,
        routing_teacher_kl_weight=1.0,
        flow_curriculum_start_step=20,
        flow_curriculum_ramp_steps=10,
        flow_time_embedding_mode="continuous",
        flow_time_sampling_mode="openpi_beta",
        flow_float32_path=True,
        flow_zero_init_output=False,
    )
    hidden, proprio, projector = _routing_inputs()
    target = torch.randn(2, NUM_ACTIONS_CHUNK, 7)
    head.set_routing_step(15)
    head.train()
    loss, metrics = head.flow_matching_loss(hidden, target, proprio, projector)

    assert torch.isfinite(loss)
    assert metrics["routing_curriculum_phase"] == 2
    assert metrics["routing_curriculum_teacher_kl"] >= 0
    assert torch.isfinite(metrics["routing_curriculum_teacher_entropy"])
    loss.backward()
    assert torch.count_nonzero(head.skill_query_projection[-1].weight.grad).item() > 0

    head.eval()
    with torch.no_grad():
        head._extract_state_emb(hidden, proprio, projector, target_actions=target)
    inference_weights = head.last_routing_diagnostics["layer_weights"]
    assert torch.all(inference_weights > 0)
