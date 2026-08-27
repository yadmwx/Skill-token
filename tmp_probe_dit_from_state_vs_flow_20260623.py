import numpy as np
import torch

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    initialize_model,
    load_initial_states,
    prepare_observation,
    validate_config,
)
from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla
from experiments.robot.robot_utils import set_seed_everywhere


DIT_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
    "--image_aug--VLA-Adapter-DIT12-fullaction-adaptive-object-1000-20260623--2000_chkpt"
)
FLOW_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0"
    "--image_aug--VLA-Adapter-FlowMLP-object-1000-20260618_231916--1000_chkpt"
)


def make_cfg(ckpt: str, head: str):
    kwargs = dict(
        model_family="openvla",
        pretrained_checkpoint=ckpt,
        task_suite_name="libero_object",
        task_ids="0",
        center_crop=True,
        num_trials_per_task=1,
        action_head_type=head,
        use_adaptive_bridge=True,
        bridge_mode="adaptive",
        fixed_layer_index=-1,
        use_depth_interface=False,
        depth_interface_mode="none",
        unnorm_key="libero_object_no_noops",
        use_proprio=True,
        num_images_in_input=2,
        use_film=False,
        use_minivlm=True,
        use_pro_version=True,
        use_wandb=False,
        seed=7,
    )
    if head == "DIT":
        cfg = GenerateConfig(
            **kwargs,
            dit_num_blocks=12,
            dit_num_inference_steps=20,
            dit_num_inference_samples=1,
            dit_anchor_blend=0.0,
            dit_anchor_blend_was_set=True,
            dit_disable_inference_anchor=False,
        )
    else:
        cfg = GenerateConfig(
            **kwargs,
            flowmlp_num_inference_steps=5,
            flowmlp_num_inference_samples=8,
            flowmlp_anchor_blend=0.0,
        )
    validate_config(cfg)
    return cfg


def summarize(name, tensor):
    t = tensor.float()
    print(
        f"{name}: shape={tuple(tensor.shape)} mean={t.mean().item():.6f} "
        f"abs_mean={t.abs().mean().item():.6f} std={t.std(unbiased=False).item():.6f} norm={t.norm().item():.6f}"
    )


def build_obs(raw_obs):
    observation, _ = prepare_observation(raw_obs, 224)
    observation["state"] = np.concatenate(
        (
            raw_obs["robot0_eef_pos"],
            quat2axisangle(raw_obs["robot0_eef_quat"]),
            raw_obs["robot0_gripper_qpos"],
        )
    )
    return observation


def build_inputs(cfg, model, processor, observation, task_description):
    device = next(model.parameters()).device
    all_images = [observation["full_image"]]
    if cfg.num_images_in_input > 1:
        all_images.extend([observation[k] for k in observation.keys() if "wrist" in k])
    all_images = prepare_images_for_vla(all_images, cfg)
    primary_image = all_images.pop(0)
    prompt = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\nWhat action should the robot take to {task_description.lower()}?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = processor(prompt, primary_image).to(device, dtype=torch.bfloat16)
    if all_images:
        wrist_inputs = [processor(prompt, image).to(device, dtype=torch.bfloat16) for image in all_images]
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"]] + [item["pixel_values"] for item in wrist_inputs], dim=1)

    proprio = observation["state"].copy()
    proprio_norm_stats = model.norm_stats[cfg.unnorm_key]["proprio"]
    proprio = normalize_proprio(proprio, proprio_norm_stats)
    return inputs, proprio


def get_hidden_states(pack, cfg, observation, task_description):
    model, action_head, depth_interface, proprio_projector, noisy_action_projector, processor = pack
    inputs, proprio = build_inputs(cfg, model, processor, observation, task_description)
    with torch.no_grad():
        _, actions_hidden_states = model.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            action_head=action_head,
            depth_interface=depth_interface,
            use_film=cfg.use_film,
        )
    proprio_t = torch.as_tensor(proprio, device=actions_hidden_states.device, dtype=actions_hidden_states.dtype)
    return actions_hidden_states, proprio_t


def extract_dit_state_emb(dit_head, actions_hidden_states):
    task_hidden_states = actions_hidden_states[:, :, : dit_head.num_task_tokens, :]
    action_hidden_states = actions_hidden_states[:, :, dit_head.num_task_tokens :, :]
    task_tokens, layer_weights = dit_head.velocity_network.task_layer_selector(task_hidden_states)
    weights = layer_weights.view(task_hidden_states.shape[0], task_hidden_states.shape[1], 1, 1)
    action_tokens = (weights * action_hidden_states).sum(dim=1)
    bsz, num_tokens, dim = action_tokens.shape
    state_emb = action_tokens.reshape(bsz, 8, num_tokens // 8, dim).mean(dim=2)
    return state_emb, task_tokens, action_tokens


def rollout_dit_from_state(dit_head, state_emb, proprio, proprio_projector, z0, num_steps):
    z = z0.clone()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((z.shape[0],), i / num_steps, device=z.device, dtype=torch.float32)
        r = torch.zeros_like(t)
        v = dit_head.predict_velocity_from_state(z, t, r, state_emb, proprio, proprio_projector)
        z = z + dt * v
    return z


def rollout_flow_from_state(flow_head, state_emb, z0, num_steps):
    z = z0.clone().to(dtype=state_emb.dtype)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((z.shape[0],), i / num_steps, device=z.device, dtype=z.dtype)
        v = flow_head.forward(state_emb, z, t)
        z = z + dt * v
    return z


def main():
    set_seed_everywhere(7)
    cfg_dit = make_cfg(DIT_CKPT, "DIT")
    cfg_flow = make_cfg(FLOW_CKPT, "FlowMLP")
    dit_pack = initialize_model(cfg_dit)
    flow_pack = initialize_model(cfg_flow)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg_dit.task_suite_name]()
    task = task_suite.get_task(0)
    initial_states, _ = load_initial_states(cfg_dit, task_suite, 0, log_file=None)
    env, task_description = get_libero_env(task, cfg_dit.model_family, resolution=cfg_dit.env_img_res)
    try:
        env.reset()
        obs = env.set_init_state(initial_states[0])
        for _ in range(cfg_dit.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg_dit.model_family))
        observation = build_obs(obs)
    finally:
        env.close()

    dit_hidden, dit_proprio = get_hidden_states(dit_pack, cfg_dit, observation, task_description)
    flow_hidden, flow_proprio = get_hidden_states(flow_pack, cfg_flow, observation, task_description)

    _, dit_head, _, dit_proprio_projector, _, _ = dit_pack
    _, flow_head, _, flow_proprio_projector, _, _ = flow_pack

    model_dtype = next(dit_head.time_encoder.parameters()).dtype
    z0 = torch.randn((1, 8, 7), device=dit_hidden.device, dtype=model_dtype)
    t0 = torch.zeros((1,), device=dit_hidden.device, dtype=torch.float32)
    r0 = torch.zeros((1,), device=dit_hidden.device, dtype=torch.float32)

    with torch.no_grad():
        dit_state_emb, dit_task_tokens, dit_action_tokens = extract_dit_state_emb(dit_head, dit_hidden)
        flow_state_emb, _, _ = flow_head._extract_state_emb(flow_hidden, flow_proprio, flow_proprio_projector)
        scale = flow_state_emb.float().norm() / dit_state_emb.float().norm().clamp_min(1e-8)
        dit_state_emb_scaled = dit_state_emb * scale.to(dtype=dit_state_emb.dtype)
        dit_v0_from_state = dit_head.predict_velocity_from_state(
            z0, t0, r0, dit_state_emb, dit_proprio, dit_proprio_projector
        )
        dit_v0_from_state_scaled = dit_head.predict_velocity_from_state(
            z0, t0, r0, dit_state_emb_scaled, dit_proprio, dit_proprio_projector
        )
        flow_v0 = flow_head.forward(flow_state_emb, z0.to(dtype=flow_state_emb.dtype), t0.to(dtype=flow_state_emb.dtype))
        dit_roll5_from_state = rollout_dit_from_state(
            dit_head, dit_state_emb, dit_proprio, dit_proprio_projector, z0, num_steps=5
        )
        dit_roll5_from_state_scaled = rollout_dit_from_state(
            dit_head, dit_state_emb_scaled, dit_proprio, dit_proprio_projector, z0, num_steps=5
        )
        flow_roll5 = rollout_flow_from_state(flow_head, flow_state_emb, z0, num_steps=5)

    summarize("dit_state_emb", dit_state_emb)
    summarize("dit_state_emb_scaled", dit_state_emb_scaled)
    summarize("flow_state_emb", flow_state_emb)
    summarize("dit_task_tokens", dit_task_tokens)
    summarize("dit_action_tokens", dit_action_tokens)
    summarize("z0", z0)
    summarize("dit_v0_from_state", dit_v0_from_state)
    summarize("dit_v0_from_state_scaled", dit_v0_from_state_scaled)
    summarize("flow_v0", flow_v0)
    summarize("v0_from_state_diff", dit_v0_from_state.float() - flow_v0.float())
    summarize("v0_from_state_scaled_diff", dit_v0_from_state_scaled.float() - flow_v0.float())
    summarize("dit_roll5_from_state", dit_roll5_from_state)
    summarize("dit_roll5_from_state_scaled", dit_roll5_from_state_scaled)
    summarize("flow_roll5", flow_roll5)
    summarize("roll5_from_state_diff", dit_roll5_from_state.float() - flow_roll5.float())
    summarize("roll5_from_state_scaled_diff", dit_roll5_from_state_scaled.float() - flow_roll5.float())
    print(f"state_emb_scale_factor={float(scale.item()):.6f}")

    print("dit_v0_from_state_first")
    print(np.array2string(dit_v0_from_state[0].detach().float().cpu().numpy(), precision=4, separator=","))
    print("dit_v0_from_state_scaled_first")
    print(np.array2string(dit_v0_from_state_scaled[0].detach().float().cpu().numpy(), precision=4, separator=","))
    print("flow_v0_first")
    print(np.array2string(flow_v0[0].detach().float().cpu().numpy(), precision=4, separator=","))
    print("dit_roll5_from_state_first")
    print(np.array2string(dit_roll5_from_state[0].detach().float().cpu().numpy(), precision=4, separator=","))
    print("dit_roll5_from_state_scaled_first")
    print(np.array2string(dit_roll5_from_state_scaled[0].detach().float().cpu().numpy(), precision=4, separator=","))
    print("flow_roll5_first")
    print(np.array2string(flow_roll5[0].detach().float().cpu().numpy(), precision=4, separator=","))


if __name__ == "__main__":
    main()
