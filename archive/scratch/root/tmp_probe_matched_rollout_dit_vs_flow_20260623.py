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


def summarize(name, tensor):
    tensor_f = tensor.float()
    print(
        f"{name}: shape={tuple(tensor.shape)} "
        f"mean={tensor_f.mean().item():.6f} abs_mean={tensor_f.abs().mean().item():.6f} "
        f"std={tensor_f.std(unbiased=False).item():.6f} norm={tensor_f.norm().item():.6f}"
    )


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
    return actions_hidden_states, torch.as_tensor(proprio, device=actions_hidden_states.device, dtype=actions_hidden_states.dtype)


def rollout_dit(action_head, actions_hidden_states, proprio, proprio_projector, z0, num_steps):
    z = z0.clone()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((z.shape[0],), i / num_steps, device=z.device, dtype=torch.float32)
        r = torch.zeros_like(t)
        v = action_head.predict_velocity(z, t, r, actions_hidden_states, proprio, proprio_projector)
        z = z + dt * v
    return z


def rollout_flow(action_head, actions_hidden_states, proprio, proprio_projector, z0, num_steps):
    state_emb, _, _ = action_head._extract_state_emb(actions_hidden_states, proprio, proprio_projector)
    z = z0.clone()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((z.shape[0],), i / num_steps, device=z.device, dtype=z.dtype)
        v = action_head.forward(state_emb, z, t)
        z = z + dt * v
    return z, state_emb


def rollout_dit_from_state(action_head, state_emb, proprio, proprio_projector, z0, num_steps):
    z = z0.clone()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((z.shape[0],), i / num_steps, device=z.device, dtype=torch.float32)
        r = torch.zeros_like(t)
        v = action_head.predict_velocity_from_state(z, t, r, state_emb, proprio, proprio_projector)
        z = z + dt * v
    return z


def split_dit_context(action_head, actions_hidden_states, proprio, proprio_projector):
    velocity_network = action_head.velocity_network
    task_hidden_states = actions_hidden_states[:, :, : action_head.num_task_tokens, :]
    action_hidden_states = actions_hidden_states[:, :, action_head.num_task_tokens :, :]
    num_layers = task_hidden_states.shape[1]

    if not velocity_network.use_adaptive_bridge:
        layer_idx = velocity_network.fixed_layer_index if velocity_network.fixed_layer_index >= 0 else num_layers // 2
        task_tokens = task_hidden_states[:, layer_idx, :, :]
        action_tokens = action_hidden_states[:, layer_idx, :, :]
    elif velocity_network.bridge_mode == "adaptive":
        task_tokens, layer_weights = velocity_network.task_layer_selector(task_hidden_states)
        weights = layer_weights.view(task_hidden_states.shape[0], num_layers, 1, 1)
        action_tokens = (weights * action_hidden_states).sum(dim=1)
    elif velocity_network.bridge_mode == "uniform":
        task_tokens = task_hidden_states.mean(dim=1)
        action_tokens = action_hidden_states.mean(dim=1)
    else:
        raise ValueError(f"Unsupported DIT bridge mode for probe: {velocity_network.bridge_mode}")

    if proprio.ndim == 1:
        proprio = proprio.unsqueeze(0)
    proprio_features = proprio_projector(proprio.reshape(proprio.shape[0], -1)).unsqueeze(1)
    return task_tokens, action_tokens, proprio_features


def align_action_tokens_to_chunk(action_tokens):
    bsz, num_tokens, dim = action_tokens.shape
    if num_tokens % 8 != 0:
        raise ValueError(f"Expected action token count divisible by 8, got {num_tokens}")
    return action_tokens.reshape(bsz, 8, num_tokens // 8, dim).mean(dim=2)


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
        dit_v0 = dit_head.predict_velocity(z0, t0, r0, dit_hidden, dit_proprio, dit_proprio_projector)
        dit_task_tokens, dit_action_tokens, dit_proprio_features = split_dit_context(
            dit_head, dit_hidden, dit_proprio, dit_proprio_projector
        )
        dit_state_emb = align_action_tokens_to_chunk(dit_action_tokens)
        dit_t_emb = dit_head.time_encoder(t0.to(dtype=model_dtype))
        dit_r_emb = dit_head.target_t_encoder(r0.to(dtype=model_dtype))
        zeros = torch.zeros((1,), device=dit_hidden.device)

        def run_dit_variant(task_tokens, action_tokens):
            context = torch.cat([task_tokens, action_tokens, dit_proprio_features], dim=1)
            return dit_head.velocity_network.ditx(
                sample=z0,
                timestep=zeros,
                target_t=zeros,
                vis_cond=context,
                timestep_emb=dit_t_emb,
                target_t_emb=dit_r_emb,
            )

        dit_v0_zero_task = run_dit_variant(torch.zeros_like(dit_task_tokens), dit_action_tokens)
        dit_v0_zero_action = run_dit_variant(dit_task_tokens, torch.zeros_like(dit_action_tokens))
        dit_v0_from_state = dit_head.predict_velocity_from_state(
            z0,
            t0,
            r0,
            dit_state_emb,
            dit_proprio,
            dit_proprio_projector,
        )
        flow_state_emb, _, _ = flow_head._extract_state_emb(flow_hidden, flow_proprio, flow_proprio_projector)
        flow_v0 = flow_head.forward(flow_state_emb, z0.to(dtype=flow_state_emb.dtype), t0.to(dtype=flow_state_emb.dtype))

        dit_roll5 = rollout_dit(dit_head, dit_hidden, dit_proprio, dit_proprio_projector, z0, num_steps=5)
        dit_roll5_from_state = rollout_dit_from_state(
            dit_head,
            dit_state_emb,
            dit_proprio,
            dit_proprio_projector,
            z0,
            num_steps=5,
        )
        flow_roll5, _ = rollout_flow(flow_head, flow_hidden, flow_proprio, flow_proprio_projector, z0.to(dtype=flow_state_emb.dtype), num_steps=5)

    summarize("dit_hidden", dit_hidden)
    summarize("flow_hidden", flow_hidden)
    summarize("dit_state_emb", dit_state_emb)
    summarize("flow_state_emb", flow_state_emb)
    summarize("z0", z0)
    summarize("dit_v0", dit_v0)
    summarize("dit_v0_zero_task", dit_v0_zero_task)
    summarize("dit_v0_zero_action", dit_v0_zero_action)
    summarize("dit_v0_from_state", dit_v0_from_state)
    summarize("flow_v0", flow_v0)
    summarize("dit_roll5", dit_roll5)
    summarize("dit_roll5_from_state", dit_roll5_from_state)
    summarize("flow_roll5", flow_roll5)
    summarize("v0_diff", (dit_v0.float() - flow_v0.float()))
    summarize("v0_zero_task_diff", (dit_v0_zero_task.float() - flow_v0.float()))
    summarize("v0_zero_action_diff", (dit_v0_zero_action.float() - flow_v0.float()))
    summarize("v0_from_state_diff", (dit_v0_from_state.float() - flow_v0.float()))
    summarize("roll5_diff", (dit_roll5.float() - flow_roll5.float()))
    summarize("roll5_from_state_diff", (dit_roll5_from_state.float() - flow_roll5.float()))

    print("z0_first")
    print(np.array2string(z0[0].detach().float().cpu().numpy(), precision=4, separator=","))
    print("dit_v0_first")
    print(np.array2string(dit_v0[0].detach().float().cpu().numpy(), precision=4, separator=","))
    print("flow_v0_first")
    print(np.array2string(flow_v0[0].detach().float().cpu().numpy(), precision=4, separator=","))
    print("dit_roll5_first")
    print(np.array2string(dit_roll5[0].detach().float().cpu().numpy(), precision=4, separator=","))
    print("dit_roll5_from_state_first")
    print(np.array2string(dit_roll5_from_state[0].detach().float().cpu().numpy(), precision=4, separator=","))
    print("flow_roll5_first")
    print(np.array2string(flow_roll5[0].detach().float().cpu().numpy(), precision=4, separator=","))


if __name__ == "__main__":
    main()
