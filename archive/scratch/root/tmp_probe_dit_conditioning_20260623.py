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
from experiments.robot.openvla_utils import prepare_images_for_vla, normalize_proprio
from experiments.robot.robot_utils import set_seed_everywhere


DIT_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
    "--image_aug--VLA-Adapter-DIT12-fullaction-adaptive-object-1000-20260623--2000_chkpt"
)


def make_cfg():
    cfg = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=DIT_CKPT,
        task_suite_name="libero_object",
        task_ids="0",
        center_crop=True,
        num_trials_per_task=1,
        action_head_type="DIT",
        use_adaptive_bridge=True,
        bridge_mode="adaptive",
        dit_num_blocks=12,
        dit_num_inference_steps=20,
        dit_num_inference_samples=1,
        dit_anchor_blend=0.0,
        dit_anchor_blend_was_set=True,
        dit_disable_inference_anchor=False,
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


def aggregate_tokens(action_head, task_hidden_states, action_hidden_states):
    velocity_network = action_head.velocity_network
    num_layers = task_hidden_states.shape[1]
    if not velocity_network.use_adaptive_bridge:
        layer_idx = velocity_network.fixed_layer_index if velocity_network.fixed_layer_index >= 0 else num_layers // 2
        task_tokens = task_hidden_states[:, layer_idx, :, :]
        action_tokens = action_hidden_states[:, layer_idx, :, :]
        layer_weights = None
    elif velocity_network.bridge_mode == "adaptive":
        task_tokens, layer_weights = velocity_network.task_layer_selector(task_hidden_states)
        weights = layer_weights.view(task_hidden_states.shape[0], num_layers, 1, 1)
        action_tokens = (weights * action_hidden_states).sum(dim=1)
    else:
        raise ValueError(f"Unsupported bridge mode for probe: {velocity_network.bridge_mode}")
    return task_tokens, action_tokens, layer_weights


def probe_velocity(action_head, task_tokens, action_tokens, proprio_features):
    batch_size = task_tokens.shape[0]
    model_dtype = next(action_head.time_encoder.parameters()).dtype
    z = torch.randn(
        (batch_size, 8, action_head.action_dim),
        device=task_tokens.device,
        dtype=model_dtype,
    )
    t = torch.full((batch_size,), 0.0, device=task_tokens.device, dtype=model_dtype)
    r = torch.full((batch_size,), 0.0, device=task_tokens.device, dtype=model_dtype)
    t_emb = action_head.time_encoder(t)
    r_emb = action_head.target_t_encoder(r)
    timestep = torch.zeros(batch_size, device=task_tokens.device)
    target_t = torch.zeros(batch_size, device=task_tokens.device)

    def run_with_context(context):
        return action_head.velocity_network.ditx(
            sample=z,
            timestep=timestep,
            target_t=target_t,
            vis_cond=context,
            timestep_emb=t_emb,
            target_t_emb=r_emb,
        )

    full_v = run_with_context(torch.cat([task_tokens, action_tokens, proprio_features], dim=1))
    zero_task_v = run_with_context(torch.cat([torch.zeros_like(task_tokens), action_tokens, proprio_features], dim=1))
    zero_action_v = run_with_context(torch.cat([task_tokens, torch.zeros_like(action_tokens), proprio_features], dim=1))
    summarize("velocity_full", full_v)
    summarize("velocity_zero_task", zero_task_v)
    summarize("velocity_zero_action", zero_action_v)
    print("velocity_full_first")
    print(np.array2string(full_v[0].detach().float().cpu().numpy(), precision=4, separator=","))


def main():
    set_seed_everywhere(7)
    cfg = make_cfg()
    model, action_head, depth_interface, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(0)
    initial_states, _ = load_initial_states(cfg, task_suite, 0, log_file=None)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    try:
        env.reset()
        obs = env.set_init_state(initial_states[0])
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        observation = build_obs(obs)
    finally:
        env.close()

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

    print(f"actions_hidden_states shape={tuple(actions_hidden_states.shape)}")
    task_hidden_states = actions_hidden_states[:, :, : action_head.num_task_tokens, :]
    action_hidden_states = actions_hidden_states[:, :, action_head.num_task_tokens :, :]
    summarize("task_hidden_states", task_hidden_states)
    summarize("action_hidden_states", action_hidden_states)

    task_tokens, action_tokens, layer_weights = aggregate_tokens(action_head, task_hidden_states, action_hidden_states)
    if layer_weights is not None:
        lw_tensor = layer_weights[0].detach().to(dtype=torch.float32).cpu()
        lw = np.asarray(lw_tensor.tolist(), dtype=np.float32)
        print("layer_weights")
        print(np.array2string(lw, precision=4, separator=","))
        top_idx = np.argsort(-lw)[:5]
        print(f"top_layers={top_idx.tolist()} top_weights={[float(lw[i]) for i in top_idx]}")

    summarize("task_tokens_agg", task_tokens)
    summarize("action_tokens_agg", action_tokens)

    proprio_tensor = torch.as_tensor(proprio, device=actions_hidden_states.device, dtype=actions_hidden_states.dtype).reshape(1, -1)
    proprio_features = proprio_projector(proprio_tensor).unsqueeze(1)
    summarize("proprio_features", proprio_features)

    context = torch.cat([task_tokens, action_tokens, proprio_features], dim=1)
    summarize("context_full", context)
    print(f"context_lengths task={task_tokens.shape[1]} action={action_tokens.shape[1]} proprio={proprio_features.shape[1]} total={context.shape[1]}")

    print("segment_norms")
    print(
        {
            "task_mean_token_norm": float(task_tokens.float().norm(dim=-1).mean().item()),
            "action_mean_token_norm": float(action_tokens.float().norm(dim=-1).mean().item()),
            "proprio_token_norm": float(proprio_features.float().norm(dim=-1).mean().item()),
        }
    )

    probe_velocity(action_head, task_tokens, action_tokens, proprio_features)


if __name__ == "__main__":
    main()
