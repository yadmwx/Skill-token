import numpy as np

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    initialize_model,
    load_initial_states,
    validate_config,
)
from experiments.robot.robot_utils import get_action, set_seed_everywhere


PURE_1000 = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
    "--image_aug--VLA-Adapter-DIT12-fullaction-pure-gripfix-nozeroinit-object-1000-20260623--1000_chkpt"
)
PURE_2000 = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
    "--image_aug--VLA-Adapter-DIT12-fullaction-pure-gripfix-nozeroinit-object-1000-20260623--2000_chkpt"
)
FLOW = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0"
    "--image_aug--VLA-Adapter-FlowMLP-object-1000-20260618_231916--1000_chkpt"
)


def make_cfg(ckpt, head):
    cfg = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=ckpt,
        task_suite_name="libero_object",
        center_crop=True,
        num_trials_per_task=1,
        action_head_type=head,
        use_adaptive_bridge=True,
        bridge_mode="adaptive",
        fixed_layer_index=-1,
        dit_num_blocks=12,
        dit_num_inference_steps=20,
        dit_num_inference_samples=1,
        dit_supervised_anchor_weight=0.0,
        dit_anchor_blend=0.0,
        dit_anchor_blend_was_set=True,
        dit_detach_flow_conditioning=True,
        dit_disable_inference_anchor=True,
        dit_pure_inference=True,
        dit_zero_init_adaln=False,
        dit_zero_init_output=False,
        flow_ratio=1.0,
        use_depth_interface=False,
        depth_interface_mode="none",
        unnorm_key="libero_object_no_noops",
        seed=7,
        use_wandb=False,
    )
    validate_config(cfg)
    return cfg


def collect(name, pack, cfg, initial_states, task, task_description, num_episodes=3):
    model, action_head, depth_interface, proprio_projector, noisy_action_projector, processor = pack
    env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    preds = []
    try:
        for ep in range(num_episodes):
            env.reset()
            obs = env.set_init_state(initial_states[ep])
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            obs = dict(obs)
            obs["full_image"] = get_libero_image(obs)
            obs["wrist_image"] = get_libero_wrist_image(obs)
            obs["state"] = np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            )
            action = get_action(
                cfg,
                model,
                obs,
                task_description,
                processor=processor,
                action_head=action_head,
                depth_interface=depth_interface,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
                use_minivlm=cfg.use_minivlm,
            )
            preds.append(np.asarray(action))
    finally:
        env.close()

    arr = np.stack(preds)
    first = arr[:, 0, :]
    flat = arr.reshape(-1, arr.shape[-1])
    print(f"MODEL {name}")
    print(f"SHAPE {arr.shape}")
    print("FIRST_MEAN", np.array2string(first.mean(axis=0), precision=4, separator=","))
    print("FIRST_STD ", np.array2string(first.std(axis=0), precision=4, separator=","))
    print("ALL_MEAN  ", np.array2string(flat.mean(axis=0), precision=4, separator=","))
    print("ALL_STD   ", np.array2string(flat.std(axis=0), precision=4, separator=","))
    print("GRIP_VALUES", np.unique(flat[:, -1], return_counts=True))
    print(f"ABS_MEAN {np.abs(flat).mean():.6f}")
    print(f"GRIP_MEAN {flat[:, -1].mean():.6f}")
    print(f"GRIP_STD {flat[:, -1].std():.6f}")
    print()
    return arr


def compare(label, a, b):
    diff = np.abs(a - b)
    print(f"{label} mean_abs {float(diff.mean()):.6f}")
    print(f"{label} max_abs {float(diff.max()):.6f}")
    print("DIM_MEAN_ABS", np.array2string(diff.reshape(-1, diff.shape[-1]).mean(axis=0), precision=4, separator=","))
    print()


def main():
    set_seed_everywhere(7)
    cfgs = {
        "PURE_1000": make_cfg(PURE_1000, "DIT"),
        "PURE_2000": make_cfg(PURE_2000, "DIT"),
        "FLOW": make_cfg(FLOW, "FlowMLP"),
    }
    loaded = {name: initialize_model(cfg) for name, cfg in cfgs.items()}
    task_suite = benchmark.get_benchmark_dict()[cfgs["PURE_1000"].task_suite_name]()
    task = task_suite.get_task(0)
    initial_states, _ = load_initial_states(cfgs["PURE_1000"], task_suite, 0, log_file=None)

    results = {}
    for name, cfg in cfgs.items():
        results[name] = collect(name, loaded[name], cfg, initial_states, task, task.language)

    compare("PURE_2000_vs_1000", results["PURE_2000"], results["PURE_1000"])
    compare("PURE_1000_vs_FLOW", results["PURE_1000"], results["FLOW"])
    compare("PURE_2000_vs_FLOW", results["PURE_2000"], results["FLOW"])


if __name__ == "__main__":
    main()
