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


RESIDUAL_DIT_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
    "--image_aug--VLA-Adapter-DIT12-residual-anchor-object-1000-aggfix-hfpathfix-aqsave-20260622--1000_chkpt"
)
FULLACTION_DIT_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
    "--image_aug--VLA-Adapter-DIT12-fullaction-adaptive-object-1000-20260623--2000_chkpt"
)
FLOW_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0"
    "--image_aug--VLA-Adapter-FlowMLP-object-1000-20260618_231916--1000_chkpt"
)


def make_cfg(
    ckpt: str,
    head: str,
    *,
    bridge_mode: str = "adaptive",
    dit_steps: int = 20,
    disable_anchor: bool = False,
):
    cfg = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=ckpt,
        task_suite_name="libero_object",
        center_crop=True,
        num_trials_per_task=1,
        action_head_type=head,
        use_adaptive_bridge=True,
        bridge_mode=bridge_mode,
        dit_num_blocks=12,
        dit_num_inference_steps=dit_steps,
        dit_num_inference_samples=8,
        dit_anchor_blend=0.0,
        dit_anchor_blend_was_set=True,
        dit_disable_inference_anchor=disable_anchor,
        flowmlp_anchor_blend=0.0,
        use_depth_interface=False,
        depth_interface_mode="none",
        unnorm_key="libero_object_no_noops",
        seed=7,
        use_wandb=False,
    )
    validate_config(cfg)
    return cfg


def collect_actions(name, pack, cfg, initial_states, task, task_description, num_episodes=3):
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
    print(f"MODEL {name}")
    print(f"SHAPE {arr.shape}")
    first = arr[:, 0, :]
    flat = arr.reshape(-1, arr.shape[-1])
    print("FIRST_MEAN", np.array2string(first.mean(axis=0), precision=4, separator=","))
    print("FIRST_STD ", np.array2string(first.std(axis=0), precision=4, separator=","))
    print("ALL_MEAN  ", np.array2string(flat.mean(axis=0), precision=4, separator=","))
    print("ALL_STD   ", np.array2string(flat.std(axis=0), precision=4, separator=","))
    print(f"ABS_MEAN {np.abs(flat).mean():.6f}")
    print(f"GRIP_MEAN {flat[:, -1].mean():.6f}")
    print(f"GRIP_STD {flat[:, -1].std():.6f}")
    print()
    return arr


def compare(label, a, b):
    absdiff = np.abs(a - b)
    print(f"{label} mean_abs {float(absdiff.mean()):.6f}")
    print(f"{label} max_abs {float(absdiff.max()):.6f}")
    print()


def main():
    set_seed_everywhere(7)
    configs = {
        "DIT_RESIDUAL_NOANCHOR": make_cfg(
            RESIDUAL_DIT_CKPT,
            "DIT",
            bridge_mode="adaptive",
            dit_steps=20,
            disable_anchor=True,
        ),
        "DIT_FULLACTION_2000": make_cfg(
            FULLACTION_DIT_CKPT,
            "DIT",
            bridge_mode="adaptive",
            dit_steps=20,
            disable_anchor=False,
        ),
        "FLOW_BASELINE": make_cfg(
            FLOW_CKPT,
            "FlowMLP",
            bridge_mode="adaptive",
            dit_steps=20,
            disable_anchor=False,
        ),
    }
    loaded = {name: initialize_model(cfg) for name, cfg in configs.items()}
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[configs["DIT_FULLACTION_2000"].task_suite_name]()
    task = task_suite.get_task(0)
    task_description = task.language
    initial_states, _ = load_initial_states(configs["DIT_FULLACTION_2000"], task_suite, 0, log_file=None)

    results = {}
    for name, cfg in configs.items():
        results[name] = collect_actions(
            name,
            loaded[name],
            cfg,
            initial_states,
            task,
            task_description,
            num_episodes=3,
        )

    compare(
        "DIT_FULLACTION_vs_FLOW",
        results["DIT_FULLACTION_2000"],
        results["FLOW_BASELINE"],
    )
    compare(
        "DIT_RESIDUAL_NOANCHOR_vs_FLOW",
        results["DIT_RESIDUAL_NOANCHOR"],
        results["FLOW_BASELINE"],
    )
    compare(
        "DIT_FULLACTION_vs_DIT_RESIDUAL_NOANCHOR",
        results["DIT_FULLACTION_2000"],
        results["DIT_RESIDUAL_NOANCHOR"],
    )


if __name__ == "__main__":
    main()
