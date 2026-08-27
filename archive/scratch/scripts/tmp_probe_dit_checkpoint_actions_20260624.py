import argparse
from pathlib import Path

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


FLOW_BASELINE_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0"
    "--image_aug--VLA-Adapter-FlowMLP-object-1000-20260618_231916--1000_chkpt"
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def make_cfg(args, ckpt, head):
    cfg = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=ckpt,
        task_suite_name=args.task_suite,
        center_crop=True,
        num_trials_per_task=1,
        action_head_type=head,
        use_adaptive_bridge=args.use_adaptive_bridge,
        bridge_mode=args.bridge_mode,
        fixed_layer_index=args.fixed_layer_index,
        dit_num_blocks=args.dit_num_blocks,
        dit_num_inference_steps=args.dit_num_inference_steps,
        dit_num_inference_samples=args.dit_num_inference_samples,
        dit_supervised_anchor_weight=0.0,
        dit_anchor_blend=0.0,
        dit_anchor_blend_was_set=True,
        dit_detach_flow_conditioning=args.dit_detach_flow_conditioning,
        dit_condition_mode=args.dit_condition_mode,
        dit_include_prompt_tokens=args.dit_include_prompt_tokens,
        dit_disable_inference_anchor=True,
        dit_pure_inference=True,
        dit_use_state_conditioning=False,
        dit_zero_init_adaln=args.dit_zero_init_adaln,
        dit_zero_init_output=args.dit_zero_init_output,
        flow_ratio=1.0,
        use_depth_interface=False,
        depth_interface_mode="none",
        unnorm_key=args.unnorm_key,
        seed=args.seed,
        use_wandb=False,
    )
    validate_config(cfg)
    return cfg


def collect_actions(name, pack, cfg, initial_states, task, task_description, episodes):
    model, action_head, depth_interface, proprio_projector, noisy_action_projector, processor = pack
    env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    preds = []
    try:
        for ep in range(episodes):
            env.reset()
            obs = env.set_init_state(initial_states[ep])
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            obs = dict(obs)
            obs["full_image"] = get_libero_image(obs)
            obs["wrist_image"] = get_libero_wrist_image(obs)
            obs["state"] = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
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
    print(f"CHECKPOINT {cfg.pretrained_checkpoint}")
    if cfg.action_head_type.upper() == "DIT":
        print(f"DIT_CONDITION_MODE {cfg.dit_condition_mode}")
        print(f"DIT_INCLUDE_PROMPT_TOKENS {cfg.dit_include_prompt_tokens}")
    print(f"SHAPE {arr.shape}")
    print("FIRST_MEAN", np.array2string(first.mean(axis=0), precision=5, separator=","))
    print("FIRST_STD ", np.array2string(first.std(axis=0), precision=5, separator=","))
    print("ALL_MEAN  ", np.array2string(flat.mean(axis=0), precision=5, separator=","))
    print("ALL_STD   ", np.array2string(flat.std(axis=0), precision=5, separator=","))
    print("ALL_MIN   ", np.array2string(flat.min(axis=0), precision=5, separator=","))
    print("ALL_MAX   ", np.array2string(flat.max(axis=0), precision=5, separator=","))
    print(f"ABS_MEAN {float(np.abs(flat).mean()):.6f}")
    print(f"NON_GRIP_ABS_MEAN {float(np.abs(flat[:, :-1]).mean()):.6f}")
    print(f"GRIP_MEAN {float(flat[:, -1].mean()):.6f}")
    print(f"GRIP_STD {float(flat[:, -1].std()):.6f}")
    print(f"GRIP_MIN {float(flat[:, -1].min()):.6f}")
    print(f"GRIP_MAX {float(flat[:, -1].max()):.6f}")
    print()
    return arr


def print_compare(label, lhs, rhs):
    diff = np.abs(lhs - rhs)
    flat = diff.reshape(-1, diff.shape[-1])
    print(f"COMPARE {label}")
    print(f"MEAN_ABS {float(diff.mean()):.6f}")
    print(f"MAX_ABS {float(diff.max()):.6f}")
    print("DIM_MEAN_ABS", np.array2string(flat.mean(axis=0), precision=5, separator=","))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--task_suite", default="libero_object")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--unnorm_key", default="libero_object_no_noops")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dit_num_blocks", type=int, default=12)
    parser.add_argument("--dit_num_inference_steps", type=int, default=20)
    parser.add_argument("--dit_num_inference_samples", type=int, default=1)
    parser.add_argument("--dit_detach_flow_conditioning", type=str2bool, default=False)
    parser.add_argument("--dit_condition_mode", default="full", choices=["full", "task_only"])
    parser.add_argument("--dit_include_prompt_tokens", type=str2bool, default=False)
    parser.add_argument("--dit_zero_init_adaln", type=str2bool, default=False)
    parser.add_argument("--dit_zero_init_output", type=str2bool, default=False)
    parser.add_argument("--use_adaptive_bridge", type=str2bool, default=True)
    parser.add_argument("--bridge_mode", default="adaptive")
    parser.add_argument("--fixed_layer_index", type=int, default=-1)
    parser.add_argument("--compare_flow", type=str2bool, default=True)
    parser.add_argument("--flow_checkpoint", default=FLOW_BASELINE_CKPT)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)

    set_seed_everywhere(args.seed)
    dit_cfg = make_cfg(args, args.checkpoint, "DIT")
    configs = {"DIT": dit_cfg}
    if args.compare_flow:
        configs["FLOW_BASELINE"] = make_cfg(args, args.flow_checkpoint, "FlowMLP")

    loaded = {name: initialize_model(cfg) for name, cfg in configs.items()}
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    initial_states, _ = load_initial_states(dit_cfg, task_suite, args.task_id, log_file=None)
    episodes = min(args.episodes, len(initial_states))

    results = {}
    for name, cfg in configs.items():
        results[name] = collect_actions(
            name,
            loaded[name],
            cfg,
            initial_states,
            task,
            task.language,
            episodes,
        )

    if "FLOW_BASELINE" in results:
        print_compare("DIT_vs_FLOW_BASELINE", results["DIT"], results["FLOW_BASELINE"])


if __name__ == "__main__":
    main()
