import copy
import os

import numpy as np

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
    process_action,
    validate_config,
)
from experiments.robot.robot_utils import get_action, set_seed_everywhere


CKPTS = {
    "statepos": (
        "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
        "--image_aug--VLA-Adapter-DIT12-statecond-sqrtgroup-nozeroinit-proprioadd-statepos-object-1000-20260623--1000_chkpt"
    ),
    "fullaction_adaptive": (
        "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
        "--image_aug--VLA-Adapter-DIT12-fullaction-adaptive-object-1000-20260623--1000_chkpt"
    ),
    "fresh_anchor_gated": (
        "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
        "--image_aug--VLA-Adapter-DIT12-residual-anchor-object-1000-aggfix-hfpathfix-aqsave-20260622--1000_chkpt"
    ),
    "pure_gripfix": (
        "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
        "--image_aug--VLA-Adapter-DIT12-fullaction-pure-gripfix-nozeroinit-object-1000-20260623--1000_chkpt"
    ),
}


def make_cfg(kind):
    ckpt = CKPTS[kind]
    common = dict(
        model_family="openvla",
        pretrained_checkpoint=ckpt,
        task_suite_name="libero_object",
        task_ids="0",
        center_crop=True,
        num_trials_per_task=1,
        action_head_type="DIT",
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
        dit_num_blocks=12,
        dit_num_inference_steps=20,
        dit_num_inference_samples=1,
        dit_anchor_blend=0.0,
        dit_anchor_blend_was_set=True,
        dit_detach_flow_conditioning=True,
    )
    if kind == "statepos":
        common.update(
            use_adaptive_bridge=True,
            bridge_mode="adaptive",
            dit_supervised_anchor_weight=0.0,
            dit_disable_inference_anchor=False,
            dit_use_state_conditioning=True,
            dit_state_scale_mode="sqrt_group",
            dit_state_proprio_mode="add",
            dit_state_use_chunk_pos=True,
            dit_zero_init_adaln=False,
            dit_zero_init_output=False,
        )
    elif kind == "fullaction_adaptive":
        common.update(
            use_adaptive_bridge=True,
            bridge_mode="adaptive",
            dit_supervised_anchor_weight=0.0,
            dit_disable_inference_anchor=False,
            dit_use_state_conditioning=False,
            dit_state_scale_mode="none",
            dit_zero_init_adaln=True,
            dit_zero_init_output=True,
        )
    elif kind == "fresh_anchor_gated":
        common.update(
            use_adaptive_bridge=True,
            bridge_mode="adaptive_gated",
            dit_supervised_anchor_weight=1.0,
            dit_anchor_gripper_weight=1.0,
            dit_anchor_gripper_bce_weight=0.2,
            dit_disable_inference_anchor=False,
            dit_use_state_conditioning=False,
            dit_state_scale_mode="none",
            dit_zero_init_adaln=True,
            dit_zero_init_output=True,
        )
    elif kind == "pure_gripfix":
        common.update(
            use_adaptive_bridge=True,
            bridge_mode="adaptive",
            dit_supervised_anchor_weight=0.0,
            dit_anchor_gripper_weight=1.0,
            dit_anchor_gripper_bce_weight=0.0,
            dit_disable_inference_anchor=True,
            dit_pure_inference=True,
            dit_use_state_conditioning=False,
            dit_state_scale_mode="none",
            dit_zero_init_adaln=False,
            dit_zero_init_output=False,
        )
    else:
        raise ValueError(f"Unknown probe kind: {kind}")

    cfg = GenerateConfig(**common)
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


def set_pure(action_head, enabled):
    if hasattr(action_head, "pure_inference"):
        action_head.pure_inference = enabled
    if hasattr(action_head, "disable_inference_anchor"):
        action_head.disable_inference_anchor = enabled
    if hasattr(action_head, "velocity_network") and hasattr(action_head.velocity_network, "pure_inference"):
        action_head.velocity_network.pure_inference = enabled


def collect_actions(name, cfg, pack, observation, task_description, pure):
    model, action_head, depth_interface, proprio_projector, noisy_action_projector, processor = pack
    set_seed_everywhere(123)
    set_pure(action_head, pure)
    cfg = copy.copy(cfg)
    cfg.dit_pure_inference = pure
    cfg.dit_disable_inference_anchor = bool(getattr(cfg, "dit_disable_inference_anchor", False) or pure)
    actions = get_action(
        cfg,
        model,
        observation,
        task_description,
        processor=processor,
        action_head=action_head,
        depth_interface=depth_interface,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        use_film=cfg.use_film,
        use_minivlm=cfg.use_minivlm,
    )
    raw = np.stack([np.asarray(a, dtype=np.float32) for a in actions], axis=0)
    exec_actions = np.stack([process_action(a.copy(), cfg.model_family) for a in raw], axis=0)
    label = f"{name}_{'PURE' if pure else 'NORMAL'}"
    print(f"MODEL {label}")
    print("RAW_ALL")
    print(np.array2string(raw, precision=4, separator=","))
    print("EXEC_ALL")
    print(np.array2string(exec_actions, precision=4, separator=","))
    print(
        "RAW_STATS "
        f"mean={raw.mean():.6f} abs_mean={np.abs(raw).mean():.6f} "
        f"std={raw.std():.6f} min={raw.min():.6f} max={raw.max():.6f}"
    )
    print("RAW_FIRST6_ABS_MEAN", np.array2string(np.abs(raw[:, :6]).mean(axis=0), precision=4, separator=","))
    print("GRIPPER_RAW", np.array2string(raw[:, -1], precision=4, separator=","))
    print("GRIPPER_EXEC", np.array2string(exec_actions[:, -1], precision=4, separator=","))
    print()
    return raw


def main():
    kind = os.environ.get("DIT_PROBE_KIND", "statepos")
    set_seed_everywhere(7)
    cfg = make_cfg(kind)
    pack = initialize_model(cfg)

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

    normal = collect_actions(kind, cfg, pack, observation, task_description, pure=False)
    pure = collect_actions(kind, cfg, pack, observation, task_description, pure=True)
    diff = pure - normal
    print("PURE_MINUS_NORMAL")
    print(np.array2string(diff, precision=4, separator=","))
    print(
        "DIFF_STATS "
        f"mean={diff.mean():.6f} abs_mean={np.abs(diff).mean():.6f} "
        f"std={diff.std():.6f} min={diff.min():.6f} max={diff.max():.6f}"
    )


if __name__ == "__main__":
    main()
