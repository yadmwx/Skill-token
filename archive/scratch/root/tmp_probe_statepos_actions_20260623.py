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


DIT_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
    "--image_aug--VLA-Adapter-DIT12-statecond-sqrtgroup-nozeroinit-proprioadd-statepos-object-1000-20260623--1000_chkpt"
)
FLOW_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0"
    "--image_aug--VLA-Adapter-FlowMLP-object-1000-20260618_231916--1000_chkpt"
)


def make_dit_cfg():
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
        dit_disable_inference_anchor=False,
        dit_detach_flow_conditioning=True,
        dit_use_state_conditioning=True,
        dit_state_scale_mode="sqrt_group",
        dit_state_proprio_mode="add",
        dit_state_use_chunk_pos=True,
        dit_zero_init_adaln=False,
        dit_zero_init_output=False,
    )
    validate_config(cfg)
    return cfg


def make_flow_cfg():
    cfg = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=FLOW_CKPT,
        task_suite_name="libero_object",
        task_ids="0",
        center_crop=True,
        num_trials_per_task=1,
        action_head_type="FlowMLP",
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
        flowmlp_num_inference_steps=5,
        flowmlp_num_inference_samples=8,
        flowmlp_anchor_blend=0.0,
    )
    validate_config(cfg)
    return cfg


def build_obs(raw_obs):
    observation, _ = prepare_observation(raw_obs, 224)
    observation["state"] = np.concatenate(
        (raw_obs["robot0_eef_pos"], quat2axisangle(raw_obs["robot0_eef_quat"]), raw_obs["robot0_gripper_qpos"])
    )
    return observation


def dump_model_actions(name, cfg, pack, observation, task_description):
    model, action_head, depth_interface, proprio_projector, noisy_action_projector, processor = pack
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
    arr = np.stack([np.asarray(a, dtype=np.float32) for a in actions], axis=0)
    exec_arr = np.stack([process_action(a.copy(), cfg.model_family) for a in arr], axis=0)
    print(f"MODEL {name}")
    print("RAW_ALL")
    print(np.array2string(arr, precision=4, separator=","))
    print("EXEC_ALL")
    print(np.array2string(exec_arr, precision=4, separator=","))
    print("RAW_STATS")
    print(
        f"mean={arr.mean():.6f} abs_mean={np.abs(arr).mean():.6f} "
        f"std={arr.std():.6f} min={arr.min():.6f} max={arr.max():.6f}"
    )
    print("GRIPPER_RAW", np.array2string(arr[:, -1], precision=4, separator=","))
    print("GRIPPER_EXEC", np.array2string(exec_arr[:, -1], precision=4, separator=","))
    print()


def main():
    set_seed_everywhere(7)
    cfg_dit = make_dit_cfg()
    cfg_flow = make_flow_cfg()
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

    dump_model_actions("DIT_STATEPOS", cfg_dit, dit_pack, observation, task_description)
    dump_model_actions("FLOWMLP", cfg_flow, flow_pack, observation, task_description)


if __name__ == "__main__":
    main()
