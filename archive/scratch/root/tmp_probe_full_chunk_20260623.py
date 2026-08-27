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
    dit_samples: int = 1,
    flow_samples: int = 8,
    debug_group_dit_tokens: bool = False,
):
    cfg = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=ckpt,
        task_suite_name="libero_object",
        task_ids="0",
        center_crop=True,
        num_trials_per_task=1,
        action_head_type=head,
        use_adaptive_bridge=True,
        bridge_mode="adaptive",
        dit_num_blocks=12,
        dit_num_inference_steps=20,
        dit_num_inference_samples=dit_samples,
        dit_anchor_blend=0.0,
        dit_anchor_blend_was_set=True,
        dit_disable_inference_anchor=False,
        flowmlp_num_inference_steps=5,
        flowmlp_num_inference_samples=flow_samples,
        flowmlp_anchor_blend=0.0,
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
        debug_dit_group_action_tokens_to_chunk=debug_group_dit_tokens,
    )
    validate_config(cfg)
    return cfg


def build_obs(raw_obs):
    observation, _ = prepare_observation(raw_obs, 224)
    # prepare_observation already builds the policy state; keep an extra sanity print-friendly raw state shape
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
    print()


def main():
    set_seed_everywhere(7)
    cfg_dit = make_cfg(DIT_CKPT, "DIT", dit_samples=1)
    cfg_dit_grouped = make_cfg(DIT_CKPT, "DIT", dit_samples=1, debug_group_dit_tokens=True)
    cfg_flow = make_cfg(FLOW_CKPT, "FlowMLP", flow_samples=8)
    dit_pack = initialize_model(cfg_dit)
    dit_grouped_pack = initialize_model(cfg_dit_grouped)
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

    dump_model_actions("DIT_S1", cfg_dit, dit_pack, observation, task_description)
    dump_model_actions("DIT_S1_GROUPED", cfg_dit_grouped, dit_grouped_pack, observation, task_description)
    dump_model_actions("FLOW_S8", cfg_flow, flow_pack, observation, task_description)


if __name__ == "__main__":
    main()
