import copy
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
    process_action,
    validate_config,
)
from experiments.robot.robot_utils import get_action, set_seed_everywhere


FULLACTION_DIT_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
    "--image_aug--VLA-Adapter-DIT12-fullaction-adaptive-object-1000-20260623--2000_chkpt"
)
FLOW_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0"
    "--image_aug--VLA-Adapter-FlowMLP-object-1000-20260618_231916--1000_chkpt"
)


def make_cfg(ckpt: str, head: str, *, inference_samples: int):
    cfg = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=ckpt,
        task_suite_name="libero_object",
        center_crop=True,
        num_trials_per_task=1,
        action_head_type=head,
        use_adaptive_bridge=True,
        bridge_mode="adaptive",
        dit_num_blocks=12,
        dit_num_inference_steps=20,
        dit_num_inference_samples=inference_samples,
        dit_anchor_blend=0.0,
        dit_anchor_blend_was_set=True,
        dit_disable_inference_anchor=False,
        flowmlp_num_inference_steps=5,
        flowmlp_num_inference_samples=inference_samples,
        flowmlp_anchor_blend=0.0,
        use_depth_interface=False,
        depth_interface_mode="none",
        unnorm_key="libero_object_no_noops",
        seed=7,
        use_wandb=False,
    )
    validate_config(cfg)
    return cfg


def build_obs(raw_obs):
    obs = dict(raw_obs)
    obs["full_image"] = get_libero_image(obs)
    obs["wrist_image"] = get_libero_wrist_image(obs)
    obs["state"] = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )
    return obs


def collect_first_action(name, cfg, pack, obs, task_description):
    model, action_head, depth_interface, proprio_projector, noisy_action_projector, processor = pack

    normal_obs = copy.deepcopy(obs)
    normal_actions = get_action(
        cfg,
        model,
        normal_obs,
        task_description,
        processor=processor,
        action_head=action_head,
        depth_interface=depth_interface,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        use_film=cfg.use_film,
        use_minivlm=cfg.use_minivlm,
    )
    normal_first = np.asarray(normal_actions[0], dtype=np.float32)
    exec_first = process_action(normal_first.copy(), cfg.model_family)

    original_unnorm = model._unnormalize_actions
    model._unnormalize_actions = lambda normalized_actions, unnorm_key=None: normalized_actions
    try:
        raw_obs = copy.deepcopy(obs)
        raw_actions = get_action(
            cfg,
            model,
            raw_obs,
            task_description,
            processor=processor,
            action_head=action_head,
            depth_interface=depth_interface,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
            use_minivlm=cfg.use_minivlm,
        )
        raw_first = np.asarray(raw_actions[0], dtype=np.float32)
    finally:
        model._unnormalize_actions = original_unnorm

    print(f"MODEL {name}")
    print("RAW_FIRST   ", np.array2string(raw_first, precision=4, separator=","))
    print("UNNORM_FIRST", np.array2string(normal_first, precision=4, separator=","))
    print("EXEC_FIRST  ", np.array2string(exec_first, precision=4, separator=","))
    print()


def main():
    set_seed_everywhere(7)
    configs = {
        "DIT_FULLACTION_S8": make_cfg(FULLACTION_DIT_CKPT, "DIT", inference_samples=8),
        "DIT_FULLACTION_S1": make_cfg(FULLACTION_DIT_CKPT, "DIT", inference_samples=1),
        "FLOW_BASELINE_S8": make_cfg(FLOW_CKPT, "FlowMLP", inference_samples=8),
    }
    loaded = {name: initialize_model(cfg) for name, cfg in configs.items()}

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_object"]()
    task = task_suite.get_task(0)
    task_description = task.language
    initial_states, _ = load_initial_states(configs["DIT_FULLACTION_S8"], task_suite, 0, log_file=None)
    env, _ = get_libero_env(task, configs["DIT_FULLACTION_S8"].model_family, resolution=configs["DIT_FULLACTION_S8"].env_img_res)
    try:
        env.reset()
        obs = env.set_init_state(initial_states[0])
        for _ in range(configs["DIT_FULLACTION_S8"].num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action(configs["DIT_FULLACTION_S8"].model_family))
        obs = build_obs(obs)
    finally:
        env.close()

    for name, cfg in configs.items():
        collect_first_action(name, cfg, loaded[name], obs, task_description)


if __name__ == "__main__":
    main()
