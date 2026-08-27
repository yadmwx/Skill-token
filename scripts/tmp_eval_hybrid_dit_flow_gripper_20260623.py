import numpy as np
import tqdm

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    TASK_MAX_STEPS,
    initialize_model,
    load_initial_states,
    prepare_observation,
    process_action,
    validate_config,
)
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere


DIT_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r32+dropout-0.0"
    "--image_aug--VLA-Adapter-DIT12-fullaction-adaptive-object-1000-20260623--2000_chkpt"
)
FLOW_CKPT = (
    "outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0"
    "--image_aug--VLA-Adapter-FlowMLP-object-1000-20260618_231916--1000_chkpt"
)


def make_cfg(ckpt: str, head: str, *, dit_samples: int = 1, flow_samples: int = 8):
    cfg = GenerateConfig(
        model_family="openvla",
        pretrained_checkpoint=ckpt,
        task_suite_name="libero_object",
        task_ids="0",
        center_crop=True,
        num_trials_per_task=3,
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
    )
    validate_config(cfg)
    return cfg


def hybrid_action(cfg_dit, dit_pack, cfg_flow, flow_pack, observation, task_description):
    dit_model, dit_head, dit_depth, dit_prop, dit_noisy, dit_proc = dit_pack
    flow_model, flow_head, flow_depth, flow_prop, flow_noisy, flow_proc = flow_pack

    dit_actions = get_action(
        cfg_dit,
        dit_model,
        observation,
        task_description,
        processor=dit_proc,
        action_head=dit_head,
        depth_interface=dit_depth,
        proprio_projector=dit_prop,
        noisy_action_projector=dit_noisy,
        use_film=cfg_dit.use_film,
        use_minivlm=cfg_dit.use_minivlm,
    )
    flow_actions = get_action(
        cfg_flow,
        flow_model,
        observation,
        task_description,
        processor=flow_proc,
        action_head=flow_head,
        depth_interface=flow_depth,
        proprio_projector=flow_prop,
        noisy_action_projector=flow_noisy,
        use_film=cfg_flow.use_film,
        use_minivlm=cfg_flow.use_minivlm,
    )

    merged = []
    for dit_a, flow_a in zip(dit_actions, flow_actions):
        dit_a = np.asarray(dit_a, dtype=np.float32)
        flow_a = np.asarray(flow_a, dtype=np.float32)
        hybrid = dit_a.copy()
        hybrid[-1] = flow_a[-1]
        merged.append(hybrid)
    return merged


def run_episode(cfg_dit, dit_pack, cfg_flow, flow_pack, env, task_description, resize_size, initial_state):
    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue = []
    max_steps = TASK_MAX_STEPS[cfg_dit.task_suite_name]
    t = 0

    while t < max_steps + cfg_dit.num_steps_wait:
        if t < cfg_dit.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action(cfg_dit.model_family))
            if done:
                return True
            t += 1
            continue

        observation, _ = prepare_observation(obs, resize_size)
        if not action_queue:
            action_queue = hybrid_action(cfg_dit, dit_pack, cfg_flow, flow_pack, observation, task_description)

        action = action_queue.pop(0)
        action = process_action(action, cfg_dit.model_family)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True
        t += 1

    return False


def main():
    set_seed_everywhere(7)
    cfg_dit = make_cfg(DIT_CKPT, "DIT", dit_samples=1)
    cfg_flow = make_cfg(FLOW_CKPT, "FlowMLP", flow_samples=8)

    dit_pack = initialize_model(cfg_dit)
    flow_pack = initialize_model(cfg_flow)
    resize_size = get_image_resize_size(cfg_dit)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg_dit.task_suite_name]()
    task = task_suite.get_task(0)
    initial_states, _ = load_initial_states(cfg_dit, task_suite, 0, log_file=None)
    env, task_description = get_libero_env(task, cfg_dit.model_family, resolution=cfg_dit.env_img_res)

    try:
        successes = 0
        for episode_idx in tqdm.tqdm(range(cfg_dit.num_trials_per_task)):
            success = run_episode(
                cfg_dit,
                dit_pack,
                cfg_flow,
                flow_pack,
                env,
                task_description,
                resize_size,
                initial_states[episode_idx],
            )
            print(f"EPISODE {episode_idx + 1} success={success}")
            successes += int(success)
        print(f"HYBRID_SUCCESS {successes}/{cfg_dit.num_trials_per_task}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
