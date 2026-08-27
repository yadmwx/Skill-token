"""Flow-GRPO fine-tuning script for LIBERO with diffusion action head.

This script assumes you have already fine-tuned / loaded a diffusion-enabled
VLA-Adapter checkpoint (LoRA on the VLM, diffusion action head weights). It
freezes the VLM backbone and optimises only the diffusion action head using the
Flow-GRPO trainer we implemented earlier.
"""


import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    DATE_TIME,
    get_libero_env,
)
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    TaskSuite,
    TASK_MAX_STEPS,
    initialize_model,
    validate_config,
)
from experiments.robot.openvla_utils import (
    normalize_proprio,
    prepare_images_for_vla,
)
from experiments.robot.rl.flow_grpo_trainer import (
    FlowGRPOTrainer,
    FlowGRPOTrainerConfig,
    FlowGRPORolloutBatch,
)
from prismatic.models.action_heads import DiffusionActionHead, FlowGRPOSample
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


@dataclass
class FlowGRPOTrainConfig:
    """High-level configuration for Flow-GRPO fine-tuning."""

    pretrained_checkpoint: str
    output_dir: str = "outputs/flow_grpo"
    task_suite: str = "libero_spatial"
    num_iterations: int = 500
    learning_rate: float = 5e-5
    clip_epsilon: float = 0.2
    entropy_coef: float = 1e-3
    normalize_advantage: bool = True
    reward_baseline: str = "mean"
    num_diffusion_steps: int = 50
    use_pro_version: bool = True
    device: str = "cuda"
    seed: int = 42
    log_interval: int = 10
    save_interval: int = 100
    max_env_steps: Optional[int] = None
    train_proprio_projector: bool = False
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_generate_config(cfg: FlowGRPOTrainConfig) -> GenerateConfig:
    gen_cfg = GenerateConfig()
    gen_cfg.model_family = "openvla"
    gen_cfg.pretrained_checkpoint = cfg.pretrained_checkpoint
    gen_cfg.use_l1_regression = False
    gen_cfg.use_diffusion = True
    gen_cfg.num_diffusion_steps = cfg.num_diffusion_steps
    gen_cfg.use_minivlm = True
    gen_cfg.use_proprio = True
    gen_cfg.task_suite_name = TaskSuite(cfg.task_suite).value
    gen_cfg.use_pro_version = cfg.use_pro_version
    gen_cfg.phase = "RL-Training"
    gen_cfg.num_open_loop_steps = NUM_ACTIONS_CHUNK
    gen_cfg.skip_action_head_loading = True
    gen_cfg.allow_partial_action_head_load = True
    validate_config(gen_cfg)
    return gen_cfg


def prepare_policy_inputs(
    cfg: GenerateConfig,
    model: torch.nn.Module,
    processor,
    obs: Dict,
    task_label: str,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Optional[np.ndarray]]:
    """Prepare transformer inputs and normalized proprioception."""

    primary_key_candidates = ["full_image", "agentview_image", "rgb_static"]
    primary_image = None
    for key in primary_key_candidates:
        if key in obs:
            primary_image = obs[key]
            break
    if primary_image is None:
        raise KeyError(
            f"Observation does not contain any of the expected image keys: {primary_key_candidates}. "
            f"Available keys: {list(obs.keys())}"
        )

    images: List[np.ndarray] = [primary_image]
    if cfg.num_images_in_input > 1:
        wrist_candidates = ["wrist", "eye_in_hand", "hand_image", "handcamera", "robot0_eye"]
        wrist_keys = [
            key
            for key in obs.keys()
            if key != primary_key_candidates[0]
            and any(token in key.lower() for token in wrist_candidates)
        ]
        for key in wrist_keys:
            images.append(obs[key])

        if len(images) < cfg.num_images_in_input:
            # replicate primary image to match expected count
            while len(images) < cfg.num_images_in_input:
                images.append(primary_image)

    processed_images = prepare_images_for_vla(images, cfg)
    primary_image = processed_images.pop(0)

    if cfg.use_minivlm:
        prompt = (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\nWhat action should the robot take to {task_label.lower()}?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    inputs = processor(prompt, primary_image)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(device=device, dtype=torch.bfloat16)

    if processed_images:
        wrist_tensors = [
            processor(prompt, wrist_image)["pixel_values"].to(device=device, dtype=torch.bfloat16)
            for wrist_image in processed_images
        ]
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"]] + wrist_tensors, dim=1)

    proprio = None
    if cfg.use_proprio:
        proprio_keys = ["state", "proprio", "joint_pos", "robot0_proprio-state"]
        proprio_raw = None
        for key in proprio_keys:
            if key in obs:
                proprio_raw = obs[key]
                break
        if proprio_raw is not None:
            proprio_raw = np.asarray(proprio_raw)
            if proprio_raw.shape[-1] != PROPRIO_DIM:
                alt_components: List[np.ndarray] = []
                if "robot0_eef_pos" in obs:
                    alt_components.append(np.asarray(obs["robot0_eef_pos"]).reshape(-1))
                if "robot0_eef_quat" in obs:
                    alt_components.append(np.asarray(obs["robot0_eef_quat"]).reshape(-1))
                if "robot0_gripper_qpos" in obs:
                    gripper = np.asarray(obs["robot0_gripper_qpos"]).reshape(-1)
                    alt_components.append(gripper[:1])
                if alt_components and sum(comp.shape[0] for comp in alt_components) == PROPRIO_DIM:
                    proprio_raw = np.concatenate(alt_components, axis=0)
                else:
                    if not getattr(prepare_policy_inputs, "_proprio_shape_warned", False):
                        print(
                            f"Warning: proprio dimension mismatch. Expected {PROPRIO_DIM}, "
                            f"got {proprio_raw.shape[-1]}. Using zeros."
                        )
                        prepare_policy_inputs._proprio_shape_warned = True
                    proprio_raw = np.zeros(PROPRIO_DIM, dtype=np.float32)

            stats = model.norm_stats[cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(proprio_raw, stats)
        else:
            if not getattr(prepare_policy_inputs, "_proprio_missing_warned", False):
                print(
                    f"Warning: observation missing proprio keys {proprio_keys}; "
                    "proprio input will be zero-initialized."
                )
                prepare_policy_inputs._proprio_missing_warned = True
            proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)

    return inputs, proprio


def to_tensor(data: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(data, dtype=dtype, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def save_action_head(action_head: DiffusionActionHead, output_dir: Path, step: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"action_head_step{step:06d}.pt"
    torch.save({"state_dict": action_head.state_dict(), "step": step, "timestamp": DATE_TIME}, save_path)
    print(f"[Flow-GRPO] Saved action head checkpoint to {save_path}")


def run_training(cfg: FlowGRPOTrainConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    gen_cfg = build_generate_config(cfg)
    model, action_head, proprio_projector, _, processor = initialize_model(gen_cfg)

    model = model.to(device)
    action_head = action_head.to(device)
    proprio_projector = proprio_projector.to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if not cfg.train_proprio_projector:
        proprio_projector.eval()
        for param in proprio_projector.parameters():
            param.requires_grad = False

    action_head.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, action_head.parameters()), lr=cfg.learning_rate
    )

    trainer = FlowGRPOTrainer(
        action_head=action_head,
        proprio_projector=proprio_projector,
        optimizer=optimizer,
        config=FlowGRPOTrainerConfig(
            clip_epsilon=cfg.clip_epsilon,
            normalize_advantage=cfg.normalize_advantage,
            entropy_coef=cfg.entropy_coef,
            reward_baseline=cfg.reward_baseline,
            max_grad_norm=0.5,
        ),
        device=device,
    )

    suite_cls = benchmark.get_benchmark(TaskSuite(cfg.task_suite).value)
    suite = suite_cls()
    tasks = suite.tasks

    max_env_steps = cfg.max_env_steps or TASK_MAX_STEPS[TaskSuite(cfg.task_suite)]
    output_dir = Path(cfg.output_dir)

    use_wandb = cfg.use_wandb and cfg.wandb_entity and cfg.wandb_project
    if use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or f"flow-grpo-{cfg.task_suite}-{DATE_TIME}",
            config=vars(cfg),
        )

    episode_counter = 0
    for iteration in range(1, cfg.num_iterations + 1):
        task = tasks[iteration % len(tasks)]
        env, task_desc = get_libero_env(task, model_family=gen_cfg.model_family, resolution=gen_cfg.env_img_res)
        obs = env.reset()
        if iteration == 1:
            print("[Flow-GRPO][DEBUG] observation keys:", list(obs.keys()))
            print(
                "[Flow-GRPO][DEBUG] proprio-like keys present:",
                [k for k in obs.keys() if any(token in k.lower() for token in ["state", "proprio", "joint"])],
            )

        done = False
        step_count = 0
        episode_return = 0.0

        while not done and step_count < max_env_steps:
            policy_inputs, proprio_norm = prepare_policy_inputs(
                gen_cfg, model, processor, obs, task_desc, device
            )

            actions_array, actions_hidden_states = model.predict_action(
                **policy_inputs,
                unnorm_key=gen_cfg.unnorm_key,
                do_sample=False,
                proprio=proprio_norm,
                proprio_projector=proprio_projector,
                action_head=action_head,
                use_film=gen_cfg.use_film,
            )

            actions_hidden_states = actions_hidden_states.detach()
            batch_size = actions_hidden_states.shape[0]
            if proprio_norm is not None:
                proprio_tensor = to_tensor(
                    proprio_norm, device=device, dtype=actions_hidden_states.dtype
                )
                if proprio_tensor.shape[0] != batch_size:
                    proprio_tensor = proprio_tensor.expand(batch_size, -1)
            else:
                proprio_tensor = torch.zeros(
                    (batch_size, PROPRIO_DIM),
                    device=device,
                    dtype=actions_hidden_states.dtype,
                )

            flow_sample: FlowGRPOSample = action_head.sample_with_logprob(
                actions_hidden_states=actions_hidden_states,
                proprio=proprio_tensor,
                proprio_projector=proprio_projector,
            )

            chunk_rewards: List[float] = []
            chunk_success_flags: List[float] = []
            for chunk_idx in range(NUM_ACTIONS_CHUNK):
                action_tensor = flow_sample.actions[chunk_idx]
                action_np = (
                    action_tensor.detach()
                    .to(dtype=torch.float32)
                    .cpu()
                    .numpy()
                    .tolist()
                )
                env_step_result = env.step(action_np)
                if len(env_step_result) == 5:
                    obs, reward, terminated, truncated, info = env_step_result
                else:
                    obs, reward, done_flag, info = env_step_result
                    terminated = done_flag
                    truncated = False
                chunk_rewards.append(reward)
                success_flag = float(info.get("success", False)) if isinstance(info, dict) else 0.0
                chunk_success_flags.append(success_flag)
                episode_return += reward
                step_count += 1
                if terminated or truncated or step_count >= max_env_steps:
                    done = True
                    break

            chunk_success_rate = (
                float(sum(chunk_success_flags)) / len(chunk_success_flags)
                if chunk_success_flags
                else 0.0
            )
            episode_success = float(any(chunk_success_flags))

            rewards_tensor = torch.tensor(
                [episode_success], device=device, dtype=torch.float32
            )

            batch = FlowGRPORolloutBatch(
                actions_hidden_states=actions_hidden_states.detach(),
                proprio=proprio_tensor.detach(),
                rewards=rewards_tensor,
                flow_sample=flow_sample,
            )

            metrics = trainer.train_step(batch)

            if iteration % cfg.log_interval == 0:
                print(
                    f"[Iter {iteration}] episode_success={episode_success:.3f} "
                    f"chunk_success_rate={chunk_success_rate:.3f} "
                    f"reward_sum={sum(chunk_rewards):.3f} "
                    f"loss={metrics['loss']:.4f} "
                    f"kl={metrics['approx_kl']:.4f}"
                )
            if use_wandb:
                wandb.log(
                    {
                        "iteration": iteration,
                        "episode_success": episode_success,
                        "chunk_success_rate": chunk_success_rate,
                        "reward_sum": sum(chunk_rewards),
                        "loss": metrics["loss"],
                        "policy_loss": metrics["policy_loss"],
                        "entropy": metrics["entropy"],
                        "approx_kl": metrics["approx_kl"],
                    },
                    step=iteration,
                )

        env.close()
        episode_counter += 1

        if iteration % cfg.save_interval == 0:
            save_action_head(action_head, output_dir, iteration)

    save_action_head(action_head, output_dir, cfg.num_iterations)
    print("[Flow-GRPO] Training complete.")
    if use_wandb and wandb.run is not None:
        wandb.finish()


def parse_args() -> FlowGRPOTrainConfig:
    parser = argparse.ArgumentParser(description="Flow-GRPO fine-tuning for diffusion action head")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True, help="Path to diffusion checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/flow_grpo")
    parser.add_argument("--task_suite", type=str, default="libero_spatial", choices=[s.value for s in TaskSuite])
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=1e-3)
    parser.add_argument("--normalize_advantage", action="store_true")
    parser.add_argument("--no-normalize_advantage", dest="normalize_advantage", action="store_false")
    parser.add_argument("--reward_baseline", type=str, default="mean", choices=["mean", "none"])
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--use_pro_version", action="store_true")
    parser.add_argument("--no-use_pro_version", dest="use_pro_version", action="store_false")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_env_steps", type=int, default=None)
    parser.add_argument("--train_proprio_projector", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    parser.set_defaults(normalize_advantage=True, use_pro_version=True)
    args = parser.parse_args()

    return FlowGRPOTrainConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        output_dir=args.output_dir,
        task_suite=args.task_suite,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        normalize_advantage=args.normalize_advantage,
        reward_baseline=args.reward_baseline,
        num_diffusion_steps=args.num_diffusion_steps,
        use_pro_version=args.use_pro_version,
        device=args.device,
        seed=args.seed,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        max_env_steps=args.max_env_steps,
        train_proprio_projector=args.train_proprio_projector,
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


def main() -> None:
    cfg = parse_args()
    run_training(cfg)


if __name__ == "__main__":
    main()

