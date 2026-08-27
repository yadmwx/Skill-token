import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.robot.openvla_utils import (
    get_action_head,
    get_depth_interface,
    get_processor,
    get_proprio_projector,
    get_vla,
)
from experiments.robot.robot_utils import set_seed_everywhere
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, STOP_INDEX
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def make_cfg(args):
    action_type = args.action_head_type.upper()
    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.checkpoint,
        task_suite_name=args.task_suite,
        center_crop=True,
        action_head_type=args.action_head_type,
        num_images_in_input=args.num_images_in_input,
        use_proprio=args.use_proprio,
        use_minivlm=args.use_minivlm,
        use_film=False,
        unnorm_key=args.unnorm_key,
        seed=args.seed,
        use_wandb=False,
        use_adaptive_bridge=args.use_adaptive_bridge,
        bridge_mode=args.bridge_mode,
        fixed_layer_index=args.fixed_layer_index,
        use_depth_interface=args.use_depth_interface,
        depth_interface_mode=args.depth_interface_mode,
        depth_interface_max_layers=args.depth_interface_max_layers,
        depth_interface_add_proprio=args.depth_interface_add_proprio,
        flowmlp_num_inference_steps=args.flowmlp_num_inference_steps,
        flowmlp_num_inference_samples=args.flowmlp_num_inference_samples,
        flowmlp_supervised_anchor_weight=args.flowmlp_supervised_anchor_weight,
        flowmlp_anchor_blend=args.flowmlp_anchor_blend,
        flowmlp_anchor_gripper_weight=args.flowmlp_anchor_gripper_weight,
        flowmlp_anchor_gripper_bce_weight=args.flowmlp_anchor_gripper_bce_weight,
        flowmlp_detach_flow_conditioning=args.flowmlp_detach_flow_conditioning,
        dit_num_blocks=args.dit_num_blocks,
        dit_num_inference_steps=args.dit_num_inference_steps,
        dit_num_inference_samples=args.dit_num_inference_samples,
        dit_supervised_anchor_weight=args.dit_supervised_anchor_weight,
        dit_anchor_blend=args.dit_anchor_blend,
        dit_anchor_blend_was_set=True,
        dit_anchor_gripper_weight=args.dit_anchor_gripper_weight,
        dit_anchor_gripper_bce_weight=args.dit_anchor_gripper_bce_weight,
        dit_flow_xyz_loss_weight=args.dit_flow_xyz_loss_weight,
        dit_flow_rot_loss_weight=args.dit_flow_rot_loss_weight,
        dit_flow_gripper_loss_weight=args.dit_flow_gripper_loss_weight,
        dit_flow_gripper_bce_weight=args.dit_flow_gripper_bce_weight,
        dit_flow_gripper_bce_logit_scale=args.dit_flow_gripper_bce_logit_scale,
        dit_flow_gripper_bce_balanced=args.dit_flow_gripper_bce_balanced,
        dit_gripper_head_weight=args.dit_gripper_head_weight,
        dit_gripper_head_override=args.dit_gripper_head_override,
        dit_clip_normalized_actions=args.dit_clip_normalized_actions,
        dit_detach_flow_conditioning=args.dit_detach_flow_conditioning,
        dit_disable_inference_anchor=args.dit_disable_inference_anchor,
        dit_pure_inference=args.dit_pure_inference,
        dit_zero_init_adaln=args.dit_zero_init_adaln,
        dit_zero_init_output=args.dit_zero_init_output,
        dit_use_state_conditioning=args.dit_use_state_conditioning,
        dit_state_scale_mode=args.dit_state_scale_mode,
        dit_state_proprio_mode=args.dit_state_proprio_mode,
        dit_state_use_chunk_pos=args.dit_state_use_chunk_pos,
        dit_state_include_task_tokens=args.dit_state_include_task_tokens,
        dit_condition_mode=args.dit_condition_mode,
        dit_condition_injection_mode=args.dit_condition_injection_mode,
        dit_include_prompt_tokens=args.dit_include_prompt_tokens,
        dit_task_token_mode=args.dit_task_token_mode,
        debug_dit_group_action_tokens_to_chunk=args.debug_dit_group_action_tokens_to_chunk,
        save_version="vla-adapter",
        phase="Inference",
        use_pro_version=True,
        load_in_8bit=False,
        load_in_4bit=False,
        use_diffusion=False,
        use_l1_regression=(action_type == "MLP"),
        use_flow_matching=(action_type in {"DIT", "FLOWMLP"}),
        flow_matching_head_type=("ditx" if action_type == "DIT" else "mlp"),
    )
    return cfg


def initialize_probe_model(cfg):
    model = get_vla(cfg)
    model.set_version(cfg.save_version)
    processor = get_processor(cfg)
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion or cfg.use_flow_matching:
        action_head = get_action_head(cfg, model.llm_dim, model=model)
    depth_interface = get_depth_interface(cfg, model.llm_dim, model=model)
    return model, action_head, depth_interface, proprio_projector, processor


def get_num_patches(model):
    target = getattr(model, "module", model)
    return int(target.vision_backbone.get_num_patches() * target.vision_backbone.get_num_images_in_input())


def module_or_self(module):
    return getattr(module, "module", module)


@torch.no_grad()
def predict_batch(cfg, model, action_head, depth_interface, proprio_projector, batch):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
    labels = batch["labels"].to(device)
    target_actions = batch["actions"].to(device, dtype=torch.float32)

    proprio = None
    if cfg.use_proprio:
        proprio = batch["proprio"].to(device, dtype=torch.bfloat16)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
            output_projector_features=True,
            proprio=proprio if cfg.use_proprio else None,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=cfg.use_film,
        )

    num_patches = get_num_patches(model)
    if getattr(output, "projector_features", None) is not None:
        num_patches = int(output.projector_features.shape[1])

    gt_token_ids = labels[:, 1:]
    current_action_mask = get_current_action_mask(gt_token_ids)
    next_actions_mask = get_next_actions_mask(gt_token_ids)
    all_actions_mask = current_action_mask | next_actions_mask

    action_indices_list = [torch.where(all_actions_mask[b])[0] for b in range(all_actions_mask.shape[0])]
    expected = ACTION_DIM * NUM_ACTIONS_CHUNK
    if any(idx.numel() != expected for idx in action_indices_list):
        got = [int(idx.numel()) for idx in action_indices_list]
        raise RuntimeError(f"Expected {expected} action tokens per sample, got {got}")

    action_idx = torch.stack(action_indices_list, dim=0)
    mm_pos = action_idx + 1 + num_patches

    include_prompt_tokens = cfg.action_head_type.upper() == "DIT" and bool(cfg.dit_include_prompt_tokens)
    prompt_mm_pos = None
    prompt_valid = None
    prompt_lengths = None
    task_token_mode = str(getattr(cfg, "dit_task_token_mode", "vision_prompt"))
    if task_token_mode not in {"vision_prompt", "vision_only", "prompt_only", "last_prompt"}:
        raise ValueError(f"Unsupported dit_task_token_mode: {task_token_mode}")
    needs_prompt_tokens = include_prompt_tokens or task_token_mode in {"prompt_only", "last_prompt"}
    if needs_prompt_tokens:
        prompt_candidate_mask = (
            (~all_actions_mask)
            & (gt_token_ids != STOP_INDEX)
            & attention_mask[:, 1:].bool()
        )
        prompt_indices_list = [torch.where(prompt_candidate_mask[b])[0] for b in range(prompt_candidate_mask.shape[0])]
        prompt_lengths = torch.tensor(
            [int(idx.numel()) for idx in prompt_indices_list],
            device=gt_token_ids.device,
            dtype=torch.long,
        )
        prompt_count = max(int(idx.numel()) for idx in prompt_indices_list)
        if prompt_count <= 0:
            raise RuntimeError("No valid prompt tokens found for DIT prompt conditioning.")
        prompt_idx_rows = []
        prompt_valid_rows = []
        for idx in prompt_indices_list:
            num_valid = int(idx.numel())
            if num_valid < prompt_count:
                pad = idx.new_zeros(prompt_count - num_valid)
                prompt_idx_rows.append(torch.cat([idx, pad], dim=0))
                prompt_valid_rows.append(
                    torch.cat(
                        [
                            torch.ones(num_valid, device=idx.device, dtype=torch.bool),
                            torch.zeros(prompt_count - num_valid, device=idx.device, dtype=torch.bool),
                        ],
                        dim=0,
                    )
                )
            else:
                prompt_idx_rows.append(idx)
                prompt_valid_rows.append(torch.ones(prompt_count, device=idx.device, dtype=torch.bool))
        prompt_idx = torch.stack(prompt_idx_rows, dim=0)
        prompt_mm_pos = prompt_idx + 1 + num_patches
        prompt_valid = torch.stack(prompt_valid_rows, dim=0)

    packed_layers = []
    task_token_count = num_patches
    for layer_h in output.hidden_states:
        batch_size, _, hidden_dim = layer_h.shape
        vision_latents = layer_h[:, 1 : 1 + num_patches, :]
        task_latents = vision_latents
        if needs_prompt_tokens:
            prompt_gather_index = prompt_mm_pos.unsqueeze(-1).expand(batch_size, prompt_mm_pos.shape[1], hidden_dim)
            prompt_latents = layer_h.gather(dim=1, index=prompt_gather_index)
            prompt_latents = prompt_latents * prompt_valid.unsqueeze(-1).to(dtype=prompt_latents.dtype)
        if task_token_mode == "vision_only":
            task_latents = vision_latents
        elif task_token_mode == "prompt_only":
            task_latents = prompt_latents
        elif task_token_mode == "last_prompt":
            last_prompt_idx = (prompt_lengths - 1).view(batch_size, 1, 1).expand(batch_size, 1, hidden_dim)
            task_latents = prompt_latents.gather(dim=1, index=last_prompt_idx)
        elif include_prompt_tokens:
            task_latents = torch.cat([task_latents, prompt_latents], dim=1)
        task_token_count = int(task_latents.shape[1])

        gather_index = mm_pos.unsqueeze(-1).expand(batch_size, expected, hidden_dim)
        action_latents = layer_h.gather(dim=1, index=gather_index)
        packed_layers.append(torch.cat([task_latents.unsqueeze(1), action_latents.unsqueeze(1)], dim=2).to(torch.bfloat16))

    hidden_states = torch.cat(packed_layers, dim=1)
    action_head_obj = module_or_self(action_head)
    if hasattr(action_head_obj, "task_token_mode"):
        action_head_obj.task_token_mode = task_token_mode
    if (needs_prompt_tokens or task_token_mode != "vision_prompt") and hasattr(action_head_obj, "set_num_task_tokens"):
        action_head_obj.set_num_task_tokens(task_token_count)
    layer_selector_info = {}
    velocity_network = getattr(action_head_obj, "velocity_network", None)
    selector = getattr(velocity_network, "task_layer_selector", None)
    if selector is not None:
        task_states = hidden_states[:, :, :task_token_count, :]
        task_mask = task_states.float().abs().sum(dim=-1) > 0
        _, unmasked_weights = selector(task_states)
        _, masked_weights = selector(task_states, task_mask=task_mask)
        weight_delta = (masked_weights.float() - unmasked_weights.float()).abs()
        layer_selector_info = {
            "task_padding_frac": float((~task_mask).float().mean().detach().cpu()),
            "layer_weight_delta_mean": float(weight_delta.mean().detach().cpu()),
            "layer_weight_delta_max": float(weight_delta.max().detach().cpu()),
        }
    split_info = {
        "include_prompt_tokens": include_prompt_tokens,
        "task_token_mode": task_token_mode,
        "condition_mode": str(getattr(cfg, "dit_condition_mode", "full")),
        "condition_injection_mode": str(getattr(cfg, "dit_condition_injection_mode", "cross_attn")),
        "clip_normalized_actions": bool(getattr(cfg, "dit_clip_normalized_actions", False)),
        "debug_group_action_tokens_to_chunk": bool(
            getattr(cfg, "debug_dit_group_action_tokens_to_chunk", False)
        ),
        "task_token_count": int(task_token_count),
        "action_token_count": int(expected),
        "packed_token_count": int(hidden_states.shape[2]),
        "action_head_num_task_tokens": int(getattr(action_head_obj, "num_task_tokens", -1)),
        "anchor_head_num_task_tokens": int(
            getattr(getattr(action_head_obj, "anchor_head", None), "num_task_tokens", -1)
        ),
        "velocity_num_task_tokens": int(getattr(getattr(action_head_obj, "velocity_network", None), "num_task_tokens", -1)),
        "ditx_num_task_tokens": int(
            getattr(getattr(getattr(action_head_obj, "velocity_network", None), "ditx", None), "num_task_tokens", -1)
        ),
        **layer_selector_info,
    }

    teacher_flow_info = {}
    if cfg.action_head_type.upper() == "DIT" and hasattr(action_head_obj, "predict_velocity"):
        model_dtype = next(action_head_obj.time_encoder.parameters()).dtype
        batch_size = target_actions.shape[0]
        x0 = torch.randn_like(target_actions, device=target_actions.device, dtype=model_dtype)
        x1 = target_actions.to(dtype=model_dtype)
        flow_x1 = x1
        anchor_actions = None
        anchor_head = getattr(action_head_obj, "anchor_head", None)
        if anchor_head is not None and float(getattr(action_head_obj, "supervised_anchor_weight", 0.0)) > 0.0:
            anchor_actions = anchor_head.predict_action(
                hidden_states,
                proprio=proprio if cfg.use_proprio else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                phase="Inference",
            ).to(dtype=model_dtype)
            flow_x1 = x1 - anchor_actions.detach()
        t = torch.full((batch_size,), 0.5, device=target_actions.device, dtype=model_dtype)
        t_view = t.view(batch_size, 1, 1)
        x_t = t_view * flow_x1 + (1.0 - t_view) * x0
        if getattr(action_head_obj.flow_cfg, "sample_target_t_mode", "relative") == "relative":
            r = torch.zeros_like(t)
        else:
            r = t
        v_pred = action_head_obj.predict_velocity(
            x_t,
            t,
            r,
            hidden_states,
            proprio=proprio if cfg.use_proprio else None,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
            detach_conditioning=bool(getattr(action_head_obj, "detach_flow_conditioning", False)),
        )
        recon = x_t + (1.0 - t_view) * v_pred
        if anchor_actions is not None:
            recon = recon + anchor_actions
        recon_diff = (recon.float() - target_actions.float()).abs()
        v_target = flow_x1 - x0
        velocity_diff = (v_pred.float() - v_target.float()).abs()
        teacher_flow_info = {
            "teacher_flow_recon_mae": float(recon_diff.mean().detach().cpu()),
            "teacher_flow_recon_non_grip_mae": float(recon_diff[..., :-1].mean().detach().cpu()),
            "teacher_flow_recon_grip_mae": float(recon_diff[..., -1].mean().detach().cpu()),
            "teacher_flow_velocity_mae": float(velocity_diff.mean().detach().cpu()),
        }
        zero_hidden_states = torch.zeros_like(hidden_states)
        v_zero_condition = action_head_obj.predict_velocity(
            x_t,
            t,
            r,
            zero_hidden_states,
            proprio=proprio if cfg.use_proprio else None,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
            detach_conditioning=bool(getattr(action_head_obj, "detach_flow_conditioning", False)),
        )
        recon_zero_condition = x_t + (1.0 - t_view) * v_zero_condition
        if anchor_actions is not None:
            recon_zero_condition = recon_zero_condition + anchor_actions
        teacher_flow_info.update(
            {
                "teacher_flow_condition_zero_velocity_delta": float(
                    (v_pred.float() - v_zero_condition.float()).abs().mean().detach().cpu()
                ),
                "teacher_flow_condition_zero_recon_delta": float(
                    (recon.float() - recon_zero_condition.float()).abs().mean().detach().cpu()
                ),
            }
        )
        if batch_size > 1:
            shuffled_hidden_states = hidden_states.roll(shifts=1, dims=0)
            shuffled_proprio = proprio.roll(shifts=1, dims=0) if cfg.use_proprio and proprio is not None else None
            v_shuffled_condition = action_head_obj.predict_velocity(
                x_t,
                t,
                r,
                shuffled_hidden_states,
                proprio=shuffled_proprio if cfg.use_proprio else None,
                proprio_projector=proprio_projector if cfg.use_proprio else None,
                detach_conditioning=bool(getattr(action_head_obj, "detach_flow_conditioning", False)),
            )
            recon_shuffled_condition = x_t + (1.0 - t_view) * v_shuffled_condition
            if anchor_actions is not None:
                recon_shuffled_condition = recon_shuffled_condition + anchor_actions
            teacher_flow_info.update(
                {
                    "teacher_flow_condition_shuffle_velocity_delta": float(
                        (v_pred.float() - v_shuffled_condition.float()).abs().mean().detach().cpu()
                    ),
                    "teacher_flow_condition_shuffle_recon_delta": float(
                        (recon.float() - recon_shuffled_condition.float()).abs().mean().detach().cpu()
                    ),
                }
            )

    if depth_interface is not None:
        depth_out = module_or_self(depth_interface)(
            hidden_states,
            proprio=proprio if cfg.use_proprio else None,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
        )
        pred = action_head_obj.predict_action_from_state(
            depth_out.state_emb,
            proprio=proprio if cfg.use_proprio else None,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
        )
    else:
        pred = action_head_obj.predict_action(
            hidden_states,
            proprio=proprio if cfg.use_proprio else None,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
            phase="Inference",
        )

    return pred.float().detach().cpu(), target_actions.float().detach().cpu(), split_info, teacher_flow_info


def summarize(name, array):
    flat = array.reshape(-1, array.shape[-1])
    print(name, np.array2string(flat, precision=5, separator=","))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--action_head_type", default="DIT", choices=["MLP", "DIT", "FlowMLP"])
    parser.add_argument("--data_root_dir", default="data/libero")
    parser.add_argument("--dataset_name", default="libero_object_no_noops")
    parser.add_argument("--task_suite", default="libero_object")
    parser.add_argument("--unnorm_key", default="libero_object_no_noops")
    parser.add_argument("--num_batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--shuffle_buffer_size", type=int, default=1000)
    parser.add_argument("--image_aug", type=str2bool, default=False)
    parser.add_argument("--num_images_in_input", type=int, default=2)
    parser.add_argument("--use_proprio", type=str2bool, default=True)
    parser.add_argument("--use_minivlm", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use_adaptive_bridge", type=str2bool, default=True)
    parser.add_argument("--bridge_mode", default="adaptive")
    parser.add_argument("--fixed_layer_index", type=int, default=-1)
    parser.add_argument("--use_depth_interface", type=str2bool, default=False)
    parser.add_argument("--depth_interface_mode", default="none")
    parser.add_argument("--depth_interface_max_layers", type=int, default=64)
    parser.add_argument("--depth_interface_add_proprio", type=str2bool, default=True)
    parser.add_argument("--flowmlp_num_inference_steps", type=int, default=5)
    parser.add_argument("--flowmlp_num_inference_samples", type=int, default=1)
    parser.add_argument("--flowmlp_supervised_anchor_weight", type=float, default=0.0)
    parser.add_argument("--flowmlp_anchor_blend", type=float, default=0.0)
    parser.add_argument("--flowmlp_anchor_gripper_weight", type=float, default=1.0)
    parser.add_argument("--flowmlp_anchor_gripper_bce_weight", type=float, default=0.0)
    parser.add_argument("--flowmlp_detach_flow_conditioning", type=str2bool, default=False)
    parser.add_argument("--dit_num_blocks", type=int, default=12)
    parser.add_argument("--dit_num_inference_steps", type=int, default=20)
    parser.add_argument("--dit_num_inference_samples", type=int, default=1)
    parser.add_argument("--dit_supervised_anchor_weight", type=float, default=0.0)
    parser.add_argument("--dit_anchor_blend", type=float, default=0.0)
    parser.add_argument("--dit_anchor_gripper_weight", type=float, default=1.0)
    parser.add_argument("--dit_anchor_gripper_bce_weight", type=float, default=0.0)
    parser.add_argument("--dit_flow_xyz_loss_weight", type=float, default=1.0)
    parser.add_argument("--dit_flow_rot_loss_weight", type=float, default=1.0)
    parser.add_argument("--dit_flow_gripper_loss_weight", type=float, default=1.0)
    parser.add_argument("--dit_flow_gripper_bce_weight", type=float, default=0.0)
    parser.add_argument("--dit_flow_gripper_bce_logit_scale", type=float, default=1.0)
    parser.add_argument("--dit_flow_gripper_bce_balanced", type=str2bool, default=False)
    parser.add_argument("--dit_gripper_head_weight", type=float, default=0.0)
    parser.add_argument("--dit_gripper_head_override", type=str2bool, default=False)
    parser.add_argument("--dit_clip_normalized_actions", type=str2bool, default=False)
    parser.add_argument("--dit_detach_flow_conditioning", type=str2bool, default=False)
    parser.add_argument("--dit_disable_inference_anchor", type=str2bool, default=False)
    parser.add_argument("--dit_pure_inference", type=str2bool, default=False)
    parser.add_argument("--dit_zero_init_adaln", type=str2bool, default=False)
    parser.add_argument("--dit_zero_init_output", type=str2bool, default=False)
    parser.add_argument("--dit_use_state_conditioning", type=str2bool, default=False)
    parser.add_argument("--dit_state_scale_mode", default="none")
    parser.add_argument("--dit_state_proprio_mode", default="concat")
    parser.add_argument("--dit_state_use_chunk_pos", type=str2bool, default=False)
    parser.add_argument("--dit_state_include_task_tokens", type=str2bool, default=False)
    parser.add_argument("--dit_condition_mode", default="full", choices=["full", "task_only"])
    parser.add_argument(
        "--dit_condition_injection_mode",
        default="cross_attn",
        choices=["cross_attn", "joint_prefix", "action_expert_prefix"],
    )
    parser.add_argument("--dit_include_prompt_tokens", type=str2bool, default=False)
    parser.add_argument(
        "--dit_task_token_mode",
        default="vision_prompt",
        choices=["vision_prompt", "vision_only", "prompt_only", "last_prompt"],
    )
    parser.add_argument("--debug_dit_group_action_tokens_to_chunk", type=str2bool, default=False)
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(args.checkpoint)

    set_seed_everywhere(args.seed)
    cfg = make_cfg(args)
    model, action_head, depth_interface, proprio_projector, processor = initialize_probe_model(cfg)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=args.num_images_in_input > 1,
        use_proprio=args.use_proprio,
        use_minivlm=args.use_minivlm,
    )
    dataset = RLDSDataset(
        args.data_root_dir,
        args.dataset_name,
        batch_transform,
        resize_resolution=tuple(module_or_self(model).config.image_sizes),
        shuffle_buffer_size=args.shuffle_buffer_size,
        image_aug=args.image_aug,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=0, pin_memory=True)

    preds = []
    targets = []
    split_infos = []
    teacher_flow_infos = []
    iterator = iter(dataloader)
    for _ in range(args.num_batches):
        batch = next(iterator)
        pred, target, split_info, teacher_flow_info = predict_batch(
            cfg,
            model,
            action_head,
            depth_interface,
            proprio_projector,
            batch,
        )
        preds.append(pred)
        targets.append(target)
        split_infos.append(split_info)
        if teacher_flow_info:
            teacher_flow_infos.append(teacher_flow_info)

    pred = torch.cat(preds, dim=0).numpy()
    target = torch.cat(targets, dim=0).numpy()
    diff = pred - target
    abs_diff = np.abs(diff)

    print(f"CHECKPOINT {args.checkpoint}")
    print(f"ACTION_HEAD {args.action_head_type}")
    print(f"SHAPE pred={pred.shape} target={target.shape}")
    if split_infos:
        first_split = split_infos[0]
        print(
            "SPLIT_INFO "
            + " ".join(f"{key}={value}" for key, value in first_split.items())
        )
    print("DIM_NAMES", ",".join(DIM_NAMES))
    summarize("PRED_MEAN", pred.mean(axis=(0, 1), keepdims=True))
    summarize("PRED_STD ", pred.std(axis=(0, 1), keepdims=True))
    summarize("PRED_MIN ", pred.min(axis=(0, 1), keepdims=True))
    summarize("PRED_MAX ", pred.max(axis=(0, 1), keepdims=True))
    print(f"PRED_OOR_FRAC {float((np.abs(pred) > 1.0).mean()):.6f}")
    summarize("GT_MEAN  ", target.mean(axis=(0, 1), keepdims=True))
    summarize("GT_STD   ", target.std(axis=(0, 1), keepdims=True))
    summarize("GT_MIN   ", target.min(axis=(0, 1), keepdims=True))
    summarize("GT_MAX   ", target.max(axis=(0, 1), keepdims=True))
    summarize("MAE_DIM  ", abs_diff.mean(axis=(0, 1), keepdims=True))
    summarize("BIAS_DIM ", diff.mean(axis=(0, 1), keepdims=True))
    print(f"MAE_ALL {float(abs_diff.mean()):.6f}")
    print(f"MAE_NON_GRIP {float(abs_diff[..., :-1].mean()):.6f}")
    print(f"MAE_GRIP {float(abs_diff[..., -1].mean()):.6f}")
    if teacher_flow_infos:
        keys = teacher_flow_infos[0].keys()
        for key in keys:
            print(f"{key.upper()} {float(np.mean([info[key] for info in teacher_flow_infos])):.6f}")
    print("MAE_BY_STEP")
    print(np.array2string(abs_diff.mean(axis=0), precision=5, separator=","))
    print("BIAS_BY_STEP")
    print(np.array2string(diff.mean(axis=0), precision=5, separator=","))


if __name__ == "__main__":
    main()
