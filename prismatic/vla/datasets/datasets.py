"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type, Optional, List
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder, QwenPromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    NormalizationType,
    PROPRIO_DIM,
    STOP_INDEX,
)
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.real_robot_resampling import resample_episode_to_fixed_hz



@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    use_wrist_image: bool = False
    use_proprio: bool = False
    use_minivlm: bool = False


    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, current_action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        actions = rlds_batch["action"]

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")

        # Get future action chunk
        future_actions = rlds_batch["action"][1:]

        # Direct HF checkpoints require preserving raw action token IDs; the
        # text round-trip branch loses the 56 action positions under Qwen2.
        if True:
            self.prompt_builder_fn = QwenPromptBuilder
            prompt_builder = self.prompt_builder_fn("openvla")
            # Get action chunk string
            future_actions_string = self.action_tokenizer(future_actions, True)
            current_action_string = self.action_tokenizer(current_action, True)

            action_chunk_string = [current_action_string] + future_actions_string
            flattened_action_chunk_string = [item for sublist in action_chunk_string for item in sublist]
            expected_action_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK
            if len(flattened_action_chunk_string) != expected_action_tokens:
                raise ValueError(
                    f"Expected {expected_action_tokens} action tokens for "
                    f"{NUM_ACTIONS_CHUNK}x{ACTION_DIM} action chunk, got "
                    f"{len(flattened_action_chunk_string)}"
                )

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": ''},
            ]

            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            prompt = prompt_builder.get_prompt() #e.g. 'In: What action should the robot take to put both the cream cheese box and the butter in the basket?\nOut: 希</s>'
            input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids

            if len(input_ids) >= 3:
                del input_ids[-3] 
                del input_ids[-2] 
                del input_ids[-1] 

            input_ids = input_ids + flattened_action_chunk_string
            labels = list(input_ids)
            action_chunk_len = expected_action_tokens

        else:
            future_actions_string = ''.join(self.action_tokenizer(future_actions, use_minivlm=False))

            # Get action chunk string
            current_action_string = self.action_tokenizer(current_action, use_minivlm=False)
            action_chunk_string = current_action_string + future_actions_string
            action_chunk_len = len(action_chunk_string)

            conversation = [
                {"from": "human", "value": f"What action should the robot take to {lang}?"},
                {"from": "gpt", "value": action_chunk_string[0]},
            ]
            # remove action token
            # conversation = [
            #     {"from": "human", "value": f"What action should the robot take to {lang}?"},
            #     {"from": "gpt", "value": ""},
            # ]
            action_chunk_len = 1


            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])
            prompt = prompt_builder.get_prompt() #e.g. 'In: What action should the robot take to put both the cream cheese box and the butter in the basket?\nOut: 希</s>'
            # Tokenize (w/ `base_tokenizer`)
            input_ids = self.base_tokenizer(prompt, add_special_tokens=True).input_ids
            labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        timestep = rlds_batch["observation"].get("timestep", -1)
        if hasattr(timestep, "item"):
            timestep = timestep.item()
        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=dataset_name,
            task_language=lang,
            timestep=int(timestep),
            actions=actions,
        )

        # Add additional inputs
        if self.use_wrist_image:
            all_wrist_pixels = []
            for k in rlds_batch["observation"].keys():
                if "wrist" in k:
                    img_wrist = Image.fromarray(rlds_batch["observation"][k][0])
                    pixel_values_wrist = self.image_transform(img_wrist)
                    all_wrist_pixels.append(pixel_values_wrist)
            return_dict["pixel_values_wrist"] = torch.cat(all_wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            proprio = rlds_batch["observation"]["proprio"]
            return_dict["proprio"] = proprio

        return return_dict
    
    

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        if "aloha" in self.data_mix:
            load_camera_views = ("primary", "left_wrist", "right_wrist")
        else:
            load_camera_views = ("primary", "wrist")

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=NUM_ACTIONS_CHUNK-1,      # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class RealRobotDataset(Dataset):
    """
    本地真机数据集，支持从关节角度数据转换为末端位姿增量（EEF Delta）。

    数据格式约定：
      - root_dir 下每个 episode_xxxxx 目录结构为:
            images/              # 主相机图像
            images_wrist/        # 腕相机图像（可选）
            actions.npz          # 关节角度 [joint1-6, gripper]，单位度（不再直接使用）
            states.npz           # 原始相机帧、逐相机时间戳、完整机器人状态历史
            task.txt             # 语言指令

    动作空间转换（关键）：
      - 默认构造严格的 10 Hz episode 时间轴
      - 双相机按时间戳配对，机器人状态从完整回调历史插值到统一时间轴
      - eef_delta: 前六维为 pose(t+0.1s) - pose(t)
      - absolute_joint: 前六维为 joint(t+0.1s) 的绝对关节角（统一转换为 rad）
      - 第七维均为 t+0.1s 的实际夹爪开口目标，范围 0--1000；部署时按绝对位置发送
      - 掉帧或状态时间断档会切开连续段，动作块不会跨越断档

    与 LIBERO 对齐：
      - LIBERO action: EEF_POS = Delta XYZ (3) + Delta RPY (3) + Gripper (1)
      - 本数据集转换后格式完全一致
    """

    def __init__(
        self,
        root_dir: Path,
        dataset_name: str,
        batch_transform: RLDSBatchTransform,
        target_hz: float = 10.0,
        max_camera_error_s: float = 0.04,
        max_camera_skew_s: float = 0.03,
        max_state_gap_s: float = 0.05,
        action_mode: str = "eef_delta",
        joint_unit: str = "degree",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.dataset_name = str(dataset_name)
        self.batch_transform = batch_transform
        self.target_hz = float(target_hz)
        self.action_dt = 1.0 / self.target_hz
        self.max_camera_error_s = float(max_camera_error_s)
        self.max_camera_skew_s = float(max_camera_skew_s)
        self.max_state_gap_s = float(max_state_gap_s)
        self.action_mode = str(action_mode)
        self.joint_unit = str(joint_unit)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"RealRobotDataset root_dir does not exist: {self.root_dir}")

        self._episodes = []
        all_actions = []
        all_proprios = []
        num_transitions = 0
        num_source_episodes = 0
        num_legacy_episodes = 0
        rejected_camera_error = 0
        rejected_camera_skew = 0
        rejected_state_gap = 0

        for ep_dir in sorted(self.root_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue

            task_path = ep_dir / "task.txt"
            states_path = ep_dir / "states.npz"
            images_dir = ep_dir / "images"
            if not (task_path.exists() and states_path.exists() and images_dir.exists()):
                continue

            # 语言指令
            language = task_path.read_text(encoding="utf-8").strip().encode("utf-8")

            # 图像路径
            main_imgs = sorted(
                [p for p in images_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            )
            if not main_imgs:
                continue

            # 腕相机图像（可选）
            wrist_imgs: List[Path] = []
            wrist_dir = ep_dir / "images_wrist"
            if wrist_dir.exists():
                wrist_imgs = sorted(
                    [p for p in wrist_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
                )

            with np.load(states_path, allow_pickle=False) as states_npz:
                if not all(k in states_npz for k in ("joint", "pose", "proprio")):
                    continue

                raw_joint = states_npz["joint"].astype(np.float32)
                raw_pose = states_npz["pose"].astype(np.float32)
                raw_proprio = states_npz["proprio"].astype(np.float32)
                raw_count = min(len(main_imgs), raw_pose.shape[0], raw_joint.shape[0])
                if wrist_imgs:
                    raw_count = min(raw_count, len(wrist_imgs))
                if raw_count < 2:
                    continue
                main_imgs = main_imgs[:raw_count]
                wrist_imgs = wrist_imgs[:raw_count]

                is_timestamp_v2 = all(
                    key in states_npz
                    for key in (
                        "camera_timestamp",
                        "camera_valid",
                        "joint_history",
                        "joint_history_timestamp",
                        "pose_history",
                        "pose_history_timestamp",
                    )
                )

                if is_timestamp_v2:
                    camera_timestamps = states_npz["camera_timestamp"].astype(np.float64)[:raw_count]
                    camera_valid = states_npz["camera_valid"].astype(bool)[:raw_count]
                    camera_count = 2 if wrist_imgs else 1
                    camera_timestamps = camera_timestamps[:, :camera_count]
                    camera_valid = camera_valid[:, :camera_count]
                    pose_history = states_npz["pose_history"].astype(np.float32)
                    pose_history_timestamps = states_npz["pose_history_timestamp"].astype(np.float64)
                    joint_history = states_npz["joint_history"].astype(np.float32)
                    joint_history_timestamps = states_npz["joint_history_timestamp"].astype(np.float64)
                    if "gripper_history" in states_npz and "gripper_history_timestamp" in states_npz:
                        gripper_history = states_npz["gripper_history"].astype(np.float32)
                        gripper_history_timestamps = states_npz["gripper_history_timestamp"].astype(np.float64)
                    else:
                        gripper_history = None
                        gripper_history_timestamps = None
                else:
                    # Compatibility path for v1 episodes: it can still create a
                    # real 10 Hz grid, but only from the old per-camera-loop state
                    # snapshots rather than the complete 200 Hz callback stream.
                    num_legacy_episodes += 1
                    if "receive_timestamp" in states_npz:
                        legacy_timestamps = states_npz["receive_timestamp"].astype(np.float64)[:raw_count]
                    else:
                        legacy_timestamps = np.arange(raw_count, dtype=np.float64) / 30.0
                    camera_count = 2 if wrist_imgs else 1
                    camera_timestamps = np.repeat(legacy_timestamps[:, None], camera_count, axis=1)
                    camera_valid = np.ones_like(camera_timestamps, dtype=bool)
                    pose_history = raw_pose[:raw_count]
                    pose_history_timestamps = legacy_timestamps
                    joint_history = raw_joint[:raw_count]
                    joint_history_timestamps = legacy_timestamps
                    gripper_history = raw_proprio[:raw_count, 6:8]
                    gripper_history_timestamps = legacy_timestamps

            segments, diagnostics = resample_episode_to_fixed_hz(
                camera_timestamps=camera_timestamps,
                camera_valid=camera_valid,
                pose_timestamps=pose_history_timestamps,
                poses=pose_history,
                joint_timestamps=joint_history_timestamps,
                joints=joint_history,
                gripper_timestamps=gripper_history_timestamps,
                grippers=gripper_history,
                target_hz=self.target_hz,
                max_camera_error_s=self.max_camera_error_s,
                max_camera_skew_s=self.max_camera_skew_s,
                max_state_gap_s=self.max_state_gap_s,
                action_mode=self.action_mode,
                joint_unit=self.joint_unit,
            )
            rejected_camera_error += diagnostics.rejected_camera_error
            rejected_camera_skew += diagnostics.rejected_camera_skew
            rejected_state_gap += diagnostics.rejected_state_gap
            if not segments:
                print(f"[RealRobotDataset] Skipping {ep_dir.name}: no valid {self.target_hz:g} Hz samples")
                continue

            num_source_episodes += 1
            for segment in segments:
                source_image_indices = segment["source_image_indices"].astype(np.int64)
                actions = segment["actions"]
                proprio = segment["proprio"]
                self._episodes.append(
                    dict(
                        language=language,
                        actions=actions,
                        pose=segment["pose"],
                        proprio=proprio,
                        images=[main_imgs[i] for i in source_image_indices],
                        images_wrist=[wrist_imgs[i] for i in source_image_indices] if wrist_imgs else [],
                        timestamps=segment["timestamps"],
                        camera_error_s=segment["camera_error_s"],
                        camera_skew_s=segment["camera_skew_s"],
                    )
                )
                all_actions.append(actions)
                all_proprios.append(proprio)
                num_transitions += actions.shape[0]

        if not self._episodes:
            raise RuntimeError(f"No valid episodes found under {self.root_dir}")

        # 计算动作的逐维归一化统计量
        all_actions_arr = np.concatenate(all_actions, axis=0)
        self._action_stats = {
            "min": all_actions_arr.min(axis=0).astype(np.float32),
            "max": all_actions_arr.max(axis=0).astype(np.float32),
            "mean": all_actions_arr.mean(axis=0).astype(np.float32),
            "std": all_actions_arr.std(axis=0).astype(np.float32),
            "q01": np.quantile(all_actions_arr, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(all_actions_arr, 0.99, axis=0).astype(np.float32),
        }

        # proprio 归一化统计量
        all_proprios_arr = np.concatenate(all_proprios, axis=0)
        self._proprio_stats = {
            "min": all_proprios_arr.min(axis=0).astype(np.float32),
            "max": all_proprios_arr.max(axis=0).astype(np.float32),
            "q01": np.quantile(all_proprios_arr, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(all_proprios_arr, 0.99, axis=0).astype(np.float32),
        }

        # Dataset 统计信息
        self.dataset_statistics = {
            self.dataset_name: {
                "action": {k: v.copy() for k, v in self._action_stats.items()},
                "proprio": {k: v.copy() for k, v in self._proprio_stats.items()},
                "num_trajectories": num_source_episodes,
                "num_contiguous_segments": len(self._episodes),
                "num_transitions": num_transitions,
                "target_hz": self.target_hz,
                "action_dt_seconds": self.action_dt,
                "action_mode": self.action_mode,
                "joint_unit": "radian" if self.action_mode == "absolute_joint" else None,
                "action_layout": (
                    ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
                    if self.action_mode == "absolute_joint"
                    else ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
                ),
                "proprio_layout": (
                    ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
                    + ["gripper_position", "gripper_force"][: max(0, all_proprios_arr.shape[1] - 6)]
                    if self.action_mode == "absolute_joint"
                    else ["x", "y", "z", "roll", "pitch", "yaw"]
                    + ["gripper_position", "gripper_force"][: max(0, all_proprios_arr.shape[1] - 6)]
                ),
                "timestamp_alignment": {
                    "max_camera_error_seconds": self.max_camera_error_s,
                    "max_camera_skew_seconds": self.max_camera_skew_s,
                    "max_state_gap_seconds": self.max_state_gap_s,
                },
            }
        }

        # 构建索引
        self._indices: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(self._episodes):
            T = ep["actions"].shape[0]
            for t in range(T):
                self._indices.append((ep_idx, t))

        print(
            f"[RealRobotDataset] Loaded {num_source_episodes} episodes as {len(self._episodes)} "
            f"contiguous segments, {num_transitions} transitions at {self.target_hz:g} Hz"
        )
        if num_legacy_episodes:
            print(
                f"[RealRobotDataset] WARNING: {num_legacy_episodes} legacy episodes lack full-rate state "
                "history; resampling used per-frame snapshots"
            )
        print(
            "[RealRobotDataset] Alignment rejects: "
            f"camera_error={rejected_camera_error}, camera_skew={rejected_camera_skew}, "
            f"state_gap={rejected_state_gap}"
        )
        print(f"[RealRobotDataset] Action mode: {self.action_mode}")
        print(
            f"[RealRobotDataset] Action q01={self._action_stats['q01']} "
            f"q99={self._action_stats['q99']}"
        )

    def __len__(self) -> int:
        return len(self._indices)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """使用 q01/q99 将动作归一化到 [-1, 1]，与 LIBERO 的 BOUNDS_Q99 一致。"""
        low = self._action_stats["q01"]
        high = self._action_stats["q99"]
        denom = high - low
        denom[denom == 0] = 1.0
        x = 2.0 * (action - low) / denom - 1.0
        x = np.clip(x, -1.0, 1.0)
        zeros_mask = self._action_stats["min"] == self._action_stats["max"]
        x[:, zeros_mask] = 0.0
        return x.astype(np.float32)

    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        """将本体状态归一化到 [-1, 1]。"""
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            low, high = self._proprio_stats["q01"], self._proprio_stats["q99"]
        else:
            low, high = self._proprio_stats["min"], self._proprio_stats["max"]
        denom = high - low
        denom[denom == 0] = 1.0
        x = 2.0 * (proprio.astype(np.float32) - low) / denom - 1.0
        x = np.clip(x, -1.0, 1.0)
        zeros_mask = self._proprio_stats["min"] == self._proprio_stats["max"]
        x[zeros_mask] = 0.0
        return x.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_idx, t = self._indices[idx]
        ep = self._episodes[ep_idx]

        actions = ep["actions"]
        T = actions.shape[0]

        # 动作窗口：[t, t+NUM_ACTIONS_CHUNK)
        end = min(t + NUM_ACTIONS_CHUNK, T)
        window = actions[t:end]
        if end - t < NUM_ACTIONS_CHUNK:
            pad_count = NUM_ACTIONS_CHUNK - (end - t)
            if self.action_mode == "absolute_joint":
                # An absolute target remains valid until replaced. Zero padding
                # would instead teach every arm joint and the gripper to go to zero.
                pad = np.repeat(window[-1:], pad_count, axis=0)
            else:
                pad = np.zeros((pad_count, ACTION_DIM), dtype=window.dtype)
            window = np.concatenate([window, pad], axis=0)
        window = self._normalize_action(window)

        # 图像
        img = Image.open(ep["images"][min(t, len(ep["images"]) - 1)]).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)
        obs: Dict[str, Any] = {
            "image_primary": np.repeat(img_np[None, ...], NUM_ACTIONS_CHUNK, axis=0),
        }

        if ep["images_wrist"]:
            w_img = Image.open(ep["images_wrist"][min(t, len(ep["images_wrist"]) - 1)]).convert("RGB")
            w_np = np.array(w_img, dtype=np.uint8)
            obs["image_wrist"] = np.repeat(w_np[None, ...], NUM_ACTIONS_CHUNK, axis=0)

        # 本体状态
        raw_proprio = ep["proprio"][min(t, ep["proprio"].shape[0] - 1)]
        obs["proprio"] = self._normalize_proprio(raw_proprio)

        rlds_batch = dict(
            dataset_name=self.dataset_name,
            action=window,
            observation=obs,
            task={"language_instruction": ep["language"]},
        )

        return self.batch_transform(rlds_batch)


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
