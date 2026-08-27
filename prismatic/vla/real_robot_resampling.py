"""Timestamp-based resampling utilities for the real-robot NPZ dataset.

This module intentionally only depends on NumPy and SciPy so its alignment logic
can be tested without importing the full VLA/RLDS training stack.
"""


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class ResamplingDiagnostics:
    source_camera_samples: int
    grid_samples: int
    valid_samples: int
    rejected_camera_error: int
    rejected_camera_skew: int
    rejected_state_gap: int


def _clean_timeseries(timestamps: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite, sorted samples with duplicate timestamps removed."""
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    values = np.asarray(values)
    count = min(timestamps.shape[0], values.shape[0])
    timestamps, values = timestamps[:count], values[:count]
    finite = np.isfinite(timestamps)
    timestamps, values = timestamps[finite], values[finite]
    if timestamps.size == 0:
        return timestamps, values

    order = np.argsort(timestamps, kind="stable")
    timestamps, values = timestamps[order], values[order]
    # Keep the last callback when multiple messages have the same timestamp.
    keep = np.r_[timestamps[1:] != timestamps[:-1], True]
    return timestamps[keep], values[keep]


def _bracket_gaps(timestamps: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Width of the source interval bracketing each query (inf outside range)."""
    query = np.asarray(query, dtype=np.float64)
    gaps = np.full(query.shape, np.inf, dtype=np.float64)
    if timestamps.size < 2:
        return gaps
    right = np.searchsorted(timestamps, query, side="left")
    exact = (right < timestamps.size) & np.isclose(
        timestamps[np.minimum(right, timestamps.size - 1)], query, rtol=0.0, atol=1e-9
    )
    gaps[exact] = 0.0
    interior = (~exact) & (right > 0) & (right < timestamps.size)
    gaps[interior] = timestamps[right[interior]] - timestamps[right[interior] - 1]
    return gaps


def _interpolate_linear(timestamps: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    columns = [np.interp(query, timestamps, values[:, dim]) for dim in range(values.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def _interpolate_pose(timestamps: np.ndarray, poses: np.ndarray, query: np.ndarray) -> np.ndarray:
    positions = _interpolate_linear(timestamps, poses[:, :3], query)
    rotations = Rotation.from_euler("xyz", poses[:, 3:6])
    euler = Slerp(timestamps, rotations)(query).as_euler("xyz").astype(np.float32)
    return np.concatenate([positions, euler], axis=1).astype(np.float32)


def _nearest_indices(sorted_timestamps: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(sorted_timestamps, query, side="left")
    right = np.clip(right, 0, sorted_timestamps.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(sorted_timestamps[left] - query) <= np.abs(sorted_timestamps[right] - query)
    return np.where(choose_left, left, right)


def resample_episode_to_fixed_hz(
    *,
    camera_timestamps: np.ndarray,
    camera_valid: np.ndarray,
    pose_timestamps: np.ndarray,
    poses: np.ndarray,
    joint_timestamps: np.ndarray,
    joints: np.ndarray,
    gripper_timestamps: Optional[np.ndarray] = None,
    grippers: Optional[np.ndarray] = None,
    target_hz: float = 10.0,
    max_camera_error_s: float = 0.04,
    max_camera_skew_s: float = 0.03,
    max_state_gap_s: float = 0.05,
) -> Tuple[List[Dict[str, np.ndarray]], ResamplingDiagnostics]:
    """Resample one raw episode and split it at invalid/missing time steps.

    Each returned sample contains an observation at ``t`` and an EEF delta action
    from ``t`` to ``t + 1 / target_hz``. Returned segments are temporally
    contiguous, so action chunks never cross a dropped frame or state gap.
    """
    if target_hz <= 0:
        raise ValueError(f"target_hz must be positive, got {target_hz}")

    camera_timestamps = np.asarray(camera_timestamps, dtype=np.float64)
    if camera_timestamps.ndim == 1:
        camera_timestamps = camera_timestamps[:, None]
    camera_valid = np.asarray(camera_valid, dtype=bool)
    if camera_valid.ndim == 1:
        camera_valid = camera_valid[:, None]
    if camera_timestamps.shape != camera_valid.shape:
        raise ValueError(
            f"camera timestamp/valid shape mismatch: {camera_timestamps.shape} vs {camera_valid.shape}"
        )

    finite_camera = np.all(np.isfinite(camera_timestamps), axis=1)
    valid_camera_rows = np.all(camera_valid, axis=1) & finite_camera
    source_indices = np.flatnonzero(valid_camera_rows)
    if source_indices.size == 0:
        return [], ResamplingDiagnostics(camera_timestamps.shape[0], 0, 0, 0, 0, 0)

    pair_timestamps = camera_timestamps[source_indices].mean(axis=1)
    camera_skews = np.ptp(camera_timestamps[source_indices], axis=1)
    order = np.argsort(pair_timestamps, kind="stable")
    pair_timestamps = pair_timestamps[order]
    camera_skews = camera_skews[order]
    source_indices = source_indices[order]
    unique = np.r_[pair_timestamps[1:] != pair_timestamps[:-1], True]
    pair_timestamps = pair_timestamps[unique]
    camera_skews = camera_skews[unique]
    source_indices = source_indices[unique]

    pose_timestamps, poses = _clean_timeseries(pose_timestamps, np.asarray(poses, dtype=np.float32))
    joint_timestamps, joints = _clean_timeseries(joint_timestamps, np.asarray(joints, dtype=np.float32))
    if pose_timestamps.size < 2 or joint_timestamps.size < 2:
        return [], ResamplingDiagnostics(camera_timestamps.shape[0], 0, 0, 0, 0, 0)

    dt = 1.0 / float(target_hz)
    start = max(pair_timestamps[0], pose_timestamps[0], joint_timestamps[0])
    end = min(pair_timestamps[-1], pose_timestamps[-1] - dt, joint_timestamps[-1])
    if end < start:
        return [], ResamplingDiagnostics(camera_timestamps.shape[0], 0, 0, 0, 0, 0)

    # Starting at the overlap boundary avoids depending on Unix-time modulo while
    # still producing one strict, episode-local fixed-rate time axis.
    grid_count = int(np.floor((end - start) / dt + 1e-9)) + 1
    grid = start + np.arange(grid_count, dtype=np.float64) * dt
    nearest = _nearest_indices(pair_timestamps, grid)
    selected_camera_indices = source_indices[nearest]
    camera_error = np.abs(pair_timestamps[nearest] - grid)
    camera_skew = camera_skews[nearest]

    pose_gap_now = _bracket_gaps(pose_timestamps, grid)
    pose_gap_next = _bracket_gaps(pose_timestamps, grid + dt)
    joint_gap = _bracket_gaps(joint_timestamps, grid)
    state_ok = (
        (pose_gap_now <= max_state_gap_s)
        & (pose_gap_next <= max_state_gap_s)
        & (joint_gap <= max_state_gap_s)
    )
    camera_error_ok = camera_error <= max_camera_error_s
    camera_skew_ok = camera_skew <= max_camera_skew_s
    # A source image may not represent two target steps, even with loose tolerances.
    camera_unique = np.r_[True, selected_camera_indices[1:] != selected_camera_indices[:-1]]
    valid = state_ok & camera_error_ok & camera_skew_ok & camera_unique

    pose_now = _interpolate_pose(pose_timestamps, poses, grid)
    pose_next = _interpolate_pose(pose_timestamps, poses, grid + dt)
    joint_now = _interpolate_linear(joint_timestamps, joints, grid)

    if gripper_timestamps is not None and grippers is not None and len(gripper_timestamps) > 0:
        gripper_timestamps, grippers = _clean_timeseries(
            gripper_timestamps, np.asarray(grippers, dtype=np.float32)
        )
        if gripper_timestamps.size >= 2:
            gripper_now = _interpolate_linear(gripper_timestamps, grippers, grid)
        elif gripper_timestamps.size == 1:
            gripper_now = np.repeat(grippers[:1], grid_count, axis=0).astype(np.float32)
        else:
            gripper_now = np.zeros((grid_count, 2), dtype=np.float32)
    else:
        gripper_now = np.zeros((grid_count, 2), dtype=np.float32)

    delta_actions = np.zeros((grid_count, 7), dtype=np.float32)
    delta_actions[:, :6] = pose_next - pose_now
    # Deployment adds Euler deltas to the current Euler angles, so keep that
    # convention while avoiding a +/-pi discontinuity.
    delta_actions[:, 3:6] = (delta_actions[:, 3:6] + np.pi) % (2.0 * np.pi) - np.pi
    proprios = np.concatenate([pose_now, gripper_now], axis=1).astype(np.float32)

    segments: List[Dict[str, np.ndarray]] = []
    valid_grid_indices = np.flatnonzero(valid)
    if valid_grid_indices.size:
        boundaries = np.flatnonzero(np.diff(valid_grid_indices) != 1) + 1
        for part in np.split(valid_grid_indices, boundaries):
            if part.size == 0:
                continue
            segments.append(
                {
                    "timestamps": grid[part],
                    "source_image_indices": selected_camera_indices[part],
                    "pose": pose_now[part],
                    "joint": joint_now[part],
                    "gripper": gripper_now[part],
                    "proprio": proprios[part],
                    "delta_actions": delta_actions[part],
                    "camera_error_s": camera_error[part],
                    "camera_skew_s": camera_skew[part],
                }
            )

    diagnostics = ResamplingDiagnostics(
        source_camera_samples=camera_timestamps.shape[0],
        grid_samples=grid_count,
        valid_samples=int(valid.sum()),
        rejected_camera_error=int((~camera_error_ok).sum()),
        rejected_camera_skew=int((~camera_skew_ok).sum()),
        rejected_state_gap=int((~state_ok).sum()),
    )
    return segments, diagnostics
