"""Shared validation helpers for BPX HDF5 episodes."""

import dataclasses
import pathlib
import re
from typing import Any

import h5py
import numpy as np

from openpi.policies import bpx_constants

QPOS_KEY = "observations/qpos"
ACTION_KEY = "action"
IMAGE_DATASET_KEYS = {
    "left_eye": "observations/images/left_eye",
    "right_eye": "observations/images/right_eye",
    "left_wrist": "observations/images/left_wrist",
    "right_wrist": "observations/images/right_wrist",
}


@dataclasses.dataclass(frozen=True)
class EpisodeSummary:
    path: pathlib.Path
    frames: int
    task: str | None
    success: bool | None
    max_action_waist_abs: float
    max_action_gripper_pair_error: float


def discover_episodes(raw_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return deterministically ordered ``episode_*.hdf5`` files."""
    pattern = re.compile(r"episode_(\d+)\.hdf5")
    indexed_files = []
    for path in pathlib.Path(raw_dir).glob("episode_*.hdf5"):
        match = pattern.fullmatch(path.name)
        if match:
            indexed_files.append((int(match.group(1)), path))
    files = [path for _, path in sorted(indexed_files)]
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found under {raw_dir}.")
    return files


def _read_optional_scalar(group: h5py.File, key: str) -> Any | None:
    if key in group:
        value = group[key][()]
    elif key in group.attrs:
        value = group.attrs[key]
    else:
        return None
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def read_task(group: h5py.File, default_task: str) -> str:
    """Read an optional per-episode task, falling back to the configured task."""
    value = _read_optional_scalar(group, "task")
    task = default_task if value is None else str(value).strip()
    if not task:
        raise ValueError("Task prompt must not be empty.")
    return task


def read_success(group: h5py.File) -> bool | None:
    """Read an optional success marker."""
    value = _read_optional_scalar(group, "success")
    if value is None:
        return None
    if not isinstance(value, bool | np.bool_):
        raise ValueError(f"Optional 'success' metadata must be boolean, got {type(value).__name__}.")
    return bool(value)


def select_episodes(files: list[pathlib.Path], episodes: list[int] | None) -> list[pathlib.Path]:
    """Select requested episode indices while preserving natural file order."""
    if not episodes:
        return files
    requested = set(episodes)
    selected = [path for path in files if int(path.stem.split("_")[-1]) in requested]
    found = {int(path.stem.split("_")[-1]) for path in selected}
    if missing := sorted(requested - found):
        raise FileNotFoundError(f"Requested episode indices not found: {missing}.")
    return selected


def _require_dataset(group: h5py.File, key: str, path: pathlib.Path) -> h5py.Dataset:
    if key not in group:
        raise ValueError(f"{path}: missing HDF5 dataset '{key}'.")
    dataset = group[key]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{path}: '{key}' is not an HDF5 dataset.")
    return dataset


def validate_episode(
    path: pathlib.Path,
    *,
    require_success: bool = False,
    waist_action_tolerance: float = 1e-6,
    gripper_pair_tolerance: float = 1e-6,
) -> EpisodeSummary:
    """Validate one raw BPX episode against the agreed 23-dimensional schema."""
    path = pathlib.Path(path)
    with h5py.File(path, "r") as episode:
        qpos_dataset = _require_dataset(episode, QPOS_KEY, path)
        action_dataset = _require_dataset(episode, ACTION_KEY, path)

        if qpos_dataset.ndim != 2 or qpos_dataset.shape[1] != bpx_constants.BPX_RAW_DIM:
            raise ValueError(
                f"{path}: '{QPOS_KEY}' must have shape (T, {bpx_constants.BPX_RAW_DIM}), " f"got {qpos_dataset.shape}."
            )
        if action_dataset.shape != qpos_dataset.shape:
            raise ValueError(
                f"{path}: '{ACTION_KEY}' shape {action_dataset.shape} must match "
                f"'{QPOS_KEY}' shape {qpos_dataset.shape}."
            )
        frames = qpos_dataset.shape[0]
        if frames < 2:
            raise ValueError(f"{path}: an episode must contain at least two frames, got {frames}.")
        if not np.issubdtype(qpos_dataset.dtype, np.floating):
            raise ValueError(f"{path}: qpos must be floating point, got {qpos_dataset.dtype}.")
        if not np.issubdtype(action_dataset.dtype, np.floating):
            raise ValueError(f"{path}: action must be floating point, got {action_dataset.dtype}.")

        qpos = np.asarray(qpos_dataset[:], dtype=np.float32)
        actions = np.asarray(action_dataset[:], dtype=np.float32)
        if not np.isfinite(qpos).all():
            raise ValueError(f"{path}: qpos contains NaN or infinity.")
        if not np.isfinite(actions).all():
            raise ValueError(f"{path}: action contains NaN or infinity.")

        max_waist = float(np.max(np.abs(actions[:, : bpx_constants.BPX_WAIST_DIM])))
        if max_waist > waist_action_tolerance:
            raise ValueError(
                f"{path}: fixed-waist action magnitude {max_waist:.8g} exceeds "
                f"tolerance {waist_action_tolerance:.8g}."
            )

        pair_errors = [
            float(np.max(np.abs(actions[:, first] - actions[:, second])))
            for first, second in bpx_constants.BPX_RAW_GRIPPER_PAIRS
        ]
        max_pair_error = max(pair_errors)
        if max_pair_error > gripper_pair_tolerance:
            raise ValueError(
                f"{path}: aligned gripper action-pair error {max_pair_error:.8g} exceeds "
                f"tolerance {gripper_pair_tolerance:.8g}."
            )

        for camera_name, key in IMAGE_DATASET_KEYS.items():
            image_dataset = _require_dataset(episode, key, path)
            expected_shape = (frames, *bpx_constants.BPX_SOURCE_IMAGE_SHAPE)
            if image_dataset.shape != expected_shape:
                raise ValueError(
                    f"{path}: camera '{camera_name}' must have shape {expected_shape}, " f"got {image_dataset.shape}."
                )
            if image_dataset.dtype != np.uint8:
                raise ValueError(f"{path}: camera '{camera_name}' must be uint8, got {image_dataset.dtype}.")

        task_value = _read_optional_scalar(episode, "task")
        task = None if task_value is None else str(task_value).strip()
        success = read_success(episode)
        if require_success and success is not True:
            raise ValueError(f"{path}: success-only conversion requires success=true metadata.")

    return EpisodeSummary(
        path=path,
        frames=frames,
        task=task,
        success=success,
        max_action_waist_abs=max_waist,
        max_action_gripper_pair_error=max_pair_error,
    )


def convert_color(image: np.ndarray, color_order: str) -> np.ndarray:
    """Return an RGB image from an RGB or BGR source array."""
    if color_order == "rgb":
        return np.asarray(image)
    if color_order == "bgr":
        return np.asarray(image)[..., ::-1]
    raise ValueError(f"color_order must be 'rgb' or 'bgr', got {color_order!r}.")
