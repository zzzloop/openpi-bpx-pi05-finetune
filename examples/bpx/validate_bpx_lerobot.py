"""Validate a converted BPX LeRobot v2.1 dataset against its cleaned HDF5 source."""

import pathlib
from typing import Any

import bpx_data
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi.policies import bpx_constants


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_rgb_uint8(value: Any) -> np.ndarray:
    image = _to_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Converted image must be three-dimensional, got {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Converted image must have three channels, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        if image.size and (image.min() < 0.0 or image.max() > 1.0):
            raise ValueError("Converted floating-point image is outside [0, 1].")
        image = np.rint(image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"Converted image must be uint8 or floating point, got {image.dtype}.")
    return image


def _list_column(table, key: str, dtype: np.dtype) -> np.ndarray:
    return np.asarray(table[key].combine_chunks().to_pylist(), dtype=dtype)


def _validate_metadata(dataset: LeRobotDataset) -> None:
    expected_images = {f"observation.images.{name}" for name in bpx_constants.BPX_IMAGE_KEYS}
    actual_images = set(dataset.meta.camera_keys)
    if actual_images != expected_images:
        raise ValueError(f"Expected converted cameras {sorted(expected_images)}, got {sorted(actual_images)}.")
    if f"observation.images.{bpx_constants.BPX_DROPPED_IMAGE_KEY}" in dataset.features:
        raise ValueError("Converted dataset unexpectedly contains the dropped right-eye camera.")
    if dataset.fps != bpx_constants.BPX_FPS:
        raise ValueError(f"Expected {bpx_constants.BPX_FPS} fps, got {dataset.fps}.")
    for key in ("observation.state", "action"):
        feature = dataset.features.get(key)
        if feature is None:
            raise ValueError(f"Converted dataset is missing '{key}'.")
        if tuple(feature["shape"]) != (bpx_constants.BPX_EFFECTIVE_DIM,):
            raise ValueError(f"'{key}' must be 18-D, got shape {feature['shape']}.")
        if tuple(feature["names"]) != bpx_constants.BPX_EFFECTIVE_JOINT_NAMES:
            raise ValueError(f"'{key}' names do not match the agreed BPX effective joint order.")


def main(
    raw_dir: pathlib.Path,
    repo_id: str = "bpx/bpx_blocks",
    *,
    episodes: list[int] | None = None,
    expected_episodes: int | None = None,
    expected_frames: int | None = None,
    max_image_mae: float = 12.0,
) -> None:
    """Compare every state/action and sampled RGB frames with the cleaned source."""
    source_files = bpx_data.select_episodes(bpx_data.discover_episodes(raw_dir), episodes)
    dataset_path = pathlib.Path(HF_LEROBOT_HOME) / repo_id
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Local LeRobot dataset not found at {dataset_path}.")
    dataset = LeRobotDataset(repo_id)
    _validate_metadata(dataset)

    source_summaries = [bpx_data.validate_episode(path) for path in source_files]
    source_frames = sum(summary.frames for summary in source_summaries)
    required_episodes = len(source_files) if expected_episodes is None else expected_episodes
    required_frames = source_frames if expected_frames is None else expected_frames
    if dataset.num_episodes != required_episodes or dataset.num_frames != required_frames:
        raise ValueError(
            f"Converted counts are episodes={dataset.num_episodes}, frames={dataset.num_frames}; "
            f"expected episodes={required_episodes}, frames={required_frames}."
        )
    if source_frames != dataset.num_frames:
        raise ValueError(f"Source has {source_frames} frames but converted dataset has {dataset.num_frames}.")

    global_index = 0
    image_checks = 0
    for converted_episode, (source_path, summary) in enumerate(zip(source_files, source_summaries, strict=True)):
        parquet_path = dataset.root / dataset.meta.get_data_file_path(converted_episode)
        table = pq.read_table(
            parquet_path,
            columns=["observation.state", "action", "frame_index", "episode_index", "index", "timestamp", "task_index"],
        )
        if table.num_rows != summary.frames:
            raise ValueError(
                f"{source_path.name}: converted episode has {table.num_rows} rows, expected {summary.frames}."
            )

        with h5py.File(source_path, "r") as source:
            expected_state = bpx_constants.to_effective_joints(
                np.asarray(source[bpx_data.QPOS_KEY][:], dtype=np.float32)
            )
            expected_action = bpx_constants.to_effective_joints(
                np.asarray(source[bpx_data.ACTION_KEY][:], dtype=np.float32)
            )
            converted_state = _list_column(table, "observation.state", np.float32)
            converted_action = _list_column(table, "action", np.float32)
            np.testing.assert_array_equal(
                converted_state, expected_state, err_msg=f"{source_path.name}: state mismatch"
            )
            np.testing.assert_array_equal(
                converted_action, expected_action, err_msg=f"{source_path.name}: action mismatch"
            )

            frame_indices = _list_column(table, "frame_index", np.int64)
            episode_indices = _list_column(table, "episode_index", np.int64)
            absolute_indices = _list_column(table, "index", np.int64)
            timestamps = _list_column(table, "timestamp", np.float32)
            np.testing.assert_array_equal(frame_indices, np.arange(summary.frames))
            np.testing.assert_array_equal(episode_indices, np.full(summary.frames, converted_episode))
            np.testing.assert_array_equal(absolute_indices, np.arange(global_index, global_index + summary.frames))
            np.testing.assert_allclose(timestamps, np.arange(summary.frames) / bpx_constants.BPX_FPS, atol=1e-4)

            task = bpx_data.read_task(source, bpx_constants.BPX_DEFAULT_PROMPT)
            task_indices = _list_column(table, "task_index", np.int64)
            if any(dataset.meta.tasks[int(task_index)] != task for task_index in np.unique(task_indices)):
                raise ValueError(f"{source_path.name}: converted task prompt does not match the HDF5 source.")

            for frame_index in sorted({0, summary.frames // 2, summary.frames - 1}):
                converted_item = dataset[global_index + frame_index]
                if converted_item["task"] != task:
                    raise ValueError(f"{source_path.name}: decoded task prompt mismatch at frame {frame_index}.")
                for camera in bpx_constants.BPX_IMAGE_KEYS:
                    key = f"observation.images.{camera}"
                    converted_rgb = _to_rgb_uint8(converted_item[key])
                    source_rgb = np.asarray(source[bpx_data.IMAGE_DATASET_KEYS[camera]][frame_index])
                    if converted_rgb.shape != source_rgb.shape:
                        raise ValueError(f"{source_path.name}: {camera} shape mismatch at frame {frame_index}.")
                    rgb_mae = float(np.mean(np.abs(converted_rgb.astype(np.float32) - source_rgb.astype(np.float32))))
                    bgr_mae = float(
                        np.mean(np.abs(converted_rgb.astype(np.float32) - source_rgb[..., ::-1].astype(np.float32)))
                    )
                    if rgb_mae > max_image_mae or rgb_mae > bgr_mae + 1e-6:
                        raise ValueError(
                            f"{source_path.name}: {camera} frame {frame_index} is not consistent with source RGB "
                            f"(RGB MAE={rgb_mae:.3f}, BGR MAE={bgr_mae:.3f})."
                        )
                    image_checks += 1
        global_index += summary.frames

    print(
        f"PASS: repo_id={repo_id}, episodes={dataset.num_episodes}, frames={dataset.num_frames}, "
        f"numeric_frames_checked={global_index}, RGB_samples_checked={image_checks}"
    )


if __name__ == "__main__":
    tyro.cli(main)
