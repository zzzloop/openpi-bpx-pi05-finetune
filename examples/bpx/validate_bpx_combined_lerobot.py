#!/usr/bin/env python3
"""Validate every numeric frame and sampled RGB frames in a combined BPX dataset."""

from __future__ import annotations

from functools import lru_cache
import json
import pathlib

import bpx_data
import bpx_rosbag_data
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
import tyro
from validate_bpx_lerobot import _list_column
from validate_bpx_lerobot import _to_rgb_uint8
from validate_bpx_lerobot import _validate_metadata

from openpi.policies import bpx_constants


@lru_cache(maxsize=2)
def _load_legacy_lerobot(root: str) -> LeRobotDataset:
    return LeRobotDataset("bpx/bpx_blocks", root=pathlib.Path(root))


def _source_arrays(record: dict) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, np.ndarray]]]:
    sample_indices = {0, record["frames"] // 2, record["frames"] - 1}
    if record["source_kind"] == "hdf5":
        with h5py.File(record["path"], "r") as source:
            states = np.asarray(
                bpx_constants.to_effective_joints(np.asarray(source[bpx_data.QPOS_KEY][:], dtype=np.float32))
            )
            actions = np.asarray(
                bpx_constants.to_effective_joints(np.asarray(source[bpx_data.ACTION_KEY][:], dtype=np.float32))
            )
            samples = {
                index: {
                    camera: np.asarray(source[bpx_data.IMAGE_DATASET_KEYS[camera]][index])
                    for camera in bpx_constants.BPX_IMAGE_KEYS
                }
                for index in sample_indices
            }
            return states, actions, samples
    if record["source_kind"] == "rosbag":
        states = []
        actions = []
        samples = {}
        for index, frame in enumerate(bpx_rosbag_data.iter_bag_frames(pathlib.Path(record["path"]))):
            states.append(frame.state)
            actions.append(frame.action)
            if index in sample_indices:
                samples[index] = frame.images
        return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32), samples
    if record["source_kind"] == "lerobot":
        dataset = _load_legacy_lerobot(record["path"])
        episode_index = int(record["source_index"])
        table = pq.read_table(
            dataset.root / dataset.meta.get_data_file_path(episode_index),
            columns=["observation.state", "action"],
        )
        states = _list_column(table, "observation.state", np.float32)
        actions = _list_column(table, "action", np.float32)
        episode_start = int(dataset.episode_data_index["from"][episode_index])
        samples = {}
        for frame_index in sample_indices:
            item = dataset[episode_start + frame_index]
            samples[frame_index] = {
                camera: _to_rgb_uint8(item[f"observation.images.{camera}"]) for camera in bpx_constants.BPX_IMAGE_KEYS
            }
        return states, actions, samples
    raise ValueError(f"Unknown source kind: {record['source_kind']}")


def main(
    manifest_path: pathlib.Path,
    repo_id: str = "bpx/bpx_blocks_combined",
    *,
    source_episodes: list[int] | None = None,
    max_image_mae: float = 12.0,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["episodes"]
    if source_episodes is not None:
        records = [records[index] for index in source_episodes]
    dataset_path = pathlib.Path(HF_LEROBOT_HOME) / repo_id
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"LeRobot dataset not found: {dataset_path}")
    dataset = LeRobotDataset(repo_id)
    _validate_metadata(dataset)
    expected_frames = sum(record["frames"] for record in records)
    if dataset.num_episodes != len(records) or dataset.num_frames != expected_frames:
        raise ValueError(
            f"Dataset counts ({dataset.num_episodes}, {dataset.num_frames}) != "
            f"manifest selection ({len(records)}, {expected_frames})"
        )

    global_index = 0
    image_checks = 0
    for output_episode, record in enumerate(records):
        table = pq.read_table(
            dataset.root / dataset.meta.get_data_file_path(output_episode),
            columns=["observation.state", "action", "frame_index", "episode_index", "index", "timestamp", "task_index"],
        )
        if table.num_rows != record["frames"]:
            raise ValueError(f"Output episode {output_episode} frame count differs from manifest")
        states, actions, samples = _source_arrays(record)
        np.testing.assert_array_equal(_list_column(table, "observation.state", np.float32), states)
        np.testing.assert_array_equal(_list_column(table, "action", np.float32), actions)
        np.testing.assert_array_equal(_list_column(table, "frame_index", np.int64), np.arange(record["frames"]))
        np.testing.assert_array_equal(
            _list_column(table, "episode_index", np.int64), np.full(record["frames"], output_episode)
        )
        np.testing.assert_array_equal(
            _list_column(table, "index", np.int64), np.arange(global_index, global_index + record["frames"])
        )
        np.testing.assert_allclose(
            _list_column(table, "timestamp", np.float32),
            np.arange(record["frames"]) / bpx_constants.BPX_FPS,
            atol=1e-4,
        )
        task_indices = np.unique(_list_column(table, "task_index", np.int64))
        if any(dataset.meta.tasks[int(index)] != record["task"] for index in task_indices):
            raise ValueError(f"Output episode {output_episode} has the wrong task prompt")
        for frame_index, camera_images in samples.items():
            item = dataset[global_index + frame_index]
            if item["task"] != record["task"]:
                raise ValueError(f"Decoded task mismatch at episode={output_episode}, frame={frame_index}")
            for camera, source_rgb in camera_images.items():
                converted = _to_rgb_uint8(item[f"observation.images.{camera}"])
                rgb_mae = float(np.mean(np.abs(converted.astype(np.float32) - source_rgb.astype(np.float32))))
                bgr_mae = float(
                    np.mean(np.abs(converted.astype(np.float32) - source_rgb[..., ::-1].astype(np.float32)))
                )
                if rgb_mae > max_image_mae or rgb_mae > bgr_mae + 1e-6:
                    raise ValueError(
                        f"RGB check failed episode={output_episode}, frame={frame_index}, camera={camera}, "
                        f"rgb_mae={rgb_mae:.3f}, bgr_mae={bgr_mae:.3f}"
                    )
                image_checks += 1
        global_index += record["frames"]
        print(f"VALIDATED {output_episode + 1:03d}/{len(records):03d}", flush=True)
    print(
        f"VALIDATION PASS: repo_id={repo_id}, episodes={len(records)}, frames={global_index}, "
        f"RGB_samples={image_checks}"
    )


if __name__ == "__main__":
    tyro.cli(main)
