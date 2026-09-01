#!/usr/bin/env python3
"""Convert the audited BPX HDF5 + ROS1-bag manifest to one LeRobot v2.1 dataset."""

from __future__ import annotations

import json
import pathlib
import shutil

import bpx_data
import bpx_rosbag_data
import convert_bpx_data_to_lerobot as legacy_converter
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

from openpi.policies import bpx_constants


def _load_manifest(path: pathlib.Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema_version')}")
    episodes = manifest.get("episodes", [])
    if len(episodes) != manifest.get("episode_count"):
        raise ValueError("Manifest episode count is inconsistent")
    if [item["episode_index"] for item in episodes] != list(range(len(episodes))):
        raise ValueError("Manifest episode indices must be contiguous from zero")
    return manifest


def _hdf5_frames(record: dict):
    path = pathlib.Path(record["path"])
    summary = bpx_data.validate_episode(path)
    if summary.frames != record["frames"]:
        raise ValueError(f"{path}: manifest frame count changed")
    with h5py.File(path, "r") as episode:
        states = np.asarray(
            bpx_constants.to_effective_joints(np.asarray(episode[bpx_data.QPOS_KEY][:], dtype=np.float32)),
            dtype=np.float32,
        )
        actions = np.asarray(
            bpx_constants.to_effective_joints(np.asarray(episode[bpx_data.ACTION_KEY][:], dtype=np.float32)),
            dtype=np.float32,
        )
        images = {
            camera: episode[bpx_data.IMAGE_DATASET_KEYS[camera]] for camera in ("left_eye", "left_wrist", "right_wrist")
        }
        for index in range(summary.frames):
            yield {
                "observation.images.left_eye": bpx_data.convert_color(images["left_eye"][index], "rgb"),
                "observation.images.left_wrist": bpx_data.convert_color(images["left_wrist"][index], "rgb"),
                "observation.images.right_wrist": bpx_data.convert_color(images["right_wrist"][index], "rgb"),
                "observation.state": states[index],
                "action": actions[index],
                "task": record["task"],
            }


def _rosbag_frames(record: dict):
    path = pathlib.Path(record["path"])
    count = 0
    for frame in bpx_rosbag_data.iter_bag_frames(path):
        count += 1
        yield {
            "observation.images.left_eye": frame.images["left_eye"],
            "observation.images.left_wrist": frame.images["left_wrist"],
            "observation.images.right_wrist": frame.images["right_wrist"],
            "observation.state": frame.state,
            "action": frame.action,
            "task": record["task"],
        }
    if count != record["frames"]:
        raise ValueError(f"{path}: decoded {count} frames, manifest says {record['frames']}")


def main(
    manifest_path: pathlib.Path,
    repo_id: str = "bpx/bpx_blocks_combined",
    *,
    source_episodes: list[int] | None = None,
    overwrite: bool = False,
) -> None:
    manifest = _load_manifest(manifest_path.resolve())
    records = manifest["episodes"]
    if source_episodes is not None:
        requested = list(dict.fromkeys(source_episodes))
        invalid = sorted(set(requested) - set(range(len(records))))
        if invalid:
            raise ValueError(f"Source episode indices are out of range: {invalid}")
        records = [records[index] for index in requested]
    output_path = legacy_converter.resolve_output_path(repo_id)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_path}; pass --overwrite only after checking the target")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="bpx20260310",
        fps=bpx_constants.BPX_FPS,
        features=legacy_converter.bpx_features(),
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
    )
    total_frames = 0
    for output_episode, record in enumerate(records):
        if record["source_kind"] == "hdf5":
            frames = _hdf5_frames(record)
        elif record["source_kind"] == "rosbag":
            frames = _rosbag_frames(record)
        else:
            raise ValueError(f"Unknown source kind: {record['source_kind']}")
        frame_count = 0
        for frame in frames:
            dataset.add_frame(frame)
            frame_count += 1
        if frame_count != record["frames"]:
            raise ValueError(f"Episode {record['episode_index']}: converted {frame_count}, expected {record['frames']}")
        dataset.save_episode()
        total_frames += frame_count
        print(
            f"Saved output={output_episode} source={record['episode_index']} kind={record['source_kind']} "
            f"frames={frame_count} task={record['task']!r}",
            flush=True,
        )
    print(f"CONVERSION PASS: episodes={len(records)}, frames={total_frames}, output={output_path}")


if __name__ == "__main__":
    tyro.cli(main)
