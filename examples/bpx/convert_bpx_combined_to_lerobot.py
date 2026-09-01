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


def _create_or_resume_dataset(
    repo_id: str, records: list[dict], output_path: pathlib.Path
) -> tuple[LeRobotDataset, list[dict]]:
    legacy_prefix = []
    for record in records:
        if record["source_kind"] != "lerobot":
            break
        legacy_prefix.append(record)
    if legacy_prefix:
        if len(legacy_prefix) != 59 or [record["source_index"] for record in legacy_prefix] != list(range(59)):
            raise ValueError("The legacy LeRobot prefix must contain episodes 0..58")
        source_roots = {pathlib.Path(record["path"]).resolve() for record in legacy_prefix}
        if len(source_roots) != 1:
            raise ValueError("Legacy LeRobot records must reference one dataset root")
        legacy_root = source_roots.pop()
        print(f"Copying validated legacy LeRobot dataset without re-encoding: {legacy_root} -> {output_path}")
        shutil.copytree(legacy_root, output_path)
        dataset = LeRobotDataset(repo_id, root=output_path)
        if dataset.num_episodes != 59 or dataset.num_frames != 67_455:
            raise ValueError("Copied legacy LeRobot dataset does not have the expected 59/67,455 contract")
        remaining = records[len(legacy_prefix) :]
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="bpx20260310",
            fps=bpx_constants.BPX_FPS,
            features=legacy_converter.bpx_features(),
            use_videos=True,
            image_writer_threads=8,
            image_writer_processes=0,
        )
        remaining = records
    if dataset.image_writer is None:
        dataset.start_image_writer(num_processes=0, num_threads=8)
    return dataset, remaining


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

    if any(record["source_kind"] == "lerobot" for record in records[59:]):
        raise ValueError("Legacy LeRobot records are only allowed as the first 59 manifest entries")
    dataset, records_to_convert = _create_or_resume_dataset(repo_id, records, output_path)
    total_frames = dataset.num_frames
    for record in records_to_convert:
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
            f"Saved output={dataset.num_episodes - 1} source={record['episode_index']} kind={record['source_kind']} "
            f"frames={frame_count} task={record['task']!r}",
            flush=True,
        )
    dataset.stop_image_writer()
    if dataset.num_episodes != len(records) or total_frames != sum(record["frames"] for record in records):
        raise ValueError("Converted dataset totals do not match the selected manifest records")
    print(f"CONVERSION PASS: episodes={dataset.num_episodes}, frames={total_frames}, output={output_path}")


if __name__ == "__main__":
    tyro.cli(main)
