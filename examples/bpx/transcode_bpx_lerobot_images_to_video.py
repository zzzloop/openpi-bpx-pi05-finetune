#!/usr/bin/env python3
"""Transcode an image-in-Parquet BPX LeRobot v2.1 dataset to MP4-backed storage.

The source dataset is read only.  A separate output dataset is created and
validated before the operator may replace or delete the source directory.
This is useful for BPX datasets created by the earlier converter, whose camera
features were explicitly declared as ``dtype=image`` even though
``use_videos=True`` was passed to LeRobot.
"""

from __future__ import annotations

from concurrent import futures
import copy
import pathlib
import shutil
from typing import Any

import convert_bpx_data_to_lerobot as bpx_converter
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import encode_video_frames
import numpy as np
import pyarrow.parquet as pq
import tyro
from validate_bpx_lerobot import _list_column
from validate_bpx_lerobot import _to_rgb_uint8
from validate_bpx_lerobot import _validate_metadata

_AUTOMATIC_FEATURES = {"index", "episode_index", "frame_index", "timestamp", "task_index"}
_NUMERIC_FEATURES = ("observation.state", "action")


class _ParallelVideoLeRobotDataset(LeRobotDataset):
    """LeRobot v2.1 writer that encodes independent camera streams concurrently."""

    video_encode_workers = 1

    def encode_episode_videos(self, episode_index: int) -> dict[str, str]:
        video_keys = list(self.meta.video_keys)

        def encode_one(key: str) -> tuple[str, str]:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            if not video_path.is_file():
                image_dir = self._get_image_file_path(
                    episode_index=episode_index,
                    image_key=key,
                    frame_index=0,
                ).parent
                encode_video_frames(image_dir, video_path, self.fps, overwrite=True)
            return key, str(video_path)

        workers = min(self.video_encode_workers, len(video_keys))
        if workers <= 1:
            return dict(encode_one(key) for key in video_keys)
        with futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bpx-video") as executor:
            return dict(executor.map(encode_one, video_keys))


def _tree_size(path: pathlib.Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _sample_offsets(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError("Episode length must be positive")
    if count <= 0:
        return []
    return sorted({int(value) for value in np.linspace(0, length - 1, min(count, length))})


def _output_features(source: LeRobotDataset) -> dict[str, dict[str, Any]]:
    features = copy.deepcopy(source.features)
    for key in source.meta.camera_keys:
        features[key]["dtype"] = "video"
        # Video information is populated from the first encoded episode.
        features[key].pop("info", None)
    return features


def _validate_numeric_episode(source: LeRobotDataset, output: LeRobotDataset, episode_index: int) -> None:
    columns = [*_NUMERIC_FEATURES, "frame_index", "episode_index", "index", "timestamp", "task_index"]
    source_table = pq.read_table(source.root / source.meta.get_data_file_path(episode_index), columns=columns)
    output_table = pq.read_table(output.root / output.meta.get_data_file_path(episode_index), columns=columns)
    if source_table.num_rows != output_table.num_rows:
        raise ValueError(f"Episode {episode_index}: output row count differs from source")
    for key in _NUMERIC_FEATURES:
        np.testing.assert_array_equal(
            _list_column(output_table, key, np.float32),
            _list_column(source_table, key, np.float32),
            err_msg=f"Episode {episode_index}: {key} changed during transcoding",
        )
    for key, dtype in (
        ("frame_index", np.int64),
        ("episode_index", np.int64),
        ("index", np.int64),
        ("task_index", np.int64),
    ):
        np.testing.assert_array_equal(
            _list_column(output_table, key, dtype),
            _list_column(source_table, key, dtype),
            err_msg=f"Episode {episode_index}: {key} changed during transcoding",
        )
    np.testing.assert_allclose(
        _list_column(output_table, "timestamp", np.float32),
        _list_column(source_table, "timestamp", np.float32),
        rtol=0,
        atol=1e-7,
        err_msg=f"Episode {episode_index}: timestamp changed during transcoding",
    )


def main(
    source_repo_id: str,
    output_repo_id: str,
    *,
    overwrite: bool = False,
    image_writer_threads: int = 16,
    video_encode_workers: int = 3,
    episode_limit: int | None = None,
    validation_samples_per_episode: int = 3,
    max_image_mae: float = 12.0,
) -> None:
    """Create and validate a video-backed copy of one local BPX dataset."""
    if source_repo_id == output_repo_id:
        raise ValueError("source_repo_id and output_repo_id must differ; in-place transcoding is forbidden")
    if image_writer_threads < 1:
        raise ValueError("image_writer_threads must be at least one")
    if video_encode_workers < 1:
        raise ValueError("video_encode_workers must be at least one")
    if episode_limit is not None and episode_limit < 1:
        raise ValueError("episode_limit must be at least one when provided")
    if validation_samples_per_episode < 0:
        raise ValueError("validation_samples_per_episode cannot be negative")
    if max_image_mae < 0:
        raise ValueError("max_image_mae cannot be negative")

    source_path = bpx_converter.resolve_output_path(source_repo_id)
    output_path = bpx_converter.resolve_output_path(output_repo_id)
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_path}")
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_path}; inspect it or pass --overwrite")
        shutil.rmtree(output_path)

    source = LeRobotDataset(source_repo_id, root=source_path)
    _validate_metadata(source)
    if set(source.meta.image_keys) != set(source.meta.camera_keys):
        raise ValueError("Source must store every BPX camera as dtype=image")
    if source.meta.video_keys:
        raise ValueError("Source already contains video-backed camera features")
    episode_count = source.num_episodes if episode_limit is None else min(episode_limit, source.num_episodes)

    output = _ParallelVideoLeRobotDataset.create(
        repo_id=output_repo_id,
        root=output_path,
        robot_type=source.meta.robot_type,
        fps=source.fps,
        features=_output_features(source),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=image_writer_threads,
    )
    output.video_encode_workers = video_encode_workers

    writable_keys = [key for key in source.features if key not in _AUTOMATIC_FEATURES]
    source_global_index = 0
    try:
        for episode_index in range(episode_count):
            episode_start = int(source.episode_data_index["from"][episode_index])
            episode_end = int(source.episode_data_index["to"][episode_index])
            episode_length = episode_end - episode_start
            if episode_start != source_global_index:
                raise ValueError("Source episode index is not contiguous")

            sample_offsets = _sample_offsets(episode_length, validation_samples_per_episode)
            reference_images: dict[int, dict[str, np.ndarray]] = {}
            reference_tasks: dict[int, str] = {}
            for offset in range(episode_length):
                item = source[episode_start + offset]
                frame = {key: item[key] for key in writable_keys if key not in source.meta.camera_keys}
                for camera_key in source.meta.camera_keys:
                    frame[camera_key] = _to_rgb_uint8(item[camera_key])
                frame["task"] = item["task"]
                output.add_frame(frame)
                if offset in sample_offsets:
                    reference_images[offset] = {key: np.array(frame[key], copy=True) for key in source.meta.camera_keys}
                    reference_tasks[offset] = item["task"]

            output.save_episode()
            _validate_numeric_episode(source, output, episode_index)
            for offset in sample_offsets:
                converted = output[episode_start + offset]
                if converted["task"] != reference_tasks[offset]:
                    raise ValueError(f"Episode {episode_index}, frame {offset}: task changed during transcoding")
                for camera_key in source.meta.camera_keys:
                    decoded = _to_rgb_uint8(converted[camera_key])
                    expected = reference_images[offset][camera_key]
                    mae = float(np.mean(np.abs(decoded.astype(np.float32) - expected.astype(np.float32))))
                    if mae > max_image_mae:
                        raise ValueError(
                            f"Episode {episode_index}, frame {offset}, {camera_key}: "
                            f"video MAE {mae:.3f} exceeds {max_image_mae:.3f}"
                        )
            source_global_index = episode_end
            print(
                f"TRANSCODED {episode_index + 1}/{episode_count}: "
                f"frames={episode_length}, sampled={len(sample_offsets)}",
                flush=True,
            )
    finally:
        output.stop_image_writer()

    reloaded = LeRobotDataset(output_repo_id, root=output_path)
    _validate_metadata(reloaded)
    if reloaded.num_episodes != episode_count or reloaded.num_frames != source_global_index:
        raise ValueError("Reloaded output episode/frame totals differ from source")
    if set(reloaded.meta.video_keys) != set(source.meta.camera_keys) or reloaded.meta.image_keys:
        raise ValueError("Reloaded output is not fully video-backed")
    expected_videos = episode_count * len(source.meta.camera_keys)
    actual_videos = sum(1 for _ in output_path.rglob("*.mp4"))
    if actual_videos != expected_videos:
        raise ValueError(f"Expected {expected_videos} MP4 files, found {actual_videos}")

    source_bytes = _tree_size(source_path)
    output_bytes = _tree_size(output_path)
    ratio = output_bytes / source_bytes if source_bytes else float("nan")
    print(
        "TRANSCODE PASS: "
        f"episodes={reloaded.num_episodes}, frames={reloaded.num_frames}, videos={actual_videos}, "
        f"source_gib={source_bytes / 2**30:.2f}, output_gib={output_bytes / 2**30:.2f}, ratio={ratio:.3f}"
    )


if __name__ == "__main__":
    tyro.cli(main)
