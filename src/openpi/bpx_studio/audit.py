"""Parallel structural and numeric audit for MP4-backed LeRobot v2.1 datasets."""

from __future__ import annotations

import argparse
from concurrent import futures
import json
import pathlib
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq


def _json_lines(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _audit_parquet(path: pathlib.Path, numeric_keys: tuple[str, ...]) -> dict[str, Any]:
    columns = [*numeric_keys, "episode_index", "frame_index", "timestamp", "task_index"]
    table = pq.read_table(path, columns=columns)
    if table.num_rows < 1:
        raise ValueError(f"空 episode parquet: {path}")
    report: dict[str, Any] = {"path": str(path), "frames": table.num_rows}
    for key in numeric_keys:
        values = np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {key} 必须是有限值 (T, D) 矩阵")
        report[f"{key}_dim"] = values.shape[1]
    episode_indices = np.asarray(table["episode_index"].combine_chunks().to_pylist())
    frame_indices = np.asarray(table["frame_index"].combine_chunks().to_pylist())
    timestamps = np.asarray(table["timestamp"].combine_chunks().to_pylist(), dtype=np.float64)
    if np.unique(episode_indices).size != 1:
        raise ValueError(f"{path}: 单个 parquet 中出现多个 episode_index")
    if not np.array_equal(frame_indices, np.arange(table.num_rows)):
        raise ValueError(f"{path}: frame_index 不连续")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{path}: timestamp 不是严格递增")
    report["episode_index"] = int(episode_indices[0])
    indices, counts = np.unique(
        np.asarray(table["task_index"].combine_chunks().to_pylist(), dtype=np.int64), return_counts=True
    )
    report["task_frames"] = {str(int(i)): int(n) for i, n in zip(indices, counts, strict=True)}
    return report


def _video_path(root: pathlib.Path, info: dict[str, Any], episode_index: int, key: str) -> pathlib.Path:
    template = info.get("video_path")
    if not template:
        raise ValueError("meta/info.json 缺少 video_path 模板")
    chunk_size = int(info.get("chunks_size", 1000))
    return root / template.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
        video_key=key,
    )


def _audit_video(arguments: tuple[pathlib.Path, int, int, int, float, str]) -> None:
    path, expected_frames, height, width, fps, mode = arguments
    if not path.is_file():
        raise FileNotFoundError(path)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: 不包含视频流")
        stream = container.streams.video[0]
        if (stream.height, stream.width) != (height, width):
            raise ValueError(f"{path}: 视频尺寸 {(stream.height, stream.width)} != {(height, width)}")
        rate = float(stream.average_rate) if stream.average_rate else 0.0
        if rate and not np.isclose(rate, fps, rtol=0.02, atol=0.05):
            raise ValueError(f"{path}: fps {rate:.6g} != {fps:.6g}")
        if mode == "full":
            decoded = sum(1 for _ in container.decode(video=0))
            if decoded != expected_frames:
                raise ValueError(f"{path}: 解码帧数 {decoded} != {expected_frames}")


def audit_dataset(root: pathlib.Path, workers: int, *, video_check: str = "headers") -> dict[str, Any]:
    if video_check not in {"none", "headers", "full"}:
        raise ValueError("video_check 只能是 none、headers 或 full")
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    tasks_path = root / "meta" / "tasks.jsonl"
    for path in (info_path, episodes_path, tasks_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = _json_lines(episodes_path)
    tasks = _json_lines(tasks_path)
    features = info.get("features", {})
    fps = float(info.get("fps", 0))
    if not str(info.get("robot_type", "")).strip() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("robot_type 必须非空且 fps 必须为正数")
    numeric_keys = ("observation.state", "action")
    for key in numeric_keys:
        if key not in features:
            raise ValueError(f"缺少必要特征: {key}")
    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if len(parquet_files) != len(episodes):
        raise ValueError(f"Parquet/episode 数量不一致: {len(parquet_files)} != {len(episodes)}")
    with futures.ProcessPoolExecutor(max_workers=workers) as executor:
        reports = []
        for index, report in enumerate(
            executor.map(_audit_parquet, parquet_files, [numeric_keys] * len(parquet_files)), start=1
        ):
            reports.append(report)
            print(f"AUDITED PARQUET {index}/{len(parquet_files)}", flush=True)
    total_frames = sum(item["frames"] for item in reports)
    if total_frames != int(info.get("total_frames", -1)) or len(episodes) != int(info.get("total_episodes", -1)):
        raise ValueError("info.json 的 episode/frame 总数与实际数据不一致")
    if sorted(item["episode_index"] for item in reports) != list(range(len(episodes))):
        raise ValueError("episode_index 必须从 0 连续且唯一")
    dimensions = {key: sorted({item[f"{key}_dim"] for item in reports}) for key in numeric_keys}
    if any(len(values) != 1 for values in dimensions.values()):
        raise ValueError(f"episode 间数值维度不一致: {dimensions}")
    known_tasks = {int(item["task_index"]): str(item["task"]) for item in tasks}
    task_frames: dict[str, int] = {}
    for report in reports:
        for raw_index, count in report["task_frames"].items():
            index = int(raw_index)
            if index not in known_tasks:
                raise ValueError(f"未知 task_index: {index}")
            task = known_tasks[index]
            task_frames[task] = task_frames.get(task, 0) + count
    video_keys = sorted(key for key, feature in features.items() if feature.get("dtype") == "video")
    frames_by_episode = {item["episode_index"]: item["frames"] for item in reports}
    video_jobs = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        frame_count = frames_by_episode[episode_index]
        if int(episode.get("length", frame_count)) != frame_count:
            raise ValueError(f"episode {episode_index}: 元数据 length 与 parquet 不一致")
        for key in video_keys:
            shape = list(features[key].get("shape", []))
            if len(shape) != 3 or shape[2] != 3:
                raise ValueError(f"{key}: 视频 shape 必须为 [height, width, 3]")
            video_jobs.append(
                (_video_path(root, info, episode_index, key), frame_count, shape[0], shape[1], fps, video_check)
            )
    expected = {job[0].resolve() for job in video_jobs}
    actual = {path.resolve() for path in (root / "videos").rglob("*.mp4")} if video_keys else set()
    if actual != expected:
        raise ValueError(f"MP4 路径集合不一致: missing={len(expected - actual)}, extra={len(actual - expected)}")
    if video_check != "none" and video_jobs:
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for index, _ in enumerate(executor.map(_audit_video, video_jobs), start=1):
                print(f"AUDITED VIDEO {index}/{len(video_jobs)}", flush=True)
    return {
        "root": str(root),
        "robot_type": info["robot_type"],
        "episodes": len(episodes),
        "frames": total_frames,
        "tasks": len(tasks),
        "task_frames": task_frames,
        "numeric_dimensions": {key: values[0] for key, values in dimensions.items()},
        "fps": fps,
        "video_files": len(actual),
        "video_check": video_check,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--video-check", choices=("none", "headers", "full"), default="headers")
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    report = audit_dataset(args.dataset_root.resolve(), args.workers, video_check=args.video_check)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    print("DATASET AUDIT PASS")


if __name__ == "__main__":
    main()
