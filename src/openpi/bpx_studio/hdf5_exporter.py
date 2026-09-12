"""Read-only LeRobot v2.1 video dataset export to episode HDF5 files."""

from __future__ import annotations

import argparse
from concurrent import futures
import hashlib
import json
import os
import pathlib
import shutil
from typing import Any

import av
import h5py
import numpy as np
import pyarrow.parquet as pq


def _json_lines(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _camera_name(feature_key: str) -> str:
    return feature_key.removeprefix("observation.images.").replace("/", "_")


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


def _write_video(images, name: str, path: pathlib.Path, frames: int, dataset_kwargs: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = None
    count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if dataset is None:
                dataset = images.create_dataset(
                    name,
                    shape=(frames, *rgb.shape),
                    dtype=np.uint8,
                    chunks=(1, *rgb.shape),
                    **dataset_kwargs,
                )
            if count >= frames:
                raise ValueError(f"{path}: 视频帧数超过数据帧数 {frames}")
            dataset[count] = rgb
            count += 1
    if count != frames:
        raise ValueError(f"{path}: 解码帧数 {count} != 数据帧数 {frames}")


def _array(table, key: str, dtype) -> np.ndarray:
    return np.asarray(table[key].combine_chunks().to_pylist(), dtype=dtype)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_episode(arguments: tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict, dict, dict, str, int]) -> dict:
    source, output_root, parquet_path, info, episode, tasks, compression, level = arguments
    table = pq.read_table(parquet_path)
    if table.num_rows < 1:
        raise ValueError(f"空 episode: {parquet_path}")
    indices = np.unique(_array(table, "episode_index", np.int64))
    if indices.size != 1:
        raise ValueError(f"{parquet_path}: episode_index 不唯一")
    episode_index = int(indices[0])
    if episode_index != int(episode["episode_index"]):
        raise ValueError(f"episode 元数据顺序与 parquet 不一致: {episode_index}")
    camera_keys = sorted(key for key, feature in info["features"].items() if feature.get("dtype") == "video")
    inline_images = sorted(key for key, feature in info["features"].items() if feature.get("dtype") == "image")
    if inline_images:
        raise ValueError(f"HDF5 导出要求 MP4-backed LeRobot; 发现 image 特征: {inline_images}")
    state = _array(table, "observation.state", np.float32)
    action = _array(table, "action", np.float32)
    timestamps = _array(table, "timestamp", np.float64)
    task_indices = np.unique(_array(table, "task_index", np.int64)) if "task_index" in table.column_names else []
    task_text = [tasks[int(index)] for index in task_indices]
    output_path = output_root / f"episode_{episode_index:06d}.hdf5"
    temporary = output_path.with_suffix(".hdf5.partial")
    kwargs: dict[str, Any] = {}
    if compression != "none":
        kwargs["compression"] = compression
        if compression == "gzip":
            kwargs["compression_opts"] = level
    with h5py.File(temporary, "w") as output:
        output.attrs.update(
            source_format="lerobot_v2.1",
            robot_type=str(info.get("robot_type", "")),
            episode_index=episode_index,
            fps=float(info["fps"]),
            task=json.dumps(task_text, ensure_ascii=False),
        )
        output.create_dataset("action", data=action, **kwargs)
        observations = output.create_group("observations")
        observations.create_dataset("qpos", data=state, **kwargs)
        observations.create_dataset("timestamp", data=timestamps)
        images = observations.create_group("images")
        for key in camera_keys:
            _write_video(
                images, _camera_name(key), _video_path(source, info, episode_index, key), table.num_rows, kwargs
            )
    os.replace(temporary, output_path)
    return {
        "episode_index": episode_index,
        "frames": table.num_rows,
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
        "tasks": task_text,
    }


def export_hdf5(
    source_root: pathlib.Path,
    output_root: pathlib.Path,
    *,
    workers: int,
    compression: str,
    compression_level: int,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在: {output_root}")
    if source_root.resolve() == output_root.resolve():
        raise ValueError("输出目录不能覆盖 LeRobot 权威数据集")
    if workers < 1 or compression not in {"gzip", "lzf", "none"} or not 0 <= compression_level <= 9:
        raise ValueError("导出参数无效")
    info = json.loads((source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = _json_lines(source_root / "meta" / "episodes.jsonl")
    tasks = {int(item["task_index"]): str(item["task"]) for item in _json_lines(source_root / "meta" / "tasks.jsonl")}
    parquet_paths = sorted((source_root / "data").rglob("*.parquet"))
    if len(parquet_paths) != len(episodes):
        raise ValueError(f"Parquet/episode 数量不一致: {len(parquet_paths)} != {len(episodes)}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.partial")
    if staging.exists():
        raise FileExistsError(f"暂存目录已存在: {staging}")
    staging.mkdir(parents=True)
    try:
        jobs = [
            (source_root, staging, parquet, info, episode, tasks, compression, compression_level)
            for parquet, episode in zip(parquet_paths, episodes, strict=True)
        ]
        with futures.ProcessPoolExecutor(max_workers=workers) as executor:
            records = []
            for index, result in enumerate(executor.map(_export_episode, jobs), start=1):
                result["path"] = str(output_root / pathlib.Path(result["path"]).relative_to(staging))
                records.append(result)
                print(
                    f"EXPORTED {index}/{len(jobs)} episode={result['episode_index']} frames={result['frames']}",
                    flush=True,
                )
        manifest = {
            "source": str(source_root),
            "source_robot_type": info.get("robot_type"),
            "format": "hdf5",
            "compression": compression,
            "episodes": records,
            "total_frames": sum(item["frames"] for item in records),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=pathlib.Path)
    parser.add_argument("output_root", type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    args = parser.parse_args()
    result = export_hdf5(
        args.source_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        workers=args.workers,
        compression=args.compression,
        compression_level=args.compression_level,
    )
    print(f"HDF5 EXPORT PASS: episodes={len(result['episodes'])}, frames={result['total_frames']}")


if __name__ == "__main__":
    main()
