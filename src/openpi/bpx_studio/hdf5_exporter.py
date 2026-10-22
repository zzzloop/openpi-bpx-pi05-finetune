"""Read-only LeRobot v2.1 image/video export, preserving source frame order."""

from __future__ import annotations

import argparse
from concurrent import futures
import hashlib
import io
import json
import os
import pathlib
import shutil
from typing import Any

import av
import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


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


def _write_video(
    images, name: str, path: pathlib.Path, frames: int, dataset_kwargs: dict[str, Any], shape: tuple[int, ...]
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = None
    count = 0
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != shape:
                raise ValueError(f"{path}: 视频尺寸 {rgb.shape} != 声明尺寸 {shape}")
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


def _decode_image(cell: Any, source: pathlib.Path) -> np.ndarray:
    """Decode Hugging Face Image's {bytes, path} struct without ML imports."""
    if not isinstance(cell, dict):
        raise ValueError("image 列必须是包含 bytes/path 的图像结构")
    encoded = cell.get("bytes")
    if encoded is not None:
        image_source = io.BytesIO(encoded)
    elif cell.get("path"):
        path = (source / cell["path"]).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError(f"图像路径不在源数据集内: {path}")
        image_source = path
    else:
        raise ValueError("image 单元格的 bytes 与 path 均为空")
    with Image.open(image_source) as image:
        # Materialize while the image file is open; never apply BGR swapping,
        # geometric transforms, or lossy re-encoding to an RGB dataset.
        return np.array(image.convert("RGB"), dtype=np.uint8)


def _write_inline_images(images, name, column, source, shape, dataset_kwargs) -> None:
    dataset = images.create_dataset(
        name, shape=(len(column), *shape), dtype=np.uint8, chunks=(1, *shape), **dataset_kwargs
    )
    for index, cell in enumerate(column):
        try:
            rgb = _decode_image(cell.as_py(), source)
            if rgb.shape != shape:
                raise ValueError(f"尺寸 {rgb.shape} != 声明尺寸 {shape}")
            dataset[index] = rgb
        except Exception as error:
            raise ValueError(f"{name} frame={index}: 图像解码失败: {error}") from error


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
    camera_keys = sorted(
        key for key, feature in info["features"].items() if feature.get("dtype") in {"video", "image"}
    )
    # Retain floating precision; export is not a training transform.
    state = _array(table, "observation.state", None)
    action = _array(table, "action", None)
    for key, values in (("observation.state", state), ("action", action)):
        expected_shape = (table.num_rows, *info["features"][key]["shape"])
        dtype = np.dtype(info["features"][key]["dtype"])
        if values.shape != expected_shape or not np.issubdtype(dtype, np.floating):
            raise ValueError(f"{key}: 维度或浮点类型与元数据不符")
        value_type = getattr(table.schema.field(key).type, "value_type", None)
        if value_type is None or np.dtype(value_type.to_pandas_dtype()) != dtype:
            raise ValueError(f"{key}: Parquet 数值类型与声明不一致，拒绝隐式转换精度")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key}: 存在非有限数值")
    state = state.astype(info["features"]["observation.state"]["dtype"])
    action = action.astype(info["features"]["action"]["dtype"])
    timestamps = _array(table, "timestamp", np.float64)
    frame_indices = _array(table, "frame_index", np.int64)
    if not np.array_equal(frame_indices, np.arange(table.num_rows)):
        raise ValueError(f"{parquet_path}: frame_index 不连续，禁止猜测图像对齐关系")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{parquet_path}: timestamp 必须有限且严格递增")
    if int(episode.get("length", table.num_rows)) != table.num_rows:
        raise ValueError(f"{parquet_path}: episode length 与实际帧数不一致")
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
            task_table_json=json.dumps(tasks, ensure_ascii=False),
            source_episode_json=json.dumps(episode, ensure_ascii=False),
            image_color_order="RGB",
            features_json=json.dumps(info["features"], ensure_ascii=False),
            numeric_representation="source values; no normalization, delta conversion or joint expansion",
        )
        output.create_dataset("action", data=action, **kwargs)
        output.create_dataset("frame_index", data=frame_indices)
        if "task_index" in table.column_names:
            output.create_dataset("task_index", data=_array(table, "task_index", np.int64))
        observations = output.create_group("observations")
        observations.create_dataset("qpos", data=state, **kwargs)
        observations.create_dataset("timestamp", data=timestamps)
        images = observations.create_group("images")
        for key in camera_keys:
            feature = info["features"][key]
            shape = tuple(feature.get("shape", []))
            if len(shape) != 3 or shape[2] != 3 or any(int(size) <= 0 for size in shape):
                raise ValueError(f"{key}: 图像 shape 必须为 [height, width, 3]")
            if feature["dtype"] == "image":
                if key not in table.column_names:
                    raise ValueError(f"Parquet 缺少 image 列: {key}")
                _write_inline_images(images, _camera_name(key), table[key], source, shape, kwargs)
            else:
                _write_video(
                    images, _camera_name(key), _video_path(source, info, episode_index, key),
                    table.num_rows, kwargs, shape,
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
    episode_indices: list[int] | None = None,
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
    records_by_index = {int(episode["episode_index"]): episode for episode in episodes}
    if len(records_by_index) != len(episodes):
        raise ValueError("episode 元数据存在重复索引")
    selected = set(records_by_index) if episode_indices is None else set(episode_indices)
    if not selected or selected - records_by_index.keys():
        raise ValueError("指定 episode 为空或不存在于数据集")
    if any(index < 0 for index in selected):
        raise ValueError("episode 索引不能为负数")
    # Map files to metadata by their actual scalar index, not JSONL ordering.
    selected_pairs = []
    seen = set()
    for parquet in parquet_paths:
        index_values = pq.read_table(parquet, columns=["episode_index"])["episode_index"].unique().to_pylist()
        if len(index_values) != 1 or int(index_values[0]) in seen:
            raise ValueError(f"episode_index 不唯一或重复: {parquet}")
        index = int(index_values[0])
        seen.add(index)
        if index not in records_by_index:
            raise ValueError(f"episode_index 缺少元数据: {index}")
        if index in selected:
            selected_pairs.append((parquet, records_by_index[index]))
    if selected - seen:
        raise ValueError(f"指定 episode 缺少 Parquet: {sorted(selected - seen)}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.partial")
    if staging.exists():
        raise FileExistsError(f"暂存目录已存在: {staging}")
    staging.mkdir(parents=True)
    try:
        jobs = [
            (source_root, staging, parquet, info, episode, tasks, compression, compression_level)
            for parquet, episode in selected_pairs
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
            "selected_episode_indices": sorted(selected),
            "image_color_order": "RGB",
            "source_features": info["features"],
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
    parser.add_argument("--episodes", nargs="+", type=int, help="仅导出指定原始 episode 索引；不重新编号")
    args = parser.parse_args()
    result = export_hdf5(
        args.source_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        workers=args.workers,
        compression=args.compression,
        compression_level=args.compression_level,
        episode_indices=args.episodes,
    )
    print(f"HDF5 EXPORT PASS: episodes={len(result['episodes'])}, frames={result['total_frames']}")


if __name__ == "__main__":
    main()
