"""Bounded, resumable LeRobot v2.1 image-to-video conversion without ML imports.

One worker handles one episode, one camera at a time. Parquet image columns are
read in small batches, never materialized as a whole dataset or PNG cache.
"""

from __future__ import annotations

import argparse
from concurrent import futures
import contextlib
import copy
import faulthandler
from fractions import Fraction
import hashlib
import json
import multiprocessing
import os
import pathlib
import signal
import socket
import threading
import time


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: pathlib.Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(path: pathlib.Path) -> dict:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _inside(root: pathlib.Path, relative: str) -> pathlib.Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"路径超出数据集目录: {path}")
    return path


@contextlib.contextmanager
def _heartbeat(stage: str):
    stopped = threading.Event()
    started = time.monotonic()

    def emit():
        while not stopped.wait(15):
            _log(f"WAIT {stage}; elapsed={time.monotonic() - started:.0f}s")

    thread = threading.Thread(target=emit, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=1)


@contextlib.contextmanager
def _lock(work: pathlib.Path, resume: bool):
    # Keep the lock outside the directory we atomically rename on success.
    path = work.with_name(f"{work.name}.lock")
    if path.exists():
        previous = _json(path)
        if not resume or previous.get("host") != socket.gethostname():
            raise ValueError(f"转换目录被占用或需显式 --resume: {path}")
        try:
            os.kill(int(previous["pid"]), 0)
        except ProcessLookupError:
            path.unlink()  # Only our dead worker's small lock, never source data.
        else:
            raise ValueError(f"转换进程仍存活，禁止启动第二份: pid={previous['pid']}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _worker_init() -> None:
    # Limit nested pools before NumPy/Arrow are imported by spawn children.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def _strip_image_metadata(table, camera_keys):
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        document = json.loads(metadata[b"huggingface"])
        features = document.get("info", {}).get("features", {})
        for key in camera_keys:
            features.pop(key, None)
        metadata[b"huggingface"] = json.dumps(document).encode("utf-8")
    return table.replace_schema_metadata(metadata)


def _numeric_stats(table, features: dict) -> dict:
    import numpy as np

    result = {}
    for key in table.column_names:
        if features.get(key, {}).get("dtype") == "string":
            continue
        values = np.asarray(table[key].to_pylist())
        if not np.issubdtype(values.dtype, np.number) and values.dtype != np.bool_:
            continue
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key}: 存在非有限数值")
        # LeRobot scalar statistics have shape (1,), vector statistics (D,).
        keep = values.ndim == 1
        result[key] = {
            "min": np.min(values, axis=0, keepdims=keep).tolist(),
            "max": np.max(values, axis=0, keepdims=keep).tolist(),
            "mean": np.mean(values, axis=0, keepdims=keep).tolist(),
            "std": np.std(values, axis=0, keepdims=keep).tolist(),
            "count": [len(values)],
        }
    return result


def _encode_camera(job: dict, key: str, frames: int) -> tuple[dict, dict, dict]:
    import av
    import numpy as np
    import pyarrow.parquet as pq

    from openpi.bpx_studio.hdf5_exporter import _decode_image

    source = pathlib.Path(job["source"])
    work = pathlib.Path(job["work"])
    info, options = job["info"], job["options"]
    episode = int(job["episode"]["episode_index"])
    height, width, channels = info["features"][key]["shape"]
    if channels != 3 or height % 2 or width % 2:
        raise ValueError(f"{key}: H.264/yuv420p 需要偶数宽高与三通道，禁止自动缩放")
    video_relative = (
        f"videos/chunk-{episode // info.get('chunks_size', 1000):03d}/{key}/episode_{episode:06d}.mp4"
    )
    video_path = _inside(work, video_relative)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = video_path.with_suffix(".mp4.partial")
    rate = Fraction(str(info["fps"]))
    time_base = 1 / rate
    sample_offsets = set(np.linspace(0, frames - 1, min(3, frames), dtype=int).tolist())
    reference = {}
    count = 0
    last_log = time.monotonic()
    _log(f"EPISODE {episode} ENCODE {key}; frames={frames}, pid={os.getpid()}")
    parquet = pq.ParquetFile(job["parquet"])
    with av.open(str(temporary), mode="w", format="mp4") as container:
        stream = container.add_stream(
            "libx264", rate=rate,
            options={"crf": str(options["crf"]), "preset": "fast", "g": "2", "bf": "0"},
        )
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.codec_context.thread_count = options["encoder_threads"]
        stream.codec_context.time_base = time_base
        for batch in parquet.iter_batches(batch_size=options["batch_frames"], columns=[key], use_threads=False):
            for cell in batch.column(0):
                rgb = _decode_image(cell.as_py(), source)
                if rgb.shape != (height, width, 3):
                    raise ValueError(f"episode={episode}, {key}, frame={count}: 图像尺寸不符")
                if count in sample_offsets:
                    reference[count] = rgb.copy()
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts, frame.time_base = count, time_base
                for packet in stream.encode(frame):
                    container.mux(packet)
                count += 1
                if time.monotonic() - last_log >= 15:
                    _log(f"EPISODE {episode} {key}: encoded={count}/{frames}")
                    last_log = time.monotonic()
        for packet in stream.encode():
            container.mux(packet)
    if count != frames:
        raise ValueError(f"episode={episode}, {key}: 图像数 {count} != {frames}")
    _log(f"EPISODE {episode} VERIFY {key}; full decode + timestamps + RGB samples")
    count, maximum_mae = 0, 0.0
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    low, high = np.full(3, 255, dtype=np.uint8), np.zeros(3, dtype=np.uint8)
    with av.open(str(temporary)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if float(stream.average_rate) != float(rate):
            raise ValueError(f"{key}: 视频 FPS 与源数据不一致")
        video_info = {
            "video.height": height, "video.width": width, "video.fps": float(rate),
            "video.channels": 3, "video.codec": stream.codec_context.name,
            "video.pix_fmt": stream.pix_fmt, "video.is_depth_map": False, "has_audio": False,
        }
        for frame in container.decode(video=0):
            if frame.pts is None or abs(float(frame.pts * frame.time_base) - count / float(rate)) > 1e-4:
                raise ValueError(f"episode={episode}, {key}, frame={count}: 视频时间戳错位")
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (height, width, 3):
                raise ValueError(f"episode={episode}, {key}: 解码尺寸错误")
            if count in reference:
                mae = float(np.abs(rgb.astype(np.float32) - reference[count]).mean())
                maximum_mae = max(maximum_mae, mae)
                if mae > options["max_image_mae"]:
                    raise ValueError(f"episode={episode}, {key}, frame={count}: RGB MAE={mae:.3f} 超限")
            low = np.minimum(low, rgb.min(axis=(0, 1)))
            high = np.maximum(high, rgb.max(axis=(0, 1)))
            values = rgb.astype(np.float64)
            sums += values.sum(axis=(0, 1))
            squares += np.einsum("hwc,hwc->c", values, values)
            count += 1
    if count != frames:
        raise ValueError(f"episode={episode}, {key}: 视频解码帧数 {count} != {frames}")
    pixels = frames * height * width
    mean = sums / pixels
    stats = {
        "min": (low.astype(float) / 255).reshape(3, 1, 1).tolist(),
        "max": (high.astype(float) / 255).reshape(3, 1, 1).tolist(),
        "mean": (mean / 255).reshape(3, 1, 1).tolist(),
        "std": (np.sqrt(np.maximum(0, squares / pixels - mean * mean)) / 255).reshape(3, 1, 1).tolist(),
        "count": [frames],
    }
    os.replace(temporary, video_path)
    artifact = {"path": video_relative, "bytes": video_path.stat().st_size, "sha256": _hash(video_path)}
    return artifact, video_info, {"stats": stats, "max_sampled_rgb_mae": maximum_mae}


def _completed(job: dict) -> dict | None:
    work = pathlib.Path(job["work"])
    path = work / "conversion_records" / f"episode_{int(job['episode']['episode_index']):06d}.json"
    if not path.is_file():
        return None
    result = _json(path)
    if result.get("source_signature") != job["signature"]:
        return None
    for artifact in result.get("artifacts", []):
        file = _inside(work, artifact["path"])
        if not file.is_file() or file.stat().st_size != artifact["bytes"] or _hash(file) != artifact["sha256"]:
            return None
    return result if result.get("artifacts") else None


def _transcode_episode(job: dict) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    source_path = pathlib.Path(job["parquet"])
    if _signature(source_path) != job["signature"]:
        raise ValueError(f"源文件在转换前发生变化: {source_path}")
    restored = _completed(job)
    if restored is not None:
        _log(f"EPISODE {restored['episode_index']} RESUMED; output hashes verified")
        return restored
    info = job["info"]
    cameras = job["cameras"]
    episode = int(job["episode"]["episode_index"])
    _log(f"EPISODE {episode} READ numeric columns; pid={os.getpid()}")
    schema = pq.read_schema(source_path)
    columns = [key for key in schema.names if key not in cameras]
    table = pq.read_table(source_path, columns=columns, use_threads=False)
    frames = table.num_rows
    if frames < 1 or frames != int(job["episode"]["length"]):
        raise ValueError(f"episode={episode}: length 与 Parquet 不一致")
    for key in ("episode_index", "frame_index", "index", "timestamp", "task_index", "observation.state", "action"):
        if key not in table.column_names:
            raise ValueError(f"episode={episode}: 缺少 {key}")
    if table["episode_index"].unique().to_pylist() != [episode]:
        raise ValueError(f"episode={episode}: episode_index 错位")
    if table["frame_index"].to_pylist() != list(range(frames)):
        raise ValueError(f"episode={episode}: frame_index 不连续")
    if table["index"].to_pylist() != list(range(job["global_start"], job["global_start"] + frames)):
        raise ValueError(f"episode={episode}: 全局 index 不连续")
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=float)
    if not np.allclose(timestamps, np.arange(frames) / info["fps"], rtol=0, atol=1e-4):
        raise ValueError(f"episode={episode}: 源时间戳不满足固定 FPS；拒绝悄悄重采样")
    if set(table["task_index"].to_pylist()) - set(job["task_indices"]):
        raise ValueError(f"episode={episode}: 未知任务索引")
    stats = _numeric_stats(table, info["features"])
    for key in ("observation.state", "action"):
        values = np.asarray(table[key].to_pylist())
        if values.shape != (frames, *info["features"][key]["shape"]):
            raise ValueError(f"episode={episode}: {key} 维度不符")
        value_type = getattr(table.schema.field(key).type, "value_type", None)
        if value_type is None or np.dtype(value_type.to_pandas_dtype()) != np.dtype(info["features"][key]["dtype"]):
            raise ValueError(f"episode={episode}: {key} 浮点类型与声明不符")
    artifacts, camera_info, quality = [], {}, {}
    for key in cameras:
        artifact, camera_info[key], checked = _encode_camera(job, key, frames)
        artifacts.append(artifact)
        stats[key] = checked["stats"]
        quality[key] = checked["max_sampled_rgb_mae"]
    output = _inside(pathlib.Path(job["work"]), job["data_relative"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.partial")
    written = _strip_image_metadata(table, cameras)
    pq.write_table(written, temporary, compression="snappy")
    reloaded = pq.read_table(temporary, use_threads=False)
    if not table.equals(reloaded, check_metadata=False):
        raise ValueError(f"episode={episode}: 非图像列发生改变（包括夹爪、语言编号、时间戳）")
    if _signature(source_path) != job["signature"]:
        raise ValueError(f"源文件在转换期间发生变化: {source_path}")
    os.replace(temporary, output)
    artifacts.append({"path": job["data_relative"], "bytes": output.stat().st_size, "sha256": _hash(output)})
    result = {
        "episode_index": episode, "frames": frames, "stats": stats, "camera_info": camera_info,
        "sampled_rgb_mae": quality, "source_signature": job["signature"], "artifacts": artifacts,
    }
    record = pathlib.Path(job["work"]) / "conversion_records" / f"episode_{episode:06d}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    _write_json(record, result)
    _log(f"EPISODE {episode} DONE; frames={frames}, all non-image columns identical")
    return result


def transcode_dataset(
    source: pathlib.Path, output: pathlib.Path, *, workers: int = 32, encoder_threads: int = 2,
    batch_frames: int = 8, crf: int = 18, max_image_mae: float = 12, resume: bool = False,
    episode_limit: int | None = None,
) -> dict:
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    work = output.with_name(f".{output.name}.transcoding")
    _log(f"SOURCE {source}")
    _log(f"OUTPUT {output}; staging={work}")
    if source.is_relative_to(output) or output.is_relative_to(source) or work == source:
        raise ValueError("源目录、输出目录及暂存目录不能相同或互相嵌套")
    if output.exists():
        raise FileExistsError(f"最终输出已存在，拒绝覆盖: {output}")
    if not 1 <= workers <= 256 or not 1 <= encoder_threads <= 32 or not 1 <= batch_frames <= 128:
        raise ValueError("workers=1..256, encoder_threads=1..32, batch_frames=1..128")
    if not 0 <= crf <= 51 or not 0 <= max_image_mae < float("inf"):
        raise ValueError("CRF 必须为 0..51，RGB MAE 阈值必须有限且非负")
    _log("PREFLIGHT read local metadata only; no Hub/network, no torch/JAX/datasets")
    info = _json(source / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("仅支持每 episode 一个 Parquet 的 LeRobot v2.1")
    episodes = sorted(_jsonl(source / "meta/episodes.jsonl"), key=lambda row: int(row["episode_index"]))
    tasks = _jsonl(source / "meta/tasks.jsonl")
    task_indices = [int(task["task_index"]) for task in tasks]
    if len(set(task_indices)) != len(tasks) or any(index < 0 for index in task_indices):
        raise ValueError("任务表包含重复或负数索引")
    cameras = sorted(key for key, feature in info["features"].items() if feature.get("dtype") == "image")
    if not cameras or any(feature.get("dtype") == "video" for feature in info["features"].values()):
        raise ValueError("源必须是 image-backed 数据集；已有 video 特征时拒绝重复压缩")
    if [int(row["episode_index"]) for row in episodes] != list(range(len(episodes))):
        raise ValueError("episode 编号必须从 0 连续且唯一")
    if len(episodes) != info.get("total_episodes") or sum(int(row["length"]) for row in episodes) != info.get("total_frames"):
        raise ValueError("episode/frame 总数与 info.json 不一致")
    if not isinstance(info.get("chunks_size", 1000), int) or info.get("chunks_size", 1000) <= 0:
        raise ValueError("chunks_size 必须为正整数")
    if not isinstance(info.get("fps"), (int, float)) or not 0 < info["fps"] < float("inf"):
        raise ValueError("FPS 必须为有限正数")
    for key in cameras:
        shape = info["features"][key].get("shape", [])
        if (len(shape) != 3 or shape[2] != 3
                or any(not isinstance(size, int) or size <= 0 for size in shape)
                or shape[0] % 2 or shape[1] % 2):
            raise ValueError(f"{key}: 图像 shape 必须为正整数 [H,W,3]，且 H/W 为偶数；不会自动缩放")
    if episode_limit is not None:
        if episode_limit < 1:
            raise ValueError("episode_limit 必须为正数")
        episodes = episodes[:episode_limit]
    if not episodes:
        raise ValueError("数据集没有 episode")
    options = {"encoder_threads": encoder_threads, "batch_frames": batch_frames, "crf": crf, "max_image_mae": max_image_mae}
    metadata_hash = {
        str(path.relative_to(source)): _hash(path) for path in sorted((source / "meta").rglob("*")) if path.is_file()
    }
    identity = {"version": 1, "source": str(source), "metadata_sha256": metadata_hash,
                "options": options, "episode_indices": [row["episode_index"] for row in episodes]}
    if work.exists():
        if not resume or not (work / "conversion_identity.json").is_file():
            raise FileExistsError(f"暂存目录存在；确认同一任务后使用 --resume: {work}")
        if _json(work / "conversion_identity.json") != identity:
            raise ValueError("源元数据或编码设置已变化，不能复用此暂存目录")
    else:
        work.mkdir(parents=True)
        _write_json(work / "conversion_identity.json", identity)
    with _lock(work, resume):
        _log(f"LOAD lightweight Arrow/PyAV; workers={workers}, encoder_threads={encoder_threads}, batch_frames={batch_frames}")
        with _heartbeat("import Arrow/PyAV"):
            import av
            av.codec.Codec("libx264", "w")  # Fail before scheduling if encoder is unavailable.
        jobs, offset = [], 0
        for episode in episodes:
            index = int(episode["episode_index"])
            relative = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet").format(
                episode_chunk=index // info.get("chunks_size", 1000), episode_index=index,
            )
            parquet = _inside(source, relative)
            if not parquet.is_file():
                raise FileNotFoundError(f"本地源 Parquet 不存在（不会联网下载）: {parquet}")
            jobs.append({"source": str(source), "work": str(work), "parquet": str(parquet), "data_relative": relative,
                         "signature": _signature(parquet), "episode": episode, "global_start": offset,
                         "cameras": cameras, "info": info, "options": options,
                         "task_indices": task_indices})
            offset += int(episode["length"])
        _log(f"START {len(jobs)} episodes, {offset} frames, cameras={len(cameras)}; no PNG or Arrow dataset cache")
        results = []
        with futures.ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init,
        ) as pool:
            iterator = iter(jobs)
            pending = {pool.submit(_transcode_episode, job) for job in [next(iterator) for _ in range(min(workers, len(jobs)))]}
            while pending:
                done, pending = futures.wait(pending, timeout=15, return_when=futures.FIRST_COMPLETED)
                if not done:
                    _log(f"WAIT episode workers; completed={len(results)}/{len(jobs)}, active={len(pending)}")
                for task in done:
                    try:
                        results.append(task.result())
                    except BaseException:
                        _log("ERROR episode worker failed; stop scheduling, retain validated episodes for --resume")
                        for outstanding in pending:
                            outstanding.cancel()
                        raise
                    _log(f"TRANSCODED {len(results)}/{len(jobs)}; completed episode={results[-1]['episode_index']}")
                    job = next(iterator, None)
                    if job is not None:
                        pending.add(pool.submit(_transcode_episode, job))
        results.sort(key=lambda record: record["episode_index"])
        _log("FINALIZE metadata, language table and decoded-video statistics")
        (work / "meta").mkdir(exist_ok=True)
        import shutil

        for path in sorted((source / "meta").rglob("*")):
            relative = path.relative_to(source / "meta")
            if path.is_file() and str(relative) not in {"info.json", "episodes.jsonl", "episodes_stats.jsonl", "stats.json"}:
                destination = work / "meta" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        converted = copy.deepcopy(info)
        for key in cameras:
            converted["features"][key]["dtype"] = "video"
            converted["features"][key]["info"] = results[0]["camera_info"][key]
        converted.update(total_episodes=len(episodes), total_frames=offset, total_videos=len(episodes) * len(cameras),
                         total_chunks=(len(episodes) - 1) // info.get("chunks_size", 1000) + 1,
                         chunks_size=info.get("chunks_size", 1000),
                         data_path=info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"),
                         video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                         splits={"train": f"0:{len(episodes)}"})
        if len(episodes) == info["total_episodes"]:
            converted["splits"] = copy.deepcopy(info.get("splits", converted["splits"]))
        _write_json(work / "meta/info.json", converted)
        (work / "meta/episodes.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episodes), encoding="utf-8",
        )
        (work / "meta/episodes_stats.jsonl").write_text(
            "".join(json.dumps({"episode_index": row["episode_index"], "stats": row["stats"]}) + "\n" for row in results),
            encoding="utf-8",
        )
        _log("AUDIT artifact coverage, source signatures and metadata; every video was fully decoded by workers")
        expected_artifacts = {artifact["path"] for row in results for artifact in row["artifacts"]}
        actual_artifacts = {str(path.relative_to(work)).replace(os.sep, "/")
                            for folder, suffix in (("data", "*.parquet"), ("videos", "*.mp4"))
                            for path in (work / folder).rglob(suffix)}
        if actual_artifacts != expected_artifacts:
            raise ValueError("输出 Parquet/MP4 路径集合与已验证轨迹不一致")
        for job in jobs:
            if _signature(pathlib.Path(job["parquet"])) != job["signature"]:
                raise ValueError(f"源 Parquet 在转换期间发生变化: {job['parquet']}")
        if {str(path.relative_to(source)): _hash(path) for path in sorted((source / "meta").rglob("*")) if path.is_file()} != metadata_hash:
            raise ValueError("转换期间源元数据发生变化，拒绝发布最终目录")
        report = {
            "status": "pass", "source": str(source), "output": str(output), "workers": workers,
            "encoder_threads": encoder_threads, "episodes": len(episodes), "frames": offset,
            "videos": len(episodes) * len(cameras), "codec": "h264", "crf": crf, "lossy_video": True,
            "non_image_columns": "all values and Arrow types exactly equal to source",
            "video_validation": "all frames decoded; dimensions, FPS and presentation timestamps checked",
            "image_validation": "3 RGB samples per camera/episode, not pixel-exact equivalence",
            "max_image_mae": max_image_mae, "metadata_sha256": metadata_hash,
            "audit": {"artifact_coverage": "pass", "all_episode_columns": "exact",
                      "all_videos_decoded": True, "source_signatures_unchanged": True},
        }
        _write_json(work / "transcode_validation.json", report)
        # Publish while still holding the external lock. Failed runs retain
        # only the clearly named .transcoding directory for explicit resume.
        if output.exists():
            raise FileExistsError(f"转换期间出现同名输出，拒绝覆盖: {output}")
        os.replace(work, output)
    _log(f"TRANSCODE PASS: episodes={len(episodes)}, frames={offset}, videos={report['videos']}, output={output}")
    return report


def main() -> None:
    _log(f"BOOT streaming converter pid={os.getpid()}; no LeRobot/Hugging Face initialization")
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "/scratch/bpx_datasets/lerobot"))
    parser.add_argument("--source-root", type=pathlib.Path, default=home / "bpx/bpx_blocks_combined")
    parser.add_argument("--output-root", type=pathlib.Path, default=home / "bpx/bpx_blocks_combined_video")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--max-image-mae", type=float, default=12)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        transcode_dataset(
            args.source_root, args.output_root, workers=args.workers, encoder_threads=args.encoder_threads,
            batch_frames=args.batch_frames, crf=args.crf, max_image_mae=args.max_image_mae,
            episode_limit=args.episode_limit, resume=args.resume,
        )
    except Exception as error:
        _log(f"ERROR {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
