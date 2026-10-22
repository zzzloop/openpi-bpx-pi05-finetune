"""CPU tests for conversion options and guards without importing robot/ML runtimes."""

import ast
from concurrent import futures
import pathlib
from types import SimpleNamespace

import av
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


_SOURCE = pathlib.Path(__file__).with_name("transcode_bpx_lerobot_images_to_video.py")


def _definitions(*names, **bindings):
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    scope = {"LeRobotDataset": object, "pathlib": pathlib, "futures": futures, "np": np, "pq": pq, **bindings}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(_SOURCE), "exec"), scope)
    return scope


def test_parallel_encoder_settings_produce_real_h264_mp4(tmp_path):
    expected = np.full((32, 48, 3), [190, 70, 20], dtype=np.uint8)
    inputs = {}
    for camera in ("left_eye", "left_wrist", "right_wrist"):
        image_dir = tmp_path / "images" / camera
        image_dir.mkdir(parents=True)
        Image.fromarray(expected).save(image_dir / "frame_000000.png")
        inputs[camera] = image_dir
    calls = []

    def encoder(image_dir, output_path, fps, **options):
        calls.append(options)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(output_path), "w") as output:
            stream = output.add_stream(options["vcodec"], rate=fps, options={"crf": str(options["crf"]), "g": str(options["g"])})
            stream.width, stream.height, stream.pix_fmt = 48, 32, options["pix_fmt"]
            with Image.open(image_dir / "frame_000000.png") as image:
                frame = av.VideoFrame.from_image(image)
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)

    cls = _definitions("_ParallelVideoLeRobotDataset", encode_video_frames=encoder)["_ParallelVideoLeRobotDataset"]
    dataset = cls()
    dataset.root, dataset.fps = tmp_path, 30
    dataset.video_encode_workers = 3
    dataset.meta = SimpleNamespace(
        video_keys=list(inputs),
        get_video_file_path=lambda episode, camera: pathlib.Path("videos") / camera / f"episode_{episode:06d}.mp4",
    )
    dataset._get_image_file_path = lambda *, image_key, **kwargs: inputs[image_key] / "frame_000000.png"
    result = dataset.encode_episode_videos(0)
    assert len(calls) == len(result) == 3
    for options in calls:
        assert options == {"vcodec": "h264", "pix_fmt": "yuv420p", "g": 2, "crf": 18, "overwrite": True}
    for path in result.values():
        with av.open(path) as container:
            assert container.streams.video[0].codec_context.name == "h264"
            frames = list(container.decode(video=0))
            assert len(frames) == 1
            actual = frames[0].to_ndarray(format="rgb24")
            assert np.mean(np.abs(actual.astype(float) - expected)) < 12


def test_numeric_validation_catches_changed_claw_and_task(tmp_path):
    scope = _definitions(
        "_validate_numeric_episode", _NUMERIC_FEATURES=("observation.state", "action"),
        _list_column=lambda table, key, dtype: np.asarray(table[key].to_pylist(), dtype=dtype),
    )
    columns = {
        "observation.state": [[0.] * 18] * 2, "action": [[0.] * 18] * 2,
        "frame_index": [0, 1], "episode_index": [0, 0], "index": [0, 1],
        "timestamp": [0., 1 / 30], "task_index": [0, 0],
    }
    source, output = tmp_path / "source", tmp_path / "output"
    for root in (source, output):
        root.mkdir()
        pq.write_table(pa.table(columns), root / "episode.parquet")
    meta = SimpleNamespace(get_data_file_path=lambda episode: "episode.parquet")
    left, right = SimpleNamespace(root=source, meta=meta), SimpleNamespace(root=output, meta=meta)
    check = scope["_validate_numeric_episode"]
    check(left, right, 0)
    for key, bad_values in (("action", [[0.] * 7 + [.08] + [0.] * 10] * 2), ("task_index", [1, 1])):
        pq.write_table(pa.table({**columns, key: bad_values}), output / "episode.parquet")
        with pytest.raises(AssertionError):
            check(left, right, 0)


def test_offline_launcher_uses_server_environment_and_scratch_cache():
    project_root = _SOURCE.parents[2]
    launcher = (project_root / "scripts/transcode_bpx_video.sh").read_text(encoding="utf-8")
    assert ".venv/bin/python" in launcher
    assert "/scratch/bpx_datasets" in launcher
    assert "HF_DATASETS_CACHE" in launcher
    assert "--image-writer-threads 32 --video-encode-workers 3" in launcher
    assert "uv run" not in launcher and "uv sync" not in launcher and "pip install" not in launcher
