from __future__ import annotations

import hashlib
import io
import json
import pathlib

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from openpi.bpx_studio.commands import build_hdf5_export
from openpi.bpx_studio.hdf5_exporter import _decode_image
from openpi.bpx_studio.hdf5_exporter import export_hdf5


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root, *, storage="image", bad_image=False, bad_frame_index=False):
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {
        "observation.state": {"dtype": "float32", "shape": [18]},
        "action": {"dtype": "float64", "shape": [18]},
        "observation.images.left_eye": {"dtype": storage, "shape": [4, 6, 3]},
    }
    _write_json(root / "meta/info.json", {
        "fps": 30, "robot_type": "bpx", "features": features,
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mkv",
    })
    # Deliberately unsorted metadata: export must bind by actual episode ID.
    (root / "meta/episodes.jsonl").write_text(
        '\n'.join(json.dumps({"episode_index": i, "length": 3}) for i in (1, 0)), encoding="utf-8"
    )
    (root / "meta/tasks.jsonl").write_text(
        '\n'.join(json.dumps({"task_index": i, "task": text}) for i, text in enumerate(("red", "blue"))),
        encoding="utf-8",
    )
    expected = {}
    for episode in range(2):
        rgb = np.zeros((3, 4, 6, 3), dtype=np.uint8)
        rgb[..., 0] = 200 + episode  # Distinct red/blue verifies RGB, not BGR.
        rgb[..., 2] = 13
        rgb[1, ..., 1] = 50
        state = np.arange(54, dtype=np.float32).reshape(3, 18) / 10 + episode
        action = np.arange(54, dtype=np.float64).reshape(3, 18) / 7 + 1e-10
        columns = {
            "episode_index": [episode] * 3,
            "frame_index": [0, 2, 1] if bad_frame_index else [0, 1, 2],
            "timestamp": [0., 1 / 30, 2 / 30],
            "task_index": [0, 1, 0],
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 18)),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float64(), 18)),
        }
        if storage == "image":
            cells = []
            for frame in rgb:
                encoded = io.BytesIO()
                Image.fromarray(frame).save(encoded, format="PNG")
                cells.append({"bytes": b"invalid" if bad_image else encoded.getvalue(), "path": None})
            columns["observation.images.left_eye"] = cells
        else:
            path = root / f"videos/observation.images.left_eye/episode_{episode:06d}.mkv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(path), "w") as container:
                stream = container.add_stream("ffv1", rate=30)
                stream.width, stream.height, stream.pix_fmt = 6, 4, "bgr0"
                for pixels in rgb:
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        pq.write_table(pa.table(columns), root / f"data/chunk-000/episode_{episode:06d}.parquet")
        expected[episode] = rgb, state, action
    return expected


@pytest.mark.parametrize("storage", ["image", "video"])
def test_real_export_preserves_frames_rgb_precision_and_tasks(tmp_path, storage):
    source, output = tmp_path / "source", tmp_path / "export"
    expected = _fixture(source, storage=storage)
    before = {str(p): hashlib.sha256(p.read_bytes()).digest() for p in source.rglob("*") if p.is_file()}
    report = export_hdf5(source, output, workers=2, compression="gzip", compression_level=1)
    assert report["total_frames"] == 6
    for index, (rgb, state, action) in expected.items():
        with h5py.File(output / f"episode_{index:06d}.hdf5") as data:
            np.testing.assert_array_equal(data["observations/images/left_eye"][:], rgb)
            np.testing.assert_array_equal(data["observations/qpos"][:], state)
            np.testing.assert_array_equal(data["action"][:], action)
            np.testing.assert_array_equal(data["task_index"][:], [0, 1, 0])
            np.testing.assert_array_equal(data["frame_index"][:], [0, 1, 2])
            np.testing.assert_array_equal(data["observations/timestamp"][:], [0, 1 / 30, 2 / 30])
            assert data.attrs["image_color_order"] == "RGB"
            assert json.loads(data.attrs["task"]) == ["red", "blue"]
            assert data["action"].dtype == np.float64
    after = {str(p): hashlib.sha256(p.read_bytes()).digest() for p in source.rglob("*") if p.is_file()}
    assert after == before


def test_export_selected_episode_keeps_source_number(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    _fixture(source)
    report = export_hdf5(source, output, workers=1, compression="lzf", compression_level=4, episode_indices=[1])
    assert report["selected_episode_indices"] == [1]
    assert len(list(output.glob("*.hdf5"))) == 1
    assert (output / "episode_000001.hdf5").is_file()
    with pytest.raises(FileExistsError):
        export_hdf5(source, output, workers=1, compression="none", compression_level=4)


@pytest.mark.parametrize("options, message", [
    ({"bad_image": True}, "图像解码失败"),
    ({"bad_frame_index": True}, "frame_index 不连续"),
])
def test_bad_inputs_do_not_publish_partial_export(tmp_path, options, message):
    source, output = tmp_path / "source", tmp_path / "export"
    _fixture(source, **options)
    with pytest.raises(ValueError, match=message):
        export_hdf5(source, output, workers=1, compression="none", compression_level=4)
    assert not output.exists()
    assert not (tmp_path / ".export.partial").exists()


def test_image_path_decode_and_path_escape_rejection(tmp_path):
    rgb = np.full((4, 6, 3), [203, 40, 13], dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "image.png")
    np.testing.assert_array_equal(_decode_image({"bytes": None, "path": "image.png"}, tmp_path), rgb)
    with pytest.raises(ValueError, match="不在源数据集内"):
        _decode_image({"path": "../outside.png"}, tmp_path)


def test_three_camera_export_keeps_streams_distinct(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    _fixture(source)
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    expected = {}
    path = source / "data/chunk-000/episode_000000.parquet"
    table = pq.read_table(path)
    for name, color in (("left_wrist", [10, 40, 200]), ("right_wrist", [70, 230, 90])):
        key = f"observation.images.{name}"
        info["features"][key] = {"dtype": "image", "shape": [4, 6, 3]}
        rgb = np.full((4, 6, 3), color, dtype=np.uint8)
        encoded = io.BytesIO()
        Image.fromarray(rgb).save(encoded, format="PNG")
        table = table.append_column(key, pa.array([{"bytes": encoded.getvalue(), "path": None}] * 3))
        expected[name] = np.stack([rgb] * 3)
    pq.write_table(table, path)
    _write_json(info_path, info)
    export_hdf5(source, output, workers=1, compression="gzip", compression_level=1, episode_indices=[0])
    with h5py.File(output / "episode_000000.hdf5") as data:
        assert set(data["observations/images"]) == {"left_eye", "left_wrist", "right_wrist"}
        for name, frames in expected.items():
            np.testing.assert_array_equal(data[f"observations/images/{name}"][:], frames)


def test_export_command_selection_and_validation(tmp_path):
    values = {"source_root": str(tmp_path), "output_root": str(tmp_path / "export"), "episode_indices": [2, 0, 2]}
    spec = build_hdf5_export(tmp_path, values)
    assert spec.argv[-3:] == ("--episodes", "0", "2")
    for bad in ([], [-1], [True], [0.5], "0,1"):
        with pytest.raises(ValueError, match="episode 索引"):
            build_hdf5_export(tmp_path, {**values, "episode_indices": bad})
