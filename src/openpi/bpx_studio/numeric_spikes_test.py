import numpy as np
import json
import pathlib
import subprocess
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.bpx_studio.clean_spikes import _clean_parquet
from openpi.bpx_studio.numeric_spikes import repair_isolated_spikes


@pytest.mark.parametrize("values, expected, count", [
    ([0, 0.1, 9, 0.3, 0.4], [0, 0.1, 0.2, 0.3, 0.4], 1),
    ([0, -0.1, -9, -0.3, -0.4], [0, -0.1, -0.2, -0.3, -0.4], 1),
    ([0, 9, 9, 9], [0, 9, 9, 9], 0),
    ([0, 9, 9, 0], [0, 9, 9, 0], 0),
    ([0, 9, 0, 9, 0], [0, 9, 0, 9, 0], 0),
    ([9, 0, 0], [9, 0, 0], 0),
    ([0, 0, 9], [0, 0, 9], 0),
    ([0], [0], 0),
    ([0, 9], [0, 9], 0),
])
def test_excursions_steps_runs_and_boundaries(values, expected, count):
    source = np.array(values, dtype=np.float64)[:, None]
    original = source.copy()
    result = repair_isolated_spikes(source, jump_thresholds=np.array([1]), neighbor_thresholds=np.array([0.5]))
    np.testing.assert_allclose(result.repaired[:, 0], expected)
    np.testing.assert_array_equal(source, original)
    assert result.count == count
    assert result.unresolved_jump_mask.sum() == np.count_nonzero(np.abs(np.diff(expected)) > 1)


def test_wide_neighbor_threshold_does_not_smooth_monotonic_motion():
    values = np.array([[0.0], [2.0], [4.0]])
    result = repair_isolated_spikes(values, jump_thresholds=np.array([1]), neighbor_thresholds=np.array([5]))
    np.testing.assert_array_equal(result.repaired, values)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_non_finite_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="finite"):
        repair_isolated_spikes(np.zeros((3, 1)), jump_thresholds=np.array([threshold]), neighbor_thresholds=np.array([0]))


def _write_episode(path, episode_ids=None):
    precise = 0.123456789012345
    values = [[precise], [precise], [9.0], [precise], [precise]]
    table = pa.table({
        "observation.state": pa.array(values, type=pa.list_(pa.float64(), 1)),
        "action": pa.array([[0.], [0.], [5.], [5.], [5.]], type=pa.list_(pa.float64(), 1)),
        "frame_index": list(range(5)), "timestamp": [i / 30 for i in range(5)],
        "episode_index": episode_ids or [0] * 5,
        "task_index": [0] * 5,
    })
    pq.write_table(table, path)
    return precise


def test_parquet_scan_apply_preserves_precision_and_reports_steps(tmp_path):
    path = tmp_path / "episode.parquet"
    precise = _write_episode(path)
    thresholds = {key: {"jump": np.array([1]), "neighbor": np.array([0.5])}
                  for key in ("observation.state", "action")}
    original = path.read_bytes()
    scan = _clean_parquet((path, thresholds, False))
    assert path.read_bytes() == original
    assert scan["total"] == 1
    applied = _clean_parquet((path, thresholds, True))
    assert applied["counts"] == scan["counts"]
    assert applied["unresolved_jump_counts"]["action"] == 1
    table = pq.read_table(path)
    assert table["observation.state"].to_pylist() == [[precise]] * 5
    assert table["action"].to_pylist() == [[0.], [0.], [5.], [5.], [5.]]


def test_never_interpolates_across_episode_boundary(tmp_path):
    path = tmp_path / "mixed.parquet"
    _write_episode(path, [0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="episode"):
        _clean_parquet((path, {}, True))


def test_real_audit_clean_and_hdf5_pipeline(tmp_path):
    import av
    import h5py

    raw = tmp_path / "raw"
    (raw / "data").mkdir(parents=True)
    (raw / "meta").mkdir()
    (raw / "videos").mkdir()
    parquet = raw / "data" / "episode_000000.parquet"
    _write_episode(parquet)
    original = parquet.read_bytes()
    info = {"robot_type": "fixture", "fps": 30, "total_frames": 5, "total_episodes": 1,
            "video_path": "videos/{episode_index}.mp4", "features": {
                "observation.state": {"dtype": "float64", "shape": [1]},
                "action": {"dtype": "float64", "shape": [1]},
                "observation.images.front": {"dtype": "video", "shape": [16, 16, 3]},
            }}
    (raw / "meta/info.json").write_text(json.dumps(info))
    (raw / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 5}) + "\n")
    (raw / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test motion"}) + "\n")
    with av.open(str(raw / "videos/0.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({key: {"jump_thresholds": [1], "neighbor_thresholds": [0.5]}
                                      for key in ("observation.state", "action")}))
    clean = tmp_path / "clean"
    report = tmp_path / "report.json"
    export = tmp_path / "export"
    calls = [
        ("audit", [str(raw), "--workers", "1", "--video-check", "full"]),
        ("clean_spikes", [str(raw), "--thresholds", str(thresholds), "--workers", "1",
                          "--apply", "--output-root", str(clean), "--report", str(report)]),
        ("hdf5_exporter", [str(clean), str(export), "--workers", "1"]),
    ]
    for module, args in calls:
        result = subprocess.run([sys.executable, "-m", "openpi.bpx_studio." + module, *args],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    assert parquet.read_bytes() == original
    cleaned_report = json.loads(report.read_text(encoding="utf-8"))
    assert cleaned_report["clean_audit_passed"]
    assert cleaned_report["repairs"] == 1
    assert cleaned_report["unresolved_jumps"] == 1
    manifest = json.loads((export / "manifest.json").read_text())
    output = pathlib.Path(manifest["episodes"][0]["path"])
    assert output.is_file()
    with h5py.File(output) as file:
        assert file["observations/images/front"].shape == (5, 16, 16, 3)
