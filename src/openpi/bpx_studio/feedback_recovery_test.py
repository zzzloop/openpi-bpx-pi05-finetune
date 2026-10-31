"""Server-run specifications; NOT executed on the local desktop.

The transaction fixture intentionally mocks media audit: it tests numeric
integrity/metadata only, not real MP4 or LeRobot SDK compatibility.
"""

import dataclasses
import hashlib
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.bpx_studio.feedback_estimator import Episode
from openpi.bpx_studio.feedback_estimator import FEETECH_DIMS
from openpi.bpx_studio.feedback_estimator import causal_response
from openpi.bpx_studio.feedback_estimator import fit_dimension
from openpi.bpx_studio.feedback_estimator import predict_dimension
from openpi.bpx_studio.feedback_estimator import reject_calibration_outliers
from openpi.bpx_studio.feedback_estimator import repair_short_outliers
from openpi.bpx_studio.feedback_estimator import transfer_right_gripper
from openpi.bpx_studio.recover_feedback import EXTRA_FEATURES
from openpi.bpx_studio.recover_feedback import recover
from openpi.policies.bpx_constants import BPX_EFFECTIVE_JOINT_NAMES
from openpi.training.validation_split import load_split


def test_estimator_does_not_read_same_frame_or_future_commands():
    times = np.arange(30) / 30
    original = np.zeros(30)
    changed = original.copy()
    changed[10:] = 1
    for delay, tau in ((0., 0.), (0., .05), (.07, .1)):
        first = causal_response(times, original, delay, tau)
        second = causal_response(times, changed, delay, tau)
        np.testing.assert_array_equal(first[:11], second[:11])
        assert second[-1] > 0


def test_gross_gripper_outlier_is_marked_without_flattening_a_real_step():
    times = np.arange(20) / 30
    values = np.zeros((20, 18), dtype=np.float32)
    values[10:, 7] = .08
    values[4, 7] = 4.562
    bad = reject_calibration_outliers(values)
    assert bad[4, 7] and not bad[10, 7]
    output, repaired = repair_short_outliers(values, times, bad)
    assert repaired.sum() == 1 and output[4, 7] == 0
    np.testing.assert_array_equal(output[10:, 7], values[10:, 7])


def _episodes(count=4):
    episodes = {}
    times = np.arange(50) / 30
    for index in range(count):
        command = np.zeros((50, 18))
        command[:, 4] = np.sin(times * (index + 1)) * .2
        measured = command.copy()
        measured[:, 4] = causal_response(times, command[:, 4], 0., .05)
        episodes[index] = Episode(index, times.copy(), command, measured, np.ones(command.shape, dtype=bool))
    return episodes


def test_nested_validation_excludes_the_entire_outer_episode():
    model, predictions = fit_dimension(_episodes(), 4, grid=((0., 0.), (0., .05)))
    assert len(predictions) == 4
    for fold in model["validation"]["folds"]:
        assert fold["heldout_episode"] not in fold["training_episodes"]
    assert model["validation"]["aggregate"]["mae"] < 1e-6


def test_right_gripper_transfer_uses_right_commands_and_is_never_claimed_validated():
    episodes = _episodes()
    for ep in episodes.values():
        ep.measured[:, 7], ep.measured[:, 15] = .00037, .000095
    left = {"dimension": 7, "source_dimension": 7, "method": "causal_first_order_affine",
            "delay_seconds": 0., "tau_seconds": .05, "gain": 1., "bias": .00037,
            "calibration_command_min": 0., "calibration_command_max": .08,
            "limitations": [], "validation": {"mean_episode_mae": .01}}
    model = transfer_right_gripper(left, episodes)
    assert model["validation"]["kind"] == "unvalidated_transfer"
    assert model["validation"]["dynamic_right_samples"] == 0
    ep = dataclasses.replace(episodes[0], command=episodes[0].command.copy())
    ep.command[10:, 15] = .08
    result, _, review = predict_dimension(model, ep)
    assert result[-1] > .07 and review.all()
    assert np.all(ep.command[:, 7] == 0)


def _source_fixture(root):
    (root / "meta").mkdir(parents=True)
    features = {}
    rows, stats = [], []
    for index in range(6):
        n = 40
        command = np.tile(np.linspace(0, .2, n)[:, None], (1, 18)).astype(np.float32)
        command[:, [7, 15]] *= .4
        state = command.copy()
        if index >= 4:
            state[:, list(FEETECH_DIMS)] = .01
        numeric = {"observation.state": state, "action": command,
                   "bpx.real_timestamp": np.arange(n, dtype=np.float64) / 28.4,
                   "frame_index": np.arange(n, dtype=np.int64), "index": np.arange(index*n, (index+1)*n),
                   "episode_index": np.full(n, index), "timestamp": np.arange(n, dtype=np.float32)/30,
                   "task_index": np.full(n, int(index == 5))}
        arrays, hf = {}, {}
        for key, value in numeric.items():
            vector = value.ndim == 2
            dtype = str(value.dtype)
            arrays[key] = pa.array(value.tolist(), type=pa.list_(pa.float32(), 18) if vector else pa.from_numpy_dtype(value.dtype))
            val = {"dtype": dtype, "_type": "Value"}
            hf[key] = {"feature": val, "length": 18, "_type": "Sequence"} if vector else val
            features[key] = {"dtype": dtype, "shape": [18] if vector else [1],
                             "names": list(BPX_EFFECTIVE_JOINT_NAMES) if vector else None}
        table = pa.table(arrays).replace_schema_metadata({b"huggingface": json.dumps({"info": {"features": hf}}).encode()})
        target = root / "data/chunk-000" / f"episode_{index:06d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        rows.append({"episode_index": index, "length": n, "tasks": ["singleton" if index == 5 else "repeated"]})
        stats.append({"episode_index": index, "stats": {}})
    info = {"codebase_version": "v2.1", "robot_type": "bpx_v1", "fps": 30, "chunks_size": 1000,
            "total_episodes": 6, "total_frames": 240, "features": features,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
    for camera in ("left_eye", "right_eye", "left_wrist", "right_wrist"):
        key = f"observation.images.{camera}"
        features[key] = {"dtype": "video", "shape": [360, 640, 3]}
        for index in range(6):
            path = root / info["video_path"].format(episode_chunk=0, episode_index=index, video_key=key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"NOT-A-REAL-VIDEO: media audit is explicitly mocked in this numeric fixture")
    (root / "meta/info.json").write_text(json.dumps(info))
    for name, records in (("episodes.jsonl", rows), ("episodes_stats.jsonl", stats),
                          ("tasks.jsonl", [{"task_index": 0, "task": "repeated"}, {"task_index": 1, "task": "singleton"}])):
        (root / "meta" / name).write_text("".join(json.dumps(row)+"\n" for row in records))


@pytest.mark.skipif(os.name != "posix", reason="Recovery is server-only")
def test_derived_dataset_preserves_source_actions_and_measured_dimensions(tmp_path, monkeypatch):
    from openpi.bpx_studio import audit, recover_feedback
    source, output, reports = tmp_path / "raw", tmp_path / "estimated", tmp_path / "reports"
    _source_fixture(source)
    before = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source.rglob("*") if path.is_file()}
    monkeypatch.setattr(audit, "audit_dataset", lambda root, *args, **kwargs: {"root": str(root), "episodes": 6})

    def fake_fit(episodes, dim):
        return {"dimension": dim, "source_dimension": dim, "method": "past_command_baseline",
                "delay_seconds": 0., "tau_seconds": 0., "gain": 1., "bias": 0.,
                "calibration_command_min": 0., "calibration_command_max": .2,
                "validation": {"mean_episode_mae": 0.}, "limitations": []}, {}

    monkeypatch.setattr(recover_feedback, "fit_dimension", fake_fit)
    recover(source, output, reports, frozen_from=4, apply=True, accept_estimated=True, plots=False, video_check="none")
    after = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in source.rglob("*") if path.is_file()}
    assert after == before
    original = pq.ParquetFile(source / "data/chunk-000/episode_000004.parquet").read()
    derived = pq.ParquetFile(output / "data/chunk-000/episode_000004.parquet").read()
    assert derived["action"].equals(original["action"])
    assert derived["bpx.state_original"].equals(original["observation.state"])
    masks = np.asarray(derived["bpx.state_estimated_mask"].to_pylist())
    assert np.all(masks[:, list(FEETECH_DIMS)] == 1) and masks.sum() == 400
    untouched = [dim for dim in range(18) if dim not in FEETECH_DIMS]
    np.testing.assert_array_equal(np.asarray(derived["observation.state"].to_pylist())[:, untouched],
                                  np.asarray(original["observation.state"].to_pylist())[:, untouched])
    split = load_split(reports / "validation_split.json")
    assert split["validation_episodes"] == [4] and 5 in split["train_episodes"]
    assert not set(range(4)) & set(split["validation_episodes"])
    assert set(EXTRA_FEATURES).issubset(derived.column_names)
    with pytest.raises(FileExistsError):
        recover(source, output, reports, frozen_from=4, apply=True, accept_estimated=True, plots=False, video_check="none")
