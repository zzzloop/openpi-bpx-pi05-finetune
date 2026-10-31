"""Normalization progress must never be confused with dataset cache progress."""

from __future__ import annotations

import json

import pytest

from openpi.bpx_studio.progress import parse_progress_line


@pytest.mark.parametrize("line", [
    "Downloading data: 100%|██████████| 80/80 [00:00<00:00, 40000files/s]",
    "Loading dataset shards: 100%|██████████| 12/12 [00:00<00:00]",
    "Generating train split: 36603 examples [00:00, 180631.90 examples/s]",
])
def test_normalize_ignores_data_loading_bars(line):
    assert parse_progress_line(line, job_kind="normalize") is None


def test_normalize_tracks_only_real_statistics_and_reserves_final_success():
    progress = parse_progress_line("Computing stats: 50%|█████| 8358/16716 [01:00<01:00]", job_kind="normalize")
    assert progress["current"] == 8358 and progress["percent"] == 50
    complete = parse_progress_line("Computing stats: 100%|██████████| 16716/16716 [02:00<00:00]", job_kind="normalize")
    assert complete["current"] == 16716 and complete["percent"] == 99.9


def test_normalize_stage_is_separate_from_progress():
    stage = parse_progress_line("[BPX NORM] 初始化数据加载器/设备", job_kind="normalize")
    assert stage == {"kind": "phase", "phase": "初始化数据加载器/设备"}


def test_other_job_progress_and_training_metrics_are_unchanged():
    assert parse_progress_line("Downloading data: 100%|████| 80/80")["percent"] == 100
    result = parse_progress_line("Step 200: loss=0.1", job_kind="train", total_units=1000)
    assert result["current"] == 200 and result["percent"] == 20


def test_manager_keeps_phase_and_marks_complete_only_after_success(tmp_path):
    from openpi.bpx_studio.jobs import JobManager

    manager = JobManager(tmp_path)
    directory = manager.root / "test"
    directory.mkdir()
    log = directory / "job.log"
    log.write_text("[BPX NORM] 初始化\nDownloading data: 100%|████| 80/80\n", encoding="utf-8")
    record = {"id": "test", "kind": "normalize", "log_path": str(log),
              "exit_path": str(directory / "exit.json"), "progress": {"current": 0, "percent": 0},
              "log_offset": 0, "total_units": None}
    manager._consume_log(record)
    assert record["progress"] == {"current": 0, "percent": 0, "phase": "初始化"}
    with log.open("a", encoding="utf-8") as stream:
        stream.write("Computing stats: 100%|████| 10/10\n[BPX NORM] 写入统计\n")
    manager._consume_log(record)
    assert record["progress"]["percent"] == 99.9
    assert record["progress"]["phase"] == "写入统计"
    (directory / "exit.json").write_text(json.dumps({"return_code": 0}), encoding="utf-8")
    manager._finish_from_exit(record)
    assert record["status"] == "succeeded" and record["progress"]["percent"] == 100


def test_normalize_cpu_environment_does_not_mutate_parent_or_train(monkeypatch, tmp_path):
    from openpi.bpx_studio.commands import build_normalize

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    (tmp_path / "scripts").mkdir()
    for name in ("train.py", "serve_policy.py", "compute_norm_stats.py"):
        (tmp_path / "scripts" / name).touch()
    (tmp_path / "pyproject.toml").touch()
    result = build_normalize({"openpi_root": str(tmp_path), "config_name": "test", "workers": 32})
    assert result.env["JAX_PLATFORMS"] == "cpu" and result.env["CUDA_VISIBLE_DEVICES"] == ""
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3" and os.environ["JAX_PLATFORMS"] == "cuda"
    assert "--num-workers" in result.argv and "32" in result.argv
