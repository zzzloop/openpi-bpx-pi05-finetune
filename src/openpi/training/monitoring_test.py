import json
import pathlib

import numpy as np

from openpi.training import monitoring


def _metrics(scale: float) -> dict[str, np.ndarray | float]:
    return {
        "loss": 1.0 * scale,
        "loss_std": 0.2 * scale,
        "loss_min": 0.5 * scale,
        "loss_max": 1.5 * scale,
        "loss_by_horizon": np.linspace(0.5, 1.5, 30) * scale,
        "grad_norm": 2.0 * scale,
        "update_norm": 0.1 * scale,
        "param_norm": 100.0,
        "learning_rate": 2.5e-5,
    }


def test_training_monitor_persists_metrics_and_renders_plots(tmp_path: pathlib.Path):
    monitor = monitoring.TrainingMonitor(
        tmp_path,
        batch_size=32,
        plot_interval=100,
        visualization_interval=1_000,
    )

    first_paths = monitor.log_metrics(0, _metrics(1.0))
    second_paths = monitor.log_metrics(100, _metrics(0.8))

    assert all(path.is_file() for path in [*first_paths, *second_paths])
    assert (tmp_path / "training_overview_latest.png").is_file()
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [0, 100]
    assert len(records[-1]["loss_by_horizon"]) == 30
    assert (tmp_path / "metrics.csv").read_text().count("\n") == 3


def test_training_monitor_saves_camera_and_action_visualizations(tmp_path: pathlib.Path):
    monitor = monitoring.TrainingMonitor(
        tmp_path,
        batch_size=2,
        plot_interval=100,
        visualization_interval=1_000,
        action_names=tuple(f"joint_{index}" for index in range(18)),
    )
    images = {
        "base_0_rgb": np.zeros((2, 16, 24, 3), dtype=np.float32),
        "left_wrist_0_rgb": np.ones((2, 16, 24, 3), dtype=np.float32),
        "right_wrist_0_rgb": -np.ones((2, 16, 24, 3), dtype=np.float32),
    }
    actions = np.zeros((2, 30, 32), dtype=np.float32)
    actions[..., :18] = np.linspace(-1.0, 1.0, 30, dtype=np.float32)[None, :, None]

    paths = monitor.save_batch_visualizations(0, images, actions, max_samples=2)

    assert len(paths) == 2
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_training_monitor_resumes_existing_jsonl_history(tmp_path: pathlib.Path):
    first = monitoring.TrainingMonitor(
        tmp_path,
        batch_size=32,
        plot_interval=100,
        visualization_interval=1_000,
    )
    first.log_metrics(0, _metrics(1.0))
    first.log_metrics(100, _metrics(0.95))

    resumed = monitoring.TrainingMonitor(
        tmp_path,
        batch_size=32,
        plot_interval=100,
        visualization_interval=1_000,
    )
    resumed.prepare_resume(50)
    resumed.log_metrics(50, _metrics(0.9))

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [0, 50]
    assert (tmp_path / "metrics.csv").read_text().count("\n") == 3
