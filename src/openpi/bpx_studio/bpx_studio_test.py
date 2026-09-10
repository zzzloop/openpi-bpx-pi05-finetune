from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.bpx_studio import commands
from openpi.bpx_studio.api import _checkpoints
from openpi.bpx_studio.api import _config_profile
from openpi.bpx_studio.api import _next_experiment
from openpi.bpx_studio.api import create_app
from openpi.bpx_studio.api import default_paths
from openpi.bpx_studio.numeric_spikes import repair_isolated_spikes
from openpi.bpx_studio.progress import parse_progress_line
from openpi.bpx_studio.spike_viewer import load_spike_trace
from openpi.bpx_studio.spike_viewer import summarize_spike_report


def _openpi_root(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "scripts" / "train.py").touch()
    (tmp_path / "scripts" / "compute_norm_stats.py").touch()
    (tmp_path / "scripts" / "serve_policy.py").touch()
    (tmp_path / "pyproject.toml").touch()
    return tmp_path


def test_default_paths_keep_large_artifacts_under_data_root(tmp_path: pathlib.Path) -> None:
    paths = default_paths(tmp_path)
    assert paths["hf_lerobot_home"] == str(tmp_path / "lerobot")
    assert paths["checkpoint_root"] == str(tmp_path / "checkpoints")
    assert paths["state_root"] == str(tmp_path / "dashboard_state")
    assert pathlib.Path(paths["raw_dataset"]).parts[-3:] == ("lerobot", "bpx", "bpx_v2_humanoid_raw")
    assert pathlib.Path(paths["clean_dataset"]).parts[-3:] == ("lerobot", "bpx", "bpx_v2_humanoid")
    assert paths["spike_scan_report"].endswith("bpx_v2_humanoid_spikes_scan.json")
    assert paths["spike_clean_report"].endswith("bpx_v2_humanoid_spikes_clean.json")
    assert all(pathlib.Path(path).is_relative_to(tmp_path) for path in paths.values())


def test_checkpoint_discovery_ignores_incomplete_saves_and_increments_experiment(tmp_path: pathlib.Path) -> None:
    complete = tmp_path / "config" / "bpx_v2_humanoid_bs2_v001" / "200000"
    (complete / "params").mkdir(parents=True)
    (complete / "assets").mkdir()
    other_config = tmp_path / "other_config" / "legacy_experiment" / "60000"
    (other_config / "params").mkdir(parents=True)
    (other_config / "assets").mkdir()
    incomplete = tmp_path / "config" / "bpx_v2_humanoid_bs2_v002" / "25000.orbax-checkpoint-tmp-1"
    (incomplete / "params").mkdir(parents=True)
    records = _checkpoints(tmp_path)
    assert {record["path"] for record in records} == {str(complete.resolve()), str(other_config.resolve())}
    assert {record["config_name"] for record in records} == {"config", "other_config"}
    assert _next_experiment(tmp_path, "config") == "bpx_v2_humanoid_bs2_v003"


def test_config_profile_exposes_actual_dataset_and_norm_asset_binding(tmp_path: pathlib.Path) -> None:
    train_config = SimpleNamespace(
        name="pi05_bpx_full_low_mem_multitask_v2",
        assets_base_dir="./assets",
        data=SimpleNamespace(
            repo_id="bpx/bpx_blocks_multitask_v2",
            assets=SimpleNamespace(assets_dir=None, asset_id="bpx/bpx_blocks_multitask_v2"),
        ),
    )
    profile = _config_profile(
        project_root=tmp_path / "openpi",
        lerobot_root=tmp_path / "lerobot",
        train_config=train_config,
    )
    assert profile["dataset_path"] == str((tmp_path / "lerobot/bpx/bpx_blocks_multitask_v2").resolve())
    assert profile["stats_paths_match"] is True
    assert profile["computed_stats_path"] == profile["loaded_stats_path"]
    assert profile["computed_stats_exists"] is False
    assert profile["stats_size_bytes"] == 0

    stats_path = pathlib.Path(str(profile["computed_stats_path"]))
    stats_path.parent.mkdir(parents=True)
    stats_path.write_text('{"norm_stats": true}\n', encoding="utf-8")
    with_stats = _config_profile(
        project_root=tmp_path / "openpi",
        lerobot_root=tmp_path / "lerobot",
        train_config=train_config,
    )
    assert with_stats["computed_stats_exists"] is True
    assert with_stats["loaded_stats_exists"] is True
    assert with_stats["stats_size_bytes"] == stats_path.stat().st_size

    train_config.data.assets.assets_dir = "./assets/shared"
    mismatched = _config_profile(
        project_root=tmp_path / "openpi",
        lerobot_root=tmp_path / "lerobot",
        train_config=train_config,
    )
    assert mismatched["stats_paths_match"] is False
    assert mismatched["computed_stats_path"] != mismatched["loaded_stats_path"]


def test_dashboard_routes_and_control_token_are_created(tmp_path: pathlib.Path) -> None:
    app = create_app(
        state_dir=tmp_path / "state",
        project_root=tmp_path / "openpi",
        data_root=tmp_path / "scratch",
    )
    routes = {route.path for route in app.routes}
    assert {
        "/",
        "/healthz",
        "/api/config-profile",
        "/api/spikes/report",
        "/api/spikes/trace",
        "/api/jobs/train",
        "/api/jobs/normalize",
    } <= routes
    assert app.state.control_token
    assert (tmp_path / "state" / ".server-token").is_file()


def test_studio_launcher_is_strictly_offline() -> None:
    project_root = pathlib.Path(__file__).resolve().parents[3]
    launcher = (project_root / "scripts" / "start_bpx_studio.sh").read_text(encoding="utf-8")
    assert '.venv/bin/python' in launcher
    assert "third_party/bpx_studio_wheels" in launcher
    assert 'export PYTHONPATH="${wheel_pythonpath}' in launcher
    assert 'exec "${PROJECT_PYTHON}" -m openpi.bpx_studio.api' in launcher
    assert "uv run" not in launcher
    assert "--with" not in launcher
    wheels = list((project_root / "third_party" / "bpx_studio_wheels").glob("*.whl"))
    assert len(wheels) == 9


def test_train_command_preserves_known_two_gpu_contract(tmp_path: pathlib.Path) -> None:
    root = _openpi_root(tmp_path / "openpi")
    checkpoint_root = tmp_path / "scratch" / "checkpoints"
    spec = commands.build_train(
        {
            "openpi_root": str(root),
            "config_name": "pi05_bpx_full_low_mem_multitask_v2",
            "exp_name": "bpx_multitask_v2_bs2",
            "checkpoint_base_dir": str(checkpoint_root),
            "cuda_devices": "2,3",
            "memory_fraction": 0.96,
            "batch_size": 2,
            "num_train_steps": 200000,
            "fsdp_devices": 2,
            "save_interval": 5000,
            "keep_period": 25000,
            "mode": "resume",
        }
    )
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert spec.env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.96"
    assert f"--checkpoint-base-dir={checkpoint_root.resolve()}" in spec.argv
    assert "--resume" in spec.argv
    assert spec.total_units == 200000


def test_train_rejects_batch_not_divisible_by_devices(tmp_path: pathlib.Path) -> None:
    root = _openpi_root(tmp_path / "openpi")
    with pytest.raises(ValueError, match="batch size"):
        commands.build_train(
            {
                "openpi_root": str(root),
                "config_name": "config",
                "exp_name": "experiment",
                "checkpoint_base_dir": str(tmp_path / "checkpoints"),
                "cuda_devices": "2,3",
                "memory_fraction": 0.96,
                "batch_size": 1,
                "num_train_steps": 1,
                "fsdp_devices": 2,
                "save_interval": 1,
                "keep_period": 1,
                "mode": "overwrite",
            }
        )


def test_policy_serve_uses_selected_complete_checkpoint(tmp_path: pathlib.Path) -> None:
    root = _openpi_root(tmp_path / "openpi")
    checkpoint = tmp_path / "checkpoints" / "config" / "experiment" / "200000"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    spec = commands.build_policy_serve(
        {
            "openpi_root": str(root),
            "config_name": "config",
            "checkpoint_dir": str(checkpoint),
            "cuda_devices": "2,3",
            "memory_fraction": 0.96,
        }
    )
    assert "--policy.config=config" in spec.argv
    assert f"--policy.dir={checkpoint.resolve()}" in spec.argv
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_isolated_spike_uses_neighbor_mean_without_touching_other_frames() -> None:
    source = np.array([[0.0, 1.0], [0.1, 1.1], [9.0, 1.2], [0.3, 1.3], [0.4, 1.4]], dtype=np.float32)
    result = repair_isolated_spikes(
        source,
        jump_thresholds=np.array([1.0, 1.0]),
        neighbor_thresholds=np.array([0.5, 0.5]),
    )
    assert result.count == 1
    np.testing.assert_allclose(result.repaired[2], [0.2, 1.2])
    np.testing.assert_array_equal(result.repaired[[0, 1, 3, 4]], source[[0, 1, 3, 4]])


def test_spike_report_summary_and_trace_show_source_and_cleaned_values(tmp_path: pathlib.Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source_root = tmp_path / "raw"
    output_root = tmp_path / "clean"
    relative = pathlib.Path("data/chunk-000/episode_000000.parquet")
    for root in (source_root, output_root):
        (root / relative.parent).mkdir(parents=True)
        (root / "meta").mkdir()
        (root / "meta/info.json").write_text(
            json.dumps(
                {
                    "features": {
                        "observation.state": {"names": ["shoulder", "elbow"]},
                        "action": {"names": ["shoulder", "elbow"]},
                    }
                }
            ),
            encoding="utf-8",
        )
    source_values = [[0.0, 1.0], [0.1, 1.1], [9.0, 1.2], [0.3, 1.3], [0.4, 1.4]]
    cleaned_values = [[0.0, 1.0], [0.1, 1.1], [0.2, 1.2], [0.3, 1.3], [0.4, 1.4]]

    def write_episode(root: pathlib.Path, values: list[list[float]]) -> None:
        table = pa.table(
            {
                "observation.state": pa.array(values, type=pa.list_(pa.float32(), 2)),
                "action": pa.array(values, type=pa.list_(pa.float32(), 2)),
                "episode_index": pa.array([0] * 5, type=pa.int64()),
                "frame_index": pa.array(range(5), type=pa.int64()),
                "timestamp": pa.array([index / 30 for index in range(5)], type=pa.float64()),
            }
        )
        pq.write_table(table, root / relative)

    write_episode(source_root, source_values)
    write_episode(output_root, cleaned_values)
    report_path = tmp_path / "spikes_clean.json"
    report_path.write_text(
        json.dumps(
            {
                "source": str(source_root),
                "output": str(output_root),
                "applied": True,
                "clean_audit_passed": True,
                "episode_files": 1,
                "repairs": 1,
                "files": [
                    {
                        "path": str(output_root / relative),
                        "episode_index": 0,
                        "repairs": [
                            {
                                "feature": "observation.state",
                                "frame_index": 2,
                                "dimension": 0,
                                "previous": 0.1,
                                "original": 9.0,
                                "following": 0.3,
                                "repaired": 0.2,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_spike_report(report_path)
    assert summary["total_repairs"] == 1
    assert summary["affected_files"] == 1
    assert summary["hits"][0]["dimension_name"] == "shoulder"
    assert summary["clean_audit_passed"] is True
    trace = load_spike_trace(report_path, file_index=0, repair_index=0, radius=2)
    assert trace["source_values"][2] == pytest.approx(9.0)
    assert trace["review_values"][2] == pytest.approx(0.2)
    assert trace["frame_vector"][0]["name"] == "shoulder"


def test_progress_parser_reads_openpi_training_metrics() -> None:
    parsed = parse_progress_line(
        "Step 200: grad_norm=1.005, learning_rate=3.784e-06, loss=0.08145",
        total_units=1000,
    )
    assert parsed is not None
    assert parsed["current"] == 200
    assert parsed["percent"] == 20.0
    assert parsed["metrics"]["loss"] == pytest.approx(0.08145)
