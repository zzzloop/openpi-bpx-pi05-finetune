from __future__ import annotations

import pathlib

import numpy as np
import pytest

from openpi.bpx_studio import commands
from openpi.bpx_studio.api import _checkpoints
from openpi.bpx_studio.api import _next_experiment
from openpi.bpx_studio.api import create_app
from openpi.bpx_studio.api import default_paths
from openpi.bpx_studio.numeric_spikes import repair_isolated_spikes
from openpi.bpx_studio.progress import parse_progress_line


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
    assert all(pathlib.Path(path).is_relative_to(tmp_path) for path in paths.values())


def test_checkpoint_discovery_ignores_incomplete_saves_and_increments_experiment(tmp_path: pathlib.Path) -> None:
    complete = tmp_path / "config" / "bpx_v2_humanoid_bs2_v001" / "200000"
    (complete / "params").mkdir(parents=True)
    (complete / "assets").mkdir()
    incomplete = tmp_path / "config" / "bpx_v2_humanoid_bs2_v002" / "25000.orbax-checkpoint-tmp-1"
    (incomplete / "params").mkdir(parents=True)
    assert [record["path"] for record in _checkpoints(tmp_path)] == [str(complete.resolve())]
    assert _next_experiment(tmp_path, "config") == "bpx_v2_humanoid_bs2_v003"


def test_dashboard_routes_and_control_token_are_created(tmp_path: pathlib.Path) -> None:
    app = create_app(
        state_dir=tmp_path / "state",
        project_root=tmp_path / "openpi",
        data_root=tmp_path / "scratch",
    )
    routes = {route.path for route in app.routes}
    assert {"/", "/healthz", "/api/jobs/train", "/api/jobs/normalize"} <= routes
    assert app.state.control_token
    assert (tmp_path / "state" / ".server-token").is_file()


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


def test_progress_parser_reads_openpi_training_metrics() -> None:
    parsed = parse_progress_line(
        "Step 200: grad_norm=1.005, learning_rate=3.784e-06, loss=0.08145",
        total_units=1000,
    )
    assert parsed is not None
    assert parsed["current"] == 200
    assert parsed["percent"] == 20.0
    assert parsed["metrics"]["loss"] == pytest.approx(0.08145)
