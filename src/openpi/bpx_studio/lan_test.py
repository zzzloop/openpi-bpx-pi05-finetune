"""Regression coverage for LAN monitoring and shared pipeline paths."""
import pathlib
import subprocess
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from openpi.bpx_studio import api
from openpi.bpx_studio import commands


def test_gpu_csv_handles_blank_lines_quoted_names_and_bad_rows(monkeypatch):
    monkeypatch.setattr(api.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout='0, "GPU, Test", 100, 200, 50, N/A\n\nbroken\n1, Second, 0, 200, 0, 40\n'))
    result = api._gpu_snapshot()
    assert len(result["gpus"]) == 2
    assert result["gpus"][0]["name"] == "GPU, Test"
    assert result["gpu_error"]


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), subprocess.TimeoutExpired("nvidia-smi", 8)])
def test_gpu_failure_is_explicit(monkeypatch, error):
    def fail(*a, **k):
        raise error
    monkeypatch.setattr(api.subprocess, "run", fail)
    result = api._gpu_snapshot()
    assert result["gpus"] == []
    assert "查询失败" in result["gpu_error"]


def test_custom_checkpoint_root_and_missing_disk_path(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "_gpu_snapshot", lambda: {"gpus": [], "gpu_error": "missing"})
    checkpoint = tmp_path / "custom" / "config" / "bpx_v2_humanoid_bs2_v001" / "100"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    app = api.create_app(state_dir=tmp_path / "state", project_root=tmp_path, data_root=tmp_path / "data")
    with TestClient(app) as client:
        response = client.get("/api/system", params={"path": str(tmp_path / "future" / "checkpoints")})
        assert response.status_code == 200
        assert response.json()["disk"]["path"] == str(tmp_path)
        assert response.json()["gpu_error"] == "missing"
        params = {"checkpoint_root": str(tmp_path / "custom")}
        records = client.get("/api/checkpoints", params=params).json()
        assert records[0]["path"] == str(checkpoint)
        params["config_name"] = "config"
        assert client.get("/api/next-experiment", params=params).json()["experiment_name"].endswith("v002")


def test_default_clean_path_matches_registered_config(tmp_path):
    import ast
    source = pathlib.Path(__file__).parents[1] / "training" / "config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    config = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and any(k.arg == "name" and isinstance(k.value, ast.Constant)
                          and k.value.value == api.DEFAULT_CONFIG_NAME for k in node.keywords))
    data = next(k.value for k in config.keywords if k.arg == "data")
    repo_id = next(k.value.value for k in data.keywords if k.arg == "repo_id")
    paths = api.default_paths(tmp_path)
    assert pathlib.Path(paths["clean_dataset"]) == pathlib.Path(paths["hf_lerobot_home"]) / repo_id


@pytest.mark.parametrize("helper", [commands._output_path, commands._existing_directory])
def test_empty_paths_are_rejected(helper):
    with pytest.raises(ValueError, match="不能为空"):
        helper(" ", "路径")


def test_existing_experiment_cannot_be_overwritten(tmp_path):
    from openpi.bpx_studio.bpx_studio_test import _openpi_root

    root = _openpi_root(tmp_path / "project")
    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / "config" / "exp").mkdir(parents=True)
    values = dict(openpi_root=str(root), config_name="config", exp_name="exp",
                  checkpoint_base_dir=str(checkpoint_root), batch_size=2, num_train_steps=10,
                  fsdp_devices=2, save_interval=5, keep_period=10, cuda_devices="2,3",
                  memory_fraction=0.96, mode="overwrite")
    with pytest.raises(ValueError, match="实验目录已存在"):
        commands.build_train(values)
    values["mode"] = "resume"
    assert "--resume" in commands.build_train(values).argv


def test_all_job_routes_build_matching_commands(tmp_path, monkeypatch):
    from openpi.bpx_studio.bpx_studio_test import _openpi_root

    root = _openpi_root(tmp_path / "project")
    raw = tmp_path / "raw"
    raw.mkdir()
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text("{}")
    checkpoint = tmp_path / "checkpoints" / "config" / "exp" / "10"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    monkeypatch.setattr(api.JobManager, "start", lambda self, kind, spec: {
        "kind": kind, "argv": list(spec.argv), "env": dict(spec.env)})
    app = api.create_app(state_dir=tmp_path / "state", project_root=root, data_root=tmp_path)
    common = dict(openpi_root=str(root), config_name="config", hf_lerobot_home=str(raw))
    spike = dict(source_root=str(raw), thresholds_path=str(thresholds),
                 report_path=str(tmp_path / "report.json"), output_root=str(tmp_path / "clean"))
    cases = [
        ("audit", dict(dataset_root=str(raw), report_path=str(tmp_path / "audit.json")), "openpi.bpx_studio.audit"),
        ("spikes/scan", spike, "openpi.bpx_studio.clean_spikes"),
        ("spikes/clean", spike, "--apply"),
        ("normalize", common, "scripts/compute_norm_stats.py"),
        ("export/hdf5", dict(source_root=str(raw), output_root=str(tmp_path / "export")), "openpi.bpx_studio.hdf5_exporter"),
        ("train", dict(**common, exp_name="new", checkpoint_base_dir=str(tmp_path / "checkpoints")), "scripts/train.py"),
        ("serve", dict(openpi_root=str(root), checkpoint_dir=str(checkpoint), config_name="config"), "scripts/serve_policy.py"),
    ]
    with TestClient(app) as client:
        for route, payload, expected in cases:
            response = client.post("/api/jobs/" + route, json=payload)
            assert response.status_code == 200, response.text
            assert expected in response.json()["argv"]
            if route in {"normalize", "train"}:
                assert response.json()["env"]["HF_LEROBOT_HOME"] == str(raw)
