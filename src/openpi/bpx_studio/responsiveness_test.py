"""Regression: config previews and hung GPU drivers cannot wedge the dashboard."""

import builtins
import dataclasses
import pathlib
import threading
import time

from fastapi.testclient import TestClient
import pytest

from openpi.bpx_studio import api
from openpi.bpx_studio.config_profile import read_config_binding
from openpi.bpx_studio.status_probe import CachedProbe


PROJECT = pathlib.Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name,repo,assets", [
    ("pi05_bpx_full", "bpx/bpx_blocks", "./assets/pi05_bpx_full"),
    ("pi05_bpx_full_low_mem", "bpx/bpx_blocks", "./assets/pi05_bpx_full"),
    ("pi05_bpx_full_low_mem_combined", "bpx/bpx_blocks_combined", None),
    ("pi05_bpx_full_low_mem_multitask_v2", "bpx/bpx_blocks_multitask_v2", None),
    ("pi05_bpx_lora", "bpx/bpx_blocks", "./assets/pi05_bpx_full"),
])
def test_reads_actual_config_bindings_without_imports(name, repo, assets):
    binding = read_config_binding(PROJECT, name)
    assert binding.data.repo_id == repo
    assert binding.data.assets.assets_dir == assets
    assert binding.assets_base_dir == "./assets"


def test_static_reader_does_not_execute_source_or_guess_dynamic_values(tmp_path):
    source = tmp_path / "src/openpi/training/config.py"
    source.parent.mkdir(parents=True)
    source.write_text('raise RuntimeError("MUST NOT EXECUTE")\n_CONFIGS = [TrainConfig(name="custom", '
                      'data=Data(repo_id="team/my_data", assets=AssetsConfig(asset_id="team/my_data")))]')
    assert read_config_binding(tmp_path, "custom").data.repo_id == "team/my_data"
    source.write_text(source.read_text().replace('repo_id="team/my_data"', 'repo_id=download_data()'))
    with pytest.raises(ValueError, match="字面量"):
        read_config_binding(tmp_path, "custom")


def test_live_config_endpoint_never_imports_training_runtime(tmp_path, monkeypatch):
    app = api.create_app(state_dir=tmp_path / "state", project_root=PROJECT, data_root=tmp_path)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("jax", "torch", "transformers", "openpi.models")) or name == "openpi.training.config":
            raise AssertionError(f"Config preview imported heavyweight runtime: {name}")
        return original(name, *args, **kwargs)

    with TestClient(app) as client:
        monkeypatch.setattr(builtins, "__import__", guarded)
        start = time.monotonic()
        response = client.get("/api/config-profile", params={"config_name": "pi05_bpx_full_low_mem_combined"})
        assert time.monotonic()-start < 2
        assert response.status_code == 200, response.text
        assert response.json()["repo_id"] == "bpx/bpx_blocks_combined"
        assert response.json()["dataset_path"] == str((tmp_path / "lerobot/bpx/bpx_blocks_combined").resolve())
        assert client.get("/api/config-profile", params={"config_name": "/scratch/not-a-config"}).status_code == 422
        assert client.get("/").status_code == 200


def test_holdout_profile_uses_the_same_pure_binding(tmp_path):
    from openpi.training.validation_split import configure
    from openpi.training.validation_split import split_digest
    import json

    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"schema_version": 1, "groups": ["a", "b"],
                                    "train_episodes": [0], "validation_episodes": [1]}))
    binding = read_config_binding(PROJECT, "pi05_bpx_full_low_mem_combined")
    selected = configure(dataclasses.replace(binding, validation_manifest=str(manifest)))
    assert selected.assets_base_dir.endswith("holdout-" + split_digest(manifest))
    app = api.create_app(state_dir=tmp_path / "state", project_root=PROJECT, data_root=tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/config-profile", params={"config_name": binding.name,
                                                             "validation_manifest": str(manifest)})
    assert response.status_code == 200
    assert f"holdout-{split_digest(manifest)}" in response.json()["loaded_stats_path"]


def test_hung_gpu_probe_is_coalesced_and_does_not_block_requests(tmp_path, monkeypatch):
    release = threading.Event()
    calls = []

    def blocked():
        calls.append(1)
        release.wait(5)
        return {"gpus": [], "gpu_error": "synthetic driver response"}

    monkeypatch.setattr(api, "_gpu_snapshot", blocked)
    app = api.create_app(state_dir=tmp_path / "state", project_root=PROJECT, data_root=tmp_path)
    try:
        with TestClient(app) as client:
            start = time.monotonic()
            for _ in range(4):
                response = client.get("/api/system")
                assert response.status_code == 200
                assert response.json()["gpu_pending"]
            assert client.get("/healthz").json() == {"ok": True}
            assert client.get("/").status_code == 200
            assert client.get("/api/config-profile", params={"config_name": api.DEFAULT_CONFIG_NAME}).status_code == 200
            assert time.monotonic()-start < 2
            assert len(calls) == 1
    finally:
        release.set()


def test_timeout_is_visible_and_probe_recovers():
    release = threading.Event()
    def blocked():
        release.wait(3)
        return {"gpus": [{"name": "recovered"}], "gpu_error": None}
    probe = CachedProbe(blocked, timeout=0.001)
    try:
        assert "超时" in probe.snapshot()["gpu_error"]
    finally:
        release.set()
    assert probe.ready.wait(2)
    assert probe.snapshot()["gpus"][0]["name"] == "recovered"
    assert not probe.snapshot()["gpu_pending"]
