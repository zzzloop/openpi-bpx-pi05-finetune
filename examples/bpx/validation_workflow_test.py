"""CPU-only contracts: no base weights, CUDA, ROS or training environment required."""

import ast
import copy
import dataclasses
import json
import pathlib
import sys
import types

import numpy as np
import pytest

from openpi.training import loss_trends
from openpi.training import validation
from openpi.training import validation_split as split


def make_dataset(root, *, groups=4):
    meta = root / "meta"
    meta.mkdir(parents=True)
    records, sources = [], []
    for task in ("red", "blue"):
        for group in range(groups):
            for aug in ("", "__aug_dark", "__aug_bright"):
                index = len(records)
                records.append({"episode_index": index, "length": 40, "tasks": [task]})
                sources.append({"episode_index": index, "frames": 40, "task": task,
                                "source_kind": "rosbag", "path": f"/source/{task}/{group}{aug}.bag"})
    (meta / "info.json").write_text(json.dumps({"total_episodes": len(records)}))
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    source = root / "source.json"
    source.write_text(json.dumps({"episodes": sources}))
    return source


def test_grouped_split_prevents_augmentation_leakage_and_is_reproducible(tmp_path):
    root = tmp_path / "dataset"
    source = make_dataset(root)
    result = split.build_split(root, tmp_path / "a.json", source_manifest=source)
    again = split.build_split(root, tmp_path / "b.json", source_manifest=source)
    assert result == again
    assert len(result["validation_episodes"]) == 6
    assert result["train_frames"] + result["validation_frames"] == 24 * 40
    assert {result["groups"][i] for i in result["train_episodes"]}.isdisjoint(
        {result["groups"][i] for i in result["validation_episodes"]})
    assert split.load_split(tmp_path / "a.json") == result
    assert split.split_digest(tmp_path / "a.json") == split.split_digest(tmp_path / "b.json")


def test_split_requires_provenance_or_explicit_independence(tmp_path):
    root = tmp_path / "dataset"
    make_dataset(root)
    with pytest.raises(ValueError, match="source manifest"):
        split.build_split(root, tmp_path / "a.json")
    split.build_split(root, tmp_path / "a.json", independent_episodes=True)
    with pytest.raises(FileExistsError):
        split.build_split(root, tmp_path / "a.json", independent_episodes=True)


def test_source_mismatch_and_rare_task_fail_explicitly(tmp_path):
    root = tmp_path / "dataset"
    source = make_dataset(root, groups=1)
    with pytest.raises(ValueError, match="two independent"):
        split.build_split(root, tmp_path / "a.json", source_manifest=source)
    content = json.loads(source.read_text())
    content["episodes"][0]["frames"] = 99
    source.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="match output"):
        split.build_split(root, tmp_path / "a.json", source_manifest=source)


def test_tampered_cross_split_group_is_rejected(tmp_path):
    source = make_dataset(tmp_path / "dataset")
    path = tmp_path / "split.json"
    result = split.build_split(source.parent, path, source_manifest=source)
    result["groups"][result["validation_episodes"][0]] = result["groups"][result["train_episodes"][0]]
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="overlap"):
        split.load_split(path)


@dataclasses.dataclass(frozen=True)
class Assets:
    assets_dir: str | None = "old_assets"
    asset_id: str = "bpx/test"


@dataclasses.dataclass(frozen=True)
class Factory:
    assets: Assets = Assets()


@dataclasses.dataclass(frozen=True)
class Config:
    validation_manifest: str | None
    data: Factory = Factory()
    assets_base_dir: str = "assets"


def test_training_only_stats_have_independent_stable_path(tmp_path):
    source = make_dataset(tmp_path / "dataset")
    path = tmp_path / "split.json"
    split.build_split(source.parent, path, source_manifest=source)
    original = Config(str(path))
    configured = split.configure(original)
    assert configured.assets_base_dir.startswith("assets")
    assert "holdout-" in configured.assets_base_dir
    assert configured.data.assets.assets_dir is None
    assert split.configure(configured) == configured
    assert original.data.assets.assets_dir == "old_assets"
    legacy = Config(None)
    assert split.configure(legacy) is legacy


def test_training_and_validation_load_disjoint_episodes_and_reject_changed_metadata(tmp_path, monkeypatch):
    root = tmp_path / "datasets" / "bpx" / "test"
    source = make_dataset(root)
    path = tmp_path / "split.json"
    result = split.build_split(root, path, source_manifest=source)
    module = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
    module.HF_LEROBOT_HOME = tmp_path / "datasets"
    monkeypatch.setitem(sys.modules, module.__name__, module)

    @dataclasses.dataclass
    class Data:
        repo_id: str = "bpx/test"
        episode_indices: tuple | None = None
        balance_by_task: bool = True

    config = Config(str(path))
    train = split.select_data_config(config, Data())
    val = split.select_data_config(config, Data(), "validation")
    assert train.episode_indices == tuple(result["train_episodes"])
    assert val.episode_indices == tuple(result["validation_episodes"])
    assert train.balance_by_task and not val.balance_by_task
    (root / "meta/info.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="metadata changed"):
        split.select_data_config(config, Data())


def test_evaluation_history_tracks_best_raw_value_and_resume(tmp_path):
    identity = {"seed": 42, "split": "test"}
    history = validation.EvaluationHistory(tmp_path, identity, start_step=0, resume=False)
    for step, train, val in ((0, 1, 1), (100, .8, .7), (200, .7, .8), (300, .6, .9), (400, .5, 1.0)):
        record = history.append(step, train, val, train_stderr=.01, val_stderr=.02,
                                seconds=.1, checkpoint_saved=step % 200 == 0)
    assert record["overfit_warning"]
    assert record["best_validation_step"] == 100
    assert record["best_evaluated_saved_step"] == 200
    resumed = validation.EvaluationHistory(tmp_path, identity, start_step=201, resume=True)
    assert [r["step"] for r in resumed.records] == [0, 100, 200]
    with pytest.raises(ValueError, match="protocol"):
        validation.EvaluationHistory(tmp_path, {"seed": 43}, start_step=201, resume=True)


def test_evaluation_metrics_reject_nonfinite_values():
    mean, se = validation.evaluate_batches([1, 2, 3], lambda i, batch: batch)
    assert mean == 2 and se > 0
    with pytest.raises(FloatingPointError):
        validation.evaluate_batches([1], lambda i, batch: np.nan)


def test_fixed_reference_samples_cover_tasks_without_replacement():
    inventory = [("common", 100), ("rare", 8), ("common", 50)]
    indices = validation.reference_frame_indices(inventory, 16, 42)
    assert indices == validation.reference_frame_indices(inventory, 16, 42)
    assert len(set(indices)) == 16
    assert sum(100 <= index < 108 for index in indices) == 8
    with pytest.raises(ValueError, match="validation_batches"):
        validation.reference_frame_indices(inventory, 1, 42)


def test_training_dataset_selection_keeps_whole_episode_and_original_chunk_source():
    source = pathlib.Path(__file__).resolve().parents[2] / "src/openpi/training/data_loader.py"
    function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "create_torch_dataset")

    class Dataset:
        def __init__(self, *args, **kwargs):
            self.episode_data_index = {"from": [0, 3, 7], "to": [3, 7, 9]}

        def __getitem__(self, index):
            return {"original_query_index": index}

    class Subset:
        def __init__(self, dataset, indices):
            self.dataset, self.indices = dataset, indices

    namespace = {"lerobot_dataset": types.SimpleNamespace(LeRobotDatasetMetadata=lambda _: types.SimpleNamespace(fps=30),
                                                        LeRobotDataset=Dataset),
                 "torch": types.SimpleNamespace(utils=types.SimpleNamespace(data=types.SimpleNamespace(Subset=Subset)))}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    config = types.SimpleNamespace(repo_id="test/data", action_sequence_keys=("action",), episode_indices=(0, 2), prompt_from_task=False)
    dataset = namespace["create_torch_dataset"](config, 30, None)
    assert dataset.indices == [0, 1, 2, 7, 8]
    assert dataset.dataset[7]["original_query_index"] == 7


def test_dashboard_commands_use_the_same_manifest_for_stats_and_training(tmp_path):
    from openpi.bpx_studio.commands import build_normalize, build_train, build_validation_split

    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    for name in ("compute_norm_stats.py", "serve_policy.py", "train.py"):
        (root / "scripts" / name).touch()
    (root / "pyproject.toml").touch()
    source = make_dataset(tmp_path / "dataset")
    manifest = tmp_path / "split.json"
    split.build_split(source.parent, manifest, source_manifest=source)
    values = {"openpi_root": str(root), "config_name": "test", "validation_manifest": str(manifest),
              "exp_name": "new", "checkpoint_base_dir": str(tmp_path / "checkpoints"), "cuda_devices": "2,3",
              "memory_fraction": .96, "batch_size": 2, "fsdp_devices": 2, "num_train_steps": 1000,
              "save_interval": 5000, "keep_period": 25000, "validation_interval": 1000, "validation_batches": 16}
    assert str(manifest) in build_normalize(values).argv
    train = build_train(values)
    assert f"--validation-manifest={manifest}" in train.argv
    assert "--validation-interval=1000" in train.argv
    assert train.env["CUDA_VISIBLE_DEVICES"] == "2,3"
    spec = build_validation_split(root, {"dataset_root": str(source.parent), "output_path": str(tmp_path / "future.json"), "source_manifest": str(source)})
    assert "openpi.training.validation_split" in spec.argv


def test_forward_evaluation_does_not_modify_params_or_use_training_mode():
    source = pathlib.Path(__file__).resolve().parents[2] / "scripts/train.py"
    function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "eval_step")
    params = {"value": np.array([1., 2.])}
    before = copy.deepcopy(params)

    class Model:
        def eval(self):
            self.evaluation = True

        def compute_loss(self, rng, observation, actions, *, train):
            assert self.evaluation and train is False
            return params["value"] + actions

    namespace = {"nnx": types.SimpleNamespace(merge=lambda definition, p: Model()), "jnp": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["eval_step"](None, params, None, (None, np.array([3., 4.]))) == 5
    np.testing.assert_array_equal(params["value"], before["value"])


def test_trend_removes_isolated_and_short_bursts_without_mutating_raw():
    values = np.ones(45)
    values[[12, 25, 26]] = 100
    original = values.copy()
    trend, outliers = loss_trends.robust_trend(values)
    assert outliers[[12, 25, 26]].all()
    np.testing.assert_allclose(trend, 1)
    np.testing.assert_array_equal(values, original)


def test_trend_preserves_divergence_missing_values_and_rejects_bad_options():
    values = np.r_[np.ones(15), np.full(15, 3.), np.nan]
    trend, _ = loss_trends.robust_trend(values)
    assert trend[-2] > 2.8
    assert np.isnan(trend[-1])
    with pytest.raises(ValueError):
        loss_trends.robust_trend(values, window=4)


def test_monitor_compares_raw_eval_losses_and_preserves_all_peaks(tmp_path):
    from openpi.training.monitoring import TrainingMonitor

    monitor = TrainingMonitor(tmp_path, batch_size=2, plot_interval=1000, visualization_interval=1000)
    for step, value in ((0, 1.), (100, 10.)):
        paths = monitor.log_metrics(step, {"loss": value, "train_eval_loss": .1, "val_loss": .2,
                                          "generalization_gap": .1, "val_stderr": .01}, force_plot=True)
    records = [json.loads(line) for line in monitor.jsonl_path.read_text().splitlines()]
    assert records[-1]["loss"] == 10.
    assert records[-1]["val_loss"] == .2
    assert (tmp_path / "generalization_latest.png").is_file()
    assert all(p.is_file() for p in paths)
