"""Contract tests without OpenPI models, a real dataset, or GPUs."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from openpi.bpx_studio import commands
from openpi.bpx_studio.config_profile import read_config_binding
from openpi.training import bpx_counterfactual as cf
from openpi.training import validation_split


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    meta = root / "meta"
    meta.mkdir(parents=True)
    features = {name: {"shape": [18]} for name in ("observation.state", "action")}
    features.update({f"observation.images.{name}": {"dtype": "video"}
                     for name in ("left_eye", "left_wrist", "right_wrist")})
    (meta / "info.json").write_text(json.dumps({"fps": 30, "features": features}))
    tasks = ["Pick red", "Pick blue", "Pick red", "Pick blue"]
    rows = [{"episode_index": i, "length": 80 + i, "tasks": [tasks[i % 4]]} for i in range(12)]
    (meta / "episodes.jsonl").write_text("\n".join(map(json.dumps, rows)))
    groups = [
        {"group_id": f"g{i}", "family": "task3", "episodes": [4*i+j for j in (2, 3, 0, 1)],
         "windows": [[5, 10], [10, 20], [15, 20], [20, 25]], "attention_semantic": [[0, 1], [2, 3]],
         "attention_swap": [[1, 2]]}
        for i in range(3)
    ]
    groups[1]["split_group"] = groups[0]["group_id"]
    manifest = tmp_path / "quads.json"
    payload = {"schema_version": 1, "metadata_sha256": validation_split.inventory(root)[1], "groups": groups}
    manifest.write_text(json.dumps(payload))
    return root, manifest, payload


def test_split_binding_sampling_and_resume(dataset, tmp_path):
    root, manifest, _ = dataset
    output = tmp_path / "validation.json"
    split = cf.build_split(root, manifest, output)
    train = split["train_episodes"]
    assert {split["groups"][i] for i in train}.isdisjoint({split["groups"][i] for i in split["validation_episodes"]})
    assert (0 in train) == (4 in train)
    quads, rows = cf.bound_quads(manifest, root, output, require_attention=True)
    sampler = cf.QuadBatchSampler(quads, rows, train, seed=42)
    it = iter(sampler)
    batches = [next(it) for _ in range(11)]
    for offset in (3, 10):
        sampler.start_step = offset
        assert next(iter(sampler)) == batches[offset]
    for step, batch in enumerate(batches):
        assert len(batch) == 4
        if not cf.is_aligned_step(step):
            assert all(0 <= index < sampler.frame_count and mask is None for index, mask in batch)
            continue
        assert any(
            all(sampler.offsets[i] + lo <= index <= sampler.offsets[i] + hi and mask == q.attention
                for (index, mask), i, (lo, hi) in zip(batch, q.episodes, q.windows, strict=True))
            for q in quads
        )
    cases = cf.heldout_probe_cases(split, rows, 16)
    assert all(c["episode_index"] in split["validation_episodes"] and
               c["counterfactual_prompt"] != rows[c["episode_index"]]["tasks"][0] and
               not c["semantic_reviewed"] for c in cases)
    with pytest.raises(FileExistsError):
        cf.build_split(root, manifest, output)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="exact quadruple"):
        cf.bound_quads(manifest, root, output)


def test_whole_trajectory_covers_every_training_frame_including_prelude_and_tail(dataset, tmp_path):
    root, manifest, _ = dataset
    output = tmp_path / "validation.json"
    train = cf.build_split(root, manifest, output)["train_episodes"]
    quads, rows = cf.bound_quads(manifest, root, output)
    sampler = cf.QuadBatchSampler(quads, rows, train, seed=42)
    batches = iter(sampler)
    frames = []
    for step in range(2 * ((sampler.frame_count + 3) // 4)):
        batch = next(batches)
        if not cf.is_aligned_step(step):
            frames.extend(index for index, _ in batch)
    assert sorted(frames[:sampler.frame_count]) == list(range(sampler.frame_count))


@pytest.mark.parametrize(("key", "value", "error"), [
    ("attention_swap", [[0, 3]], "Task 3"),
    ("windows", [[0, 1000]] * 4, "real 30-frame"),
    ("episodes", [0, 0, 2, 3], "four distinct"),
    ("episodes", [0, 1, 3, 2], "matching task labels"),
])
def test_reject_invalid_annotations(dataset, key, value, error):
    root, manifest, payload = dataset
    payload["groups"][0][key] = value
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=error):
        cf.load_quads(manifest, root)


def test_train_and_norm_commands_are_opt_in(dataset, tmp_path):
    dataset_root, quad, _ = dataset
    split = tmp_path / "validation.json"
    cf.build_split(dataset_root, quad, split)
    project = Path(__file__).resolve().parents[3]
    params = tmp_path / "stage1/18000/params"
    params.mkdir(parents=True)
    values = dict(openpi_root=str(project), exp_name="exp", checkpoint_base_dir=str(tmp_path / "checkpoints"),
                  memory_fraction=.96, num_train_steps=10, save_interval=10, keep_period=10, mode="overwrite",
                  cuda_devices="0,1,2,3", fsdp_devices=4, batch_size=4, validation_manifest=str(split),
                  counterfactual_manifest=str(quad), init_checkpoint=str(params.parent))
    for config_name in cf.CONFIGS:
        binding = read_config_binding(project, config_name)
        assert binding.name == config_name and binding.data.repo_id == "bpx/bpx_v1_counterfactual_v001"
        spec = commands.build_train(dict(values, config_name=config_name))
        assert spec.env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert f"--counterfactual-manifest={quad}" in spec.argv and "--batch-size=4" in spec.argv
        preflight = commands.build_train(dict(values, config_name=config_name, stage2_smoke_test=True))
        assert "--stage2-smoke-test" in preflight.argv and "--overwrite" not in preflight.argv
        assert preflight.total_units == 2 and preflight.env["WANDB_MODE"] == "disabled"
        assert "WANDB_DIR" not in preflight.env
        with pytest.raises(ValueError, match="GPU 预检"):
            commands.build_train(dict(values, config_name=config_name, stage2_smoke_test=True, mode="resume"))
        norm = commands.build_normalize(dict(values, config_name=config_name))
        assert "--validation-manifest" in norm.argv
        with pytest.raises(ValueError, match="第二阶段"):
            commands.build_train(dict(values, config_name=config_name, cuda_devices="0,0,2,3"))
        with pytest.raises(ValueError, match="四元组"):
            commands.build_normalize(dict(values, config_name=config_name, validation_manifest=""))
    legacy = commands.build_train(dict(
        values, config_name="pi05_bpx_full_low_mem_v1_base_estimated", cuda_devices="2,3",
        fsdp_devices=2, batch_size=2, validation_manifest="", init_checkpoint=""
    ))
    assert not any("counterfactual" in item for item in legacy.argv)
    assert legacy.env["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_freeze_exact_vlm_paths():
    for key in ("img/encoderblock/posemb", "llm/embedder/input_embedding",
                "llm/layers/attn/kv_einsum/w", "llm/final_norm/scale"):
        assert cf.is_vlm_parameter(("PaliGemma", *key.split("/")))
    for key in ("PaliGemma/llm/layers/attn/q_einsum_1/w", "PaliGemma/llm/layers/mlp_1/linear",
                "PaliGemma/llm/final_norm_1/Dense_0/kernel", "action_out_proj/kernel", "time_mlp_out/bias"):
        assert not cf.is_vlm_parameter(tuple(key.split("/")))


@pytest.fixture
def replay_dataset(dataset):
    root, manifest, payload = dataset
    meta = root / "meta"
    records, _ = validation_split.inventory(root)
    records.extend({"episode_index": i, "length": 40, "tasks": [f"Extra task {i}"]} for i in range(12, 16))
    (meta / "episodes.jsonl").write_text("\n".join(map(json.dumps, records)))
    provenance = [dict(episode_index=i, source_episode=i + 2, task=r["tasks"][0], frames=r["length"],
                       old_base_episode=i - 13 if i in (13, 14) else None,
                       estimated_coordinates=10 if i in (12, 13) else 0)
                  for i, r in enumerate(records)]
    p = meta / "bpx_episode_provenance.jsonl"
    p.write_text("\n".join(map(json.dumps, provenance)))
    (meta / "bpx_feedback_recovery.json").write_text("{}")
    payload.update(schema_version=2, sampling_protocol=cf.REPLAY_PROTOCOL,
                   metadata_sha256=validation_split.inventory(root)[1],
                   provenance_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                   heldout_split_groups=["g2"], old_replay_episodes=[13], excluded_episodes=[14, 15])
    for q in payload["groups"]:
        q["source_episodes"] = [i + 2 for i in q["episodes"]]
    manifest.write_text(json.dumps(payload))
    return root, manifest, payload


def test_replay_exact_ratio_full_coverage_and_resume(replay_dataset, tmp_path):
    root, manifest, _ = replay_dataset
    path = tmp_path / "replay-split.json"
    split = cf.build_split(root, manifest, path)
    assert split["train_episodes"] == [*range(8), 12, 13]
    assert split["validation_episodes"] == list(range(8, 12))
    assert split["excluded_episodes"] == [14, 15]
    quads, rows = cf.bound_quads(manifest, root, path, require_attention=True)
    sampler = cf.QuadBatchSampler(quads, rows, split["train_episodes"], 42, old_replay_episodes=[13])
    it = iter(sampler)
    batches = [next(it) for _ in range(400)]
    old_start = sampler.offsets[13]
    old_draws, new_draws = [], []
    for step, batch in enumerate(batches):
        aligned = cf.is_aligned_step(step, cf.REPLAY_PROTOCOL)
        assert aligned == (step % 4 == 3)
        assert all(0 <= i < sampler.frame_count and (mask is not None) == aligned for i, mask in batch)
        if aligned:
            assert all(i < sampler.offsets[12] for i, _ in batch)
        else:
            for i, _ in batch:
                (old_draws if i >= old_start else new_draws).append(i)
    assert sum(i >= old_start for b in batches[:20] for i, _ in b) == 12
    assert sum(i >= old_start for b in batches[20:40] for i, _ in b) == 12
    assert sorted(old_draws[:40]) == list(range(old_start, old_start + 40))
    assert sorted(new_draws[:old_start]) == list(range(old_start))
    for step in (0, 1, 3, 4, 19, 20, 39, 398, 399):
        sampler.start_step = step
        assert next(iter(sampler)) == batches[step]
    # The split must match the reviewed binding, even when a modified split
    # still partitions the dataset and holds out complete quads.
    split["old_replay_episodes"] = [12]
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="partition differs"):
        cf.bound_quads(manifest, root, path)


def test_replay_dashboard_and_norm_dataset_binding(replay_dataset, tmp_path, monkeypatch):
    root, quad, _ = replay_dataset
    path = tmp_path / "replay-split.json"
    split = cf.build_split(root, quad, path)
    project = Path(__file__).resolve().parents[3]
    params = tmp_path / "stage1/18000/params"
    params.mkdir(parents=True)
    options = dict(openpi_root=str(project), exp_name="replay", checkpoint_base_dir=str(tmp_path / "checkpoints"),
                       num_train_steps=100, save_interval=10, keep_period=10, memory_fraction=.96,
                       mode="overwrite", cuda_devices="0,1,2,3",
                   fsdp_devices=4, batch_size=4, validation_manifest=str(path), counterfactual_manifest=str(quad),
                   init_checkpoint=str(params.parent), hf_lerobot_home=str(root.parent), stage2_smoke_test=True)
    for recipe in cf.CONFIGS:
        values = dict(options, config_name=recipe)
        train, norm = commands.build_train(values), commands.build_normalize(values)
        assert train.total_units == 4
        assert train.env["HF_LEROBOT_HOME"] == norm.env["HF_LEROBOT_HOME"] == str(root.parent)
        two_gpu = dict(values, cuda_devices="2,3", fsdp_devices=2)
        if recipe == cf.QUAD_DIFF_CONFIG:
            with pytest.raises(ValueError, match=r"GPU 0,1,2,3 / FSDP=4"):
                commands.build_train(two_gpu)
        else:
            preflight = commands.build_train(two_gpu)
            assert preflight.argv == tuple("--fsdp-devices=2" if arg == "--fsdp-devices=4" else arg
                                           for arg in train.argv)
            assert preflight.env == dict(train.env, CUDA_VISIBLE_DEVICES="2,3")
            assert preflight.total_units == 4
            full = commands.build_train(dict(two_gpu, stage2_smoke_test=False))
            assert "--batch-size=4" in full.argv and "--fsdp-devices=2" in full.argv
            assert full.total_units == options["num_train_steps"]
        with pytest.raises(ValueError, match="global BS=4"):
            commands.build_train(dict(two_gpu, batch_size=2))
    monkeypatch.setitem(sys.modules, "lerobot.common.datasets.lerobot_dataset",
                        SimpleNamespace(HF_LEROBOT_HOME=root.parent))
    from dataclasses import dataclass

    @dataclass
    class DataConfig:
        repo_id: str = root.name
        episode_indices: tuple | None = None
        balance_by_task: bool = False

    config = SimpleNamespace(validation_manifest=path)
    assert validation_split.select_data_config(config, DataConfig()).episode_indices == tuple(split["train_episodes"])
    provenance = root / "meta/bpx_episode_provenance.jsonl"
    provenance.write_text(provenance.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance differs"):
        validation_split.select_data_config(config, DataConfig())


@pytest.mark.parametrize("recipe", cf.CONFIGS)
def test_training_topology_shared_with_cli(recipe):
    cf.validate_training_topology(recipe, batch_size=4, fsdp_devices=4, gpu_count=4)
    if recipe == cf.QUAD_DIFF_CONFIG:
        with pytest.raises(ValueError, match="four GPUs"):
            cf.validate_training_topology(recipe, batch_size=4, fsdp_devices=2, gpu_count=2)
    else:
        cf.validate_training_topology(recipe, batch_size=4, fsdp_devices=2, gpu_count=2)
    for batch_size, fsdp, count in ((2, 2, 2), (4, 4, 2), (4, 2, 4), (4, 1, 1)):
        with pytest.raises(ValueError, match="global BS=4"):
            cf.validate_training_topology(recipe, batch_size=batch_size, fsdp_devices=fsdp, gpu_count=count)


@pytest.mark.parametrize("tamper", ["source_order", "estimated", "old_in_quad", "provenance"])
def test_replay_rejects_wrong_role_sources(replay_dataset, tamper):
    root, manifest, payload = replay_dataset
    p = root / "meta/bpx_episode_provenance.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    group = payload["groups"][0]
    i = group["episodes"][0]
    if tamper == "source_order":
        group["source_episodes"].reverse()
    elif tamper in {"estimated", "old_in_quad"}:
        rows[i]["estimated_coordinates" if tamper == "estimated" else "old_base_episode"] = 1
        p.write_text("\n".join(map(json.dumps, rows)))
        payload["provenance_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    else:
        p.write_text(p.read_text() + "\n")
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="aligned roles|provenance has changed"):
        cf.load_quads(manifest, root)
