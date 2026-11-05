"""Contract tests without OpenPI models, a real dataset, or GPUs."""

import json
from pathlib import Path

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
    sampler.start_step = 10
    assert next(iter(sampler)) == batches[-1]
    for batch in batches:
        assert len(batch) == 4
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
