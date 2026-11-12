import dataclasses
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import bpx_counterfactual as cf
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_task_balanced_frame_weights_give_each_task_equal_mass():
    inventory = [("common", 3), ("rare", 2), ("common", 1)]

    weights = _data_loader.task_balanced_frame_weights(inventory)

    np.testing.assert_allclose(weights, [0.25, 0.25, 0.25, 0.5, 0.5, 0.25])
    np.testing.assert_allclose(weights[:3].sum() + weights[-1], 1.0)
    np.testing.assert_allclose(weights[3:5].sum(), 1.0)


@pytest.mark.parametrize("recipe", cf.CONFIGS)
def test_stage2_loader_builds_and_collates_recipe_tokens(recipe, monkeypatch):
    mask = np.array([True, False])
    tokenizer = SimpleNamespace(
        tokenize=lambda prompt, _state: (np.array([len(prompt), 0]), mask),
        task_token_mask=lambda _prompt, _state: mask,
    )
    tokenize = transforms.TokenizePrompt(tokenizer, discrete_state_input=True)
    data = _config.DataConfig(repo_id="test/dataset", episode_indices=(0, 1, 2, 3, 4),
                              model_transforms=transforms.Group(inputs=(tokenize,)))
    config = SimpleNamespace(name=recipe, counterfactual_manifest="quads.json", validation_manifest="split.json",
                             seed=42, num_workers=0, model=SimpleNamespace(action_horizon=30))
    quad = cf.Quad("q", "bottle", (0, 1, 2, 3), ((0, 0),) * 4, (True, False, True, False), "q")
    monkeypatch.setattr(cf, "bound_quads", lambda *args, **kwargs: ((quad,), [{"length": 30}] * 5))
    monkeypatch.setattr(_data_loader._validation_split, "load_split",
                        lambda _path: {"schema_version": 2, "old_replay_episodes": [4]})

    class Frames:
        def __len__(self):
            return 150

        def __getitem__(self, _index):
            return {"prompt": "pick bottle", "state": np.zeros(18), "actions": np.zeros((30, 32))}

    monkeypatch.setattr(_data_loader, "create_torch_dataset", lambda *args: Frames())
    monkeypatch.setattr(_data_loader, "transform_dataset", lambda samples, data: _data_loader.TransformedDataset(
        samples, data.model_transforms.inputs,
    ))
    monkeypatch.setattr(_data_loader._model.Observation, "from_dict", staticmethod(lambda batch: batch))

    def batches(dataset, *, batch_sampler, **_kwargs):
        for batch in batch_sampler:
            yield _data_loader._collate_fn([dataset[item] for item in batch])

    monkeypatch.setattr(_data_loader, "TorchDataLoader", batches)
    loader = iter(_data_loader._create_stage2_loader(config, data, sharding=None, num_batches=4))
    ordinary, _, _, aligned = (next(loader)[2] for _ in range(4))
    mode = cf.CONFIGS[recipe]
    if mode == "policy_cfg":
        assert ordinary.keys() == aligned.keys() == {"null_prompt", "null_prompt_mask"}
        np.testing.assert_array_equal(ordinary["null_prompt"], np.zeros((4, 2)))
    elif mode == "ccfa_ours":
        assert ordinary.keys() == {"task_token_mask"}
        assert aligned.keys() == {"task_token_mask", "attention_edges"}
        np.testing.assert_array_equal(aligned["attention_edges"], np.tile(quad.attention, (4, 1)))
    else:
        assert ordinary == aligned == {}
