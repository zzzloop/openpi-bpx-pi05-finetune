"""Test the zero-state serving path with a synthetic checkpoint and sampler."""

from types import SimpleNamespace

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import bpx_constants
from openpi.policies import bpx_policy
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config


@pytest.mark.parametrize("config_name", ["pi05_bpx_cf_naive_full", "pi05_bpx_cf_policy_cfg"])
def test_zero_state_model_input_preserves_live_images_language_and_delta_decode(config_name, tmp_path, monkeypatch):
    cfg = config.get_config(config_name)
    seen = []

    class Tokenizer:
        def tokenize(self, prompt, state):
            return np.array([len(prompt), round(float(state[0]) * 100)]), np.ones(2, dtype=bool)

    def sample(_key, obs, **kwargs):
        seen.append((obs, kwargs))
        return np.zeros((1, 30, 32), dtype=np.float32)

    dim = bpx_constants.BPX_EFFECTIVE_DIM
    stats = transforms.NormStats(mean=np.full(dim, 0.25), std=np.ones(dim), q01=np.full(dim, -1.0),
                                 q99=np.full(dim, 3.0))
    action_stats = transforms.NormStats(mean=np.zeros(dim), std=np.ones(dim), q01=np.full(dim, -1.0),
                                        q99=np.ones(dim))
    data = config.DataConfig(
        use_quantile_norm=True,
        data_transforms=transforms.Group(
            inputs=(bpx_policy.BpxInputs(),),
            outputs=(transforms.AbsoluteActions(bpx_constants.BPX_DELTA_ACTION_MASK), bpx_policy.BpxOutputs()),
        ),
        model_transforms=transforms.Group(inputs=(transforms.TokenizePrompt(Tokenizer(), discrete_state_input=True),
                                                  transforms.PadStatesAndActions(32))),
    )
    monkeypatch.setattr(type(cfg.data), "create", lambda *args: data)
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda _: tmp_path)
    monkeypatch.setattr(policy_config._model, "restore_params", lambda *args, **kwargs: {})
    monkeypatch.setattr(type(cfg.model), "load", lambda *args: SimpleNamespace(sample_actions=sample))
    monkeypatch.setattr(nnx_utils, "module_jit", lambda fn: fn)
    kwargs = {"norm_stats": {"state": stats, "actions": action_stats}}
    normal = policy_config.create_trained_policy(cfg, tmp_path, **kwargs)
    zero = policy_config.create_trained_policy(cfg, tmp_path, bpx_zero_state=True, **kwargs)

    obs = bpx_policy.make_bpx_example()
    first = {**obs, "qpos": np.zeros(dim, dtype=np.float32), "prompt": "pick red"}
    second = {**first, "qpos": np.full(dim, 0.5, dtype=np.float32)}
    a = normal.infer(first)["actions"]
    b = normal.infer(second)["actions"]
    c = zero.infer(first)["actions"]
    d = zero.infer(second)["actions"]
    np.testing.assert_allclose(c, a)
    np.testing.assert_allclose(d, b)
    np.testing.assert_array_less(0, np.abs(d[:, 3] - c[:, 3]))
    original, changed, ablated_a, ablated_b = (item[0] for item in seen)
    np.testing.assert_allclose(ablated_b.state, original.state)
    np.testing.assert_allclose(ablated_a.state[:, :dim], -0.5, atol=1e-6)
    np.testing.assert_array_equal(ablated_a.tokenized_prompt, ablated_b.tokenized_prompt)
    assert not np.array_equal(changed.tokenized_prompt, ablated_b.tokenized_prompt)
    assert int(ablated_b.tokenized_prompt[0, 0]) == len("pick red")
    for camera in original.images:
        np.testing.assert_array_equal(ablated_b.images[camera], changed.images[camera])
    if config_name.endswith("policy_cfg"):
        np.testing.assert_array_equal(seen[2][1]["null_prompt"], seen[3][1]["null_prompt"])
        assert not np.array_equal(seen[1][1]["null_prompt"], seen[3][1]["null_prompt"])
    assert "bpx_input_mode" not in normal.metadata
    assert zero.metadata["bpx_input_mode"] == "zero_state"
    assert not np.any(first["qpos"]) and np.all(second["qpos"] == 0.5)
