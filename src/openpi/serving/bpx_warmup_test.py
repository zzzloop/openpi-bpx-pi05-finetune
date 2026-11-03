"""Server-side tests use synthetic arrays only; never load a model checkpoint."""
import numpy as np
import pytest

from openpi.serving.websocket_policy_server import _warmup_bpx_policy


class FakePolicy:
    def __init__(self, invalid=False):
        self.calls = 0
        self.invalid = invalid

    def infer(self, observation):
        self.calls += 1
        assert observation["qpos"].shape == (23,)
        assert set(observation["images"]) == {"left_eye", "left_wrist", "right_wrist"}
        assert all(image.shape == (360, 640, 3) and image.dtype == np.uint8
                   for image in observation["images"].values())
        return {"actions": np.full((30, 23), np.nan if self.invalid else 0.0)}


def test_bpx_warmup_finishes_twice_without_transport():
    policy = FakePolicy()
    _warmup_bpx_policy(policy, {"raw_action_dim": 23, "effective_action_dim": 18, "action_horizon": 30})
    assert policy.calls == 2


def test_non_bpx_policy_is_untouched():
    policy = FakePolicy()
    _warmup_bpx_policy(policy, {})
    assert policy.calls == 0


def test_invalid_warmup_fails_before_server_can_listen():
    with pytest.raises(ValueError, match="invalid actions"):
        _warmup_bpx_policy(FakePolicy(invalid=True),
                           {"raw_action_dim": 23, "effective_action_dim": 18, "action_horizon": 30})
