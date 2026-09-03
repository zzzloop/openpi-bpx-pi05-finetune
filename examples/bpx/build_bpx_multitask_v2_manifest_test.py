from build_bpx_multitask_v2_manifest import _select_representative_indices
from build_bpx_multitask_v2_manifest import _trajectory_signature
import numpy as np
import pytest


def test_trajectory_signature_has_fixed_shape_and_rejects_bad_inputs():
    state = np.arange(6 * 18, dtype=np.float32).reshape(6, 18)
    action = state + 0.25
    signature = _trajectory_signature(state, action, points=4)
    assert signature.shape == (4 * 18 * 3 + 1,)
    assert np.all(np.isfinite(signature))
    with pytest.raises(ValueError, match="matching"):
        _trajectory_signature(state[:, :-1], action[:, :-1], points=4)


def test_representative_selection_is_deterministic_and_covers_separated_groups():
    signatures = np.asarray([[0.0], [0.1], [10.0], [10.1], [20.0], [20.1]])
    first = _select_representative_indices(signatures, keep=3)
    second = _select_representative_indices(signatures, keep=3)
    assert first == second
    assert {index // 2 for index in first} == {0, 1, 2}
