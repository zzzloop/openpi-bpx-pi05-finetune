import numpy as np
import pytest
import smoke_test_bpx_checkpoint

from openpi.policies import bpx_constants


def _valid_actions() -> np.ndarray:
    actions = np.zeros((30, bpx_constants.BPX_RAW_DIM), dtype=np.float32)
    actions[:, 10] = actions[:, 11] = 0.21
    actions[:, 19] = actions[:, 20] = 0.34
    return actions


def test_validate_output_contract_accepts_explicitly_broadcast_waist():
    actions = _valid_actions()

    smoke_test_bpx_checkpoint.validate_output_contract(actions, (30, bpx_constants.BPX_RAW_DIM))


def test_validate_output_contract_rejects_nonzero_waist():
    actions = _valid_actions()
    actions[5, 1] = 0.1

    with pytest.raises(AssertionError, match="Frozen waist"):
        smoke_test_bpx_checkpoint.validate_output_contract(actions, (30, bpx_constants.BPX_RAW_DIM))


def test_validate_output_contract_rejects_unequal_gripper_slots():
    actions = _valid_actions()
    actions[7, 11] = 0.2

    with pytest.raises(AssertionError, match="Gripper slots 10/11"):
        smoke_test_bpx_checkpoint.validate_output_contract(actions, (30, bpx_constants.BPX_RAW_DIM))
