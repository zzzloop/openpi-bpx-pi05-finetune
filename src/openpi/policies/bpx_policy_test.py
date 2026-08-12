import numpy as np
import pytest

from openpi.policies import bpx_constants
from openpi.policies import bpx_policy


def test_bpx_layout_is_consistent():
    assert bpx_constants.BPX_RAW_DIM == 23
    assert bpx_constants.BPX_EFFECTIVE_DIM == 18
    assert bpx_constants.BPX_EFFECTIVE_JOINT_NAMES[0] == "ArmL02_Joint"
    assert bpx_constants.BPX_EFFECTIVE_JOINT_NAMES[7] == "LeftGripper_1DoF"
    assert bpx_constants.BPX_EFFECTIVE_JOINT_NAMES[15] == "RightGripper_1DoF"
    assert bpx_constants.BPX_EFFECTIVE_JOINT_NAMES[-1] == "Head03_Joint"
    assert len(bpx_constants.BPX_DELTA_ACTION_MASK) == bpx_constants.BPX_EFFECTIVE_DIM


def test_to_effective_joints_drops_waist_and_sums_gripper_pairs():
    raw = np.arange(46, dtype=np.float32).reshape(2, 23)
    effective = bpx_constants.to_effective_joints(raw)

    assert effective.shape == (2, 18)
    np.testing.assert_array_equal(effective[:, :7], raw[:, 3:10])
    np.testing.assert_array_equal(effective[:, 7], raw[:, 10] + raw[:, 11])
    np.testing.assert_array_equal(effective[:, 8:15], raw[:, 12:19])
    np.testing.assert_array_equal(effective[:, 15], raw[:, 19] + raw[:, 20])
    np.testing.assert_array_equal(effective[:, 16:18], raw[:, 21:23])


def test_to_effective_joints_rejects_scalar():
    with pytest.raises(ValueError, match="scalar"):
        bpx_constants.to_effective_joints(np.asarray(1.0))


def test_expand_bpx_actions_restores_order_and_splits_grippers():
    effective = np.arange(36, dtype=np.float32).reshape(2, 18)
    raw = bpx_constants.expand_bpx_actions(effective)

    assert raw.shape == (2, 23)
    np.testing.assert_array_equal(raw[:, :3], np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(raw[:, 3:10], effective[:, :7])
    np.testing.assert_allclose(raw[:, 10], effective[:, 7] * 0.5)
    np.testing.assert_allclose(raw[:, 11], effective[:, 7] * 0.5)
    np.testing.assert_array_equal(raw[:, 12:19], effective[:, 8:15])
    np.testing.assert_allclose(raw[:, 19], effective[:, 15] * 0.5)
    np.testing.assert_allclose(raw[:, 20], effective[:, 15] * 0.5)
    np.testing.assert_array_equal(raw[:, 21:23], effective[:, 16:18])
    np.testing.assert_allclose(raw[:, 10] + raw[:, 11], effective[:, 7])
    np.testing.assert_allclose(raw[:, 19] + raw[:, 20], effective[:, 15])


def test_raw_effective_raw_round_trip_preserves_equal_gripper_slots():
    raw = np.arange(46, dtype=np.float32).reshape(2, 23)
    raw[:, :3] = 0
    raw[:, 11] = raw[:, 10]
    raw[:, 20] = raw[:, 19]

    restored = bpx_constants.expand_bpx_actions(bpx_constants.to_effective_joints(raw))
    np.testing.assert_array_equal(restored, raw)


def test_bpx_inputs_accept_raw_qpos_and_ignore_right_eye():
    raw_qpos = np.arange(23, dtype=np.float32)
    raw_actions = np.arange(46, dtype=np.float32).reshape(2, 23)
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    transformed = bpx_policy.BpxInputs()(
        {
            "qpos": raw_qpos,
            "actions": raw_actions,
            "images": {
                "left_eye": image,
                "right_eye": image,
                "left_wrist": image,
                "right_wrist": image,
            },
            "prompt": "Do the task.",
        }
    )

    np.testing.assert_array_equal(transformed["state"], bpx_constants.to_effective_joints(raw_qpos))
    np.testing.assert_array_equal(transformed["actions"], bpx_constants.to_effective_joints(raw_actions))
    assert set(transformed["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert transformed["prompt"] == "Do the task."


def test_bpx_outputs_expand_model_action_to_raw_23():
    model_actions = np.zeros((4, 32), dtype=np.float32)
    model_actions[:, 7] = 0.84
    model_actions[:, 15] = 0.62
    output = bpx_policy.BpxOutputs()({"actions": model_actions})["actions"]

    assert output.shape == (4, 23)
    np.testing.assert_array_equal(output[:, :3], np.zeros((4, 3), dtype=np.float32))
    np.testing.assert_allclose(output[:, 10:12], 0.42)
    np.testing.assert_allclose(output[:, 19:21], 0.31)


def test_bpx_inputs_reject_missing_camera():
    with pytest.raises(ValueError, match="Missing BPX cameras"):
        bpx_policy.BpxInputs()(
            {
                "qpos": np.zeros(23, dtype=np.float32),
                "images": {
                    "left_eye": np.zeros((12, 16, 3), dtype=np.uint8),
                    "left_wrist": np.zeros((12, 16, 3), dtype=np.uint8),
                },
            }
        )
