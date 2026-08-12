import clean_bpx_hdf5
import numpy as np


def _episode(frames: int = 15) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.zeros((frames, 23), dtype=np.float32)
    actions = np.zeros_like(qpos)
    return qpos, actions


def test_repairs_single_joint_spike_with_neighbor_mean():
    qpos, actions = _episode()
    qpos[:, 7] = np.linspace(0.0, 0.14, len(qpos))
    qpos[7, 7] = -3.0

    repaired, repairs, rejected = clean_bpx_hdf5.repair_isolated_qpos_spikes(qpos, actions)

    assert len(repairs) == 1
    assert rejected == 0
    assert repairs[0].frame_index == 7
    assert repairs[0].joint_index == 7
    np.testing.assert_allclose(repaired[7, 7], (qpos[6, 7] + qpos[8, 7]) * 0.5)


def test_repairs_consecutive_and_alternating_spikes():
    qpos, actions = _episode()
    qpos[5:7, 16] = -3.0
    qpos[9, 21] = -3.0
    qpos[11, 21] = -3.0

    repaired, repairs, _ = clean_bpx_hdf5.repair_isolated_qpos_spikes(qpos, actions)

    assert {(repair.frame_index, repair.joint_index) for repair in repairs} == {
        (5, 16),
        (6, 16),
        (9, 21),
        (11, 21),
    }
    np.testing.assert_array_equal(repaired[:, :7], qpos[:, :7])


def test_repairs_smaller_gripper_feedback_spike_in_both_raw_slots():
    qpos, actions = _episode()
    qpos[:, 10:12] = 0.01
    qpos[7, 10:12] = 0.15

    repaired, repairs, _ = clean_bpx_hdf5.repair_isolated_qpos_spikes(qpos, actions)

    assert {(repair.frame_index, repair.joint_index) for repair in repairs} == {(7, 10), (7, 11)}
    np.testing.assert_allclose(repaired[7, 10:12], 0.01)


def test_does_not_repair_sustained_step_change():
    qpos, actions = _episode()
    qpos[7:, 7] = 1.0

    repaired, repairs, _ = clean_bpx_hdf5.repair_isolated_qpos_spikes(qpos, actions)

    assert not repairs
    np.testing.assert_array_equal(repaired, qpos)
