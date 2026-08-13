import pathlib

import bpx_data
import h5py
import numpy as np
import pytest


def _write_scalar(path: pathlib.Path, key: str, value) -> None:
    with h5py.File(path, "w") as episode:
        episode.create_dataset(key, data=value)


@pytest.mark.parametrize("value", [True, False, np.bool_(1), np.bool_(0)])
def test_read_success_accepts_only_boolean_metadata(tmp_path: pathlib.Path, value):
    path = tmp_path / "episode.hdf5"
    _write_scalar(path, "success", value)

    with h5py.File(path, "r") as episode:
        assert bpx_data.read_success(episode) is bool(value)


@pytest.mark.parametrize("value", [0, 1, "false", b"true"])
def test_read_success_rejects_non_boolean_metadata(tmp_path: pathlib.Path, value):
    path = tmp_path / "episode.hdf5"
    _write_scalar(path, "success", value)

    with h5py.File(path, "r") as episode, pytest.raises(ValueError, match="must be boolean"):
        bpx_data.read_success(episode)


def test_select_episodes_preserves_natural_order_and_rejects_missing():
    files = [pathlib.Path("episode_2.hdf5"), pathlib.Path("episode_10.hdf5")]

    assert bpx_data.select_episodes(files, None) == files
    assert bpx_data.select_episodes(files, []) == files
    assert bpx_data.select_episodes(files, [10, 2]) == files
    with pytest.raises(FileNotFoundError, match=r"\[3\]"):
        bpx_data.select_episodes(files, [3])


def test_read_policy_observation_uses_real_frame_and_drops_right_eye(tmp_path: pathlib.Path):
    path = tmp_path / "episode_0.hdf5"
    frames = 2
    with h5py.File(path, "w") as episode:
        episode.create_dataset("observations/qpos", data=np.arange(frames * 23, dtype=np.float32).reshape(frames, 23))
        episode.create_dataset("action", data=np.zeros((frames, 23), dtype=np.float32))
        episode.create_dataset("task", data="Put the blocks in the bucket.")
        for index, key in enumerate(bpx_data.IMAGE_DATASET_KEYS.values()):
            episode.create_dataset(key, shape=(frames, 360, 640, 3), dtype=np.uint8, fillvalue=index)

    observation, resolved_index = bpx_data.read_policy_observation(path, -1)

    assert resolved_index == 1
    np.testing.assert_array_equal(observation["qpos"], np.arange(23, 46, dtype=np.float32))
    assert set(observation["images"]) == {"left_eye", "left_wrist", "right_wrist"}
    assert "right_eye" not in observation["images"]
    assert observation["prompt"] == "Put the blocks in the bucket."
