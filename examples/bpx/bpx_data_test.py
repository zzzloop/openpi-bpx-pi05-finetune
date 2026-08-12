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
