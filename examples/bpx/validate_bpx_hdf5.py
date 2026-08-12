"""Validate raw BPX HDF5 episodes before LeRobot conversion.

Run from the OpenPI repository root:

    uv run examples/bpx/validate_bpx_hdf5.py --raw-dir ../raw_bpx
"""

import argparse
import pathlib

import bpx_data
import h5py
import numpy as np

from openpi.policies import bpx_constants


def _best_feedback_lag(actions: np.ndarray, qpos: np.ndarray, max_lag: int = 30) -> tuple[int, float]:
    effective_actions = bpx_constants.to_effective_joints(actions)
    effective_qpos = bpx_constants.to_effective_joints(qpos)
    losses = []
    for lag in range(max_lag + 1):
        if lag == 0:
            predicted, reached = effective_actions, effective_qpos
        else:
            predicted, reached = effective_actions[:-lag], effective_qpos[lag:]
        losses.append(float(np.mean(np.square(predicted - reached))))
    best = int(np.argmin(losses))
    return best, losses[best]


def main(raw_dir: pathlib.Path, action_horizon: int = 30) -> None:
    files = bpx_data.discover_episodes(raw_dir)
    total_frames = 0
    all_states = []
    all_actions = []
    print(f"Validating {len(files)} BPX episode(s) at {bpx_constants.BPX_FPS} Hz")
    for path in files:
        summary = bpx_data.validate_episode(path)
        if summary.frames < action_horizon:
            raise ValueError(f"{path}: {summary.frames} frames is shorter than action_horizon={action_horizon}.")
        with h5py.File(path, "r") as episode:
            qpos = np.asarray(episode[bpx_data.QPOS_KEY][:], dtype=np.float32)
            actions = np.asarray(episode[bpx_data.ACTION_KEY][:], dtype=np.float32)
        lag, lag_mse = _best_feedback_lag(actions, qpos)
        all_states.append(bpx_constants.to_effective_joints(qpos))
        all_actions.append(bpx_constants.to_effective_joints(actions))
        total_frames += summary.frames
        success_text = "directory-convention" if summary.success is None else str(summary.success)
        print(
            f"  OK {path.name}: frames={summary.frames}, duration={summary.frames / bpx_constants.BPX_FPS:.2f}s, "
            f"success={success_text}, best_action_to_qpos_lag={lag} frame(s), lag_mse={lag_mse:.8f}"
        )

    states = np.concatenate(all_states)
    actions = np.concatenate(all_actions)
    print(
        f"\nTotal: episodes={len(files)}, frames={total_frames}, "
        f"duration={total_frames / bpx_constants.BPX_FPS / 60:.2f} min"
    )
    print("\nEffective 18-D statistics (before delta conversion and quantile normalization):")
    print("idx  joint                                      state_std   action_std  action_min  action_max")
    for index, name in enumerate(bpx_constants.BPX_EFFECTIVE_JOINT_NAMES):
        print(
            f"{index:2d}   {name:40s} "
            f"{states[:, index].std():10.6f} {actions[:, index].std():10.6f} "
            f"{actions[:, index].min():10.6f} {actions[:, index].max():10.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=30)
    arguments = parser.parse_args()
    main(arguments.raw_dir, action_horizon=arguments.action_horizon)
