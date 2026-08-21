"""Run one offline BPX inference from a saved OpenPI checkpoint."""

import pathlib

import bpx_data
import numpy as np
import tyro

from openpi.policies import bpx_constants
from openpi.policies import policy_config
from openpi.training import config as config_module


def main(
    config_name: str,
    checkpoint_dir: pathlib.Path,
    episode_path: pathlib.Path,
    *,
    frame_index: int = 0,
) -> None:
    """Run a real HDF5 observation through a full or LoRA checkpoint."""
    valid_configs = {"pi05_bpx_full", "pi05_bpx_full_low_mem", "pi05_bpx_lora"}
    if config_name not in valid_configs:
        raise ValueError(f"config_name must be one of {sorted(valid_configs)}.")
    observation, resolved_index = bpx_data.read_policy_observation(episode_path, frame_index)
    train_config = config_module.get_config(config_name)
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
    result = policy.infer(observation)
    actions = np.asarray(result.get("actions"))
    expected_shape = (train_config.model.action_horizon, bpx_constants.BPX_RAW_DIM)
    if actions.shape != expected_shape:
        raise ValueError(f"Expected checkpoint output shape {expected_shape}, got {actions.shape}.")
    if not np.isfinite(actions).all():
        raise ValueError("Checkpoint output contains NaN or infinity.")
    np.testing.assert_allclose(
        actions[:, : bpx_constants.BPX_WAIST_DIM],
        np.asarray(bpx_constants.BPX_DEFAULT_WAIST_POSITION),
        rtol=0.0,
        atol=1e-7,
        err_msg="Frozen waist output does not match the configured hold position.",
    )
    for first, second in bpx_constants.BPX_RAW_GRIPPER_PAIRS:
        np.testing.assert_allclose(
            actions[:, first],
            actions[:, second],
            rtol=0.0,
            atol=1e-7,
            err_msg=f"Gripper slots {first}/{second} are not equal halves of the physical command.",
        )

    effective = bpx_constants.to_effective_joints(actions)
    print(
        f"PASS: config={config_name}, episode={episode_path.name}, frame={resolved_index}, "
        f"prompt={observation['prompt']!r}, output_shape={actions.shape}, effective_shape={effective.shape}, "
        f"finite={np.isfinite(actions).all()}, head_shape={actions[:, 21:23].shape}"
    )


if __name__ == "__main__":
    tyro.cli(main)
