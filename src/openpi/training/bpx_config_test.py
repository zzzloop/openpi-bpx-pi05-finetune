import dataclasses
import pathlib

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import bpx_constants
from openpi.policies import bpx_policy
from openpi.training import config as config_module
from openpi.training import optimizer


@pytest.mark.parametrize(
    ("config_name", "batch_size", "num_train_steps", "peak_lr", "ema_decay"),
    [
        ("pi05_bpx_full", 64, 20_000, 2.5e-5, 0.999),
        ("pi05_bpx_full_low_mem", 2, 20_000, 2.5e-5, None),
        ("pi05_bpx_lora", 128, 30_000, 5e-5, None),
    ],
)
def test_bpx_train_configs_load_with_expected_contract(
    tmp_path: pathlib.Path,
    config_name: str,
    batch_size: int,
    num_train_steps: int,
    peak_lr: float,
    ema_decay: float | None,
):
    train_config = config_module.get_config(config_name)
    data_factory = dataclasses.replace(
        train_config.data,
        assets=config_module.AssetsConfig(assets_dir=str(tmp_path), asset_id="bpx/bpx_blocks"),
    )
    data_config = data_factory.create(tmp_path, train_config.model)

    assert train_config.model.model_type.name == "PI05"
    assert train_config.model.action_dim == 32
    assert train_config.model.action_horizon == 30
    assert train_config.batch_size == batch_size
    assert train_config.num_train_steps == num_train_steps
    assert train_config.ema_decay == ema_decay
    assert train_config.monitoring_enabled
    assert train_config.monitoring_plot_interval == 500
    assert train_config.monitoring_visualization_interval == 1_000
    assert train_config.monitoring_max_samples == 2
    assert train_config.monitoring_action_names == bpx_constants.BPX_EFFECTIVE_JOINT_NAMES
    assert isinstance(train_config.lr_schedule, optimizer.CosineDecaySchedule)
    assert train_config.lr_schedule.peak_lr == peak_lr
    assert data_config.repo_id == "bpx/bpx_blocks"
    assert data_config.asset_id == "bpx/bpx_blocks"
    assert data_config.prompt_from_task
    assert data_config.use_quantile_norm
    assert data_config.action_sequence_keys == ("action",)
    assert [type(item) for item in data_config.data_transforms.inputs] == [
        bpx_policy.BpxInputs,
        transforms.DeltaActions,
    ]
    assert [type(item) for item in data_config.data_transforms.outputs] == [
        transforms.AbsoluteActions,
        bpx_policy.BpxOutputs,
    ]
    assert tuple(data_config.data_transforms.inputs[1].mask) == bpx_constants.BPX_DELTA_ACTION_MASK
    assert isinstance(data_config.model_transforms.inputs[-1], transforms.PadStatesAndActions)
    assert data_config.model_transforms.inputs[-1].model_action_dim == 32


def test_bpx_low_memory_full_config_only_drops_ema_and_reduces_default_batch():
    full_config = config_module.get_config("pi05_bpx_full")
    low_memory_config = config_module.get_config("pi05_bpx_full_low_mem")

    assert low_memory_config.model == full_config.model
    assert low_memory_config.data == full_config.data
    assert low_memory_config.lr_schedule == full_config.lr_schedule
    assert low_memory_config.optimizer == full_config.optimizer
    assert low_memory_config.weight_loader == full_config.weight_loader
    assert low_memory_config.ema_decay is None
    assert low_memory_config.batch_size == 2


def test_bpx_data_transform_order_round_trips_absolute_actions(tmp_path: pathlib.Path):
    train_config = config_module.get_config("pi05_bpx_full")
    data_factory = dataclasses.replace(
        train_config.data,
        assets=config_module.AssetsConfig(assets_dir=str(tmp_path), asset_id="bpx/bpx_blocks"),
    )
    data_config = data_factory.create(tmp_path, train_config.model)
    state = np.linspace(-0.5, 0.5, bpx_constants.BPX_EFFECTIVE_DIM, dtype=np.float32)
    absolute_actions = np.broadcast_to(state, (train_config.model.action_horizon, state.size)).copy()
    absolute_actions[:, 0] += 0.1
    absolute_actions[:, 7] = 0.84
    absolute_actions[:, 15] = 0.62
    expected = bpx_constants.expand_bpx_actions(absolute_actions)
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    dataset_item = {
        "observation.images.left_eye": image,
        "observation.images.left_wrist": image,
        "observation.images.right_wrist": image,
        "observation.state": state,
        "action": absolute_actions,
        "prompt": bpx_constants.BPX_DEFAULT_PROMPT,
    }

    transformed = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])(
        dataset_item
    )

    np.testing.assert_allclose(transformed["actions"][:, 0], 0.1)
    np.testing.assert_allclose(transformed["actions"][:, 7], 0.84)
    np.testing.assert_allclose(transformed["actions"][:, 15], 0.62)

    restored = transforms.compose(data_config.data_transforms.outputs)(
        {"state": transformed["state"], "actions": transformed["actions"]}
    )["actions"]
    np.testing.assert_allclose(restored, expected, rtol=0.0, atol=1e-7)
