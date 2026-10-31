"""Server-side CPU regression for the unbatched probe/Observation boundary."""

import numpy as np
import pytest

from openpi.training.selection_probes import _probe_observation


def _frame(mask):
    return {
        "image": {name: np.zeros((4, 4, 3), dtype=np.float32)
                  for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        "image_mask": {name: mask for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
        "state": np.arange(32, dtype=np.float64),
        "actions": np.zeros((30, 32), dtype=np.float32),
        "tokenized_prompt": np.arange(200, dtype=np.int64),
        "tokenized_prompt_mask": np.ones(200, dtype=np.bool_),
    }


@pytest.mark.parametrize("mask", [np.True_, np.False_, True, False, np.asarray(True), np.asarray(False)])
def test_scalar_masks_become_zero_dimensional_boolean_arrays(mask):
    sample = _frame(mask)
    observation = _probe_observation(sample)
    for array in observation.image_masks.values():
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.bool_ and array.shape == ()
        assert bool(array) == bool(mask)
    assert observation.state.shape == (32,)
    assert observation.state.dtype == np.float64
    np.testing.assert_array_equal(observation.state, sample["state"])
    np.testing.assert_array_equal(observation.tokenized_prompt, sample["tokenized_prompt"])
    np.testing.assert_array_equal(observation.tokenized_prompt_mask, sample["tokenized_prompt_mask"])
    assert observation.tokenized_prompt.dtype == np.int64


def test_preparation_does_not_modify_input_or_action_targets():
    sample = _frame(np.True_)
    sample["image"]["base_0_rgb"] = np.zeros((4, 4, 3), dtype=np.uint8)
    state = sample["state"].copy()
    targets = sample["actions"].copy()
    observation = _probe_observation(sample)
    assert sample["image"]["base_0_rgb"].dtype == np.uint8
    assert isinstance(sample["image_mask"]["base_0_rgb"], np.bool_)
    assert observation.images["base_0_rgb"].dtype == np.float32
    np.testing.assert_array_equal(observation.images["base_0_rgb"], -np.ones((4, 4, 3), dtype=np.float32))
    np.testing.assert_array_equal(sample["state"], state)
    np.testing.assert_array_equal(sample["actions"], targets)


def test_incorrect_mask_dtype_is_not_silently_coerced_or_validation_disabled():
    from jaxtyping import TypeCheckError

    with pytest.raises(TypeCheckError, match="image_masks"):
        _probe_observation(_frame(np.int64(1)))


def test_batched_arrays_keep_their_existing_batch_axis():
    sample = _frame(np.asarray([True, False]))
    sample["image"] = {name: np.stack([image, image]) for name, image in sample["image"].items()}
    sample["state"] = np.stack([sample["state"], sample["state"]])
    sample["tokenized_prompt"] = np.stack([sample["tokenized_prompt"], sample["tokenized_prompt"]])
    sample["tokenized_prompt_mask"] = np.stack([sample["tokenized_prompt_mask"], sample["tokenized_prompt_mask"]])
    observation = _probe_observation(sample)
    assert observation.state.shape == (2, 32)
    assert observation.images["base_0_rgb"].shape == (2, 4, 4, 3)
    np.testing.assert_array_equal(observation.image_masks["base_0_rgb"], [True, False])
