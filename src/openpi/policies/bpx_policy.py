"""OpenPI input and output transforms for the BPX robot.

Training uses an 18-dimensional effective state/action vector:

* left arm: 7 absolute joint positions
* left gripper: 1 physical value (raw JawBlock01 + JawBlock02)
* right arm: 7 absolute joint positions
* right gripper: 1 physical value (raw JawBlock03 + JawBlock04)
* active head: 2 absolute joint positions

The three fixed waist joints are removed before normalization. The π0.5 model
pads the effective vector from 18 to its native 32 dimensions. During
inference, :class:`BpxOutputs` truncates back to 18, splits each physical
gripper value equally across its raw pair, and inserts the fixed waist pose to
return the original 23-element order.
"""

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.policies import bpx_constants


def make_bpx_example() -> dict:
    """Return a raw-format policy input for smoke tests and debugging."""
    return {
        "qpos": np.zeros((bpx_constants.BPX_RAW_DIM,), dtype=np.float32),
        "images": {
            key: np.random.randint(256, size=bpx_constants.BPX_SOURCE_IMAGE_SHAPE, dtype=np.uint8)
            for key in bpx_constants.BPX_IMAGE_KEYS
        },
        "prompt": bpx_constants.BPX_DEFAULT_PROMPT,
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert a LeRobot or runtime image to uint8 HWC RGB."""
    parsed = np.asarray(image)
    if parsed.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {parsed.shape}.")
    if parsed.shape[0] == 3 and parsed.shape[-1] != 3:
        parsed = einops.rearrange(parsed, "c h w -> h w c")
    if parsed.shape[-1] != 3:
        raise ValueError(f"Expected a three-channel image, got shape {parsed.shape}.")
    if np.issubdtype(parsed.dtype, np.floating):
        if not np.isfinite(parsed).all():
            raise ValueError("Image contains NaN or infinity.")
        if parsed.size and (parsed.min() < 0.0 or parsed.max() > 1.0):
            raise ValueError("Floating-point images must be scaled to [0, 1].")
        parsed = np.rint(parsed * 255.0).astype(np.uint8)
    elif parsed.dtype != np.uint8:
        raise ValueError(f"Expected uint8 or floating-point image, got {parsed.dtype}.")
    return parsed


@dataclasses.dataclass(frozen=True)
class BpxInputs(transforms.DataTransformFn):
    """Map canonical BPX observations into π0.5 model inputs.

    The canonical dictionary contains ``qpos``, an ``images`` dictionary with
    ``left_eye``, ``left_wrist``, and ``right_wrist``, and a language prompt.
    ``qpos`` may be raw-23 at inference or effective-18 after dataset repacking.
    Training actions may likewise be raw-23 or effective-18.
    """

    expected_cameras: ClassVar[tuple[str, ...]] = bpx_constants.BPX_IMAGE_KEYS
    ignored_cameras: ClassVar[tuple[str, ...]] = (bpx_constants.BPX_DROPPED_IMAGE_KEY,)

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        missing = set(self.expected_cameras) - set(images)
        if missing:
            raise ValueError(f"Missing BPX cameras {tuple(sorted(missing))}.")
        unknown = set(images) - set(self.expected_cameras) - set(self.ignored_cameras)
        if unknown:
            raise ValueError(f"Unexpected BPX cameras {tuple(sorted(unknown))}.")

        left_eye = _parse_image(images["left_eye"])
        left_wrist = _parse_image(images["left_wrist"])
        right_wrist = _parse_image(images["right_wrist"])
        inputs = {
            "image": {
                "base_0_rgb": left_eye,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": np.asarray(bpx_constants.to_effective_joints(data["qpos"]), dtype=np.float32),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(
                bpx_constants.to_effective_joints(data["actions"]),
                dtype=np.float32,
            )
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class BpxOutputs(transforms.DataTransformFn):
    """Return absolute BPX actions in the original 23-element robot order."""

    waist_position: tuple[float, float, float] = bpx_constants.BPX_DEFAULT_WAIST_POSITION

    def __call__(self, data: dict) -> dict:
        effective = np.asarray(data["actions"][..., : bpx_constants.BPX_EFFECTIVE_DIM])
        return {
            "actions": bpx_constants.expand_bpx_actions(
                effective,
                waist_position=self.waist_position,
            )
        }
