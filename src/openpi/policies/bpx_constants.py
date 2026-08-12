"""BPX joint, camera, and action-space constants.

The teleoperation HDF5 files use the agreed raw 23-element order. The three
waist/trunk dimensions are frozen. Each physical gripper is stored twice in
the raw file, with the two slots summing to the physical one-DoF gripper
value. The model therefore uses an 18-element representation:

    left arm 7 + left gripper 1 + right arm 7 + right gripper 1 + head 2

Inference expands the 18 model outputs back to the original 23-element robot
interface by inserting the waist hold pose and splitting each physical
gripper value equally across its two raw slots.
"""

from collections.abc import Sequence

import numpy as np

BPX_RAW_JOINT_NAMES: tuple[str, ...] = (
    "FoldingModularJoint02_Joint",
    "FoldingModularJoint03_Joint",
    "Trunk_Joint",
    "ArmL02_Joint",
    "ArmL03_Joint",
    "ArmL04_Joint",
    "ArmL05_Joint",
    "ArmL06_Joint",
    "ArmL07_Joint",
    "ArmL08_Joint",
    "JawBlock01_Joint",
    "JawBlock02_Joint",
    "ArmR02_Joint",
    "ArmR03_Joint",
    "ArmR04_Joint",
    "ArmR05_Joint",
    "ArmR06_Joint",
    "ArmR07_Joint",
    "ArmR08_Joint",
    "JawBlock03_Joint",
    "JawBlock04_Joint",
    "Head02_Joint",
    "Head03_Joint",
)

BPX_WAIST_DIM = 3
BPX_RAW_DIM = len(BPX_RAW_JOINT_NAMES)
BPX_RAW_GRIPPER_PAIRS: tuple[tuple[int, int], ...] = ((10, 11), (19, 20))

BPX_EFFECTIVE_JOINT_NAMES: tuple[str, ...] = (
    *BPX_RAW_JOINT_NAMES[3:10],
    "LeftGripper_1DoF",
    *BPX_RAW_JOINT_NAMES[12:19],
    "RightGripper_1DoF",
    *BPX_RAW_JOINT_NAMES[21:23],
)
BPX_EFFECTIVE_DIM = len(BPX_EFFECTIVE_JOINT_NAMES)
BPX_EFFECTIVE_GRIPPER_INDICES: tuple[int, int] = (7, 15)

# Delta conversion is applied to arm and head joint targets. Physical gripper
# values remain absolute, matching the OpenPI convention.
BPX_DELTA_ACTION_MASK: tuple[bool, ...] = (
    *(True for _ in range(7)),
    False,
    *(True for _ in range(7)),
    False,
    True,
    True,
)

BPX_IMAGE_KEYS: tuple[str, ...] = ("left_eye", "left_wrist", "right_wrist")
BPX_DROPPED_IMAGE_KEY = "right_eye"
BPX_FPS = 30
BPX_SOURCE_IMAGE_SHAPE = (360, 640, 3)
BPX_DEFAULT_PROMPT = "Place all the colored blocks on the table into the bucket."
BPX_DEFAULT_WAIST_POSITION = (0.0, 0.0, 0.0)

if BPX_RAW_DIM != 23 or BPX_EFFECTIVE_DIM != 18:
    raise RuntimeError("Unexpected BPX raw/effective joint dimensions.")
if len(BPX_DELTA_ACTION_MASK) != BPX_EFFECTIVE_DIM:
    raise RuntimeError("BPX delta-action mask does not match the effective action dimension.")


def to_effective_joints(values: np.ndarray) -> np.ndarray:
    """Convert raw-23 BPX values to the physical 18-dimensional representation.

    Raw gripper slots are summed because each pair jointly represents the
    physical one-DoF gripper value. Already-effective inputs are returned
    unchanged.
    """
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("Expected a BPX joint vector or batch, got a scalar.")
    if array.shape[-1] == BPX_EFFECTIVE_DIM:
        return array
    if array.shape[-1] != BPX_RAW_DIM:
        raise ValueError(
            f"Expected BPX joint dimension {BPX_RAW_DIM} (raw) or {BPX_EFFECTIVE_DIM} (effective), "
            f"got shape {array.shape}."
        )

    left_gripper = array[..., 10:12].sum(axis=-1, keepdims=True)
    right_gripper = array[..., 19:21].sum(axis=-1, keepdims=True)
    return np.concatenate(
        [
            array[..., 3:10],
            left_gripper,
            array[..., 12:19],
            right_gripper,
            array[..., 21:23],
        ],
        axis=-1,
    )


def expand_bpx_actions(
    actions: np.ndarray,
    waist_position: Sequence[float] = BPX_DEFAULT_WAIST_POSITION,
) -> np.ndarray:
    """Expand physical 18-D actions to the original 23-element robot order.

    Each physical gripper command is divided equally between its two raw
    slots, making their sum equal to the model's one-DoF value.
    """
    effective = np.asarray(to_effective_joints(actions))
    waist = np.asarray(waist_position, dtype=effective.dtype)
    if waist.shape != (BPX_WAIST_DIM,):
        raise ValueError(f"waist_position must have shape ({BPX_WAIST_DIM},), got {waist.shape}.")

    prefix = np.broadcast_to(waist, (*effective.shape[:-1], BPX_WAIST_DIM))
    left_half = effective[..., 7:8] * 0.5
    right_half = effective[..., 15:16] * 0.5
    return np.concatenate(
        [
            prefix,
            effective[..., 0:7],
            left_half,
            left_half,
            effective[..., 8:15],
            right_half,
            right_half,
            effective[..., 16:18],
        ],
        axis=-1,
    )
