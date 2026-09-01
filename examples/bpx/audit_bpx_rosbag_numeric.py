#!/usr/bin/env python3
"""Audit BPX ROS1 bag numeric topics without modifying source data."""

# ruff: noqa: PLC0415 -- rosbags is loaded from the explicitly audited runtime path.

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

NUMERIC_TOPICS = (
    "/data/action",
    "/data/past_action",
    "/data/joint_position",
    "/data/joint_velocity",
)
GRIPPER_INDICES = (10, 11, 19, 20)
TRAINING_DIM = 23


@dataclasses.dataclass(frozen=True)
class SpikeCandidate:
    frame_index: int
    joint_index: int
    original_value: float
    previous_frame_index: int
    previous_value: float
    next_frame_index: int
    next_value: float
    replacement_value: float
    deviation_from_local_median: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rosbags-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def find_isolated_spikes(values):
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    if values.ndim != 2 or values.shape[1] < TRAINING_DIM:
        raise ValueError(f"Expected (T, >= {TRAINING_DIM}) values, got {values.shape}.")
    inspected = np.asarray(values[:, :TRAINING_DIM], dtype=np.float64)
    radius = 3
    spike_thresholds = np.full(TRAINING_DIM, 0.25, dtype=np.float64)
    bridge_thresholds = np.full(TRAINING_DIM, 0.25, dtype=np.float64)
    spike_thresholds[list(GRIPPER_INDICES)] = 0.02
    bridge_thresholds[list(GRIPPER_INDICES)] = 0.05

    padded = np.pad(inspected, ((radius, radius), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, 2 * radius + 1, axis=0)
    rolling_median = np.median(windows, axis=-1)
    deviations = np.abs(inspected - rolling_median)
    candidate_mask = deviations > spike_thresholds
    repairs = []
    rejected = 0
    max_reference_distance = radius + 1
    for joint_index in range(TRAINING_DIM):
        for frame_value in np.flatnonzero(candidate_mask[:, joint_index]):
            frame_index = int(frame_value)
            previous_index = frame_index - 1
            while (
                previous_index >= 0
                and candidate_mask[previous_index, joint_index]
                and frame_index - previous_index <= max_reference_distance
            ):
                previous_index -= 1
            next_index = frame_index + 1
            while (
                next_index < inspected.shape[0]
                and candidate_mask[next_index, joint_index]
                and next_index - frame_index <= max_reference_distance
            ):
                next_index += 1
            if not (
                previous_index >= 0
                and next_index < inspected.shape[0]
                and not candidate_mask[previous_index, joint_index]
                and not candidate_mask[next_index, joint_index]
                and frame_index - previous_index <= max_reference_distance
                and next_index - frame_index <= max_reference_distance
            ):
                rejected += 1
                continue
            previous_value = float(inspected[previous_index, joint_index])
            next_value = float(inspected[next_index, joint_index])
            if abs(previous_value - next_value) > bridge_thresholds[joint_index]:
                rejected += 1
                continue
            repairs.append(
                SpikeCandidate(
                    frame_index=frame_index,
                    joint_index=joint_index,
                    original_value=float(inspected[frame_index, joint_index]),
                    previous_frame_index=previous_index,
                    previous_value=previous_value,
                    next_frame_index=next_index,
                    next_value=next_value,
                    replacement_value=(previous_value + next_value) * 0.5,
                    deviation_from_local_median=float(deviations[frame_index, joint_index]),
                )
            )
    return repairs, rejected


def load_numeric_topics(path: Path):
    import numpy as np
    from rosbags.highlevel import AnyReader

    values = {topic: [] for topic in NUMERIC_TOPICS}
    timestamps = {topic: [] for topic in NUMERIC_TOPICS}
    with AnyReader([path]) as reader:
        selected = [connection for connection in reader.connections if connection.topic in values]
        for connection, timestamp, rawdata in reader.messages(connections=selected):
            message = reader.deserialize(rawdata, connection.msgtype)
            values[connection.topic].append(np.asarray(message.data, dtype=np.float64))
            timestamps[connection.topic].append(timestamp)
    arrays = {topic: np.stack(items) for topic, items in values.items()}
    time_arrays = {topic: np.asarray(items, dtype=np.int64) for topic, items in timestamps.items()}
    return arrays, time_arrays


def candidate_with_command_context(candidate: SpikeCandidate, arrays) -> dict[str, object]:
    frame = candidate.frame_index
    joint = candidate.joint_index
    action = arrays["/data/action"]
    past_action = arrays["/data/past_action"]
    return {
        **dataclasses.asdict(candidate),
        "action_value": float(action[frame, joint]),
        "action_previous_value": float(action[max(0, frame - 1), joint]),
        "action_next_value": float(action[min(action.shape[0] - 1, frame + 1), joint]),
        "past_action_value": float(past_action[frame, joint]),
        "command_supports_original": bool(
            abs(action[frame, joint] - candidate.original_value)
            < abs(action[frame, joint] - candidate.replacement_value)
        ),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.rosbags_path.resolve()))
    import numpy as np

    root = args.root.resolve()
    bags = sorted(path for path in root.rglob("*.bag") if "_bpx_audit" not in path.parts)
    reports = []
    total_qpos_candidates = 0
    total_action_candidates = 0
    qpos_by_joint = [0] * TRAINING_DIM
    action_by_joint = [0] * TRAINING_DIM
    global_min = {topic: np.full(25, np.inf) for topic in NUMERIC_TOPICS}
    global_max = {topic: np.full(25, -np.inf) for topic in NUMERIC_TOPICS}

    for index, path in enumerate(bags, start=1):
        arrays, timestamps = load_numeric_topics(path)
        shapes = {topic: list(array.shape) for topic, array in arrays.items()}
        if len({tuple(shape) for shape in shapes.values()}) != 1:
            raise RuntimeError(f"{path}: numeric topic shapes differ: {shapes}")
        shape = next(iter(arrays.values())).shape
        if len(shape) != 2 or shape[1] != 25:
            raise RuntimeError(f"{path}: expected all numeric topics to have shape (T, 25), got {shapes}")
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise RuntimeError(f"{path}: at least one numeric topic contains NaN or infinity")

        base_timestamps = timestamps[NUMERIC_TOPICS[0]]
        exact_timestamp_alignment = all(
            np.array_equal(base_timestamps, timestamps[topic]) for topic in NUMERIC_TOPICS[1:]
        )
        for topic, array in arrays.items():
            global_min[topic] = np.minimum(global_min[topic], array.min(axis=0))
            global_max[topic] = np.maximum(global_max[topic], array.max(axis=0))

        qpos_candidates, qpos_rejected = find_isolated_spikes(arrays["/data/joint_position"])
        action_candidates, action_rejected = find_isolated_spikes(arrays["/data/action"])
        for candidate in qpos_candidates:
            qpos_by_joint[candidate.joint_index] += 1
        for candidate in action_candidates:
            action_by_joint[candidate.joint_index] += 1
        total_qpos_candidates += len(qpos_candidates)
        total_action_candidates += len(action_candidates)

        past_action = arrays["/data/past_action"]
        action = arrays["/data/action"]
        same_frame_error = float(np.max(np.abs(past_action - action)))
        shifted_error = float(np.max(np.abs(past_action[1:] - action[:-1]))) if shape[0] > 1 else 0.0
        reports.append(
            {
                "path": str(path.relative_to(root)),
                "frames": shape[0],
                "numeric_timestamps_exactly_aligned": exact_timestamp_alignment,
                "last_two_max_abs": {topic: float(np.max(np.abs(array[:, 23:25]))) for topic, array in arrays.items()},
                "gripper_pair_max_abs_difference": {
                    topic: {
                        "left": float(np.max(np.abs(array[:, 10] - array[:, 11]))),
                        "right": float(np.max(np.abs(array[:, 19] - array[:, 20]))),
                    }
                    for topic, array in arrays.items()
                },
                "past_action_relation": {
                    "max_abs_same_frame": same_frame_error,
                    "max_abs_one_frame_shift": shifted_error,
                },
                "qpos_candidate_count": len(qpos_candidates),
                "qpos_rejected_count": qpos_rejected,
                "qpos_candidates": [candidate_with_command_context(item, arrays) for item in qpos_candidates],
                "action_candidate_count": len(action_candidates),
                "action_rejected_count": action_rejected,
                "action_candidates": [dataclasses.asdict(item) for item in action_candidates],
            }
        )
        print(
            f"[{index:03d}/{len(bags):03d}] {path.relative_to(root)}: "
            f"frames={shape[0]} qpos={len(qpos_candidates)} action={len(action_candidates)}",
            flush=True,
        )

    report = {
        "root": str(root),
        "bag_count": len(bags),
        "training_dimensions": list(range(TRAINING_DIM)),
        "ignored_pending_confirmation_dimensions": [23, 24],
        "thresholds": {
            "joint_spike_rad": 0.25,
            "joint_bridge_rad": 0.25,
            "gripper_spike_rad": 0.02,
            "gripper_bridge_rad": 0.05,
            "window_radius_frames": 3,
        },
        "global_min": {topic: values.tolist() for topic, values in global_min.items()},
        "global_max": {topic: values.tolist() for topic, values in global_max.items()},
        "qpos_candidate_count": total_qpos_candidates,
        "qpos_candidates_by_joint": qpos_by_joint,
        "action_candidate_count": total_action_candidates,
        "action_candidates_by_joint": action_by_joint,
        "bags": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
