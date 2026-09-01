"""Shared ROS1 bag validation and frame decoding for BPX demonstrations."""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import pathlib

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Reader

from openpi.policies import bpx_constants

ACTION_TOPIC = "/data/action"
PAST_ACTION_TOPIC = "/data/past_action"
QPOS_TOPIC = "/data/joint_position"
VELOCITY_TOPIC = "/data/joint_velocity"
IMAGE_TOPICS = {
    "left_eye": "/data/left_eye_image",
    "right_eye": "/data/right_eye_image",
    "left_wrist": "/data/left_wrist_image",
    "right_wrist": "/data/right_wrist_image",
}
REQUIRED_TOPICS = {
    ACTION_TOPIC: "std_msgs/msg/Float64MultiArray",
    PAST_ACTION_TOPIC: "std_msgs/msg/Float64MultiArray",
    QPOS_TOPIC: "std_msgs/msg/Float64MultiArray",
    VELOCITY_TOPIC: "std_msgs/msg/Float64MultiArray",
    **dict.fromkeys(IMAGE_TOPICS.values(), "sensor_msgs/msg/Image"),
}
BPX_BAG_DIM = 25
WHEEL_INDICES = (23, 24)


@dataclasses.dataclass(frozen=True)
class BagSummary:
    path: pathlib.Path
    frames: int
    messages: int
    start_time_ns: int
    end_time_ns: int
    bytes: int
    observed_average_fps: float


@dataclasses.dataclass(frozen=True)
class BagFrame:
    timestamp_ns: int
    state: np.ndarray
    action: np.ndarray
    images: dict[str, np.ndarray]


def _brightness_lut(gain: float) -> np.ndarray:
    return np.clip(np.rint(np.arange(256, dtype=np.float64) * gain), 0, 255).astype(np.uint8)


def validate_brightness_variants(
    source_path: pathlib.Path,
    dark_path: pathlib.Path,
    bright_path: pathlib.Path,
    *,
    dark_gain: float,
    bright_gain: float,
) -> None:
    """Prove that augmented bags only apply their audited brightness LUT."""
    dark_lut = _brightness_lut(dark_gain)
    bright_lut = _brightness_lut(bright_gain)
    paths = tuple(map(pathlib.Path, (source_path, dark_path, bright_path)))
    with (
        AnyReader([paths[0]]) as source_reader,
        AnyReader([paths[1]]) as dark_reader,
        AnyReader([paths[2]]) as bright_reader,
    ):
        for source_item, dark_item, bright_item in zip(
            source_reader.messages(), dark_reader.messages(), bright_reader.messages(), strict=True
        ):
            source_connection, source_timestamp, source_data = source_item
            dark_connection, dark_timestamp, dark_data = dark_item
            bright_connection, bright_timestamp, bright_data = bright_item
            contract = (source_connection.topic, source_connection.msgtype, source_timestamp)
            if contract != (dark_connection.topic, dark_connection.msgtype, dark_timestamp) or contract != (
                bright_connection.topic,
                bright_connection.msgtype,
                bright_timestamp,
            ):
                raise ValueError(f"{source_path}: augmented topic/type/timestamp sequence differs")
            if source_connection.topic not in IMAGE_TOPICS.values():
                if bytes(source_data) != bytes(dark_data) or bytes(source_data) != bytes(bright_data):
                    raise ValueError(f"{source_path}: augmented non-image payload was modified")
                continue
            source_message = source_reader.deserialize(source_data, source_connection.msgtype)
            dark_message = dark_reader.deserialize(dark_data, dark_connection.msgtype)
            bright_message = bright_reader.deserialize(bright_data, bright_connection.msgtype)
            metadata = (
                source_message.height,
                source_message.width,
                source_message.encoding,
                source_message.is_bigendian,
                source_message.step,
            )
            if metadata != (
                dark_message.height,
                dark_message.width,
                dark_message.encoding,
                dark_message.is_bigendian,
                dark_message.step,
            ) or metadata != (
                bright_message.height,
                bright_message.width,
                bright_message.encoding,
                bright_message.is_bigendian,
                bright_message.step,
            ):
                raise ValueError(f"{source_path}: augmented image metadata differs")
            source_values = np.asarray(source_message.data, dtype=np.uint8)
            np.testing.assert_array_equal(np.asarray(dark_message.data, dtype=np.uint8), dark_lut[source_values])
            np.testing.assert_array_equal(np.asarray(bright_message.data, dtype=np.uint8), bright_lut[source_values])


def validate_bag_metadata(path: pathlib.Path) -> BagSummary:
    """Validate the indexed ROS1 bag contract without decoding image payloads."""
    path = pathlib.Path(path)
    with Reader(path) as reader:
        connections_by_topic = {connection.topic: connection for connection in reader.connections}
        if set(connections_by_topic) != set(REQUIRED_TOPICS):
            missing = sorted(set(REQUIRED_TOPICS) - set(connections_by_topic))
            extra = sorted(set(connections_by_topic) - set(REQUIRED_TOPICS))
            raise ValueError(f"{path}: topic contract mismatch; missing={missing}, extra={extra}")
        counts = set()
        for topic, expected_type in REQUIRED_TOPICS.items():
            connection = connections_by_topic[topic]
            if connection.msgtype != expected_type:
                raise ValueError(f"{path}: {topic} has type {connection.msgtype}, expected {expected_type}")
            counts.add(connection.msgcount)
        if len(counts) != 1:
            raise ValueError(f"{path}: required topics do not have identical frame counts: {sorted(counts)}")
        frames = counts.pop()
        if frames < 2:
            raise ValueError(f"{path}: expected at least two frames, got {frames}")
        if reader.message_count != frames * len(REQUIRED_TOPICS):
            raise ValueError(
                f"{path}: total message count {reader.message_count} does not equal "
                f"{frames} frames x {len(REQUIRED_TOPICS)} topics"
            )
        duration_ns = reader.end_time - reader.start_time
        if duration_ns <= 0:
            raise ValueError(f"{path}: non-positive bag duration")
        observed_fps = (frames - 1) * 1e9 / duration_ns
        # Bag start/end times include recording-boundary scheduling effects, so
        # this average is only a coarse corruption signal. The converter uses
        # the confirmed 30 Hz frame contract and validates per-frame alignment.
        if not 20.0 <= observed_fps <= 40.0:
            raise ValueError(f"{path}: implausible average frame rate {observed_fps:.3f} Hz")
        return BagSummary(
            path=path,
            frames=frames,
            messages=reader.message_count,
            start_time_ns=reader.start_time,
            end_time_ns=reader.end_time,
            bytes=path.stat().st_size,
            observed_average_fps=observed_fps,
        )


def _decode_numeric(reader: AnyReader, item, path: pathlib.Path, topic: str) -> np.ndarray:
    connection, rawdata = item
    message = reader.deserialize(rawdata, connection.msgtype)
    values = np.asarray(message.data, dtype=np.float64)
    if values.shape != (BPX_BAG_DIM,):
        raise ValueError(f"{path}: {topic} must contain {BPX_BAG_DIM} values, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: {topic} contains NaN or infinity")
    if not np.array_equal(values[list(WHEEL_INDICES)], np.zeros(len(WHEEL_INDICES))):
        raise ValueError(f"{path}: wheel dimensions 23:25 must remain exactly zero")
    return values


def _decode_image(reader: AnyReader, item, path: pathlib.Path, camera: str) -> np.ndarray:
    connection, rawdata = item
    message = reader.deserialize(rawdata, connection.msgtype)
    expected_encoding = "rgb8" if camera in {"left_eye", "right_eye"} else "bgr8"
    if (message.height, message.width, message.encoding, message.step) != (
        *bpx_constants.BPX_SOURCE_IMAGE_SHAPE[:2],
        expected_encoding,
        1920,
    ):
        raise ValueError(
            f"{path}: {camera} metadata is "
            f"{message.height}x{message.width}/{message.encoding}/step={message.step}, expected "
            f"360x640/{expected_encoding}/step=1920"
        )
    image = np.asarray(message.data, dtype=np.uint8)
    if image.size != int(np.prod(bpx_constants.BPX_SOURCE_IMAGE_SHAPE)):
        raise ValueError(f"{path}: {camera} payload has {image.size} bytes, expected 691200")
    image = image.reshape(bpx_constants.BPX_SOURCE_IMAGE_SHAPE)
    if expected_encoding == "bgr8":
        image = image[..., ::-1]
    return np.ascontiguousarray(image)


def _decode_group(reader: AnyReader, path: pathlib.Path, timestamp: int, group: dict[str, tuple]) -> BagFrame:
    if set(group) != set(REQUIRED_TOPICS):
        missing = sorted(set(REQUIRED_TOPICS) - set(group))
        extra = sorted(set(group) - set(REQUIRED_TOPICS))
        raise ValueError(f"{path}: timestamp {timestamp} topic mismatch; missing={missing}, extra={extra}")

    qpos_25 = _decode_numeric(reader, group[QPOS_TOPIC], path, QPOS_TOPIC)
    action_25 = _decode_numeric(reader, group[ACTION_TOPIC], path, ACTION_TOPIC)
    _decode_numeric(reader, group[PAST_ACTION_TOPIC], path, PAST_ACTION_TOPIC)
    _decode_numeric(reader, group[VELOCITY_TOPIC], path, VELOCITY_TOPIC)
    if np.max(np.abs(action_25[: bpx_constants.BPX_WAIST_DIM])) > 1e-6:
        raise ValueError(f"{path}: fixed-waist action is nonzero at timestamp {timestamp}")
    for first, second in bpx_constants.BPX_RAW_GRIPPER_PAIRS:
        if abs(action_25[first] - action_25[second]) > 1e-6:
            raise ValueError(f"{path}: action gripper pair differs at timestamp {timestamp}")

    decoded_images = {
        camera: _decode_image(reader, group[topic], path, camera) for camera, topic in IMAGE_TOPICS.items()
    }
    images = {
        camera: image for camera, image in decoded_images.items() if camera != bpx_constants.BPX_DROPPED_IMAGE_KEY
    }
    return BagFrame(
        timestamp_ns=timestamp,
        state=np.asarray(bpx_constants.to_effective_joints(qpos_25[: bpx_constants.BPX_RAW_DIM]), dtype=np.float32),
        action=np.asarray(bpx_constants.to_effective_joints(action_25[: bpx_constants.BPX_RAW_DIM]), dtype=np.float32),
        images=images,
    )


def iter_bag_frames(path: pathlib.Path) -> Iterator[BagFrame]:
    """Yield validated, RGB, effective-18 frames from one synchronized ROS1 bag."""
    path = pathlib.Path(path)
    with AnyReader([path]) as reader:
        selected = [connection for connection in reader.connections if connection.topic in REQUIRED_TOPICS]
        current_timestamp: int | None = None
        group: dict[str, tuple] = {}
        previous_timestamp: int | None = None
        for connection, timestamp, rawdata in reader.messages(connections=selected):
            if current_timestamp is None:
                current_timestamp = timestamp
            if timestamp != current_timestamp:
                if previous_timestamp is not None and current_timestamp <= previous_timestamp:
                    raise ValueError(f"{path}: frame timestamps are not strictly increasing")
                yield _decode_group(reader, path, current_timestamp, group)
                previous_timestamp = current_timestamp
                current_timestamp = timestamp
                group = {}
            if connection.topic in group:
                raise ValueError(f"{path}: duplicate topic {connection.topic} at timestamp {timestamp}")
            group[connection.topic] = (connection, rawdata)
        if current_timestamp is not None:
            if previous_timestamp is not None and current_timestamp <= previous_timestamp:
                raise ValueError(f"{path}: frame timestamps are not strictly increasing")
            yield _decode_group(reader, path, current_timestamp, group)
