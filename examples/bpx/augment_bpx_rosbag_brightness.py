#!/usr/bin/env python3
"""Append two brightness-only variants beside every cleaned BPX ROS1 bag."""

# ruff: noqa: PLC0415 -- rosbags is loaded from the explicitly audited runtime path.

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import sys

IMAGE_TOPICS = {
    "/data/left_eye_image",
    "/data/right_eye_image",
    "/data/left_wrist_image",
    "/data/right_wrist_image",
}
EXPECTED_IMAGE_SHAPE = (360, 640, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rosbags-path", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dark-gain-min", type=float, default=0.45)
    parser.add_argument("--dark-gain-max", type=float, default=0.90)
    parser.add_argument("--bright-gain-min", type=float, default=1.10)
    parser.add_argument("--bright-gain-max", type=float, default=1.35)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    return parser.parse_args()


def normalized_relative(value: str | Path) -> str:
    return str(value).replace("\\", "/")


def make_lut(gain: float):
    import numpy as np

    if not 0.0 < gain < 4.0:
        raise ValueError(f"Brightness gain must be in (0, 4), got {gain}")
    return np.clip(np.rint(np.arange(256, dtype=np.float64) * gain), 0, 255).astype(np.uint8)


def gains_for_bag(relative_path: str, args: argparse.Namespace) -> tuple[float, float]:
    import numpy as np

    if not 0.0 < args.dark_gain_min <= args.dark_gain_max < 1.0:
        raise ValueError("Dark gain range must satisfy 0 < min <= max < 1")
    if not 1.0 < args.bright_gain_min <= args.bright_gain_max < 4.0:
        raise ValueError("Bright gain range must satisfy 1 < min <= max < 4")
    digest = hashlib.sha256(f"{args.seed}:{relative_path}".encode()).digest()
    bag_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    generator = np.random.default_rng(bag_seed)
    return (
        float(generator.uniform(args.dark_gain_min, args.dark_gain_max)),
        float(generator.uniform(args.bright_gain_min, args.bright_gain_max)),
    )


def validate_image(message, source: Path, topic: str) -> None:
    import numpy as np

    if (message.height, message.width) != EXPECTED_IMAGE_SHAPE[:2]:
        raise RuntimeError(f"{source} {topic}: expected 360x640, got {message.height}x{message.width}")
    if message.encoding not in {"rgb8", "bgr8"}:
        raise RuntimeError(f"{source} {topic}: expected rgb8/bgr8, got {message.encoding!r}")
    if message.step != EXPECTED_IMAGE_SHAPE[1] * EXPECTED_IMAGE_SHAPE[2]:
        raise RuntimeError(f"{source} {topic}: unexpected row step {message.step}")
    values = np.asarray(message.data)
    if values.dtype != np.uint8 or values.size != EXPECTED_IMAGE_SHAPE[0] * message.step:
        raise RuntimeError(f"{source} {topic}: unexpected image payload dtype/size {values.dtype}/{values.size}")


def add_connections(reader, writer):
    result = {}
    for connection in reader.connections:
        ext = connection.ext
        result[connection.id] = writer.add_connection(
            connection.topic,
            connection.msgtype,
            typestore=reader.typestore,
            callerid=ext.callerid,
            latching=ext.latching,
        )
    return result


def output_paths(source: Path) -> tuple[Path, Path]:
    return (
        source.with_name(f"{source.stem}__aug_dark.bag"),
        source.with_name(f"{source.stem}__aug_bright.bag"),
    )


def write_variants(source: Path, dark_output: Path, bright_output: Path, dark_lut, bright_lut):
    import numpy as np
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag1 import Writer

    dark_partial = dark_output.with_name(dark_output.name + ".partial")
    bright_partial = bright_output.with_name(bright_output.name + ".partial")
    if dark_partial.exists() or bright_partial.exists():
        raise FileExistsError(f"Stale partial output exists for {source}; inspect it before retrying")

    dark_writer = Writer(dark_partial)
    bright_writer = Writer(bright_partial)
    dark_writer.set_compression(Writer.CompressionFormat.LZ4)
    bright_writer.set_compression(Writer.CompressionFormat.LZ4)
    pixel_count = 0
    source_sum = 0
    dark_sum = 0
    bright_sum = 0
    image_messages = 0
    with AnyReader([source]) as reader, dark_writer, bright_writer:
        dark_connections = add_connections(reader, dark_writer)
        bright_connections = add_connections(reader, bright_writer)
        for connection, timestamp, rawdata in reader.messages():
            dark_data = rawdata
            bright_data = rawdata
            if connection.topic in IMAGE_TOPICS:
                message = reader.deserialize(rawdata, connection.msgtype)
                validate_image(message, source, connection.topic)
                values = np.asarray(message.data, dtype=np.uint8)
                dark_values = dark_lut[values]
                bright_values = bright_lut[values]
                dark_message = dataclasses.replace(message, data=dark_values)
                bright_message = dataclasses.replace(message, data=bright_values)
                dark_data = reader.typestore.serialize_ros1(dark_message, connection.msgtype)
                bright_data = reader.typestore.serialize_ros1(bright_message, connection.msgtype)
                pixel_count += values.size
                source_sum += int(values.sum(dtype=np.uint64))
                dark_sum += int(dark_values.sum(dtype=np.uint64))
                bright_sum += int(bright_values.sum(dtype=np.uint64))
                image_messages += 1
            dark_writer.write(dark_connections[connection.id], timestamp, dark_data)
            bright_writer.write(bright_connections[connection.id], timestamp, bright_data)

    if image_messages == 0 or pixel_count == 0:
        raise RuntimeError(f"{source}: no image messages found")
    dark_partial.replace(dark_output)
    bright_partial.replace(bright_output)
    return {
        "image_messages": image_messages,
        "source_pixel_mean": source_sum / pixel_count,
        "dark_pixel_mean": dark_sum / pixel_count,
        "bright_pixel_mean": bright_sum / pixel_count,
    }


def verify_variants(source: Path, dark_output: Path, bright_output: Path, dark_lut, bright_lut) -> None:
    import numpy as np
    from rosbags.highlevel import AnyReader

    with (
        AnyReader([source]) as source_reader,
        AnyReader([dark_output]) as dark_reader,
        AnyReader([bright_output]) as bright_reader,
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
                raise RuntimeError(f"{source}: augmented topic/type/timestamp sequence differs")
            if source_connection.topic not in IMAGE_TOPICS:
                if bytes(source_data) != bytes(dark_data) or bytes(source_data) != bytes(bright_data):
                    raise RuntimeError(f"{source}: non-image topic {source_connection.topic} was modified")
                continue

            source_message = source_reader.deserialize(source_data, source_connection.msgtype)
            dark_message = dark_reader.deserialize(dark_data, dark_connection.msgtype)
            bright_message = bright_reader.deserialize(bright_data, bright_connection.msgtype)
            validate_image(source_message, source, source_connection.topic)
            for candidate in (dark_message, bright_message):
                if (
                    candidate.height != source_message.height
                    or candidate.width != source_message.width
                    or candidate.encoding != source_message.encoding
                    or candidate.step != source_message.step
                ):
                    raise RuntimeError(f"{source}: augmented image metadata differs")
            source_values = np.asarray(source_message.data, dtype=np.uint8)
            np.testing.assert_array_equal(np.asarray(dark_message.data), dark_lut[source_values])
            np.testing.assert_array_equal(np.asarray(bright_message.data), bright_lut[source_values])


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.rosbags_path.resolve()))
    root = args.root.resolve()
    requested = {normalized_relative(value) for value in args.only}
    sources = sorted(
        path
        for path in root.rglob("*.bag")
        if "__aug_" not in path.stem
        and "_bpx_audit" not in path.parts
        and (not requested or normalized_relative(path.relative_to(root)) in requested)
    )
    if requested:
        found = {normalized_relative(path.relative_to(root)) for path in sources}
        missing = requested - found
        if missing:
            raise FileNotFoundError(f"Requested bags not found: {sorted(missing)}")

    records = []
    for index, source in enumerate(sources, start=1):
        free_gib = shutil.disk_usage(root).free / 2**30
        if free_gib < args.min_free_gib:
            raise RuntimeError(f"Only {free_gib:.2f} GiB free before {source}; stopping safely")
        relative_key = normalized_relative(source.relative_to(root))
        dark_gain, bright_gain = gains_for_bag(relative_key, args)
        dark_lut = make_lut(dark_gain)
        bright_lut = make_lut(bright_gain)
        dark_output, bright_output = output_paths(source)
        if dark_output.exists() != bright_output.exists() and not args.overwrite_existing:
            raise RuntimeError(f"Only one augmentation exists for {source}; inspect before retrying")
        if dark_output.exists() and bright_output.exists() and not args.overwrite_existing:
            method = "existing"
            stats = None
        else:
            method = "overwrite" if dark_output.exists() or bright_output.exists() else "write"
            stats = write_variants(source, dark_output, bright_output, dark_lut, bright_lut)
        if args.verify:
            verify_variants(source, dark_output, bright_output, dark_lut, bright_lut)
        record = {
            "source": relative_key,
            "dark": normalized_relative(dark_output.relative_to(root)),
            "bright": normalized_relative(bright_output.relative_to(root)),
            "method": method,
            "dark_gain": dark_gain,
            "bright_gain": bright_gain,
            "source_bytes": source.stat().st_size,
            "dark_bytes": dark_output.stat().st_size,
            "bright_bytes": bright_output.stat().st_size,
            "pixel_stats": stats,
        }
        records.append(record)
        print(
            f"[{index:03d}/{len(sources):03d}] {record['source']}: {method} "
            f"ratios=({record['dark_bytes'] / record['source_bytes']:.3f}, "
            f"{record['bright_bytes'] / record['source_bytes']:.3f}) free={shutil.disk_usage(root).free / 2**30:.2f} GiB",
            flush=True,
        )

    audit_dir = root / "_bpx_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "root": str(root),
        "dark_gain_range": [args.dark_gain_min, args.dark_gain_max],
        "bright_gain_range": [args.bright_gain_min, args.bright_gain_max],
        "seed": args.seed,
        "source_bags": len(records),
        "generated_bags": len(records) * 2,
        "no_crop": True,
        "no_resize": True,
        "image_topics": sorted(IMAGE_TOPICS),
        "records": records,
    }
    (audit_dir / "brightness_augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"DONE source_bags={len(records)} generated={len(records) * 2} "
        f"free={shutil.disk_usage(root).free / 2**30:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
