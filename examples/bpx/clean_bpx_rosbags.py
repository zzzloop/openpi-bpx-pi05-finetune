#!/usr/bin/env python3
"""Create a transaction-safe cleaned BPX ROS1 bag dataset.

Source bags are never modified. Bags without approved repairs are hard-linked
when possible; bags with repairs are rewritten with only `/data/joint_position`
changed. All other serialized messages and all timestamps are preserved.
"""

# ruff: noqa: PLC0415 -- rosbags is loaded from the explicitly audited runtime path.

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import sys

QPOS_TOPIC = "/data/joint_position"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--rosbags-path", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    return parser.parse_args()


def normalized_relative(value: str | Path) -> str:
    return str(value).replace("\\", "/")


def load_repairs(path: Path) -> tuple[dict[str, dict[int, dict[int, float]]], int]:
    report = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[int, dict[int, float]]] = {}
    repair_count = 0
    for bag in report["bags"]:
        frame_map: dict[int, dict[int, float]] = {}
        for candidate in bag["qpos_candidates"]:
            if candidate.get("command_supports_original", False):
                continue
            frame = int(candidate["frame_index"])
            joint = int(candidate["joint_index"])
            frame_map.setdefault(frame, {})[joint] = float(candidate["replacement_value"])
            repair_count += 1
        result[normalized_relative(bag["path"])] = frame_map
    return result, repair_count


def add_connections(reader, writer):
    connection_map = {}
    for connection in reader.connections:
        ext = connection.ext
        connection_map[connection.id] = writer.add_connection(
            connection.topic,
            connection.msgtype,
            typestore=reader.typestore,
            callerid=ext.callerid,
            latching=ext.latching,
        )
    return connection_map


def rewrite_bag(source: Path, partial: Path, repairs: dict[int, dict[int, float]]) -> int:
    import numpy as np
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag1 import Writer

    writer = Writer(partial)
    writer.set_compression(Writer.CompressionFormat.LZ4)
    qpos_index = 0
    applied = 0
    with AnyReader([source]) as reader, writer:
        connection_map = add_connections(reader, writer)
        for connection, timestamp, rawdata in reader.messages():
            output_data = rawdata
            if connection.topic == QPOS_TOPIC:
                if qpos_index in repairs:
                    message = reader.deserialize(rawdata, connection.msgtype)
                    values = np.asarray(message.data, dtype=np.float64).copy()
                    if values.shape != (25,):
                        raise RuntimeError(
                            f"{source}: qpos frame {qpos_index} has shape {values.shape}, expected (25,)"
                        )
                    for joint, replacement in repairs[qpos_index].items():
                        values[joint] = replacement
                        applied += 1
                    message = dataclasses.replace(message, data=values)
                    output_data = reader.typestore.serialize_ros1(message, connection.msgtype)
                qpos_index += 1
            writer.write(connection_map[connection.id], timestamp, output_data)
    if applied != sum(len(items) for items in repairs.values()):
        raise RuntimeError(f"{source}: expected to apply {sum(map(len, repairs.values()))} repairs, applied {applied}")
    return applied


def verify_bag(source: Path, output: Path, repairs: dict[int, dict[int, float]]) -> None:
    import numpy as np
    from rosbags.highlevel import AnyReader

    qpos_index = 0
    with AnyReader([source]) as source_reader, AnyReader([output]) as output_reader:
        source_messages = source_reader.messages()
        output_messages = output_reader.messages()
        for source_item, output_item in zip(source_messages, output_messages, strict=True):
            source_connection, source_timestamp, source_data = source_item
            output_connection, output_timestamp, output_data = output_item
            if (
                source_connection.topic != output_connection.topic
                or source_connection.msgtype != output_connection.msgtype
                or source_timestamp != output_timestamp
            ):
                raise RuntimeError(f"{output}: topic/type/timestamp sequence differs from source")
            if source_connection.topic != QPOS_TOPIC:
                if bytes(source_data) != bytes(output_data):
                    raise RuntimeError(f"{output}: unmodified topic {source_connection.topic} differs from source")
                continue

            source_message = source_reader.deserialize(source_data, source_connection.msgtype)
            output_message = output_reader.deserialize(output_data, output_connection.msgtype)
            expected = np.asarray(source_message.data, dtype=np.float64).copy()
            for joint, replacement in repairs.get(qpos_index, {}).items():
                expected[joint] = replacement
            np.testing.assert_array_equal(np.asarray(output_message.data), expected)
            qpos_index += 1
    if qpos_index == 0:
        raise RuntimeError(f"{output}: no {QPOS_TOPIC} messages found")


def hardlink_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.rosbags_path.resolve()))

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("output-root must be a separate sibling directory, not source-root or its child")
    output_root.mkdir(parents=True, exist_ok=True)

    repair_map, approved_repair_count = load_repairs(args.audit_report)
    requested = {normalized_relative(value) for value in args.only}
    source_bags = sorted(
        path
        for path in source_root.rglob("*.bag")
        if "_bpx_audit" not in path.parts
        and (not requested or normalized_relative(path.relative_to(source_root)) in requested)
    )
    if requested:
        found = {normalized_relative(path.relative_to(source_root)) for path in source_bags}
        missing = requested - found
        if missing:
            raise FileNotFoundError(f"Requested bags not found: {sorted(missing)}")

    free_gib = shutil.disk_usage(output_root).free / 2**30
    if free_gib < args.min_free_gib:
        raise RuntimeError(f"Only {free_gib:.2f} GiB free; require at least {args.min_free_gib:.2f} GiB")

    records = []
    for index, source in enumerate(source_bags, start=1):
        relative = source.relative_to(source_root)
        key = normalized_relative(relative)
        repairs = repair_map.get(key, {})
        output = output_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".partial")
        if partial.exists():
            raise FileExistsError(f"Stale partial output exists; inspect it before retrying: {partial}")

        if output.exists():
            method = "existing"
            if args.verify:
                verify_bag(source, output, repairs)
        elif repairs:
            applied = rewrite_bag(source, partial, repairs)
            partial.replace(output)
            method = "rewrite"
            if applied != sum(len(items) for items in repairs.values()):
                raise RuntimeError(f"{output}: repair count mismatch")
            if args.verify:
                verify_bag(source, output, repairs)
        else:
            method = hardlink_or_copy(source, output)
            if args.verify and not output.samefile(source):
                verify_bag(source, output, repairs)

        record = {
            "source": key,
            "output": normalized_relative(output.relative_to(output_root)),
            "method": method,
            "repair_count": sum(len(items) for items in repairs.values()),
            "source_bytes": source.stat().st_size,
            "output_bytes": output.stat().st_size,
        }
        records.append(record)
        print(
            f"[{index:03d}/{len(source_bags):03d}] {key}: method={method} "
            f"repairs={record['repair_count']} output={record['output_bytes'] / 2**20:.1f} MiB",
            flush=True,
        )

    task_language = {
        "抓橙色方块": "Pick up the orange block and place it in the bucket.",
        "抓红色方块": "Pick up the red block and place it in the bucket.",
        "抓蓝色方块": "Pick up the blue block and place it in the bucket.",
        "抓绿色方块": "Pick up the green block and place it in the bucket.",
        "抓紫色方块": "Pick up the purple block and place it in the bucket.",
        "抓瓶子": "Pick up the bottle and place it in the bucket.",
        "抓鼠标": "Pick up the mouse and place it in the bucket.",
        "抓羽毛球": "Pick up the shuttlecock and place it in the bucket.",
        "暗光混合方块": "Place all the colored blocks on the table into the bucket.",
    }
    audit_dir = output_root / "_bpx_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "approved_repair_count_in_full_audit": approved_repair_count,
        "processed_bags": len(records),
        "applied_repair_count": sum(record["repair_count"] for record in records),
        "task_language": task_language,
        "records": records,
    }
    (audit_dir / "cleaning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (audit_dir / "task_language.json").write_text(
        json.dumps(task_language, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"DONE bags={len(records)} repairs={summary['applied_repair_count']} "
        f"free={shutil.disk_usage(output_root).free / 2**30:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
