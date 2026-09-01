#!/usr/bin/env python3
"""Read-only structural audit for a directory of ROS1 bag demonstrations."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rosbags-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--sample-payloads",
        action="store_true",
        help="Deserialize the first message from every topic in the first bag.",
    )
    return parser.parse_args()


def _sample_payloads(path: Path) -> dict[str, object]:
    from rosbags.highlevel import AnyReader  # noqa: PLC0415

    samples: dict[str, object] = {}
    with AnyReader([path]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic in samples:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            if connection.msgtype == "std_msgs/msg/Float64MultiArray":
                values = message.data.tolist()
                samples[connection.topic] = {
                    "timestamp_ns": timestamp,
                    "length": len(values),
                    "values": values,
                    "layout_dim": [
                        {"label": item.label, "size": item.size, "stride": item.stride} for item in message.layout.dim
                    ],
                    "layout_data_offset": message.layout.data_offset,
                }
            elif connection.msgtype == "sensor_msgs/msg/Image":
                samples[connection.topic] = {
                    "timestamp_ns": timestamp,
                    "height": message.height,
                    "width": message.width,
                    "encoding": message.encoding,
                    "is_bigendian": message.is_bigendian,
                    "step": message.step,
                    "data_bytes": len(message.data),
                }
            else:
                samples[connection.topic] = {
                    "timestamp_ns": timestamp,
                    "repr": repr(message),
                }
    return samples


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.rosbags_path.resolve()))
    from rosbags.rosbag1 import Reader  # noqa: PLC0415

    root = args.root.resolve()
    bags = sorted(root.rglob("*.bag"))
    topics: dict[str, dict[str, object]] = {}
    tasks: dict[str, dict[str, object]] = collections.defaultdict(lambda: {"bags": 0, "bytes": 0, "duration_s": 0.0})
    records = []

    for path in bags:
        relative = path.relative_to(root)
        task = relative.parts[0]
        with Reader(path) as reader:
            duration_s = max(0.0, (reader.end_time - reader.start_time) / 1e9)
            connections = []
            for connection in reader.connections:
                connections.append(
                    {
                        "topic": connection.topic,
                        "msgtype": connection.msgtype,
                        "messages": connection.msgcount,
                    }
                )
                summary = topics.setdefault(
                    connection.topic,
                    {"msgtypes": set(), "connections": 0, "messages": 0, "bags": set()},
                )
                summary["msgtypes"].add(connection.msgtype)
                summary["connections"] += 1
                summary["messages"] += connection.msgcount
                summary["bags"].add(str(relative))

            size = path.stat().st_size
            tasks[task]["bags"] += 1
            tasks[task]["bytes"] += size
            tasks[task]["duration_s"] += duration_s
            records.append(
                {
                    "path": str(relative),
                    "bytes": size,
                    "duration_s": duration_s,
                    "messages": reader.message_count,
                    "connections": connections,
                }
            )

    serializable_topics = {
        topic: {
            "msgtypes": sorted(value["msgtypes"]),
            "connections": value["connections"],
            "messages": value["messages"],
            "bag_count": len(value["bags"]),
        }
        for topic, value in sorted(topics.items())
    }
    report = {
        "root": str(root),
        "bag_count": len(bags),
        "total_bytes": sum(record["bytes"] for record in records),
        "tasks": dict(sorted(tasks.items())),
        "topics": serializable_topics,
        "bags": records,
    }
    if args.sample_payloads and bags:
        report["sample_bag"] = str(bags[0].relative_to(root))
        report["sample_payloads"] = _sample_payloads(bags[0])
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
