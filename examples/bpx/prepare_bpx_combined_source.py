#!/usr/bin/env python3
"""Audit the copied BPX bags and build a deterministic combined-source manifest."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import tempfile

import bpx_data
import bpx_rosbag_data
import tyro


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    directory: str
    originals: int
    task: str


TASK_SPECS = (
    TaskSpec("暗光混合方块", 15, "Place all the colored blocks on the table into the bucket."),
    TaskSpec("抓橙色方块", 12, "Pick up the orange block and place it in the bucket."),
    TaskSpec("抓红色方块", 12, "Pick up the red block and place it in the bucket."),
    TaskSpec("抓蓝色方块", 21, "Pick up the blue block and place it in the bucket."),
    TaskSpec("抓绿色方块", 6, "Pick up the green block and place it in the bucket."),
    TaskSpec("抓瓶子", 10, "Pick up the bottle and place it in the bucket."),
    TaskSpec("抓鼠标", 10, "Pick up the mouse and place it in the bucket."),
    TaskSpec("抓羽毛球", 10, "Pick up the shuttlecock and place it in the bucket."),
    TaskSpec("抓紫色方块", 12, "Pick up the purple block and place it in the bucket."),
)
VARIANTS = ("original", "dark", "bright")
EXPECTED_NEW_BAGS = 324
EXPECTED_NEW_FRAMES = 107_586
EXPECTED_LEGACY_EPISODES = 59
EXPECTED_LEGACY_FRAMES = 67_455
EXPECTED_COMBINED_EPISODES = 383
EXPECTED_COMBINED_FRAMES = 175_041
_BAG_NAME = re.compile(r"(?P<index>\d+)(?:__aug_(?P<variant>dark|bright))?\.bag")


def _variant_path(root: pathlib.Path, spec: TaskSpec, source_index: int, variant: str) -> pathlib.Path:
    suffix = "" if variant == "original" else f"__aug_{variant}"
    return root / spec.directory / f"{source_index}{suffix}.bag"


def _load_brightness_audit(root: pathlib.Path) -> dict:
    path = root / "_bpx_audit" / "brightness_augmentation_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing augmentation audit: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("source_bags") != 108 or report.get("generated_bags") != 216:
        raise ValueError("Brightness audit must describe 108 source and 216 generated bags")
    if report.get("dark_gain_range") != [0.45, 0.9] or report.get("bright_gain_range") != [1.1, 1.35]:
        raise ValueError("Brightness audit gain ranges do not match the approved contract")
    if report.get("no_crop") is not True or report.get("no_resize") is not True:
        raise ValueError("Brightness audit does not guarantee no crop/no resize")
    records = report.get("records", [])
    if len(records) != 108:
        raise ValueError(f"Brightness audit has {len(records)} records, expected 108")
    for record in records:
        for key in ("source", "dark", "bright"):
            candidate = root / record[key]
            if not candidate.is_file():
                raise FileNotFoundError(f"Audit references missing file: {candidate}")
            expected_bytes = int(record[f"{key}_bytes"])
            if candidate.stat().st_size != expected_bytes:
                raise ValueError(
                    f"Copied file size differs from augmentation audit: {candidate} "
                    f"({candidate.stat().st_size} != {expected_bytes})"
                )
        if not 0.45 <= float(record["dark_gain"]) <= 0.90:
            raise ValueError(f"Invalid dark gain in audit: {record}")
        if not 1.10 <= float(record["bright_gain"]) <= 1.35:
            raise ValueError(f"Invalid bright gain in audit: {record}")
    return report


def audit_new_bags(root: pathlib.Path, *, deep: bool) -> list[dict]:
    """Strictly audit the approximately 200 GiB copied dataset."""
    root = root.resolve()
    brightness_report = _load_brightness_audit(root)
    brightness_records = {record["source"]: record for record in brightness_report["records"]}
    if len(brightness_records) != 108:
        raise ValueError("Brightness audit contains duplicate source records")
    partials = sorted(root.rglob("*.partial"))
    if partials:
        raise ValueError(f"Found incomplete files: {[str(path) for path in partials[:10]]}")

    expected = {
        _variant_path(root, spec, source_index, variant).resolve()
        for spec in TASK_SPECS
        for source_index in range(spec.originals)
        for variant in VARIANTS
    }
    actual = {path.resolve() for path in root.rglob("*.bag") if "_bpx_audit" not in path.parts}
    if expected != actual:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise ValueError(f"Bag inventory mismatch; missing={missing[:10]}, extra={extra[:10]}")
    if len(actual) != EXPECTED_NEW_BAGS:
        raise ValueError(f"Expected {EXPECTED_NEW_BAGS} bags, found {len(actual)}")

    records: list[dict] = []
    total_frames = 0
    for spec in TASK_SPECS:
        for source_index in range(spec.originals):
            for variant in VARIANTS:
                path = _variant_path(root, spec, source_index, variant)
                match = _BAG_NAME.fullmatch(path.name)
                if match is None:
                    raise ValueError(f"Unexpected bag name: {path.name}")
                summary = bpx_rosbag_data.validate_bag_metadata(path)
                if deep:
                    decoded_frames = sum(1 for _ in bpx_rosbag_data.iter_bag_frames(path))
                    if decoded_frames != summary.frames:
                        raise ValueError(f"{path}: decoded {decoded_frames} frames, index reports {summary.frames}")
                total_frames += summary.frames
                records.append(
                    {
                        "source_kind": "rosbag",
                        "source_task": spec.directory,
                        "source_index": source_index,
                        "variant": variant,
                        "task": spec.task,
                        "path": str(path),
                        "frames": summary.frames,
                        "messages": summary.messages,
                        "bytes": summary.bytes,
                    }
                )
                print(f"NEW OK {len(records):03d}/{EXPECTED_NEW_BAGS}: {path} ({summary.frames} frames)", flush=True)
            if deep:
                source_path = _variant_path(root, spec, source_index, "original")
                relative_source = source_path.relative_to(root).as_posix()
                if relative_source not in brightness_records:
                    raise ValueError(f"Missing brightness audit record for {relative_source}")
                brightness_record = brightness_records[relative_source]
                bpx_rosbag_data.validate_brightness_variants(
                    source_path,
                    _variant_path(root, spec, source_index, "dark"),
                    _variant_path(root, spec, source_index, "bright"),
                    dark_gain=float(brightness_record["dark_gain"]),
                    bright_gain=float(brightness_record["bright_gain"]),
                )
                print(f"AUGMENTATION OK: {relative_source}", flush=True)
    if total_frames != EXPECTED_NEW_FRAMES:
        raise ValueError(f"New dataset has {total_frames} frames, expected {EXPECTED_NEW_FRAMES}")
    return records


def audit_legacy_hdf5(root: pathlib.Path) -> list[dict]:
    files = bpx_data.discover_episodes(root.resolve())
    if len(files) != EXPECTED_LEGACY_EPISODES:
        raise ValueError(f"Expected {EXPECTED_LEGACY_EPISODES} legacy episodes, found {len(files)}")
    records = []
    for expected_index, path in enumerate(files):
        if path.name != f"episode_{expected_index}.hdf5":
            raise ValueError(f"Legacy numbering is not contiguous at {path}")
        summary = bpx_data.validate_episode(path)
        if summary.success is False:
            raise ValueError(f"Legacy episode is explicitly unsuccessful: {path}")
        records.append(
            {
                "source_kind": "hdf5",
                "source_task": "legacy_blocks",
                "source_index": expected_index,
                "variant": "legacy",
                "task": "Place all the colored blocks on the table into the bucket.",
                "path": str(path.resolve()),
                "frames": summary.frames,
                "bytes": path.stat().st_size,
            }
        )
    total_frames = sum(record["frames"] for record in records)
    if total_frames != EXPECTED_LEGACY_FRAMES:
        raise ValueError(f"Legacy dataset has {total_frames} frames, expected {EXPECTED_LEGACY_FRAMES}")
    return records


def _write_json_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = pathlib.Path(handle.name)
    temporary.replace(path)


def main(
    new_bag_root: pathlib.Path,
    *,
    legacy_hdf5_root: pathlib.Path | None = None,
    output_manifest: pathlib.Path | None = None,
    deep: bool = False,
) -> None:
    """Audit new copied bags; optionally add the copied legacy set and write a manifest."""
    new_records = audit_new_bags(new_bag_root, deep=deep)
    print(
        f"NEW DATASET PASS: bags={len(new_records)}, frames={sum(item['frames'] for item in new_records)}, "
        f"bytes={sum(item['bytes'] for item in new_records)} deep={deep}",
        flush=True,
    )
    if legacy_hdf5_root is None:
        if output_manifest is not None:
            raise ValueError("--output-manifest requires --legacy-hdf5-root")
        return

    legacy_records = audit_legacy_hdf5(legacy_hdf5_root)
    records = legacy_records + new_records
    for episode_index, record in enumerate(records):
        record["episode_index"] = episode_index
    if (
        len(records) != EXPECTED_COMBINED_EPISODES
        or sum(item["frames"] for item in records) != EXPECTED_COMBINED_FRAMES
    ):
        raise AssertionError("Combined dataset totals do not match the approved contract")
    if output_manifest is None:
        raise ValueError("Combined audit requires --output-manifest")
    payload = {
        "schema_version": 1,
        "episode_count": len(records),
        "frame_count": sum(item["frames"] for item in records),
        "legacy_episode_count": len(legacy_records),
        "new_episode_count": len(new_records),
        "deep_bag_audit": deep,
        "episodes": records,
    }
    _write_json_atomic(output_manifest.resolve(), payload)
    print(f"COMBINED DATASET PASS: episodes=383, frames=175041, manifest={output_manifest.resolve()}")


if __name__ == "__main__":
    tyro.cli(main)
