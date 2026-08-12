"""Copy BPX HDF5 episodes and repair isolated qpos feedback spikes.

The source files are never modified. Each destination episode is first copied
to a temporary file, only ``observations/qpos`` is updated, and the completed
file is atomically renamed. A per-element audit report records every repair.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
from typing import Any

import bpx_data
import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from openpi.policies import bpx_constants

DEFAULT_WINDOW_RADIUS = 3
DEFAULT_JOINT_SPIKE_THRESHOLD = 0.25
DEFAULT_JOINT_BRIDGE_THRESHOLD = 0.25
DEFAULT_GRIPPER_SPIKE_THRESHOLD = 0.02
DEFAULT_GRIPPER_BRIDGE_THRESHOLD = 0.05
DEFAULT_MIN_FREE_GIB = 5.0


@dataclasses.dataclass(frozen=True)
class SpikeRepair:
    frame_index: int
    joint_index: int
    joint_name: str
    original_value: float
    previous_frame_index: int
    previous_value: float
    next_frame_index: int
    next_value: float
    replacement_value: float
    action_value: float
    spike_threshold: float
    bridge_threshold: float


def _threshold_vectors(
    joint_spike_threshold: float,
    joint_bridge_threshold: float,
    gripper_spike_threshold: float,
    gripper_bridge_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    spike_thresholds = np.full(bpx_constants.BPX_RAW_DIM, joint_spike_threshold, dtype=np.float64)
    bridge_thresholds = np.full(bpx_constants.BPX_RAW_DIM, joint_bridge_threshold, dtype=np.float64)
    for first, second in bpx_constants.BPX_RAW_GRIPPER_PAIRS:
        spike_thresholds[[first, second]] = gripper_spike_threshold
        bridge_thresholds[[first, second]] = gripper_bridge_threshold
    return spike_thresholds, bridge_thresholds


def repair_isolated_qpos_spikes(
    qpos: np.ndarray,
    actions: np.ndarray,
    *,
    window_radius: int = DEFAULT_WINDOW_RADIUS,
    joint_spike_threshold: float = DEFAULT_JOINT_SPIKE_THRESHOLD,
    joint_bridge_threshold: float = DEFAULT_JOINT_BRIDGE_THRESHOLD,
    gripper_spike_threshold: float = DEFAULT_GRIPPER_SPIKE_THRESHOLD,
    gripper_bridge_threshold: float = DEFAULT_GRIPPER_BRIDGE_THRESHOLD,
) -> tuple[np.ndarray, list[SpikeRepair], int]:
    """Replace temporally isolated qpos spikes with bracketing-point means.

    A rolling median finds candidate spikes. A candidate is repaired only when
    the nearest non-candidate samples on both sides are close to each other,
    which rejects sustained real motion and step changes. Consecutive and
    alternating short spikes are supported.

    Returns the repaired qpos array, repair records, and the number of
    candidates rejected by the bridge consistency test.
    """
    qpos_array = np.asarray(qpos)
    action_array = np.asarray(actions)
    if qpos_array.ndim != 2 or qpos_array.shape[1] != bpx_constants.BPX_RAW_DIM:
        raise ValueError(f"qpos must have shape (T, {bpx_constants.BPX_RAW_DIM}), got {qpos_array.shape}.")
    if action_array.shape != qpos_array.shape:
        raise ValueError(f"actions shape {action_array.shape} must match qpos shape {qpos_array.shape}.")
    if window_radius < 1:
        raise ValueError("window_radius must be at least 1.")
    if qpos_array.shape[0] < 2 * window_radius + 1:
        raise ValueError(f"Episode with {qpos_array.shape[0]} frames is too short for window_radius={window_radius}.")
    if not np.isfinite(qpos_array).all() or not np.isfinite(action_array).all():
        raise ValueError("qpos and actions must contain only finite values.")

    spike_thresholds, bridge_thresholds = _threshold_vectors(
        joint_spike_threshold,
        joint_bridge_threshold,
        gripper_spike_threshold,
        gripper_bridge_threshold,
    )
    padded = np.pad(qpos_array, ((window_radius, window_radius), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, 2 * window_radius + 1, axis=0)
    rolling_median = np.median(windows, axis=-1)
    candidates = np.abs(qpos_array - rolling_median) > spike_thresholds

    repaired = np.array(qpos_array, copy=True)
    repairs: list[SpikeRepair] = []
    rejected = 0
    max_reference_distance = window_radius + 1

    for joint_index in range(bpx_constants.BPX_RAW_DIM):
        for frame_index_value in np.flatnonzero(candidates[:, joint_index]):
            frame_index = int(frame_index_value)
            previous_index = frame_index - 1
            while (
                previous_index >= 0
                and candidates[previous_index, joint_index]
                and frame_index - previous_index <= max_reference_distance
            ):
                previous_index -= 1

            next_index = frame_index + 1
            while (
                next_index < qpos_array.shape[0]
                and candidates[next_index, joint_index]
                and next_index - frame_index <= max_reference_distance
            ):
                next_index += 1

            references_available = (
                previous_index >= 0
                and next_index < qpos_array.shape[0]
                and not candidates[previous_index, joint_index]
                and not candidates[next_index, joint_index]
                and frame_index - previous_index <= max_reference_distance
                and next_index - frame_index <= max_reference_distance
            )
            if not references_available:
                rejected += 1
                continue

            previous_value = float(qpos_array[previous_index, joint_index])
            next_value = float(qpos_array[next_index, joint_index])
            bridge_threshold = float(bridge_thresholds[joint_index])
            if abs(previous_value - next_value) > bridge_threshold:
                rejected += 1
                continue

            replacement = (previous_value + next_value) * 0.5
            repaired[frame_index, joint_index] = replacement
            repairs.append(
                SpikeRepair(
                    frame_index=frame_index,
                    joint_index=joint_index,
                    joint_name=bpx_constants.BPX_RAW_JOINT_NAMES[joint_index],
                    original_value=float(qpos_array[frame_index, joint_index]),
                    previous_frame_index=previous_index,
                    previous_value=previous_value,
                    next_frame_index=next_index,
                    next_value=next_value,
                    replacement_value=replacement,
                    action_value=float(action_array[frame_index, joint_index]),
                    spike_threshold=float(spike_thresholds[joint_index]),
                    bridge_threshold=bridge_threshold,
                )
            )

    return repaired, repairs, rejected


def _dataset_paths(file: h5py.File) -> list[str]:
    paths: list[str] = []

    def visitor(name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset):
            paths.append(name)

    file.visititems(visitor)
    return sorted(paths)


def _attrs_equal(first: h5py.AttributeManager, second: h5py.AttributeManager) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False
    return all(np.array_equal(np.asarray(first[key]), np.asarray(second[key])) for key in first)


def _dataset_digest(dataset: h5py.Dataset, target_block_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    if dataset.ndim == 0:
        digest.update(np.asarray(dataset[()]).tobytes())
        return digest.hexdigest()
    row_bytes = max(1, int(np.prod(dataset.shape[1:], dtype=np.int64)) * dataset.dtype.itemsize)
    block_rows = max(1, target_block_bytes // row_bytes)
    for start in range(0, dataset.shape[0], block_rows):
        block = np.ascontiguousarray(dataset[start : start + block_rows])
        digest.update(block.tobytes())
    return digest.hexdigest()


def verify_cleaned_episode(source_path: pathlib.Path, target_path: pathlib.Path, expected_qpos: np.ndarray) -> None:
    """Verify metadata and every unchanged dataset after a cleaned copy."""
    with h5py.File(source_path, "r") as source, h5py.File(target_path, "r") as target:
        if _dataset_paths(source) != _dataset_paths(target):
            raise RuntimeError(f"{target_path}: HDF5 dataset tree differs from source.")
        if not _attrs_equal(source.attrs, target.attrs):
            raise RuntimeError(f"{target_path}: root attributes differ from source.")

        for key in _dataset_paths(source):
            source_dataset = source[key]
            target_dataset = target[key]
            metadata = (
                "shape",
                "dtype",
                "chunks",
                "compression",
                "compression_opts",
                "shuffle",
                "fletcher32",
                "scaleoffset",
            )
            for field in metadata:
                if getattr(source_dataset, field) != getattr(target_dataset, field):
                    raise RuntimeError(f"{target_path}: dataset '{key}' changed {field} metadata.")
            if not _attrs_equal(source_dataset.attrs, target_dataset.attrs):
                raise RuntimeError(f"{target_path}: dataset '{key}' attributes differ from source.")
            if key == bpx_data.QPOS_KEY:
                if not np.array_equal(np.asarray(target_dataset[:]), expected_qpos):
                    raise RuntimeError(f"{target_path}: repaired qpos does not match the expected values.")
            elif _dataset_digest(source_dataset) != _dataset_digest(target_dataset):
                raise RuntimeError(f"{target_path}: unchanged dataset '{key}' content differs from source.")


def _write_json_atomic(path: pathlib.Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _record_payload(
    source_path: pathlib.Path,
    frames: int,
    repairs: list[SpikeRepair],
    rejected_candidates: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode": source_path.name,
        "source_size_bytes": source_path.stat().st_size,
        "source_mtime_ns": source_path.stat().st_mtime_ns,
        "frames": frames,
        "repair_count": len(repairs),
        "rejected_candidate_count": rejected_candidates,
        "settings": settings,
        "repairs": [dataclasses.asdict(repair) for repair in repairs],
    }


def _load_completed_record(
    record_path: pathlib.Path,
    source_path: pathlib.Path,
    target_path: pathlib.Path,
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    if not record_path.exists() or not target_path.exists():
        return None
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    source_stat = source_path.stat()
    if (
        payload.get("source_size_bytes") != source_stat.st_size
        or payload.get("source_mtime_ns") != source_stat.st_mtime_ns
    ):
        raise RuntimeError(f"{source_path}: source changed since the completed cleaning record was written.")
    recorded_settings = payload.get("settings")
    if recorded_settings is None:
        # Compatibility for datasets produced before per-episode settings were
        # recorded. The aggregate report was written from the same completed
        # records and therefore remains the authoritative run configuration.
        summary_path = record_path.parent.parent / "cleaning_summary.json"
        if not summary_path.exists():
            raise RuntimeError(f"{record_path}: missing cleaning settings and aggregate summary.")
        recorded_settings = json.loads(summary_path.read_text(encoding="utf-8")).get("settings")
    if recorded_settings != settings:
        raise RuntimeError(
            f"{record_path}: cleaning settings differ from this run; use a new output directory instead of --resume."
        )
    return payload


def _write_aggregate_reports(output_dir: pathlib.Path, records: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    rows = []
    repairs_by_joint: dict[str, int] = {}
    for record in records:
        for repair in record["repairs"]:
            rows.append(
                {
                    "episode": record["episode"],
                    "frame_index_0_based": repair["frame_index"],
                    "frame_number_1_based": repair["frame_index"] + 1,
                    **{key: value for key, value in repair.items() if key != "frame_index"},
                }
            )
            name = repair["joint_name"]
            repairs_by_joint[name] = repairs_by_joint.get(name, 0) + 1

    csv_path = output_dir / "cleaning_audit.csv"
    csv_temporary = csv_path.with_suffix(".csv.partial")
    fieldnames = [
        "episode",
        "frame_index_0_based",
        "frame_number_1_based",
        "joint_index",
        "joint_name",
        "original_value",
        "previous_frame_index",
        "previous_value",
        "next_frame_index",
        "next_value",
        "replacement_value",
        "action_value",
        "spike_threshold",
        "bridge_threshold",
    ]
    with csv_temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_temporary, csv_path)

    summary = {
        "algorithm": "rolling-median candidate detection plus bracketing valid-sample mean replacement",
        "settings": settings,
        "episode_count": len(records),
        "frame_count": sum(int(record["frames"]) for record in records),
        "repair_count": len(rows),
        "rejected_candidate_count": sum(int(record["rejected_candidate_count"]) for record in records),
        "repairs_by_joint": repairs_by_joint,
        "episodes": [
            {
                "episode": record["episode"],
                "frames": record["frames"],
                "repair_count": record["repair_count"],
                "rejected_candidate_count": record["rejected_candidate_count"],
            }
            for record in records
        ],
    }
    _write_json_atomic(output_dir / "cleaning_summary.json", summary)


def main(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    episodes: list[int] | None,
    window_radius: int,
    joint_spike_threshold: float,
    joint_bridge_threshold: float,
    gripper_spike_threshold: float,
    gripper_bridge_threshold: float,
    min_free_gib: float,
    dry_run: bool,
    resume: bool,
    verify_unchanged: bool,
) -> None:
    input_dir = pathlib.Path(input_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("input_dir and output_dir must be different; source files are never overwritten.")

    files = bpx_data.discover_episodes(input_dir)
    files = bpx_data.select_episodes(files, episodes)

    settings = {
        "window_radius": window_radius,
        "joint_spike_threshold": joint_spike_threshold,
        "joint_bridge_threshold": joint_bridge_threshold,
        "gripper_spike_threshold": gripper_spike_threshold,
        "gripper_bridge_threshold": gripper_bridge_threshold,
    }

    if dry_run:
        total_repairs = 0
        for source_path in files:
            bpx_data.validate_episode(source_path)
            with h5py.File(source_path, "r") as source:
                qpos = np.asarray(source[bpx_data.QPOS_KEY][:])
                actions = np.asarray(source[bpx_data.ACTION_KEY][:])
            _, repairs, rejected = repair_isolated_qpos_spikes(qpos, actions, **settings)
            total_repairs += len(repairs)
            print(f"DRY-RUN {source_path.name}: repairs={len(repairs)}, rejected_candidates={rejected}")
        print(f"DRY-RUN total: episodes={len(files)}, repairs={total_repairs}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = output_dir / ".cleaning_records"
    records_dir.mkdir(exist_ok=True)
    existing_outputs = list(output_dir.glob("episode_*.hdf5"))
    if existing_outputs and not resume:
        raise FileExistsError(f"{output_dir} already contains episode files; pass --resume to continue safely.")

    pending_bytes = 0
    for source_path in files:
        target_path = output_dir / source_path.name
        record_path = records_dir / f"{source_path.stem}.json"
        if resume and _load_completed_record(record_path, source_path, target_path, settings) is not None:
            continue
        if target_path.exists():
            raise FileExistsError(
                f"{target_path} exists without a matching completed record; refusing to overwrite it."
            )
        pending_bytes += source_path.stat().st_size

    free_bytes = shutil.disk_usage(output_dir).free
    required_bytes = pending_bytes + int(min_free_gib * 1024**3)
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient free space: need {required_bytes / 1024**3:.2f} GiB including reserve, "
            f"have {free_bytes / 1024**3:.2f} GiB."
        )

    selected_records: list[dict[str, Any]] = []
    for source_path in files:
        target_path = output_dir / source_path.name
        record_path = records_dir / f"{source_path.stem}.json"
        completed = _load_completed_record(record_path, source_path, target_path, settings) if resume else None
        if completed is not None:
            selected_records.append(completed)
            print(f"SKIP completed {source_path.name}: repairs={completed['repair_count']}")
            continue

        summary = bpx_data.validate_episode(source_path)
        with h5py.File(source_path, "r") as source:
            qpos = np.asarray(source[bpx_data.QPOS_KEY][:])
            actions = np.asarray(source[bpx_data.ACTION_KEY][:])
        repaired_qpos, repairs, rejected = repair_isolated_qpos_spikes(qpos, actions, **settings)

        temporary_path = output_dir / f".{source_path.name}.partial"
        if temporary_path.exists():
            if not resume:
                raise FileExistsError(f"Incomplete temporary file exists: {temporary_path}.")
            temporary_path.unlink()
        print(f"COPY {source_path.name}: frames={summary.frames}, repairs={len(repairs)}")
        shutil.copy2(source_path, temporary_path)
        with h5py.File(temporary_path, "r+") as target:
            target[bpx_data.QPOS_KEY][...] = repaired_qpos
            target.flush()
        os.replace(temporary_path, target_path)

        if verify_unchanged:
            print(f"VERIFY {source_path.name}: hashing all unchanged datasets")
            verify_cleaned_episode(source_path, target_path, repaired_qpos)

        payload = _record_payload(source_path, summary.frames, repairs, rejected, settings)
        _write_json_atomic(record_path, payload)
        selected_records.append(payload)
        print(f"DONE {source_path.name}: repairs={len(repairs)}, rejected_candidates={rejected}")

    # Include every completed record in the output directory so reports remain
    # complete after a selective pilot followed by a resumed full run.
    all_records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(records_dir.glob("episode_*.json"))]
    all_records.sort(key=lambda record: int(pathlib.Path(record["episode"]).stem.split("_")[-1]))
    _write_aggregate_reports(output_dir, all_records, settings)
    print(
        f"CLEANING COMPLETE: selected={len(selected_records)}, available={len(all_records)}, "
        f"repairs={sum(record['repair_count'] for record in all_records)}, output={output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--window-radius", type=int, default=DEFAULT_WINDOW_RADIUS)
    parser.add_argument("--joint-spike-threshold", type=float, default=DEFAULT_JOINT_SPIKE_THRESHOLD)
    parser.add_argument("--joint-bridge-threshold", type=float, default=DEFAULT_JOINT_BRIDGE_THRESHOLD)
    parser.add_argument("--gripper-spike-threshold", type=float, default=DEFAULT_GRIPPER_SPIKE_THRESHOLD)
    parser.add_argument("--gripper-bridge-threshold", type=float, default=DEFAULT_GRIPPER_BRIDGE_THRESHOLD)
    parser.add_argument("--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-unchanged", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.input_dir,
        arguments.output_dir,
        episodes=arguments.episodes,
        window_radius=arguments.window_radius,
        joint_spike_threshold=arguments.joint_spike_threshold,
        joint_bridge_threshold=arguments.joint_bridge_threshold,
        gripper_spike_threshold=arguments.gripper_spike_threshold,
        gripper_bridge_threshold=arguments.gripper_bridge_threshold,
        min_free_gib=arguments.min_free_gib,
        dry_run=arguments.dry_run,
        resume=arguments.resume,
        verify_unchanged=arguments.verify_unchanged,
    )
