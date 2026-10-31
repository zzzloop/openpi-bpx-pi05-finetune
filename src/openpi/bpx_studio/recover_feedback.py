"""Server-only, non-destructive BPX v1 estimated-feedback experiment builder.

The source remains immutable. Output observation.state is an ESTIMATE for the
frozen dimensions, with original values, masks, calibration evidence and hashes.
This module does not start training, serve policies, or communicate with ROS.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import errno
import hashlib
import json
import os
import pathlib
import random
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.bpx_studio.feedback_estimator import Episode
from openpi.bpx_studio.feedback_estimator import FEETECH_DIMS
from openpi.bpx_studio.feedback_estimator import fit_dimension
from openpi.bpx_studio.feedback_estimator import predict_dimension
from openpi.bpx_studio.feedback_estimator import reject_calibration_outliers
from openpi.bpx_studio.feedback_estimator import repair_short_outliers
from openpi.bpx_studio.feedback_estimator import transfer_right_gripper
from openpi.policies.bpx_constants import BPX_EFFECTIVE_JOINT_NAMES
from openpi.training.validation_split import inventory
from openpi.training.validation_split import load_split

EXTRA_FEATURES = {
    "bpx.state_original": "float32",
    "bpx.state_estimated_mask": "int64",
    "bpx.state_repaired_mask": "int64",
    "bpx.state_extrapolation_mask": "int64",
    "bpx.state_review_mask": "int64",
}


def _json(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing file or path escapes dataset: {relative}")
    return path


def _parquet_path(root, info, index):
    return _contained(root, info["data_path"].format(episode_index=index,
                      episode_chunk=index // int(info["chunks_size"])))


def _read_episode(root, info, record):
    index = record["episode_index"]
    table = pq.ParquetFile(_parquet_path(root, info, index)).read()
    if table.num_rows != record["length"]:
        raise ValueError(f"Episode {index}: metadata length mismatch")
    if not np.array_equal(table["frame_index"].to_numpy(), np.arange(table.num_rows)):
        raise ValueError(f"Episode {index}: non-contiguous frame indices")
    if not np.all(table["episode_index"].to_numpy() == index):
        raise ValueError(f"Episode {index}: wrong episode_index")
    for key in ("observation.state", "action"):
        field = table.schema.field(key)
        if not (pa.types.is_list(field.type) or pa.types.is_fixed_size_list(field.type)):
            raise ValueError(f"{key} must be a vector feature")
        if field.type.value_type != pa.float32():
            raise ValueError(f"{key} must preserve the known float32 BPX v1 contract")
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    command = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    # Fitting follows the actual sampling intervals, not a fabricated exact 30Hz.
    time = np.asarray(table["bpx.real_timestamp"].to_pylist(), dtype=np.float64)
    episode = Episode(index, time, command, state, np.ones(state.shape, dtype=bool))
    episode.validate()
    return table, episode


def _validate_contract(info):
    if (info.get("codebase_version") != "v2.1" or info.get("robot_type") != "bpx_v1"
            or info.get("fps") != 30):
        raise ValueError("This tool accepts only the known FIRST-generation 30Hz LeRobot v2.1 schema")
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        if feature["shape"] != [18] or feature["names"] != list(BPX_EFFECTIVE_JOINT_NAMES):
            raise ValueError(f"{key}: wrong dimension/order; do not use a ROS2/v2 dataset")
    expected_cameras = {f"observation.images.{name}" for name in ("left_eye", "right_eye", "left_wrist", "right_wrist")}
    actual_cameras = {key for key in info["features"] if key.startswith("observation.images.")}
    if actual_cameras != expected_cameras:
        raise ValueError("This recovery version expects all FOUR cameras from the audited v1 base dataset")
    for key, feature in info["features"].items():
        if key.startswith("observation.images.") and (feature["dtype"] != "video" or feature["shape"] != [360, 640, 3]):
            raise ValueError("Expected independent 640x360 MP4s, not images or a second-generation dataset")
    if any(key in info["features"] for key in EXTRA_FEATURES):
        raise ValueError("Input already contains recovery fields; use the original immutable source")


def _clean_calibration(episode):
    bad = reject_calibration_outliers(episode.measured)
    bad[:, [dim for dim in range(18) if dim not in FEETECH_DIMS]] = False
    episode.valid = ~bad
    repaired, mask = repair_short_outliers(episode.measured, episode.time, bad)
    if np.any(bad & ~mask):
        raise ValueError(f"Episode {episode.index}: some calibration outliers cannot be safely bridged; manual review required")
    records = [{"episode": episode.index, "frame": int(frame), "dimension": int(dim),
                "original": float(episode.measured[frame, dim]), "repaired": float(repaired[frame, dim])}
               for frame, dim in np.argwhere(mask)]
    return repaired, mask, records


def _draw_validation(report_dir, episodes, heldout_predictions):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "not_generated", "reason": "matplotlib unavailable; no installation attempted"}
    for dim, predictions in heldout_predictions.items():
        figure, axes = plt.subplots(len(episodes), 1, figsize=(12, 2 * len(episodes)), squeeze=False)
        for axis, (index, episode) in zip(axes[:, 0], sorted(episodes.items()), strict=True):
            actual = np.where(episode.valid[:, dim], episode.measured[:, dim], np.nan)
            axis.plot(episode.time, actual, label="measured (bad points excluded)", linewidth=1)
            axis.plot(episode.time, predictions[index], label="held-out prediction", linewidth=1)
            axis.plot(episode.time, episode.command[:, dim], label="command", alpha=0.4, linewidth=0.8)
            axis.set_title(f"episode {index}; prediction fitted WITHOUT this episode")
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(BPX_EFFECTIVE_JOINT_NAMES[dim])
        figure.tight_layout()
        figure.savefig(report_dir / f"validation_dim_{dim:02d}.png", dpi=120)
        plt.close(figure)
    return {"status": "generated", "files": len(heldout_predictions), "right_gripper_plot": "No independent closure samples"}


def _link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
            raise
        shutil.copy2(source, destination)
        if _sha(source) != _sha(destination):
            raise ValueError(f"Copied video checksum mismatch: {source}")
        return "copy"


def _stats(array):
    values = np.asarray(array)
    return {**{key: getattr(np, key)(values, axis=0).tolist() for key in ("min", "max", "mean", "std")},
            "count": [len(values)]}


def _table_with_recovery(table, state, additions):
    output = table.set_column(table.schema.get_field_index("observation.state"),
                              table.schema.field("observation.state"),
                              pa.array(state.tolist(), type=table.schema.field("observation.state").type))
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" not in metadata:
        raise ValueError("Source Parquet lacks Hugging Face feature metadata")
    hf = json.loads(metadata[b"huggingface"])
    for key, values in additions.items():
        dtype = EXTRA_FEATURES[key]
        arrow_type = pa.float32() if dtype == "float32" else pa.int64()
        output = output.append_column(key, pa.array(values.tolist(), type=pa.list_(arrow_type, 18)))
        hf["info"]["features"][key] = {"feature": {"dtype": dtype, "_type": "Value"},
                                         "length": 18, "_type": "Sequence"}
    metadata[b"huggingface"] = json.dumps(hf).encode()
    return output.replace_schema_metadata(metadata)


def _experimental_split(root, published_root, report_dir, calibration_ids, *, fraction=0.1, seed=42):
    """Singleton tasks stay train-only; calibration episodes never enter policy validation."""
    episodes, fingerprint = inventory(root)
    by_task = defaultdict(list)
    for episode in episodes:
        by_task[episode["tasks"][0]].append(episode["episode_index"])
    rng, validation, excluded = random.Random(seed), [], []
    for task, ids in sorted(by_task.items()):
        candidates = [index for index in ids if index not in calibration_ids]
        if len(ids) < 2 or not candidates:
            excluded.append({"task": task, "episodes": ids, "reason": "singleton_or_no_uncalibrated_holdout"})
            continue
        rng.shuffle(candidates)
        count = min(len(ids) - 1, len(candidates), max(1, round(len(ids) * fraction)))
        validation.extend(candidates[:count])
    if not validation:
        raise ValueError("No independent episode holdout can be formed")
    validation.sort()
    train = [index for index in range(len(episodes)) if index not in validation]
    payload = {"schema_version": 1, "metadata_sha256": fingerprint, "dataset_root": str(published_root),
               "fraction": fraction, "seed": seed, "source_manifest": None,
               "grouping": "estimated_feedback_experiment_episode_holdout",
               "groups": [f"episode:{index}" for index in range(len(episodes))],
               "train_episodes": train, "validation_episodes": validation,
               "train_frames": sum(episodes[index]["length"] for index in train),
               "validation_frames": sum(episodes[index]["length"] for index in validation),
               "calibration_episodes_always_train": sorted(calibration_ids),
               "tasks_excluded_from_validation": excluded,
               "limitations": ["Validation states may be estimated, not independent real-feedback measurements",
                               "Singleton composition tasks are NOT evaluated for generalization",
                               "No layout/session grouping metadata was provided; independence is not certified"]}
    path = report_dir / "validation_split.json"
    _json(path, payload)
    load_split(path)
    return path, payload


def _recover(source, output, report_dir, *, frozen_from=6, apply=False, accept_estimated=False,
             workers=32, video_check="headers", plots=True):
    source, output, report_dir = [path.expanduser().resolve() for path in (source, output, report_dir)]
    if any(left == right or left.is_relative_to(right) or right.is_relative_to(left)
           for left, right in ((source, output), (source, report_dir), (output, report_dir))):
        raise ValueError("Source/output/report directories must be separate and non-nested")
    if apply and not accept_estimated:
        raise ValueError("--apply also requires --accept-estimated-feedback; estimates are NOT measured truth")
    if output.exists() or report_dir.exists():
        raise FileExistsError("Output/report already exists; use a fresh version, never overwrite")
    records, fingerprint = inventory(source)
    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    _validate_contract(info)
    if not 4 <= frozen_from < len(records):
        raise ValueError("Need at least four normal calibration episodes before the frozen suffix")
    from openpi.bpx_studio.audit import audit_dataset
    source_audit = audit_dataset(source, workers, video_check=video_check)
    calibration, source_hashes, cleaning = {}, {}, []
    common_frozen = None
    for record in records:
        index = record["episode_index"]
        path = _parquet_path(source, info, index)
        source_hashes[str(path.relative_to(source))] = _sha(path)
        _, episode = _read_episode(source, info, record)
        if index < frozen_from:
            _, _, repairs = _clean_calibration(episode)
            calibration[index] = episode
            cleaning.extend(repairs)
        else:
            values = episode.measured[:, list(FEETECH_DIMS)]
            if common_frozen is None:
                common_frozen = values[0].copy()
            if not np.all(values == common_frozen):
                raise ValueError(f"Episode {index}: suffix is not the verified constant 10D failure; refusing to replace valid feedback")
    report_dir.mkdir(parents=True)
    models, predictions = {}, {}
    for dim in FEETECH_DIMS:
        if dim == 15:
            continue
        print(f"CALIBRATING {BPX_EFFECTIVE_JOINT_NAMES[dim]}", flush=True)
        models[dim], predictions[dim] = fit_dimension(calibration, dim)
    models[15] = transfer_right_gripper(models[7], calibration)
    report = {"schema_version": 1, "status": "analysis_only", "source": str(source), "output": str(output),
              "source_metadata_sha256": fingerprint, "source_parquet_sha256": source_hashes,
              "source_audit": source_audit, "calibration_episodes": sorted(calibration),
              "frozen_from_episode": frozen_from, "estimated_dimensions": list(FEETECH_DIMS),
              "measured_truth_restored": False, "calibration_repairs": cleaning,
              "models": {str(dim): {"joint": BPX_EFFECTIVE_JOINT_NAMES[dim], **model} for dim, model in models.items()},
              "limitations": ["Experimental action-derived states, not recovered measurements",
                              "Contact/loads/right-gripper dynamics are not independently validated",
                              "Do not present proxy validation loss as real robot success or VLM retention"]}
    if plots:
        report["plots"] = _draw_validation(report_dir, calibration, predictions)
    _json(report_dir / "calibration_report.json", report)
    print(f"CALIBRATION REPORT: {report_dir / 'calibration_report.json'}", flush=True)
    if not apply:
        print("ANALYSIS COMPLETE; no dataset created. Use --apply --accept-estimated-feedback with NEW output/report paths.", flush=True)
        return report

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{output.name}.estimated-", dir=output.parent))
    _json(staging / "RECOVERY_INCOMPLETE.json", {"owner": "bpx_v1_feedback_estimator", "source": str(source)})
    try:
        (staging / "meta").mkdir()
        for path in (source / "meta").iterdir():
            if path.is_file() and path.name not in ("info.json", "episodes_stats.jsonl", "stats.json"):
                shutil.copy2(path, staging / "meta" / path.name)
        original_stats = {row["episode_index"]: row["stats"] for row in map(json.loads,
                          (source / "meta/episodes_stats.jsonl").read_text(encoding="utf-8").splitlines())}
        if set(original_stats) != set(range(len(records))):
            raise ValueError("Incomplete episode stats; refusing to publish inconsistent metadata")
        for key, dtype in EXTRA_FEATURES.items():
            info["features"][key] = {"dtype": dtype, "shape": [18], "names": list(BPX_EFFECTIVE_JOINT_NAMES)}
        output_stats, modification_rows, totals = [], [], defaultdict(int)
        for record in records:
            index = record["episode_index"]
            # Read source with its original metadata, not the extended output feature schema.
            table, episode = _read_episode(source, info, record)
            original = episode.measured.astype(np.float32)
            state = original.copy()
            estimated = np.zeros(state.shape, dtype=np.int64)
            repaired = np.zeros_like(estimated)
            outside = np.zeros_like(estimated)
            review = np.zeros_like(estimated)
            if index < frozen_from:
                clean, mask, _ = _clean_calibration(episode)
                state = clean.astype(np.float32)
                repaired[:] = mask
            else:
                for dim, model in models.items():
                    values, extrapolation, uncertain = predict_dimension(model, episode)
                    state[:, dim] = values.astype(np.float32)
                    estimated[:, dim], outside[:, dim], review[:, dim] = 1, extrapolation, uncertain
            if not np.isfinite(state).all():
                raise ValueError(f"Episode {index}: nonfinite prediction")
            untouched = [dim for dim in range(18) if dim not in FEETECH_DIMS]
            if not np.array_equal(state[:, untouched], original[:, untouched]):
                raise ValueError("Non-Feetech measured feedback changed")
            additions = {"bpx.state_original": original, "bpx.state_estimated_mask": estimated,
                         "bpx.state_repaired_mask": repaired, "bpx.state_extrapolation_mask": outside,
                         "bpx.state_review_mask": review}
            changed = _table_with_recovery(table, state, additions)
            relative = _parquet_path(source, info, index).relative_to(source)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(changed, target, compression="snappy")
            readback = pq.ParquetFile(target).read()
            for name in table.column_names:
                if name != "observation.state" and not readback[name].equals(table[name]):
                    raise ValueError(f"Untouched field changed: {name}")
            if not readback.equals(changed):
                raise ValueError("Parquet readback mismatch")
            stats = dict(original_stats[index])
            stats["observation.state"] = _stats(state)
            stats.update({key: _stats(values) for key, values in additions.items()})
            output_stats.append({"episode_index": index, "stats": stats})
            for key, feature in info["features"].items():
                if feature.get("dtype") == "video":
                    relative_video = info["video_path"].format(episode_index=index,
                        episode_chunk=index // info["chunks_size"], video_key=key)
                    method = _link_or_copy(_contained(source, relative_video), staging / relative_video)
                    totals[f"videos_{method}"] += 1
            rows = {"episode": index, "estimated_coordinates": int(estimated.sum()),
                    "repaired_coordinates": int(repaired.sum()), "extrapolated_coordinates": int(outside.sum()),
                    "review_coordinates": int(review.sum())}
            modification_rows.append(rows)
            for key, value in rows.items():
                if key != "episode":
                    totals[key] += value
            print(f"WROTE {index+1}/{len(records)}: estimated={rows['estimated_coordinates']} review={rows['review_coordinates']}", flush=True)
        with (staging / "meta/episodes_stats.jsonl").open("x", encoding="utf-8") as stream:
            for row in output_stats:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        _json(staging / "meta/info.json", info)
        report.update(status="estimated_experiment_only", modifications=modification_rows, totals=dict(totals))
        _json(staging / "meta/bpx_feedback_recovery.json", report)
        post_audit = audit_dataset(staging, workers, video_check=video_check)
        # Check source numeric/metadata identities again after all writes; no
        # output file is ever opened in place on an original/hardlinked Parquet.
        if inventory(source)[1] != fingerprint or any(_sha(source / path) != digest for path, digest in source_hashes.items()):
            raise ValueError("Source changed during recovery; refusing to publish")
        split_path, split = _experimental_split(staging, output, report_dir, set(calibration))
        report.update(post_audit=post_audit, validation_manifest=str(split_path),
                      validation_episodes=split["validation_episodes"],
                      tasks_excluded_from_validation=split["tasks_excluded_from_validation"])
        # Provenance is finalized before publication, not silently overwritten.
        (staging / "meta/bpx_feedback_recovery.json").unlink()
        _json(staging / "meta/bpx_feedback_recovery.json", report)
        (staging / "RECOVERY_INCOMPLETE.json").unlink()
        if output.exists():
            raise FileExistsError(output)
        os.rename(staging, output)
        _json(report_dir / "recovery_report.json", report)
        print(f"ESTIMATED DATASET CREATED (NOT MEASURED TRUTH): {output}", flush=True)
        print(f"USE EXISTING VALIDATION MANIFEST: {split_path}", flush=True)
        return report
    except BaseException:
        print(f"Recovery incomplete; source untouched; diagnostic staging preserved at: {staging}", flush=True)
        raise


def recover(source, output, report_dir, **options):
    """Keep a recorder's existing lock for the entire read/derive transaction."""
    if os.name != "posix":
        raise RuntimeError("Run recovery on the Linux training server, not the local desktop")
    import fcntl
    lock_path = source.expanduser().resolve() / ".collector.lock"
    with contextlib.ExitStack() as stack:
        if lock_path.is_file():
            lock = stack.enter_context(lock_path.open("rb"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Source is being recorded; stop the collector and use a frozen snapshot") from error
        return _recover(source, output, report_dir, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path,
                        default=pathlib.Path("/scratch/bpx_datasets/lerobot/bpx/bpx_v1_base_raw"))
    parser.add_argument("--output-root", type=pathlib.Path,
                        default=pathlib.Path("/scratch/bpx_datasets/lerobot/bpx/bpx_v1_base_estimated_v001"))
    parser.add_argument("--report-dir", type=pathlib.Path,
                        default=pathlib.Path("/scratch/bpx_datasets/reports/bpx_v1_base_recovery_v001"))
    parser.add_argument("--frozen-from", type=int, default=6)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--video-check", choices=("none", "headers", "full"), default="headers")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--accept-estimated-feedback", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 256:
        parser.error("workers must lie in 1..256")
    recover(args.source_root, args.output_root, args.report_dir, frozen_from=args.frozen_from,
            apply=args.apply, accept_estimated=args.accept_estimated_feedback, workers=args.workers,
            video_check=args.video_check, plots=not args.no_plots)


if __name__ == "__main__":
    main()
