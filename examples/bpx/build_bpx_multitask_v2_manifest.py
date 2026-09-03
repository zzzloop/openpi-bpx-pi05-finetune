#!/usr/bin/env python3
"""Build a V2 BPX manifest with representative legacy trajectories and all new tasks."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import pathlib
import tempfile

import bpx_data
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import pyarrow.parquet as pq
import tyro
from validate_bpx_lerobot import _list_column

from openpi.policies import bpx_constants


@lru_cache(maxsize=2)
def _load_lerobot_metadata(root: str) -> LeRobotDatasetMetadata:
    return LeRobotDatasetMetadata("bpx/bpx_blocks", root=pathlib.Path(root))


def _legacy_arrays(record: dict) -> tuple[np.ndarray, np.ndarray]:
    if record["source_kind"] == "hdf5":
        with h5py.File(record["path"], "r") as episode:
            state = np.asarray(
                bpx_constants.to_effective_joints(np.asarray(episode[bpx_data.QPOS_KEY][:], dtype=np.float32))
            )
            action = np.asarray(
                bpx_constants.to_effective_joints(np.asarray(episode[bpx_data.ACTION_KEY][:], dtype=np.float32))
            )
        return state, action
    if record["source_kind"] == "lerobot":
        metadata = _load_lerobot_metadata(record["path"])
        table = pq.read_table(
            metadata.root / metadata.get_data_file_path(int(record["source_index"])),
            columns=["observation.state", "action"],
        )
        return (
            _list_column(table, "observation.state", np.float32),
            _list_column(table, "action", np.float32),
        )
    raise ValueError(f"Legacy record has unsupported source kind: {record['source_kind']}")


def _trajectory_signature(state: np.ndarray, action: np.ndarray, points: int) -> np.ndarray:
    if state.ndim != 2 or action.ndim != 2 or state.shape != action.shape or state.shape[1] != 18:
        raise ValueError(f"Expected matching [T,18] state/action arrays, got {state.shape}/{action.shape}")
    if len(state) < 2 or points < 2 or not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        raise ValueError("Trajectory signature requires finite arrays, at least two frames, and at least two points")
    sample_indices = np.rint(np.linspace(0, len(state) - 1, points)).astype(np.int64)
    sampled_state = state[sample_indices]
    sampled_action = action[sample_indices]
    residual = sampled_action - sampled_state
    return np.concatenate(
        [sampled_state.reshape(-1), sampled_action.reshape(-1), residual.reshape(-1), [np.log1p(len(state))]]
    ).astype(np.float64)


def _select_representative_indices(signatures: np.ndarray, keep: int) -> list[int]:
    """Select deterministic trajectory medoids using standardized farthest-first k-medoids."""
    signatures = np.asarray(signatures, dtype=np.float64)
    if signatures.ndim != 2 or not np.all(np.isfinite(signatures)):
        raise ValueError("Signatures must be a finite two-dimensional array")
    if keep < 1 or keep > len(signatures):
        raise ValueError(f"keep must be between 1 and {len(signatures)}")
    # A 1e-3 floor prevents encoder/feedback noise in nearly constant joint coordinates
    # from dominating the trajectory distance after standardization.
    scale = np.maximum(signatures.std(axis=0), 1e-3)
    normalized = (signatures - signatures.mean(axis=0)) / scale
    distances = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=-1)

    selected = [int(np.argmin(distances.sum(axis=1)))]
    while len(selected) < keep:
        nearest = distances[:, selected].min(axis=1)
        nearest[selected] = -np.inf
        selected.append(int(np.argmax(nearest)))

    for _ in range(100):
        labels = distances[:, selected].argmin(axis=1)
        refined = []
        for cluster_index, current_medoid in enumerate(selected):
            members = np.flatnonzero(labels == cluster_index)
            if not len(members):
                refined.append(current_medoid)
                continue
            costs = distances[np.ix_(members, members)].sum(axis=1)
            refined.append(int(members[int(np.argmin(costs))]))
        if refined == selected:
            break
        selected = refined
    if len(set(selected)) != keep:
        raise RuntimeError("Representative trajectory selection produced duplicate medoids")
    return sorted(selected)


def _write_json_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = pathlib.Path(handle.name)
    temporary.replace(path)


def main(
    source_manifest_path: pathlib.Path,
    output_manifest_path: pathlib.Path,
    *,
    legacy_keep: int = 8,
    signature_points: int = 32,
) -> None:
    source_bytes = source_manifest_path.read_bytes()
    source = json.loads(source_bytes)
    records = source.get("episodes", [])
    if len(records) != source.get("episode_count"):
        raise ValueError("Source manifest episode count is inconsistent")
    legacy_records = [record for record in records if record.get("source_task") == "legacy_blocks"]
    new_records = [record for record in records if record.get("source_task") != "legacy_blocks"]
    if len(legacy_records) != 59 or len(new_records) != 324:
        raise ValueError(f"Expected source split 59 legacy + 324 new, got {len(legacy_records)} + {len(new_records)}")
    if records != legacy_records + new_records:
        raise ValueError("Legacy records must be the source manifest prefix")

    signatures = []
    for index, record in enumerate(legacy_records):
        state, action = _legacy_arrays(record)
        if len(state) != int(record["frames"]):
            raise ValueError(f"Legacy episode {index} frame count changed")
        signatures.append(_trajectory_signature(state, action, signature_points))
        print(f"SIGNED LEGACY {index + 1:02d}/59", flush=True)
    selected_positions = _select_representative_indices(np.stack(signatures), legacy_keep)
    selected_legacy = [dict(legacy_records[index]) for index in selected_positions]
    selected_source_indices = [int(record["source_index"]) for record in selected_legacy]

    output_records = selected_legacy + [dict(record) for record in new_records]
    for episode_index, record in enumerate(output_records):
        record["episode_index"] = episode_index
    selected_legacy_frames = sum(int(record["frames"]) for record in selected_legacy)
    total_frames = sum(int(record["frames"]) for record in output_records)
    payload = {
        "schema_version": 1,
        "episode_count": len(output_records),
        "frame_count": total_frames,
        "legacy_episode_count": len(selected_legacy),
        "new_episode_count": len(new_records),
        "deep_bag_audit": source.get("deep_bag_audit", False),
        "v2_selection": {
            "method": "standardized_trajectory_signature_farthest_first_k_medoids",
            "signature_points": signature_points,
            "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "selected_legacy_source_indices": selected_source_indices,
            "selected_legacy_frames": selected_legacy_frames,
            "selected_legacy_frame_fraction": selected_legacy_frames / total_frames,
        },
        "episodes": output_records,
    }
    _write_json_atomic(output_manifest_path.resolve(), payload)
    print(
        f"V2 MANIFEST PASS: episodes={len(output_records)}, frames={total_frames}, "
        f"legacy={selected_source_indices}, legacy_frame_fraction={selected_legacy_frames / total_frames:.4%}, "
        f"manifest={output_manifest_path.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    tyro.cli(main)
