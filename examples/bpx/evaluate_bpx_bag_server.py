"""Evaluate a served BPX policy against one synchronized ROS1 bag demonstration."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import bpx_rosbag_data
import evaluate_bpx_episode_server as shared
import matplotlib as mpl
import numpy as np
from openpi_client import websocket_client_policy
import smoke_test_bpx_checkpoint
import tyro

from openpi.policies import bpx_constants

mpl.use("Agg")
from matplotlib import pyplot as plt


def _normalize_prompt(value: str) -> str:
    prompt = " ".join(value.split())
    if not prompt:
        raise ValueError("prompt must not be empty")
    return prompt


def resolve_prompt(
    bag_path: pathlib.Path,
    prompt: str | None,
    task_language_path: pathlib.Path | None,
) -> tuple[str, str]:
    """Resolve an explicit prompt or the audited task-directory mapping."""
    if prompt is not None:
        return _normalize_prompt(prompt), "explicit"

    bag_path = bag_path.resolve()
    mapping_path = task_language_path.resolve() if task_language_path is not None else None
    dataset_root: pathlib.Path | None = None
    if mapping_path is None:
        for parent in bag_path.parents:
            candidate = parent / "_bpx_audit" / "task_language.json"
            if candidate.is_file():
                mapping_path = candidate
                dataset_root = parent
                break
    elif mapping_path.parent.name == "_bpx_audit":
        dataset_root = mapping_path.parent.parent

    if mapping_path is None or not mapping_path.is_file():
        raise FileNotFoundError("No task_language.json was found; pass --prompt with the exact training instruction")
    if dataset_root is None:
        raise ValueError("Cannot infer the dataset root from task_language_path; pass --prompt explicitly")
    try:
        relative = bag_path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Bag {bag_path} is not below task-language dataset root {dataset_root}") from exc
    if len(relative.parts) < 2:
        raise ValueError(f"Bag must be inside a task directory below {dataset_root}")
    task_directory = relative.parts[0]
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or task_directory not in mapping:
        raise KeyError(f"No language mapping for task directory {task_directory!r} in {mapping_path}")
    return _normalize_prompt(mapping[task_directory]), str(mapping_path)


def _contact_positions(query_indices: np.ndarray) -> set[int]:
    positions = np.unique(np.linspace(0, len(query_indices) - 1, min(4, len(query_indices))).round().astype(int))
    return {int(query_indices[position]) for position in positions}


def _plot_camera_contact_sheet(
    output_dir: pathlib.Path,
    images_by_frame: dict[int, dict[str, np.ndarray]],
) -> None:
    selected = sorted(images_by_frame)
    figure, axes = plt.subplots(len(selected), 3, figsize=(15, 4 * len(selected)), squeeze=False)
    for row, frame_index in enumerate(selected):
        images = images_by_frame[frame_index]
        for column, camera in enumerate(bpx_constants.BPX_IMAGE_KEYS):
            axes[row, column].imshow(images[camera])
            axes[row, column].set_title(f"frame {frame_index} · {camera}")
            axes[row, column].axis("off")
    figure.suptitle("Exact RGB observations sent to the policy")
    figure.savefig(output_dir / "camera_contact_sheet.png", dpi=140)
    plt.close(figure)


def main(
    bag_path: pathlib.Path,
    *,
    prompt: str | None = None,
    task_language_path: pathlib.Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    output_dir: pathlib.Path = pathlib.Path("eval_outputs/bpx_bag_episode"),
    stride: int = 30,
    max_queries: int | None = None,
    samples_per_frame: int = 1,
    overwrite: bool = False,
) -> None:
    """Query a running policy server and compare its actions with one ROS1 bag."""
    bag_path = bag_path.resolve()
    if samples_per_frame <= 0:
        raise ValueError("samples_per_frame must be positive")
    summary = bpx_rosbag_data.validate_bag_metadata(bag_path)
    resolved_prompt, prompt_source = resolve_prompt(bag_path, prompt, task_language_path)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    query_indices = shared.select_query_indices(summary.frames, stride, max_queries)
    query_to_position = {int(frame): position for position, frame in enumerate(query_indices)}
    contact_frames = _contact_positions(query_indices)
    horizon = 30
    effective_dim = bpx_constants.BPX_EFFECTIVE_DIM
    predictions = np.full((len(query_indices), horizon, effective_dim), np.nan, dtype=np.float64)
    prediction_std = np.full_like(predictions, np.nan)
    inference_ms = np.zeros(len(query_indices), dtype=np.float64)
    server_inference_ms = np.full(len(query_indices), np.nan, dtype=np.float64)
    full_episode_states = np.empty((summary.frames, effective_dim), dtype=np.float64)
    full_episode_actions = np.empty_like(full_episode_states)
    images_by_frame: dict[int, dict[str, np.ndarray]] = {}

    policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    decoded_frames = 0
    for frame_index, frame in enumerate(bpx_rosbag_data.iter_bag_frames(bag_path)):
        if frame_index >= summary.frames:
            raise ValueError(f"Decoded more than the indexed {summary.frames} frames")
        decoded_frames += 1
        full_episode_states[frame_index] = frame.state
        full_episode_actions[frame_index] = frame.action
        if frame_index in contact_frames:
            images_by_frame[frame_index] = {key: np.array(value, copy=True) for key, value in frame.images.items()}
        query_position = query_to_position.get(frame_index)
        if query_position is None:
            continue
        observation = {
            "qpos": frame.state,
            "images": frame.images,
            "prompt": resolved_prompt,
        }
        samples = []
        sample_latencies = []
        sample_server_latencies = []
        for _ in range(samples_per_frame):
            started = time.monotonic()
            result = policy.infer(observation)
            sample_latencies.append((time.monotonic() - started) * 1000.0)
            raw_actions = np.asarray(result.get("actions"), dtype=np.float64)
            smoke_test_bpx_checkpoint.validate_output_contract(raw_actions, (horizon, bpx_constants.BPX_RAW_DIM))
            samples.append(bpx_constants.to_effective_joints(raw_actions))
            timing = result.get("server_timing", {})
            if isinstance(timing, dict) and timing.get("infer_ms") is not None:
                sample_server_latencies.append(float(timing["infer_ms"]))
        stacked = np.stack(samples)
        predictions[query_position] = stacked.mean(axis=0)
        prediction_std[query_position] = stacked.std(axis=0)
        inference_ms[query_position] = float(np.mean(sample_latencies))
        if sample_server_latencies:
            server_inference_ms[query_position] = float(np.mean(sample_server_latencies))
        print(
            f"[{query_position + 1:03d}/{len(query_indices):03d}] frame={frame_index} "
            f"round_trip_ms={inference_ms[query_position]:.1f}",
            flush=True,
        )
    if decoded_frames != summary.frames:
        raise ValueError(f"Decoded {decoded_frames} frames, indexed metadata reports {summary.frames}")

    targets = np.full_like(predictions, np.nan)
    for query_position, frame_index in enumerate(query_indices):
        valid_horizon = min(horizon, summary.frames - int(frame_index))
        targets[query_position, :valid_horizon] = full_episode_actions[frame_index : frame_index + valid_horizon]

    metrics: dict[str, Any] = shared.compute_metrics(predictions, targets, full_episode_actions)
    metrics.update(
        {
            "source_format": "ros1_bag",
            "episode_path": str(bag_path),
            "episode_frames": summary.frames,
            "observed_average_fps": summary.observed_average_fps,
            "prompt": resolved_prompt,
            "prompt_source": prompt_source,
            "server": f"{host}:{port}",
            "query_count": len(query_indices),
            "stride": stride,
            "samples_per_frame": samples_per_frame,
            "aggregate_mae_note": (
                "Raw aggregate MAE mixes radians with the two linear gripper coordinates. "
                "Use per_joint_normalized_mae for cross-joint and cross-checkpoint comparison."
            ),
            "mean_round_trip_ms": float(inference_ms.mean()),
            "mean_server_inference_ms": (
                None if np.isnan(server_inference_ms).all() else float(np.nanmean(server_inference_ms))
            ),
            "evaluation_scope": (
                "Open-loop training-fit/data-path check on a recorded expert trajectory; "
                "not a closed-loop robot success-rate measurement"
            ),
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        query_indices=query_indices,
        prediction_mean=predictions,
        prediction_std=prediction_std,
        targets=targets,
        full_episode_states=full_episode_states,
        full_episode_actions=full_episode_actions,
        inference_ms=inference_ms,
        server_inference_ms=server_inference_ms,
    )
    shared.write_csvs(output_dir, metrics)
    shared.plot_summary(output_dir, metrics, inference_ms)
    shared.plot_joint_curves(output_dir, query_indices, predictions, prediction_std, full_episode_actions)
    shared.plot_state_action_curves(output_dir, full_episode_states, full_episode_actions)
    shared.plot_chunk_overlay(output_dir, query_indices, predictions, prediction_std, targets)
    _plot_camera_contact_sheet(output_dir, images_by_frame)
    print(
        f"PASS: bag={bag_path.name}, prompt={resolved_prompt!r}, queries={len(query_indices)}, "
        f"output={output_dir}, MAE={metrics['aggregate_mae']:.6f}, "
        f"normalized_MAE={metrics['aggregate_normalized_mae']:.4f}"
    )


if __name__ == "__main__":
    tyro.cli(main)
