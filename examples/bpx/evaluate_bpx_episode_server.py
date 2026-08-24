"""Evaluate a served BPX policy against an aligned HDF5 demonstration."""

from __future__ import annotations

import csv
import json
import pathlib
import time
from typing import Any

import h5py
import matplotlib
import numpy as np
import tyro

import bpx_data
import smoke_test_bpx_checkpoint
from openpi.policies import bpx_constants
from openpi_client import websocket_client_policy

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def select_query_indices(frames: int, stride: int, max_queries: int | None) -> np.ndarray:
    if frames <= 0:
        raise ValueError("frames must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries must be positive when provided")
    indices = np.arange(0, frames, stride, dtype=np.int64)
    if max_queries is not None and indices.size > max_queries:
        positions = np.linspace(0, indices.size - 1, max_queries).round().astype(np.int64)
        indices = indices[np.unique(positions)]
    return indices


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    full_episode_actions: np.ndarray,
) -> dict[str, Any]:
    if predictions.shape != targets.shape or predictions.ndim != 3:
        raise ValueError("predictions and targets must have matching (queries, horizon, joints) shapes")
    valid = np.isfinite(targets)
    if not valid.any():
        raise ValueError("targets contain no valid values")
    errors = predictions - targets
    absolute = np.abs(errors)
    squared = errors**2

    q01, q99 = np.quantile(full_episode_actions, (0.01, 0.99), axis=0)
    robust_span = np.maximum(q99 - q01, 1e-6)
    normalized_absolute = absolute / robust_span[None, None, :]

    def masked_mean(values: np.ndarray, mask: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
        counts = mask.sum(axis=axis)
        sums = np.where(mask, values, 0.0).sum(axis=axis)
        return np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=np.float64), where=counts > 0)

    per_joint_mae = masked_mean(absolute, valid, axis=(0, 1))
    per_joint_rmse = np.sqrt(masked_mean(squared, valid, axis=(0, 1)))
    per_joint_normalized_mae = masked_mean(normalized_absolute, valid, axis=(0, 1))
    per_horizon_mae = masked_mean(absolute, valid, axis=(0, 2))
    per_horizon_normalized_mae = masked_mean(normalized_absolute, valid, axis=(0, 2))

    first_valid = valid[:, 0, :]
    first_prediction = predictions[:, 0, :]
    first_target = targets[:, 0, :]
    correlations: list[float | None] = []
    for joint_index in range(predictions.shape[-1]):
        joint_mask = first_valid[:, joint_index]
        predicted = first_prediction[joint_mask, joint_index]
        expected = first_target[joint_mask, joint_index]
        if predicted.size < 2 or np.std(predicted) < 1e-12 or np.std(expected) < 1e-12:
            correlations.append(None)
        else:
            correlations.append(float(np.corrcoef(predicted, expected)[0, 1]))

    return {
        "aggregate_mae": float(absolute[valid].mean()),
        "aggregate_rmse": float(np.sqrt(squared[valid].mean())),
        "aggregate_normalized_mae": float(normalized_absolute[valid].mean()),
        "first_step_mae": float(absolute[:, 0, :][first_valid].mean()),
        "per_joint_mae": per_joint_mae.tolist(),
        "per_joint_rmse": per_joint_rmse.tolist(),
        "per_joint_normalized_mae": per_joint_normalized_mae.tolist(),
        "per_joint_first_step_correlation": correlations,
        "per_horizon_mae": per_horizon_mae.tolist(),
        "per_horizon_normalized_mae": per_horizon_normalized_mae.tolist(),
        "episode_q01": q01.tolist(),
        "episode_q99": q99.tolist(),
        "episode_robust_span": robust_span.tolist(),
    }


def _write_csvs(output_dir: pathlib.Path, metrics: dict[str, Any]) -> None:
    with (output_dir / "per_joint_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["joint_index", "joint_name", "mae", "rmse", "normalized_mae", "first_step_correlation"])
        for index, name in enumerate(bpx_constants.BPX_EFFECTIVE_JOINT_NAMES):
            writer.writerow(
                [
                    index,
                    name,
                    metrics["per_joint_mae"][index],
                    metrics["per_joint_rmse"][index],
                    metrics["per_joint_normalized_mae"][index],
                    metrics["per_joint_first_step_correlation"][index],
                ]
            )
    with (output_dir / "per_horizon_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["horizon_index", "mae", "normalized_mae"])
        for index, (mae, normalized) in enumerate(
            zip(metrics["per_horizon_mae"], metrics["per_horizon_normalized_mae"], strict=True)
        ):
            writer.writerow([index, mae, normalized])


def _plot_summary(output_dir: pathlib.Path, metrics: dict[str, Any], inference_ms: np.ndarray) -> None:
    names = bpx_constants.BPX_EFFECTIVE_JOINT_NAMES
    figure, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    horizons = np.arange(len(metrics["per_horizon_mae"]))
    axes[0].plot(horizons, metrics["per_horizon_mae"], marker="o", label="raw MAE")
    axes[0].set(title="MAE by action-horizon index", xlabel="horizon index", ylabel="absolute joint error")
    axes[0].grid(alpha=0.25)

    joint_indices = np.arange(len(names))
    axes[1].barh(joint_indices, metrics["per_joint_normalized_mae"])
    axes[1].set_yticks(joint_indices, labels=names, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set(title="Per-joint normalized MAE", xlabel="MAE / episode q01-q99 span")
    axes[1].grid(axis="x", alpha=0.25)

    axes[2].plot(inference_ms, marker=".")
    axes[2].axhline(float(np.mean(inference_ms)), color="tab:orange", linestyle="--", label="mean")
    axes[2].set(title="Client round-trip inference latency", xlabel="query", ylabel="milliseconds")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    figure.suptitle(
        "BPX served-policy episode evaluation\n"
        f"MAE={metrics['aggregate_mae']:.5f}, RMSE={metrics['aggregate_rmse']:.5f}, "
        f"normalized MAE={metrics['aggregate_normalized_mae']:.3f}"
    )
    figure.savefig(output_dir / "summary.png", dpi=160)
    plt.close(figure)


def _plot_joint_curves(
    output_dir: pathlib.Path,
    query_indices: np.ndarray,
    prediction_mean: np.ndarray,
    prediction_std: np.ndarray,
    full_episode_actions: np.ndarray,
) -> None:
    figure, axes = plt.subplots(6, 3, figsize=(19, 20), sharex=True, constrained_layout=True)
    frame_axis = np.arange(full_episode_actions.shape[0])
    for joint_index, axis in enumerate(axes.flat):
        axis.plot(frame_axis, full_episode_actions[:, joint_index], color="0.65", linewidth=1.0, label="target")
        predicted = prediction_mean[:, 0, joint_index]
        std = prediction_std[:, 0, joint_index]
        axis.plot(query_indices, predicted, color="tab:blue", marker=".", linewidth=1.0, label="predicted h=0")
        if np.any(std > 0):
            axis.fill_between(query_indices, predicted - std, predicted + std, color="tab:blue", alpha=0.18)
        axis.set_title(f"{joint_index}: {bpx_constants.BPX_EFFECTIVE_JOINT_NAMES[joint_index]}", fontsize=9)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    figure.supxlabel("episode frame (30 Hz)")
    figure.supylabel("effective action")
    figure.savefig(output_dir / "joint_curves.png", dpi=150)
    plt.close(figure)


def _plot_chunk_overlay(
    output_dir: pathlib.Path,
    query_indices: np.ndarray,
    prediction_mean: np.ndarray,
    prediction_std: np.ndarray,
    targets: np.ndarray,
) -> None:
    query_position = len(query_indices) // 2
    valid_horizons = np.flatnonzero(np.isfinite(targets[query_position, :, 0]))
    figure, axes = plt.subplots(6, 3, figsize=(19, 20), sharex=True, constrained_layout=True)
    for joint_index, axis in enumerate(axes.flat):
        expected = targets[query_position, valid_horizons, joint_index]
        predicted = prediction_mean[query_position, valid_horizons, joint_index]
        std = prediction_std[query_position, valid_horizons, joint_index]
        axis.plot(valid_horizons, expected, color="0.35", marker="o", label="target")
        axis.plot(valid_horizons, predicted, color="tab:blue", marker=".", label="prediction")
        if np.any(std > 0):
            axis.fill_between(valid_horizons, predicted - std, predicted + std, color="tab:blue", alpha=0.18)
        axis.set_title(f"{joint_index}: {bpx_constants.BPX_EFFECTIVE_JOINT_NAMES[joint_index]}", fontsize=9)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(f"30-step chunk comparison at episode frame {query_indices[query_position]}")
    figure.supxlabel("horizon index")
    figure.supylabel("effective action")
    figure.savefig(output_dir / "chunk_overlay.png", dpi=150)
    plt.close(figure)


def _plot_camera_contact_sheet(
    output_dir: pathlib.Path,
    episode_path: pathlib.Path,
    query_indices: np.ndarray,
) -> None:
    positions = np.unique(np.linspace(0, len(query_indices) - 1, min(4, len(query_indices))).round().astype(int))
    selected_indices = query_indices[positions]
    figure, axes = plt.subplots(len(selected_indices), 3, figsize=(15, 4 * len(selected_indices)), squeeze=False)
    with h5py.File(episode_path, "r") as episode:
        for row, frame_index in enumerate(selected_indices):
            for column, camera in enumerate(bpx_constants.BPX_IMAGE_KEYS):
                image = np.asarray(episode[bpx_data.IMAGE_DATASET_KEYS[camera]][frame_index])
                axes[row, column].imshow(image)
                axes[row, column].set_title(f"frame {frame_index} · {camera}")
                axes[row, column].axis("off")
    figure.suptitle("Exact RGB observations sent to the policy")
    figure.savefig(output_dir / "camera_contact_sheet.png", dpi=140)
    plt.close(figure)


def main(
    episode_path: pathlib.Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    output_dir: pathlib.Path = pathlib.Path("eval_outputs/bpx_episode"),
    stride: int = 30,
    max_queries: int | None = None,
    samples_per_frame: int = 1,
    overwrite: bool = False,
) -> None:
    """Query an already-running policy server and compare against HDF5 actions.

    ``stride=30`` evaluates one origin per second. Use ``stride=5`` for a
    deployment-like replan cadence after the quick run passes.
    """
    if samples_per_frame <= 0:
        raise ValueError("samples_per_frame must be positive")
    summary = bpx_data.validate_episode(episode_path, require_success=False)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}.")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    query_indices = select_query_indices(summary.frames, stride, max_queries)
    horizon = 30
    effective_dim = bpx_constants.BPX_EFFECTIVE_DIM
    predictions = np.full((len(query_indices), horizon, effective_dim), np.nan, dtype=np.float64)
    prediction_std = np.full_like(predictions, np.nan)
    targets = np.full_like(predictions, np.nan)
    inference_ms = np.zeros(len(query_indices), dtype=np.float64)
    server_inference_ms = np.full(len(query_indices), np.nan, dtype=np.float64)

    policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    with h5py.File(episode_path, "r") as episode:
        prompt = bpx_data.read_task(episode, bpx_constants.BPX_DEFAULT_PROMPT)
        full_episode_actions = bpx_constants.to_effective_joints(
            np.asarray(episode[bpx_data.ACTION_KEY][:], dtype=np.float64)
        )
        for query_position, frame_index in enumerate(query_indices):
            observation = {
                "qpos": np.asarray(episode[bpx_data.QPOS_KEY][frame_index], dtype=np.float32),
                "images": {
                    camera: np.asarray(episode[bpx_data.IMAGE_DATASET_KEYS[camera]][frame_index])
                    for camera in bpx_constants.BPX_IMAGE_KEYS
                },
                "prompt": prompt,
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
            valid_horizon = min(horizon, summary.frames - int(frame_index))
            targets[query_position, :valid_horizon] = full_episode_actions[
                frame_index : frame_index + valid_horizon
            ]
            print(
                f"[{query_position + 1:03d}/{len(query_indices):03d}] frame={frame_index} "
                f"round_trip_ms={inference_ms[query_position]:.1f}"
            )

    metrics = compute_metrics(predictions, targets, full_episode_actions)
    metrics.update(
        {
            "episode_path": str(episode_path),
            "episode_frames": summary.frames,
            "prompt": prompt,
            "server": f"{host}:{port}",
            "query_count": len(query_indices),
            "stride": stride,
            "samples_per_frame": samples_per_frame,
            "mean_round_trip_ms": float(inference_ms.mean()),
            "mean_server_inference_ms": (
                None if np.isnan(server_inference_ms).all() else float(np.nanmean(server_inference_ms))
            ),
            "evaluation_scope": (
                "training-fit/data-path check if this episode was used for training; not generalization"
            ),
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        query_indices=query_indices,
        prediction_mean=predictions,
        prediction_std=prediction_std,
        targets=targets,
        inference_ms=inference_ms,
        server_inference_ms=server_inference_ms,
    )
    _write_csvs(output_dir, metrics)
    _plot_summary(output_dir, metrics, inference_ms)
    _plot_joint_curves(output_dir, query_indices, predictions, prediction_std, full_episode_actions)
    _plot_chunk_overlay(output_dir, query_indices, predictions, prediction_std, targets)
    _plot_camera_contact_sheet(output_dir, episode_path, query_indices)
    print(
        f"PASS: queries={len(query_indices)}, output={output_dir}, "
        f"MAE={metrics['aggregate_mae']:.6f}, normalized_MAE={metrics['aggregate_normalized_mae']:.4f}"
    )


if __name__ == "__main__":
    tyro.cli(main)
