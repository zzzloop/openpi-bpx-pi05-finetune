"""Durable, low-frequency training diagnostics for OpenPI runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import datetime as dt
import json
import os
import pathlib
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from openpi.training.loss_trends import robust_trend

plt.switch_backend("Agg")


_SCALAR_COLUMNS = (
    "step",
    "wall_time_utc",
    "elapsed_sec",
    "loss",
    "loss_std",
    "loss_min",
    "loss_max",
    "grad_norm",
    "update_norm",
    "param_norm",
    "learning_rate",
    "steps_per_sec",
    "samples_per_sec",
    "train_eval_loss", "val_loss", "generalization_gap", "val_stderr",
    "best_validation_step", "best_validation_loss", "overfit_warning", "evaluation_seconds",
    "action_mse", "jerk_excess", "s_cf", "s_cf_active", "grounding_margin", "instruction_specificity",
)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    if float(finite.min()) >= -1.01 and float(finite.max()) <= 1.01:
        array = (array + 1.0) * 127.5 if float(finite.min()) < 0.0 else array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), smoothed])


class TrainingMonitor:
    """Persist scalar history and periodically render compact diagnostic images."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        *,
        batch_size: int,
        plot_interval: int,
        visualization_interval: int,
        action_names: Sequence[str] = (),
    ) -> None:
        if plot_interval <= 0 or visualization_interval <= 0:
            raise ValueError("Monitoring intervals must be positive.")
        self.output_dir = pathlib.Path(output_dir)
        self.plot_dir = self.output_dir / "plots"
        self.camera_dir = self.output_dir / "camera_views"
        self.action_dir = self.output_dir / "action_targets"
        for directory in (self.output_dir, self.plot_dir, self.camera_dir, self.action_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.plot_interval = plot_interval
        self.visualization_interval = visualization_interval
        self.action_names = tuple(action_names)
        self.jsonl_path = self.output_dir / "metrics.jsonl"
        self.csv_path = self.output_dir / "metrics.csv"
        self._history = self._load_history()
        # Upgrade old CSV headers once, preserving original values.
        if self._history and self.csv_path.is_file():
            with self.csv_path.open(encoding="utf-8") as stream:
                columns = next(csv.reader(stream), [])
            if columns != list(_SCALAR_COLUMNS):
                with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=_SCALAR_COLUMNS, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(self._history)
        self._start_time = time.monotonic()
        self._last_log_time = self._start_time
        self._last_log_step: int | None = None

    def log_step_losses(self, final_step: int, losses) -> None:
        """Persist every optimizer-batch loss after the existing interval sync.

        No new model forward/backward call or per-step GPU synchronization.
        """
        values = np.asarray(losses).reshape(-1)
        path = self.output_dir / "train_loss_steps.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            for index, loss in enumerate(values):
                step = int(final_step-len(values)+1+index)
                record = {"step": step, "optimizer_step": step+1,
                          "loss": float(loss) if np.isfinite(loss) else None,
                          "nonfinite": None if np.isfinite(loss) else str(loss)}
                stream.write(json.dumps(record, allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def attach_evaluation(self, step: int, metrics) -> None:
        """Enrich an existing logged step, never create misleading duplicate loss rows."""
        matches = [row for row in self._history if row["step"] == step]
        if not matches:
            return
        matches[-1].update({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
        self._rewrite_history()

    def _rewrite_history(self):
        with self.jsonl_path.open("w", encoding="utf-8") as stream:
            for record in self._history:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=True)+"\n")
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_SCALAR_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._history)

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.exists():
            return []
        records = []
        for line_number, line in enumerate(self.jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid monitoring JSONL at line {line_number}: {self.jsonl_path}") from error
        return records

    def should_plot(self, step: int) -> bool:
        return step == 0 or step % self.plot_interval == 0

    def should_visualize(self, step: int) -> bool:
        return step == 0 or step % self.visualization_interval == 0

    def prepare_resume(self, start_step: int) -> None:
        """Discard metrics newer than the restored checkpoint before appending resumed history."""
        raw_path = self.output_dir / "train_loss_steps.jsonl"
        if raw_path.exists():
            raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            raw_path.write_text("".join(json.dumps(row)+"\n" for row in raw if row["step"] < start_step), encoding="utf-8")
        retained = [record for record in self._history if int(record["step"]) < start_step]
        if len(retained) == len(self._history):
            return
        self._history = retained
        with self.jsonl_path.open("w", encoding="utf-8") as stream:
            for record in retained:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_SCALAR_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(retained)

    def log_metrics(self, step: int, metrics: Mapping[str, Any], *, force_plot: bool = False) -> list[pathlib.Path]:
        now = time.monotonic()
        delta_steps = 0 if self._last_log_step is None else step - self._last_log_step
        delta_time = now - self._last_log_time
        steps_per_sec = float(delta_steps / delta_time) if delta_steps > 0 and delta_time > 0 else float("nan")
        record = {
            "step": int(step),
            "wall_time_utc": dt.datetime.now(dt.UTC).isoformat(),
            "elapsed_sec": float(now - self._start_time),
            "steps_per_sec": steps_per_sec,
            "samples_per_sec": steps_per_sec * self.batch_size,
        }
        for key, value in metrics.items():
            array = np.asarray(value)
            record[key] = float(array) if array.ndim == 0 else array.astype(float).tolist()
        self._append_record(record)
        self._history.append(record)
        self._last_log_time = now
        self._last_log_step = step
        if force_plot or self.should_plot(step):
            return self.render_plots(step)
        return []

    def _append_record(self, record: Mapping[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
            stream.flush()
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=_SCALAR_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(record)
            stream.flush()

    def render_plots(self, step: int) -> list[pathlib.Path]:
        if not self._history:
            return []
        figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        steps = np.asarray([record["step"] for record in self._history], dtype=np.int64)
        loss = self._series("loss")
        axes[0, 0].plot(steps, loss, alpha=0.35, label="interval mean")
        trend, outliers = robust_trend(loss)
        axes[0, 0].plot(steps, trend, linewidth=2, label="Hampel + EMA (display only)")
        if outliers.any():
            axes[0, 0].scatter(steps[outliers], loss[outliers], s=10, color="red", label="raw peaks retained")
        loss_min, loss_max = self._series("loss_min"), self._series("loss_max")
        if np.isfinite(loss_min).any() and np.isfinite(loss_max).any():
            axes[0, 0].fill_between(steps, loss_min, loss_max, alpha=0.15, label="mean interval min/max")
        axes[0, 0].set_title("Flow-matching loss")
        axes[0, 0].legend()

        for key, label in (("grad_norm", "gradient"), ("update_norm", "optimizer update")):
            values = self._series(key)
            if np.isfinite(values).any():
                axes[0, 1].plot(steps, values, label=label)
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("Gradient / update norm (log scale)")
        if axes[0, 1].lines:
            axes[0, 1].legend()

        axes[0, 2].plot(steps, self._series("param_norm"))
        axes[0, 2].set_title("Trainable kernel parameter norm")
        axes[1, 0].plot(steps, self._series("learning_rate"))
        axes[1, 0].set_title("Learning rate")
        axes[1, 1].plot(steps, self._series("samples_per_sec"))
        axes[1, 1].set_title("Training throughput (global samples/s)")

        horizon_records = [record.get("loss_by_horizon") for record in self._history]
        latest_horizon = next((values for values in reversed(horizon_records) if values is not None), None)
        if latest_horizon is not None:
            latest_horizon = np.asarray(latest_horizon, dtype=np.float64)
            axes[1, 2].plot(np.arange(latest_horizon.size), latest_horizon, marker="o", markersize=3)
        axes[1, 2].set_title("Latest loss by action-horizon index")
        for axis in axes.flat:
            axis.grid(alpha=0.25)
            axis.set_xlabel("step" if axis is not axes[1, 2] else "horizon index")
        figure.suptitle(f"OpenPI training diagnostics — step {step}", fontsize=16)

        snapshot = self.plot_dir / f"training_overview_step_{step:06d}.png"
        latest = self.output_dir / "training_overview_latest.png"
        self._save_figure(figure, snapshot)
        temporary_latest = latest.with_name(f"{latest.name}.partial")
        temporary_latest.write_bytes(snapshot.read_bytes())
        os.replace(temporary_latest, latest)
        paths = [snapshot, latest]
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        axes[0].plot(steps, loss, linewidth=1, alpha=0.65)
        axes[0].set_title("Raw training loss: all recorded peaks retained")
        axes[1].plot(steps, trend, color="darkorange", linewidth=2)
        axes[1].set_title(f"Display-only robust trend ({int(outliers.sum())} outliers; independent y scale)")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.set_xlabel("step")
        trend_path = self.output_dir / "loss_trend_latest.png"
        self._save_figure(figure, trend_path)
        paths.append(trend_path)
        comparisons = [r for r in self._history if "val_loss" in r]
        if comparisons:
            figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
            xs = [r["step"] for r in comparisons]
            for key, label in (("train_eval_loss", "train reference / eval mode"), ("val_loss", "held-out validation")):
                axes[0].plot(xs, [r[key] for r in comparisons], marker=".", label=label)
            val = np.asarray([r["val_loss"] for r in comparisons])
            se = np.asarray([r.get("val_stderr", 0) for r in comparisons])
            axes[0].fill_between(xs, val-2*se, val+2*se, alpha=0.12, label="validation +/-2 batch SE")
            best = comparisons[int(np.argmin(val))]
            axes[0].axvline(best["step"], linestyle="--", alpha=0.5, label=f"best evaluated step {best['step']}")
            axes[0].set_title("Comparable losses: same mode, fixed samples/noise")
            axes[0].legend(fontsize=8)
            axes[1].plot(xs, [r["generalization_gap"] for r in comparisons], color="purple")
            axes[1].axhline(0, color="grey", linewidth=1)
            axes[1].set_title("Validation - train reference (raw, not despiked)")
            for axis in axes:
                axis.grid(alpha=0.25)
                axis.set_xlabel("step")
            path = self.output_dir / "generalization_latest.png"
            self._save_figure(figure, path)
            paths.append(path)
        return paths

    def _series(self, key: str) -> np.ndarray:
        return np.asarray([record.get(key, np.nan) for record in self._history], dtype=np.float64)

    def save_batch_visualizations(
        self,
        step: int,
        images: Mapping[str, np.ndarray],
        actions: np.ndarray,
        *,
        max_samples: int = 2,
    ) -> list[pathlib.Path]:
        paths = [self._save_camera_montage(step, images, max_samples=max_samples)]
        if np.asarray(actions).ndim == 3:
            paths.append(self._save_action_targets(step, np.asarray(actions), max_samples=max_samples))
        return paths

    def _save_camera_montage(self, step: int, images: Mapping[str, np.ndarray], *, max_samples: int) -> pathlib.Path:
        camera_names = tuple(images)
        if not camera_names:
            raise ValueError("Cannot visualize a batch without camera images.")
        sample_count = min(max_samples, np.asarray(images[camera_names[0]]).shape[0])
        figure, axes = plt.subplots(sample_count, len(camera_names), figsize=(5 * len(camera_names), 4 * sample_count))
        axes = np.asarray(axes, dtype=object).reshape(sample_count, len(camera_names))
        for row in range(sample_count):
            for column, camera_name in enumerate(camera_names):
                axes[row, column].imshow(_to_uint8(np.asarray(images[camera_name])[row]))
                axes[row, column].set_title(f"sample {row} — {camera_name}")
                axes[row, column].axis("off")
        figure.suptitle(f"Pre-augmentation model camera batch — step {step}")
        figure.tight_layout()
        path = self.camera_dir / f"camera_step_{step:06d}.png"
        self._save_figure(figure, path)
        return path

    def _save_action_targets(self, step: int, actions: np.ndarray, *, max_samples: int) -> pathlib.Path:
        action_dim = min(actions.shape[-1], len(self.action_names) or actions.shape[-1])
        sample_count = min(max_samples, actions.shape[0])
        action_groups = [group for group in np.array_split(np.arange(action_dim), min(3, action_dim)) if group.size]
        figure, axes = plt.subplots(
            sample_count,
            len(action_groups),
            figsize=(7 * len(action_groups), 4.5 * sample_count),
            squeeze=False,
        )
        horizon = np.arange(actions.shape[1])
        for sample_index in range(sample_count):
            for group_index, group in enumerate(action_groups):
                axis = axes[sample_index, group_index]
                for action_index in group:
                    label = (
                        self.action_names[action_index]
                        if action_index < len(self.action_names)
                        else f"action_{action_index}"
                    )
                    axis.plot(horizon, actions[sample_index, :, action_index], label=label, linewidth=1)
                axis.set_title(f"sample {sample_index}: action dimensions {group[0]}-{group[-1]}")
                axis.set_xlabel("horizon index")
                axis.set_ylabel("normalized delta/absolute target")
                axis.grid(alpha=0.25)
                axis.legend(fontsize=7)
        figure.suptitle(f"Action targets seen by the model — step {step}")
        figure.tight_layout()
        path = self.action_dir / f"action_target_step_{step:06d}.png"
        self._save_figure(figure, path)
        return path

    @staticmethod
    def _save_figure(figure: plt.Figure, path: pathlib.Path) -> None:
        temporary = path.with_name(f"{path.name}.partial")
        figure.savefig(temporary, dpi=140, format="png")
        plt.close(figure)
        os.replace(temporary, path)
