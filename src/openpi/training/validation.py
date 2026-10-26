"""Fixed forward-only comparison, plus restart-safe raw validation history."""

from __future__ import annotations

import json
import math
import os
import pathlib
import time

import numpy as np


def reference_frame_indices(inventory, count: int, seed: int):
    """Task-stratified, fixed samples without replacement, in subset-local order."""
    by_task = {}
    offset = 0
    for task, length in inventory:
        by_task.setdefault(task, []).extend(range(offset, offset + length))
        offset += length
    if count < len(by_task):
        raise ValueError("Increase validation_batches: evaluation needs at least one frame per task")
    rng = np.random.default_rng(seed)
    chosen, remaining = [], []
    for index, task in enumerate(sorted(by_task)):
        candidates = np.asarray(by_task[task], dtype=np.int64)
        rng.shuffle(candidates)
        quota = count // len(by_task) + (index < count % len(by_task))
        take = min(quota, len(candidates))
        chosen.extend(candidates[:take].tolist())
        remaining.extend(candidates[take:].tolist())
    if len(chosen) < count:
        chosen.extend(rng.choice(remaining, count-len(chosen), replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def evaluate_batches(batches, evaluate):
    """The caller owns device placement; this never updates the optimizer or model."""
    values = np.asarray([float(evaluate(index, batch)) for index, batch in enumerate(batches)], dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise FloatingPointError("Validation loss is empty or non-finite; raw failures must not be smoothed away")
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0


class EvaluationHistory:
    def __init__(self, directory: pathlib.Path, identity: dict, *, start_step: int, resume: bool):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "validation.jsonl"
        self.summary_path = directory / "validation_summary.json"
        identity_path = directory / "validation_protocol.json"
        if resume:
            if not identity_path.is_file() or json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("Validation protocol changed/missing on resume; use a new experiment")
        self._atomic(identity_path, identity)
        records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()] if self.path.exists() else []
        self.records = [r for r in records if r["step"] < start_step]
        self.summary_path.unlink(missing_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, allow_nan=False) + "\n")

    @staticmethod
    def _atomic(path, payload):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)

    def append(self, step: int, train_loss: float, val_loss: float, *, train_stderr: float,
               val_stderr: float, seconds: float, checkpoint_saved: bool) -> dict:
        if not all(math.isfinite(x) for x in (train_loss, val_loss, train_stderr, val_stderr)):
            raise FloatingPointError("Non-finite evaluation metrics")
        record = {"step": step, "train_eval_loss": train_loss, "val_loss": val_loss,
                  "generalization_gap": val_loss - train_loss, "train_eval_stderr": train_stderr,
                  "val_stderr": val_stderr, "evaluation_seconds": seconds,
                  "checkpoint_saved": checkpoint_saved, "timestamp": time.time()}
        self.records.append(record)
        best = min(self.records, key=lambda r: r["val_loss"])
        recent = self.records[-3:]
        warning = (len(recent) == 3 and all(r["val_loss"] > best["val_loss"] * 1.02 for r in recent)
                   and recent[-1]["train_eval_loss"] < recent[0]["train_eval_loss"])
        saved = [r for r in self.records if r["checkpoint_saved"]]
        best_saved = min(saved, key=lambda r: r["val_loss"]) if saved else None
        summary = {"best_validation_step": best["step"], "best_validation_loss": best["val_loss"],
                   "best_evaluated_saved_step": best_saved["step"] if best_saved else None,
                   "overfit_warning": warning, "evaluations": len(self.records)}
        record.update(summary)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._atomic(self.summary_path, {**record, **summary})
        return record

    def mark_checkpoint_complete(self, step: int) -> None:
        """Selection calls this only AFTER Orbax wait_until_finished returns."""
        for record in self.records:
            if record["step"] == step:
                record["checkpoint_saved"] = True
        if self.records:
            saved = [record for record in self.records if record["checkpoint_saved"]]
            best_saved = min(saved, key=lambda record: record["val_loss"]) if saved else None
            self.records[-1]["best_evaluated_saved_step"] = best_saved["step"] if best_saved else None
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, allow_nan=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if self.records:
            self._atomic(self.summary_path, self.records[-1])
