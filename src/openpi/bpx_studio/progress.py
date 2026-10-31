from __future__ import annotations

import re
from typing import Any

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TRAIN_STEP = re.compile(r"Step\s+(?P<step>\d+):\s*(?P<metrics>.+)")
_METRIC = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[-+0-9.eE]+)")
_TQDM = re.compile(r"(?P<percent>\d{1,3})%.*?\|\s*(?P<current>\d+)/(?P<total>\d+)")
_COUNTER = re.compile(
    r"(?:EXPORTED|SCANNED|AUDITED\s+(?:PARQUET|VIDEO))\s+(?P<current>\d+)/(?P<total>\d+)",
    re.IGNORECASE,
)


def parse_progress_line(
    line: str, *, total_units: int | None = None, job_kind: str | None = None
) -> dict[str, Any] | None:
    clean = _ANSI.sub("", line).replace("\r", "").strip()
    if not clean:
        return None
    if job_kind == "normalize":
        if clean.startswith("[BPX NORM] "):
            return {"kind": "phase", "phase": clean.removeprefix("[BPX NORM] ")}
        # Hugging Face emits separate download/cache tqdm bars. Those are NOT
        # normalization progress, even when they report 80/80 or 100%.
        if not clean.startswith("Computing stats:"):
            return None
    match = _TRAIN_STEP.search(clean)
    if match:
        step = int(match.group("step"))
        metrics = {item.group("name"): float(item.group("value")) for item in _METRIC.finditer(match.group("metrics"))}
        result: dict[str, Any] = {"kind": "train", "current": step, "metrics": metrics}
        if total_units:
            result.update(total=total_units, percent=min(100.0, 100.0 * step / total_units))
        return result
    match = _TQDM.search(clean)
    if match:
        return {
            "kind": "progress",
            "current": int(match.group("current")),
            "total": int(match.group("total")),
            # Computing all batches still leaves quantiles and the output save.
            # Only JobManager's successful process exit may mark the job 100%.
            "percent": min(99.9, float(match.group("percent"))) if job_kind == "normalize"
            else float(match.group("percent")),
        }
    match = _COUNTER.search(clean)
    if match:
        current, total = int(match.group("current")), int(match.group("total"))
        return {"kind": "progress", "current": current, "total": total, "percent": 100.0 * current / total}
    return None
