"""Scan or non-destructively repair isolated numeric spikes in LeRobot data."""

from __future__ import annotations

import argparse
from concurrent import futures
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.bpx_studio.audit import audit_dataset
from openpi.bpx_studio.numeric_spikes import repair_isolated_spikes


def _load_thresholds(path: pathlib.Path) -> dict[str, dict[str, np.ndarray]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for key in ("observation.state", "action"):
        if key not in raw:
            raise ValueError(f"阈值配置缺少 {key}")
        output[key] = {
            "jump": np.asarray(raw[key]["jump_thresholds"], dtype=np.float64),
            "neighbor": np.asarray(raw[key]["neighbor_thresholds"], dtype=np.float64),
        }
    return output


def _clean_parquet(arguments: tuple[pathlib.Path, dict[str, dict[str, np.ndarray]], bool]) -> dict[str, Any]:
    path, thresholds, apply = arguments
    table = pq.read_table(path)
    counts = {}
    repairs = []
    output = table
    for key, config in thresholds.items():
        column_index = output.schema.get_field_index(key)
        if column_index < 0:
            raise ValueError(f"{path}: 缺少 {key}")
        values = np.asarray(output[key].combine_chunks().to_pylist(), dtype=np.float32)
        result = repair_isolated_spikes(
            values,
            jump_thresholds=config["jump"],
            neighbor_thresholds=config["neighbor"],
        )
        counts[key] = result.count
        for frame_index, dimension in np.argwhere(result.mask):
            frame = int(frame_index)
            axis = int(dimension)
            repairs.append(
                {
                    "feature": key,
                    "frame_index": frame,
                    "dimension": axis,
                    "previous": float(values[frame - 1, axis]),
                    "original": float(values[frame, axis]),
                    "following": float(values[frame + 1, axis]),
                    "repaired": float(result.repaired[frame, axis]),
                }
            )
        if apply and result.count:
            field = output.schema.field(column_index)
            output = output.set_column(column_index, field, pa.array(result.repaired.tolist(), type=field.type))
    if apply and sum(counts.values()):
        with tempfile.NamedTemporaryFile(prefix=path.name, suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = pathlib.Path(handle.name)
        try:
            pq.write_table(output, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(path), "counts": counts, "total": sum(counts.values()), "repairs": repairs}


def _copy_hardlinked(source: pathlib.Path, output: pathlib.Path) -> None:
    if output.exists():
        raise FileExistsError(f"输出已存在: {output}")
    staging = output.with_name(f".{output.name}.partial")
    if staging.exists():
        raise FileExistsError(f"暂存目录已存在: {staging}")
    try:
        shutil.copytree(source, staging, copy_function=os.link)
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=pathlib.Path)
    parser.add_argument("--thresholds", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output-root", type=pathlib.Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    source = args.source_root.resolve()
    target = source
    if args.apply:
        if args.output_root is None:
            parser.error("--output-root is required with --apply")
        target = args.output_root.resolve()
        _copy_hardlinked(source, target)
    thresholds = _load_thresholds(args.thresholds)
    parquet_files = sorted((target / "data").rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"数据集没有 parquet episode: {target}")
    jobs = [(path, thresholds, args.apply) for path in parquet_files]
    with futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = []
        for index, result in enumerate(executor.map(_clean_parquet, jobs), start=1):
            results.append(result)
            print(f"SCANNED {index}/{len(jobs)}", flush=True)
    report = {
        "source": str(source),
        "output": str(target) if args.apply else None,
        "applied": args.apply,
        "episode_files": len(results),
        "repairs": sum(result["total"] for result in results),
        "files": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.apply:
        audit_dataset(target, args.workers)
    summary = {key: value for key, value in report.items() if key != "files"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("SPIKE CLEAN AND AUDIT PASS" if args.apply else "SPIKE SCAN PASS")


if __name__ == "__main__":
    main()
