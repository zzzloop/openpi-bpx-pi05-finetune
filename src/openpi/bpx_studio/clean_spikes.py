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
        jump, neighbor = output[key]["jump"], output[key]["neighbor"]
        if (jump.ndim != 1 or not jump.size or jump.shape != neighbor.shape
                or not np.all(np.isfinite(jump)) or not np.all(np.isfinite(neighbor))
                or np.any(jump <= 0) or np.any(neighbor < 0)):
            raise ValueError(f"{key}: 阈值必须为等长有限向量，jump > 0 且 neighbor >= 0")
    return output


def _clean_parquet(arguments: tuple[pathlib.Path, dict[str, dict[str, np.ndarray]], bool]) -> dict[str, Any]:
    path, thresholds, apply = arguments
    table = pq.read_table(path)
    counts = {}
    repairs = []
    unresolved_jumps = []
    unresolved_counts = {}
    output = table
    episode_index = None
    if "episode_index" in table.column_names:
        episode_values = np.asarray(table["episode_index"].combine_chunks().to_pylist(), dtype=np.int64)
        if episode_values.size and np.unique(episode_values).size == 1:
            episode_index = int(episode_values[0])
        elif episode_values.size:
            raise ValueError(f"{path}: 一个 Parquet 包含多个 episode，禁止跨轨迹插值")
    for index_key in ("frame_index", "timestamp"):
        if index_key in table.column_names:
            index_values = np.asarray(table[index_key].to_pylist(), dtype=np.float64)
            if not np.all(np.isfinite(index_values)) or np.any(np.diff(index_values) <= 0):
                raise ValueError(f"{path}: {index_key} 必须有限且严格递增，禁止跨乱序帧插值")
            if index_key == "frame_index" and np.any(np.diff(index_values) != 1):
                raise ValueError(f"{path}: frame_index 有缺帧，禁止跨缺帧插值")
    for key, config in thresholds.items():
        column_index = output.schema.get_field_index(key)
        if column_index < 0:
            raise ValueError(f"{path}: 缺少 {key}")
        # Keep float64 columns intact; coercing to float32 altered untouched data.
        field = output.schema.field(column_index)
        if not (pa.types.is_list(field.type) or pa.types.is_fixed_size_list(field.type)):
            raise ValueError(f"{path}: {key} 必须为浮点向量")
        if not pa.types.is_floating(field.type.value_type):
            raise ValueError(f"{path}: {key} 必须为浮点向量")
        values = np.asarray(output[key].combine_chunks().to_pylist(), dtype=np.float64)
        result = repair_isolated_spikes(
            values,
            jump_thresholds=config["jump"],
            neighbor_thresholds=config["neighbor"],
        )
        counts[key] = result.count
        unresolved_counts[key] = int(result.unresolved_jump_mask.sum())
        # Keep reports bounded while retaining exact totals and review locations.
        for frame_index, dimension in np.argwhere(result.unresolved_jump_mask)[:200]:
            frame, axis = int(frame_index), int(dimension)
            unresolved_jumps.append({
                "feature": key, "frame_index": frame, "dimension": axis,
                "previous": float(result.repaired[frame - 1, axis]),
                "current": float(result.repaired[frame, axis]),
                "delta": float(result.repaired[frame, axis] - result.repaired[frame - 1, axis]),
            })
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
    return {
        "path": str(path),
        "episode_index": episode_index,
        "counts": counts,
        "total": sum(counts.values()),
        "repairs": repairs,
        "unresolved_jump_counts": unresolved_counts,
        "unresolved_jumps": unresolved_jumps,
    }


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
    thresholds = _load_thresholds(args.thresholds)
    target = source
    if args.apply:
        if args.output_root is None:
            parser.error("--output-root is required with --apply")
        target = args.output_root.resolve()
        _copy_hardlinked(source, target)
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
        "report_version": 3,
        "algorithm": "isolated-excursion-v2",
        "source": str(source),
        "output": str(target) if args.apply else None,
        "applied": args.apply,
        "thresholds": str(args.thresholds.resolve()),
        "clean_audit_passed": False,
        "episode_files": len(results),
        "repairs": sum(result["total"] for result in results),
        "unresolved_jumps": sum(sum(result["unresolved_jump_counts"].values()) for result in results),
        "review_note": "剩余大跳变可能为真实动作、阶跃、多帧异常或边界异常；未自动平滑，需人工复核。每文件每特征最多列出200个位置。",
        "files": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.apply:
        report["post_clean_audit"] = audit_dataset(target, args.workers)
        report["clean_audit_passed"] = True
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "files"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("SPIKE CLEAN AND AUDIT PASS" if args.apply else "SPIKE SCAN PASS")


if __name__ == "__main__":
    main()
