"""Read bounded, review-friendly views from BPX numeric spike reports."""

from __future__ import annotations

from collections import Counter
import importlib
import itertools
import json
import pathlib
from typing import Any

import numpy as np

_FEATURES = ("observation.state", "action")


def _load_report(path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".json":
        raise ValueError("毛刺报告必须是 JSON 文件")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("files"), list):
        raise ValueError(f"不是有效的 BPX 毛刺报告: {resolved}")
    return resolved, report


def _dimension_names(dataset_root: pathlib.Path, feature: str, dimension_count: int) -> list[str]:
    names: Any = None
    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        names = info.get("features", {}).get(feature, {}).get("names")
    if isinstance(names, list):
        return [str(names[index]) if index < len(names) else f"dim_{index}" for index in range(dimension_count)]
    if isinstance(names, dict):
        return [str(names.get(str(index), names.get(index, f"dim_{index}"))) for index in range(dimension_count)]
    return [f"dim_{index}" for index in range(dimension_count)]


def _relative_parquet(report: dict[str, Any], file_record: dict[str, Any]) -> tuple[pathlib.Path, pathlib.Path | None]:
    source_root = pathlib.Path(str(report["source"])).expanduser().resolve()
    output_value = report.get("output")
    output_root = pathlib.Path(str(output_value)).expanduser().resolve() if output_value else None
    recorded = pathlib.Path(str(file_record["path"])).expanduser().resolve()
    recorded_root = output_root if output_root is not None else source_root
    try:
        relative = recorded.relative_to(recorded_root)
    except ValueError as error:
        raise ValueError(f"报告中的 Parquet 不在数据集目录内: {recorded}") from error
    source_file = source_root / relative
    output_file = output_root / relative if output_root is not None else None
    if source_file.suffix.lower() != ".parquet" or not source_file.is_file():
        raise FileNotFoundError(source_file)
    if output_file is not None and not output_file.is_file():
        raise FileNotFoundError(output_file)
    return source_file, output_file


def summarize_spike_report(
    report_path: pathlib.Path,
    *,
    offset: int = 0,
    limit: int = 100,
    feature: str = "",
    dimension: int | None = None,
) -> dict[str, Any]:
    """Return a bounded page of hits plus complete, cheap aggregate counts."""
    if offset < 0 or not 1 <= limit <= 500:
        raise ValueError("offset 必须非负，limit 必须在 1 到 500 之间")
    if feature and feature not in _FEATURES:
        raise ValueError(f"未知毛刺特征: {feature}")
    if dimension is not None and dimension < 0:
        raise ValueError("dimension 必须非负")

    resolved, report = _load_report(report_path)
    source_root = pathlib.Path(str(report["source"])).expanduser().resolve()
    by_feature: Counter[str] = Counter()
    by_dimension: Counter[tuple[str, int]] = Counter()
    affected_files = 0
    matched = 0
    page: list[dict[str, Any]] = []
    maximum_dimension: dict[str, int] = {key: -1 for key in _FEATURES}

    for file_index, file_record in enumerate(report["files"]):
        repairs = file_record.get("repairs", [])
        if repairs:
            affected_files += 1
        for repair_index, repair in enumerate(repairs):
            hit_feature = str(repair["feature"])
            hit_dimension = int(repair["dimension"])
            by_feature[hit_feature] += 1
            by_dimension[(hit_feature, hit_dimension)] += 1
            maximum_dimension[hit_feature] = max(maximum_dimension.get(hit_feature, -1), hit_dimension)
            if feature and hit_feature != feature:
                continue
            if dimension is not None and hit_dimension != dimension:
                continue
            if offset <= matched < offset + limit:
                page.append(
                    {
                        **repair,
                        "file_index": file_index,
                        "repair_index": repair_index,
                        "episode_index": file_record.get("episode_index"),
                        "path": str(file_record.get("path", "")),
                    }
                )
            matched += 1

    names = {
        key: _dimension_names(source_root, key, maximum_dimension[key] + 1)
        for key in _FEATURES
        if maximum_dimension[key] >= 0
    }
    for hit in page:
        hit_names = names.get(str(hit["feature"]), [])
        axis = int(hit["dimension"])
        hit["dimension_name"] = hit_names[axis] if axis < len(hit_names) else f"dim_{axis}"

    stat = resolved.stat()
    report_identity = f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}"
    return {
        "report_path": str(resolved),
        "report_identity": report_identity,
        "source": str(source_root),
        "output": report.get("output"),
        "applied": bool(report.get("applied")),
        "clean_audit_passed": bool(report.get("clean_audit_passed")),
        "unresolved_jumps": report.get("unresolved_jumps"),
        "unresolved_examples": list(itertools.islice((
            {**hit, "episode_index": record.get("episode_index"), "path": record.get("path")}
            for record in report["files"]
            for hit in record.get("unresolved_jumps", [])
        ), 100)),
        "total_repairs": int(report.get("repairs", sum(by_feature.values()))),
        "matched_repairs": matched,
        "affected_files": affected_files,
        "episode_files": int(report.get("episode_files", len(report["files"]))),
        "by_feature": dict(by_feature),
        "by_dimension": [
            {
                "feature": key[0],
                "dimension": key[1],
                "dimension_name": names.get(key[0], [])[key[1]]
                if key[1] < len(names.get(key[0], []))
                else f"dim_{key[1]}",
                "count": count,
            }
            for key, count in sorted(by_dimension.items())
        ],
        "offset": offset,
        "limit": limit,
        "hits": page,
    }


def load_spike_trace(
    report_path: pathlib.Path,
    *,
    file_index: int,
    repair_index: int,
    radius: int = 12,
) -> dict[str, Any]:
    """Load a small source/cleaned trace and the full vector at one hit."""
    pq = importlib.import_module("pyarrow.parquet")

    if file_index < 0 or repair_index < 0:
        raise ValueError("file_index 和 repair_index 必须非负")
    if not 2 <= radius <= 100:
        raise ValueError("查看半径必须在 2 到 100 帧之间")
    _, report = _load_report(report_path)
    try:
        file_record = report["files"][file_index]
        repair = file_record["repairs"][repair_index]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("毛刺索引不存在") from error
    feature = str(repair["feature"])
    if feature not in _FEATURES:
        raise ValueError(f"报告包含未知特征: {feature}")
    dimension = int(repair["dimension"])
    center = int(repair["frame_index"])
    source_file, output_file = _relative_parquet(report, file_record)

    source_table = pq.read_table(source_file, columns=[feature, "frame_index", "timestamp"])
    source = np.asarray(source_table[feature].combine_chunks().to_pylist(), dtype=np.float64)
    if source.ndim != 2 or not 0 <= dimension < source.shape[1] or not 0 <= center < source.shape[0]:
        raise ValueError("报告中的帧或维度超出 Parquet 范围")
    cleaned = source
    if output_file is not None:
        cleaned_table = pq.read_table(output_file, columns=[feature])
        cleaned = np.asarray(cleaned_table[feature].combine_chunks().to_pylist(), dtype=np.float64)
        if cleaned.shape != source.shape:
            raise ValueError("清洗前后 Parquet 数值形状不一致")
    else:
        cleaned = source.copy()
        cleaned[center, dimension] = float(repair["repaired"])

    start = max(0, center - radius)
    stop = min(source.shape[0], center + radius + 1)
    frame_indices = np.asarray(source_table["frame_index"].combine_chunks().to_pylist(), dtype=np.int64)
    timestamps = np.asarray(source_table["timestamp"].combine_chunks().to_pylist(), dtype=np.float64)
    source_root = pathlib.Path(str(report["source"])).expanduser().resolve()
    names = _dimension_names(source_root, feature, source.shape[1])
    return {
        "feature": feature,
        "dimension": dimension,
        "dimension_name": names[dimension],
        "episode_index": file_record.get("episode_index"),
        "source_path": str(source_file),
        "output_path": str(output_file) if output_file is not None else None,
        "center_frame": center,
        "preview_only": output_file is None,
        "frames": frame_indices[start:stop].tolist(),
        "timestamps": timestamps[start:stop].tolist(),
        "source_values": source[start:stop, dimension].tolist(),
        "review_values": cleaned[start:stop, dimension].tolist(),
        "frame_vector": [
            {
                "dimension": index,
                "name": names[index],
                "source": float(source[center, index]),
                "cleaned": float(cleaned[center, index]),
            }
            for index in range(source.shape[1])
        ],
    }
