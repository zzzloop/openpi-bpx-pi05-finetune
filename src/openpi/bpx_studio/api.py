"""LAN dashboard for BPX data engineering and OpenPI training."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import json
import os
import pathlib
import shutil
import subprocess
from typing import Literal

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import Field
import uvicorn

from openpi.bpx_studio.commands import build_audit
from openpi.bpx_studio.commands import build_hdf5_export
from openpi.bpx_studio.commands import build_normalize
from openpi.bpx_studio.commands import build_policy_serve
from openpi.bpx_studio.commands import build_spike_job
from openpi.bpx_studio.commands import build_train
from openpi.bpx_studio.commands import build_validation_split
from openpi.bpx_studio.commands import build_probe_suite
from openpi.bpx_studio.jobs import JobManager
from openpi.bpx_studio.config_profile import read_config_binding
from openpi.bpx_studio.status_probe import CachedProbe
from openpi.bpx_studio.spike_viewer import load_spike_trace
from openpi.bpx_studio.spike_viewer import summarize_spike_report

DEFAULT_DATA_ROOT = pathlib.Path("/scratch/bpx_datasets")
DEFAULT_CONFIG_NAME = "pi05_bpx_full_low_mem_multitask_v2"
DEFAULT_DATASET_NAME = "bpx_v2_humanoid"
BPX_CONFIG_PURPOSES = {
    "pi05_bpx_full": "第一代 BPX 基线: bpx_blocks 数据集, 全参数微调, 默认显存需求较高。",
    "pi05_bpx_full_low_mem": "第一代 BPX 低显存全参数版本: bpx_blocks 数据集, 适配两张 48 GiB GPU。",
    "pi05_bpx_full_low_mem_combined": "第一代 BPX 合并数据版本: 原 59 条示教与新增多任务数据的全参数微调。",
    "pi05_bpx_full_low_mem_multitask_v2": "第一代 BPX 多任务 v2: 裁减固定轨迹并按任务均衡采样的低显存全参数微调。",
    "pi05_bpx_full_low_mem_v1_base_estimated": "第一代80条基础集的估计反馈试验：非真实反馈修复，使用独立数据/统计/模型，不能作实机安全或VLM保留证明。",
    "pi05_bpx_lora": "第一代 BPX LoRA 对照实验: bpx_blocks 数据集, 仅训练低秩适配参数。",
}


class DataEnvironment(BaseModel):
    hf_lerobot_home: str = ""
    hf_home: str = ""
    hf_datasets_cache: str = ""
    tmpdir: str = ""
    uv_cache_dir: str = ""
    validation_manifest: str = ""


class ValidationSplitRequest(BaseModel):
    dataset_root: str
    output_path: str
    source_manifest: str = ""
    independent_episodes: bool = False
    fraction: float = Field(default=0.1, gt=0, lt=0.5)


class AuditRequest(BaseModel):
    dataset_root: str
    report_path: str
    workers: int = Field(default=32, ge=1, le=256)
    video_check: Literal["none", "headers", "full"] = "headers"


class SpikeRequest(BaseModel):
    source_root: str
    thresholds_path: str
    report_path: str
    output_root: str = ""
    workers: int = Field(default=32, ge=1, le=256)


class NormalizeRequest(DataEnvironment):
    openpi_root: str
    config_name: str
    workers: int = Field(default=32, ge=1, le=256)


class Hdf5ExportRequest(BaseModel):
    source_root: str
    output_root: str
    workers: int = Field(default=4, ge=1, le=64)
    compression: Literal["gzip", "lzf", "none"] = "gzip"
    compression_level: int = Field(default=4, ge=0, le=9)
    episode_indices: list[int] | None = None


class TrainRequest(DataEnvironment):
    openpi_root: str
    config_name: str
    exp_name: str
    checkpoint_base_dir: str
    cuda_devices: str = "2,3"
    memory_fraction: float = Field(default=0.96, ge=0.1, le=1.0)
    batch_size: int = Field(default=2, ge=1)
    num_train_steps: int = Field(default=200000, ge=1)
    fsdp_devices: int = Field(default=2, ge=1)
    save_interval: int = Field(default=5000, ge=1)
    keep_period: int = Field(default=25000, ge=1)
    checkpoint_limit: int | None = Field(default=None, ge=1, le=100)
    mode: Literal["overwrite", "resume"] = "overwrite"
    wandb_mode: Literal["offline", "online", "disabled"] = "offline"
    validation_interval: int = Field(default=1000, ge=1)
    validation_batches: int = Field(default=16, ge=1, le=256)
    selection_suite: str = ""
    selection_keep: int = Field(default=3, ge=1, le=20)
    selection_denoise_steps: int = Field(default=10, ge=1, le=100)
    init_checkpoint: str = ""


class ProbeSuiteRequest(DataEnvironment):
    config_name: str
    output_dir: str
    case_spec: str = ""
    max_cases: int = Field(default=16, ge=1, le=256)


class ServeRequest(BaseModel):
    openpi_root: str
    config_name: str = DEFAULT_CONFIG_NAME
    checkpoint_dir: str
    cuda_devices: str = "2,3"
    memory_fraction: float = Field(default=0.96, ge=0.1, le=1.0)


def default_paths(data_root: pathlib.Path) -> dict[str, str]:
    root = data_root.expanduser().resolve()
    lerobot_root = root / "lerobot"
    raw_dataset = lerobot_root / "bpx" / f"{DEFAULT_DATASET_NAME}_raw"
    clean_dataset = lerobot_root / "bpx" / "bpx_blocks_multitask_v2"
    return {
        "data_root": str(root),
        "hf_lerobot_home": str(root / "lerobot"),
        "checkpoint_root": str(root / "checkpoints"),
        "hf_home": str(root / "cache" / "huggingface"),
        "hf_datasets_cache": str(root / "cache" / "huggingface" / "datasets"),
        "uv_cache_dir": str(root / "cache" / "uv"),
        "tmpdir": str(root / "tmp"),
        "reports_root": str(root / "reports"),
        "cleaned_root": str(root / "cleaned"),
        "exports_root": str(root / "exports"),
        "state_root": str(root / "dashboard_state"),
        "raw_dataset": str(raw_dataset),
        "clean_dataset": str(clean_dataset),
        "audit_report": str(root / "reports" / f"{DEFAULT_DATASET_NAME}_audit.json"),
        "spike_scan_report": str(root / "reports" / f"{DEFAULT_DATASET_NAME}_spikes_scan.json"),
        "spike_clean_report": str(root / "reports" / f"{DEFAULT_DATASET_NAME}_spikes_clean.json"),
        "hdf5_export": str(root / "exports" / f"{DEFAULT_DATASET_NAME}_hdf5"),
        "validation_split": str(root / "reports" / "validation_split_v1.json"),
        "selection_suite_output": str(root / "reports" / "selection_suite_v1"),
    }


def _checkpoints(checkpoint_root: pathlib.Path) -> list[dict[str, str | int]]:
    records = []
    if not checkpoint_root.is_dir():
        return records
    for params in checkpoint_root.glob("*/*/*/params"):
        step_dir = params.parent
        if not step_dir.name.isdigit() or not (step_dir / "assets").is_dir():
            continue
        records.append(
            {
                "config_name": step_dir.parents[1].name,
                "experiment_name": step_dir.parent.name,
                "step": int(step_dir.name),
                "path": str(step_dir.resolve()),
            }
        )
    for params in checkpoint_root.glob("*/*/selected_checkpoints/*/params"):
        step_dir = params.parent
        if step_dir.name.isdigit() and (step_dir / "assets").is_dir() and (step_dir / "selection.json").is_file():
            records.append({"config_name": step_dir.parents[2].name,
                            "experiment_name": step_dir.parents[1].name+" / SELECTED",
                            "step": int(step_dir.name), "path": str(step_dir.resolve())})
    return sorted(
        records,
        key=lambda item: (str(item["config_name"]), str(item["experiment_name"]), int(item["step"])),
        reverse=True,
    )


def _next_experiment(checkpoint_root: pathlib.Path, config_name: str) -> str:
    if pathlib.Path(config_name).name != config_name or not config_name:
        raise ValueError("config_name 不是安全的单级名称")
    prefix = f"{DEFAULT_DATASET_NAME}_bs2_v"
    existing = {path.name for path in (checkpoint_root / config_name).glob(f"{prefix}*")}
    for version in range(1, 10000):
        candidate = f"{prefix}{version:03d}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("无法生成新的实验编号")


def _absolute_from_project(project_root: pathlib.Path, path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path).expanduser()
    return (candidate if candidate.is_absolute() else project_root / candidate).resolve()


def _config_profile(
    *,
    project_root: pathlib.Path,
    lerobot_root: pathlib.Path,
    train_config,
) -> dict[str, object]:
    data_factory = train_config.data
    repo_id = getattr(data_factory, "repo_id", None)
    if not isinstance(repo_id, str) or not repo_id or repo_id == "fake":
        raise ValueError(f"Config '{train_config.name}' 没有可用的 LeRobot repo_id")

    dataset_path = (lerobot_root / pathlib.Path(repo_id)).resolve()
    computed_stats_dir = (
        _absolute_from_project(project_root, train_config.assets_base_dir) / train_config.name / repo_id
    )
    assets = getattr(data_factory, "assets", None)
    configured_assets_dir = getattr(assets, "assets_dir", None) if assets is not None else None
    asset_id = (getattr(assets, "asset_id", None) if assets is not None else None) or repo_id
    loaded_assets_root = (
        _absolute_from_project(project_root, configured_assets_dir)
        if configured_assets_dir
        else _absolute_from_project(project_root, train_config.assets_base_dir) / train_config.name
    )
    loaded_stats_dir = (loaded_assets_root / asset_id).resolve()
    purpose = BPX_CONFIG_PURPOSES.get(
        train_config.name,
        "自定义 OpenPI 配置; 请根据 repo_id、输入映射、动作维度和归一化资产人工复核用途。",
    )
    computed_stats_path = (computed_stats_dir / "norm_stats.json").resolve()
    loaded_stats_path = (loaded_stats_dir / "norm_stats.json").resolve()
    recovery_path = dataset_path / "meta/bpx_feedback_recovery.json"
    feedback_recovery = None
    if recovery_path.is_file():
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
        totals = recovery.get("totals", {})
        feedback_recovery = {
            "report_path": str(recovery_path),
            "status": recovery.get("status"),
            "structural_audit_passed": bool(recovery.get("post_audit"))
            and recovery.get("status") == "estimated_experiment_only"
            and pathlib.Path(str(recovery.get("output", ""))).resolve() == dataset_path,
            "estimated_coordinates": totals.get("estimated_coordinates", 0),
            "extrapolated_coordinates": totals.get("extrapolated_coordinates", 0),
            "right_gripper_dynamics_validated": False,
            "validation_manifest": recovery.get("validation_manifest"),
        }
    return {
        "config_name": train_config.name,
        "repo_id": repo_id,
        "dataset_path": str(dataset_path),
        "dataset_exists": dataset_path.is_dir(),
        "computed_stats_path": str(computed_stats_path),
        "loaded_stats_path": str(loaded_stats_path),
        "stats_paths_match": computed_stats_dir.resolve() == loaded_stats_dir,
        "computed_stats_exists": computed_stats_path.is_file(),
        "loaded_stats_exists": loaded_stats_path.is_file(),
        "stats_size_bytes": computed_stats_path.stat().st_size if computed_stats_path.is_file() else 0,
        "purpose": purpose,
        "feedback_recovery": feedback_recovery,
    }


def _gpu_snapshot() -> dict:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"gpus": [], "gpu_error": f"nvidia-smi 查询失败: {error}"}
    fields = ("index", "name", "memory_used_mib", "memory_total_mib", "utilization_percent", "temperature_c")
    gpus = []
    malformed = False
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != len(fields):
            malformed = True
            continue
        gpus.append(dict(zip(fields, (part.strip() for part in row), strict=True)))
    error = "部分 GPU 返回格式异常" if malformed else ("未检测到 NVIDIA GPU" if not gpus else None)
    return {"gpus": gpus, "gpu_error": error}



def create_app(*, state_dir: pathlib.Path, project_root: pathlib.Path, data_root: pathlib.Path) -> FastAPI:
    web_root = pathlib.Path(__file__).with_name("web")
    manager = JobManager(state_dir)
    gpu_probe = CachedProbe(_gpu_snapshot)
    app = FastAPI(title="OpenPI BPX Research Studio")

    def start(kind: str, builder):
        try:
            return manager.start(kind, builder())
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/")
    def index():
        return FileResponse(web_root / "index.html")

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    @app.get("/api/config")
    def config():
        checkpoint_root = data_root / "checkpoints"
        return {
            "default_openpi_root": str(project_root),
            "default_workers": 32,
            "default_config_name": DEFAULT_CONFIG_NAME,
            "known_config_names": list(BPX_CONFIG_PURPOSES),
            "default_experiment_name": _next_experiment(checkpoint_root, DEFAULT_CONFIG_NAME),
            **default_paths(data_root),
        }

    @app.get("/api/config-profile")
    def config_profile(config_name: str = DEFAULT_CONFIG_NAME, hf_lerobot_home: str = "", validation_manifest: str = ""):
        try:
            selected = read_config_binding(project_root, config_name)
            if validation_manifest:
                from openpi.training.validation_split import configure
                selected = configure(dataclasses.replace(selected, validation_manifest=validation_manifest))
            lerobot_root = pathlib.Path(hf_lerobot_home or data_root / "lerobot").expanduser().resolve()
            return _config_profile(project_root=project_root, lerobot_root=lerobot_root, train_config=selected)
        except (OSError, SyntaxError, ValueError, KeyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/checkpoints")
    def checkpoints(checkpoint_root: str = ""):
        root = pathlib.Path(checkpoint_root).expanduser().resolve() if checkpoint_root else data_root / "checkpoints"
        return _checkpoints(root)

    @app.get("/api/next-experiment")
    def next_experiment(config_name: str = DEFAULT_CONFIG_NAME, checkpoint_root: str = ""):
        try:
            root = pathlib.Path(checkpoint_root).expanduser().resolve() if checkpoint_root else data_root / "checkpoints"
            return {"experiment_name": _next_experiment(root, config_name)}
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/system")
    def system(path: str = ""):
        target = pathlib.Path(path).expanduser() if path else data_root
        target = target.resolve()
        requested = str(target)
        while not target.exists() and target != target.parent:
            target = target.parent
        try:
            usage = shutil.disk_usage(target)
            disk = {"path": str(target.resolve()), "total": usage.total, "used": usage.used, "free": usage.free}
        except OSError as error:
            disk = {"path": str(target), "error": str(error)}
        return {"disk": {**disk, "requested_path": requested}, **gpu_probe.snapshot()}

    @app.get("/api/spikes/report")
    def spike_report(
        path: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=500),
        feature: str = "",
        dimension: int | None = Query(default=None, ge=0),
    ):
        try:
            return summarize_spike_report(
                pathlib.Path(path), offset=offset, limit=limit, feature=feature, dimension=dimension
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/spikes/trace")
    def spike_trace(
        report_path: str,
        file_index: int = Query(ge=0),
        repair_index: int = Query(ge=0),
        radius: int = Query(default=12, ge=2, le=100),
    ):
        try:
            return load_spike_trace(
                pathlib.Path(report_path),
                file_index=file_index,
                repair_index=repair_index,
                radius=radius,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/jobs")
    def jobs():
        return manager.list()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        try:
            return manager.get(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str):
        try:
            return {"text": manager.tail(job_id)}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/jobs/{job_id}/metrics")
    def job_metrics(job_id: str, limit: int = 2000):
        try:
            return manager.metrics(job_id, limit)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    def selection_path(job_id):
        try:
            root = manager.training_directory(job_id)
            if root is None:
                raise ValueError("该任务不是训练任务")
            return root / "monitoring/selection"
        except (KeyError, ValueError, OSError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/jobs/{job_id}/selection")
    def job_selection(job_id: str):
        path = selection_path(job_id) / "recommended.json"
        if not path.is_file():
            return {"status": "not_evaluated", "checkpoint_path": None, "warnings": []}
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/jobs/{job_id}/selection-plot")
    def job_selection_plot(job_id: str):
        path = selection_path(job_id) / "selection_latest.png"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="选模图尚未生成")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/jobs/selection-suite")
    def start_probe_suite(request: ProbeSuiteRequest):
        return start("selection_suite", lambda: build_probe_suite(project_root, request.model_dump()))

    @app.post("/api/jobs/audit")
    def start_audit(request: AuditRequest):
        return start("audit", lambda: build_audit(project_root, request.model_dump()))

    @app.post("/api/jobs/spikes/scan")
    def start_spike_scan(request: SpikeRequest):
        return start("spike_scan", lambda: build_spike_job(project_root, request.model_dump(), apply=False))

    @app.post("/api/jobs/spikes/clean")
    def start_spike_clean(request: SpikeRequest):
        return start("spike_clean", lambda: build_spike_job(project_root, request.model_dump(), apply=True))

    @app.post("/api/jobs/normalize")
    def start_normalize(request: NormalizeRequest):
        return start("normalize", lambda: build_normalize(request.model_dump()))

    @app.post("/api/jobs/validation-split")
    def start_validation_split(request: ValidationSplitRequest):
        return start("validation_split", lambda: build_validation_split(project_root, request.model_dump()))

    @app.post("/api/jobs/export/hdf5")
    def start_hdf5_export(request: Hdf5ExportRequest):
        return start("hdf5_export", lambda: build_hdf5_export(project_root, request.model_dump()))

    @app.post("/api/jobs/train")
    def start_train(request: TrainRequest):
        return start("train", lambda: build_train(request.model_dump()))

    @app.post("/api/jobs/serve")
    def start_serve(request: ServeRequest):
        return start("policy_serve", lambda: build_policy_serve(request.model_dump()))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try:
            return manager.cancel(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    app.mount("/static", StaticFiles(directory=web_root), name="static")
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8866)
    parser.add_argument("--data-root", type=pathlib.Path)
    parser.add_argument("--state-dir", type=pathlib.Path)
    parser.add_argument("--openpi-root", type=pathlib.Path)
    args = parser.parse_args()
    project_root = (args.openpi_root or pathlib.Path(__file__).resolve().parents[3]).expanduser().resolve()
    data_root = (
        (args.data_root or pathlib.Path(os.environ.get("BPX_DATA_ROOT", DEFAULT_DATA_ROOT))).expanduser().resolve()
    )
    state_dir = (args.state_dir or data_root / "dashboard_state").expanduser().resolve()
    paths = default_paths(data_root)
    # Only create infrastructure directories here. Dataset/output paths must
    # remain absent until data is transferred or the corresponding job starts.
    for key in (
        "data_root",
        "hf_lerobot_home",
        "checkpoint_root",
        "hf_home",
        "hf_datasets_cache",
        "uv_cache_dir",
        "tmpdir",
        "reports_root",
        "cleaned_root",
        "exports_root",
        "state_root",
    ):
        pathlib.Path(paths[key]).mkdir(parents=True, exist_ok=True)
    app = create_app(state_dir=state_dir, project_root=project_root, data_root=data_root)
    shown_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    logging.basicConfig(level=logging.INFO)
    logging.info("OpenPI BPX Studio: http://%s:%d/ (LAN: http://<服务器IP>:%d/)", shown_host, args.port, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
