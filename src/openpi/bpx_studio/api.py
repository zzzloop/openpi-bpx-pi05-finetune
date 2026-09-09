"""Authenticated local dashboard for BPX data engineering and OpenPI training."""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import secrets
import shutil
import subprocess
from typing import Literal

from fastapi import Depends
from fastapi import FastAPI
from fastapi import Header
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
from openpi.bpx_studio.commands import build_spike_job
from openpi.bpx_studio.commands import build_train
from openpi.bpx_studio.jobs import JobManager

DEFAULT_DATA_ROOT = pathlib.Path("/scratch/bpx_datasets")


class DataEnvironment(BaseModel):
    hf_lerobot_home: str = ""
    hf_home: str = ""
    hf_datasets_cache: str = ""
    tmpdir: str = ""
    uv_cache_dir: str = ""


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
    mode: Literal["overwrite", "resume"] = "overwrite"


def default_paths(data_root: pathlib.Path) -> dict[str, str]:
    root = data_root.expanduser().resolve()
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
    }


def _load_or_create_token(state_dir: pathlib.Path) -> str:
    path = state_dir / ".server-token"
    if path.is_file() and (token := path.read_text(encoding="utf-8").strip()):
        return token
    state_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    path.write_text(token + "\n", encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)
    return token


def _gpu_snapshot() -> list[dict[str, str]]:
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
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    fields = ("index", "name", "memory_used_mib", "memory_total_mib", "utilization_percent", "temperature_c")
    return [
        dict(zip(fields, (part.strip() for part in line.split(",")), strict=True))
        for line in result.stdout.splitlines()
    ]


def create_app(*, state_dir: pathlib.Path, project_root: pathlib.Path, data_root: pathlib.Path) -> FastAPI:
    web_root = pathlib.Path(__file__).with_name("web")
    token = _load_or_create_token(state_dir)
    manager = JobManager(state_dir)
    app = FastAPI(title="OpenPI BPX Research Studio")

    def authenticate(
        x_bpx_token: str | None = Header(default=None),
        query_token: str | None = Query(default=None, alias="token"),
    ) -> None:
        if not secrets.compare_digest(x_bpx_token or query_token or "", token):
            raise HTTPException(status_code=401, detail="控制令牌无效")

    def start(kind: str, builder):
        try:
            return manager.start(kind, builder())
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/")
    async def index(_: None = Depends(authenticate)):
        return FileResponse(web_root / "index.html")

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    @app.get("/api/config", dependencies=[Depends(authenticate)])
    async def config():
        return {"default_openpi_root": str(project_root), "default_workers": 32, **default_paths(data_root)}

    @app.get("/api/system", dependencies=[Depends(authenticate)])
    async def system(path: str = ""):
        target = pathlib.Path(path).expanduser() if path else data_root
        try:
            usage = shutil.disk_usage(target.resolve())
            disk = {"path": str(target.resolve()), "total": usage.total, "used": usage.used, "free": usage.free}
        except OSError as error:
            disk = {"path": str(target), "error": str(error)}
        return {"disk": disk, "gpus": _gpu_snapshot()}

    @app.get("/api/jobs", dependencies=[Depends(authenticate)])
    async def jobs():
        return manager.list()

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(authenticate)])
    async def job(job_id: str):
        try:
            return manager.get(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/jobs/{job_id}/log", dependencies=[Depends(authenticate)])
    async def job_log(job_id: str):
        try:
            return {"text": manager.tail(job_id)}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/jobs/{job_id}/metrics", dependencies=[Depends(authenticate)])
    async def job_metrics(job_id: str, limit: int = 2000):
        try:
            return manager.metrics(job_id, limit)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/jobs/audit", dependencies=[Depends(authenticate)])
    async def start_audit(request: AuditRequest):
        return start("audit", lambda: build_audit(project_root, request.model_dump()))

    @app.post("/api/jobs/spikes/scan", dependencies=[Depends(authenticate)])
    async def start_spike_scan(request: SpikeRequest):
        return start("spike_scan", lambda: build_spike_job(project_root, request.model_dump(), apply=False))

    @app.post("/api/jobs/spikes/clean", dependencies=[Depends(authenticate)])
    async def start_spike_clean(request: SpikeRequest):
        return start("spike_clean", lambda: build_spike_job(project_root, request.model_dump(), apply=True))

    @app.post("/api/jobs/normalize", dependencies=[Depends(authenticate)])
    async def start_normalize(request: NormalizeRequest):
        return start("normalize", lambda: build_normalize(request.model_dump()))

    @app.post("/api/jobs/export/hdf5", dependencies=[Depends(authenticate)])
    async def start_hdf5_export(request: Hdf5ExportRequest):
        return start("hdf5_export", lambda: build_hdf5_export(project_root, request.model_dump()))

    @app.post("/api/jobs/train", dependencies=[Depends(authenticate)])
    async def start_train(request: TrainRequest):
        return start("train", lambda: build_train(request.model_dump()))

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(authenticate)])
    async def cancel(job_id: str):
        try:
            return manager.cancel(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    app.mount("/static", StaticFiles(directory=web_root), name="static")
    app.state.control_token = token
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
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
    for path in default_paths(data_root).values():
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    app = create_app(state_dir=state_dir, project_root=project_root, data_root=data_root)
    shown_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    logging.basicConfig(level=logging.INFO)
    logging.info("OpenPI BPX Studio: http://%s:%d/?token=%s", shown_host, args.port, app.state.control_token)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
