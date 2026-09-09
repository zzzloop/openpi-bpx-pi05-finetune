"""Persistent, restart-aware execution of validated server jobs."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

from openpi.bpx_studio.commands import CommandSpec
from openpi.bpx_studio.progress import parse_progress_line

_ACTIVE = {"queued", "running", "cancelling"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


class JobManager:
    """Run one expensive dataset or training job at a time and persist state."""

    def __init__(self, state_dir: pathlib.Path) -> None:
        self.root = state_dir / "server_jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._load()

    def _load(self) -> None:
        for path in sorted(self.root.glob("*/job.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            self._records[record["id"]] = record
            if record["status"] in _ACTIVE:
                if pathlib.Path(record["exit_path"]).is_file():
                    self._finish_from_exit(record)
                elif _process_alive(record.get("pid")):
                    record["status"] = "running"
                    self._save(record)
                    self._start_watcher(record["id"])
                else:
                    record.update(status="interrupted", ended_at=_now(), error="任务进程已不存在")
                    self._save(record)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            ordered = sorted(self._records.values(), key=lambda item: item["created_at"], reverse=True)
            return [self._public(record) for record in ordered]

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                return self._public(self._records[job_id])
            except KeyError as error:
                raise KeyError(f"未知任务: {job_id}") from error

    def start(self, kind: str, spec: CommandSpec) -> dict[str, Any]:
        with self._lock:
            if any(record["status"] in _ACTIVE for record in self._records.values()):
                raise RuntimeError("已有审计、清洗、归一化、导出或训练任务正在运行")
            job_id = uuid.uuid4().hex[:12]
            directory = self.root / job_id
            directory.mkdir()
            log_path = directory / "job.log"
            exit_path = directory / "exit.json"
            spec_path = directory / "spec.json"
            _atomic_json(
                spec_path,
                {
                    "argv": list(spec.argv),
                    "cwd": str(spec.cwd),
                    "env": dict(spec.env),
                    "log_path": str(log_path),
                    "exit_path": str(exit_path),
                },
            )
            record: dict[str, Any] = {
                "id": job_id,
                "kind": kind,
                "status": "queued",
                "command": spec.display,
                "environment": dict(spec.env),
                "cwd": str(spec.cwd),
                "created_at": _now(),
                "started_at": None,
                "ended_at": None,
                "pid": None,
                "return_code": None,
                "error": "",
                "progress": {"current": 0, "total": spec.total_units, "percent": 0.0},
                "metrics": {},
                "log_path": str(log_path),
                "exit_path": str(exit_path),
                "spec_path": str(spec_path),
                "log_offset": 0,
                "total_units": spec.total_units,
            }
            self._records[job_id] = record
            self._save(record)
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            runner_path = pathlib.Path(__file__).with_name("job_runner.py")
            with (directory / "runner.log").open("ab") as output:
                process = subprocess.Popen(
                    [sys.executable, str(runner_path), "--spec", str(spec_path)],
                    cwd=spec.cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                    creationflags=creation_flags,
                )
            self._processes[job_id] = process
            record.update(status="running", started_at=_now(), pid=process.pid)
            self._save(record)
            self._start_watcher(job_id)
            return self._public(record)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise KeyError(f"未知任务: {job_id}")
            if record["status"] not in _ACTIVE:
                raise RuntimeError("该任务已经结束")
            record["status"] = "cancelling"
            self._save(record)
            pid = record.get("pid")
            if pid and _process_alive(pid):
                if os.name == "posix":
                    os.killpg(pid, signal.SIGTERM)
                else:
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
            record.update(status="cancelled", ended_at=_now(), error="由操作者取消")
            self._processes.pop(job_id, None)
            self._save(record)
            return self._public(record)

    def tail(self, job_id: str, max_bytes: int = 65536) -> str:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise KeyError(f"未知任务: {job_id}")
            path = pathlib.Path(record["log_path"])
        if not path.is_file():
            return ""
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")

    def metrics(self, job_id: str, limit: int = 2000) -> list[dict[str, Any]]:
        if not 1 <= limit <= 10000:
            raise ValueError("指标点数量必须在 1 到 10000 之间")
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise KeyError(f"未知任务: {job_id}")
            path = pathlib.Path(record["log_path"]).with_name("metrics.jsonl")
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    def _start_watcher(self, job_id: str) -> None:
        threading.Thread(target=self._watch, args=(job_id,), daemon=True, name=f"bpx-job-{job_id}").start()

    def _watch(self, job_id: str) -> None:
        while True:
            with self._lock:
                record = self._records.get(job_id)
                if record is None or record["status"] not in _ACTIVE:
                    return
                self._consume_log(record)
                if pathlib.Path(record["exit_path"]).is_file():
                    self._finish_from_exit(record)
                    return
                process = self._processes.get(job_id)
                alive = process.poll() is None if process is not None else _process_alive(record.get("pid"))
                if not alive:
                    runner_log = self.root / job_id / "runner.log"
                    detail = runner_log.read_text(encoding="utf-8", errors="replace") if runner_log.is_file() else ""
                    error = "任务进程异常退出且没有退出记录"
                    if detail.strip():
                        error = f"{error}: {detail.strip()[-2000:]}"
                    record.update(status="failed", ended_at=_now(), error=error)
                    self._processes.pop(job_id, None)
                    self._save(record)
                    return
            time.sleep(0.5)

    def _consume_log(self, record: dict[str, Any]) -> None:
        path = pathlib.Path(record["log_path"])
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(record.get("log_offset", 0))
            content = handle.read()
            record["log_offset"] = handle.tell()
        for line in content.replace("\r", "\n").splitlines():
            parsed = parse_progress_line(line, total_units=record.get("total_units"))
            if parsed is None:
                continue
            record["progress"] = {key: parsed[key] for key in ("current", "total", "percent") if key in parsed}
            if parsed.get("metrics"):
                record["metrics"] = parsed["metrics"]
                metrics_path = path.with_name("metrics.jsonl")
                with metrics_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(parsed, ensure_ascii=False) + "\n")
        self._save(record)

    def _finish_from_exit(self, record: dict[str, Any]) -> None:
        self._consume_log(record)
        payload = json.loads(pathlib.Path(record["exit_path"]).read_text(encoding="utf-8"))
        return_code = int(payload["return_code"])
        record.update(
            status="succeeded" if return_code == 0 else "failed",
            return_code=return_code,
            error=payload.get("error", ""),
            ended_at=payload.get("ended_at", _now()),
        )
        if return_code == 0 and record.get("progress", {}).get("total"):
            record["progress"].update(current=record["progress"]["total"], percent=100.0)
        self._processes.pop(record["id"], None)
        self._save(record)

    def _save(self, record: dict[str, Any]) -> None:
        _atomic_json(self.root / record["id"] / "job.json", record)

    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        hidden = {"log_offset", "spec_path", "exit_path", "total_units"}
        return {key: value for key, value in record.items() if key not in hidden}
