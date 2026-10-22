"""Coalesce hardware probes; a wedged driver must never occupy HTTP workers."""

from __future__ import annotations

import threading
import time


class CachedProbe:
    def __init__(self, probe, *, interval=15.0, timeout=3.0):
        self.probe = probe
        self.interval = interval
        self.timeout = timeout
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.started = None
        self.updated = None
        self.active = False
        self.result = {"gpus": [], "gpu_error": "正在查询 GPU 状态"}

    def _run(self):
        try:
            result = self.probe()
        except Exception as error:
            result = {"gpus": [], "gpu_error": f"GPU 状态查询失败: {error}"}
        with self.lock:
            self.result = result
            self.updated = time.monotonic()
            self.active = False
            self.ready.set()

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            if not self.active and (self.updated is None or now-self.updated >= self.interval):
                self.active = True
                self.started = now
                self.ready.clear()
                threading.Thread(target=self._run, daemon=True, name="bpx-gpu-status").start()
        # Fast probes can populate the first response; slow probes never block the page.
        self.ready.wait(0.03)
        with self.lock:
            result = {**self.result, "gpu_pending": self.active}
            if self.active and time.monotonic()-self.started > self.timeout:
                result["gpu_error"] = "GPU 状态查询超时；使用缓存，不影响仪表盘和训练任务"
            result["gpu_cache_age_sec"] = None if self.updated is None else max(0, time.monotonic()-self.updated)
            return result
