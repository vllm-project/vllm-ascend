# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Opt-in, bounded startup evidence collection; no model/NPU imports."""

import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import psutil

logger = logging.getLogger(__name__)


def _read(path: Path) -> str:
    try:
        return path.read_text()[:32768]
    except OSError as exc:
        return f"unavailable: {exc}"


class StartupDiagnostics:
    def __init__(self, root: Path | None = None):
        self.root = root or Path("outputs/startup-diagnostics") / f"{time.time_ns()}-{os.getpid()}"
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.last_sample = -60.0
        self.last_npu_sample = -300.0
        self.sequence = 0
        self.stopped = False
        shutil.copyfile(Path(__file__).with_name("startup_diagnostics_worker.py"), self.root / "sitecustomize.py")
        packages = {}
        for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "triton-ascend", "fastapi", "starlette"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "not installed"
        self._write(
            "host.json",
            {"hostname": platform.node(), "platform": platform.platform(), "time": time.time(), "packages": packages},
        )
        logger.info("[startup-diagnostics] Evidence directory: %s", self.root)

    def _write(self, name: str, data: dict):
        (self.root / name).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def server_env(self, env: dict[str, str]) -> dict[str, str]:
        env = env.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(self.root), env.get("PYTHONPATH", "")]))
        return env

    def sample(self, pid: int, reason: str = "waiting", force: bool = False):
        """Best effort: failures are evidence gaps, never readiness failures."""
        if self.stopped:
            return
        try:
            self._sample(pid, reason, force)
        except Exception:
            logger.exception("[startup-diagnostics] Failed to collect sample")

    def _sample(self, pid: int, reason: str, force: bool):
        elapsed = time.monotonic() - self.started
        if not force and elapsed - self.last_sample < 60:
            return
        self.last_sample = elapsed
        self.sequence += 1
        process_records: list[dict[str, object]] = []
        data = {"time": time.time(), "elapsed_seconds": elapsed, "reason": reason, "processes": process_records}
        try:
            parent = psutil.Process(pid)
            processes = [parent, *parent.children(recursive=True)]
        except psutil.Error as exc:
            processes = []
            data["process_error"] = str(exc)
        for proc in processes:
            try:
                info = proc.as_dict(attrs=["pid", "ppid", "name", "cmdline", "status", "create_time", "num_threads"])
                info["cpu_times"] = proc.cpu_times()._asdict()
                info["memory"] = proc.memory_info()._asdict()
                try:
                    info["io"] = proc.io_counters()._asdict()
                    info["cpu_affinity"] = proc.cpu_affinity()
                except (psutil.Error, AttributeError) as exc:
                    info["resource_error"] = str(exc)
                proc_dir = Path(f"/proc/{proc.pid}")
                info["proc"] = {name: _read(proc_dir / name) for name in ("status", "wchan", "stack", "cgroup")}
                process_records.append(info)
            except psutil.Error as exc:
                process_records.append({"pid": proc.pid, "error": str(exc)})
        data["host"] = {
            name: _read(Path(name))
            for name in ("/proc/meminfo", "/proc/pressure/cpu", "/proc/pressure/memory", "/proc/pressure/io")
        }
        data["cgroup"] = {
            name: _read(Path("/sys/fs/cgroup") / name)
            for name in ("cpu.stat", "cpu.max", "memory.current", "memory.max", "memory.events")
        }
        if force or elapsed - self.last_npu_sample >= 300:
            self.last_npu_sample = elapsed
            try:
                result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10, check=False)
                data["npu_smi"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            except (OSError, subprocess.TimeoutExpired) as exc:
                data["npu_smi"] = {"error": str(exc)}
        self._write(f"sample-{self.sequence:04d}.json", data)
        logger.info("[startup-diagnostics] %s elapsed=%.1fs processes=%d", reason, elapsed, len(processes))

    def stop(self, pid: int, reason: str):
        if self.stopped:
            return
        self.sample(pid, reason, force=True)
        try:
            (self.root / "STOP").touch()
        except OSError:
            logger.exception("[startup-diagnostics] Failed to cancel worker trace timers")
        self.stopped = True
