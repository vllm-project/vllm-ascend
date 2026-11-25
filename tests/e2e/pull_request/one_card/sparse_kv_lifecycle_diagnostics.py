# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capture CANN logs and kernel wait state from the lifecycle test process."""

import os
import platform
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

PROC_SAMPLE_INTERVAL_SECONDS = 30
KILL_GRACE_SECONDS = 5
MAX_MATCHED_LOG_LINES = 32
MAX_TAIL_LOG_LINES = 16
NATIVE_LOG_MARKERS = (
    "getmeminfo",
    "memgetinfo",
    "setdevice",
    "aicpu",
    "hdc",
    "[error]",
    "[warning]",
    "timeout",
    "timed out",
    "ioctl",
)


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} [lifecycle native] {message}", flush=True)


def read_proc(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def sample_process(pid):
    proc = Path("/proc") / str(pid)
    main_thread = proc / "task" / str(pid)
    syscall = read_proc(main_thread / "syscall").split()
    number = syscall[0] if syscall else "unavailable"
    detail = ""
    if platform.machine() in {"aarch64", "arm64"} and number == "29" and len(syscall) > 1:
        try:
            device = os.readlink(proc / "fd" / str(int(syscall[1], 0)))
        except (OSError, ValueError):
            device = "unavailable"
        detail = f" ioctl_target={device if device.startswith('/dev/') else 'unavailable'}"
    log(f"pid={pid} wchan={read_proc(main_thread / 'wchan')} syscall={number}{detail}")
    for frame in read_proc(main_thread / "stack").splitlines()[:8]:
        log(f"pid={pid} kernel | {frame.partition(' ')[2] or frame}")


def show_native_logs(directory, pid):
    files = set(directory.rglob("*.log"))
    # CANN builds may leave device logs under the process's default log root.
    default = Path.home() / "ascend" / "log"
    for prefix in ("plog", "device"):
        files.update(default.glob(f"**/{prefix}-{pid}_*.log"))
    if not files:
        log(f"no CANN host or device log files for pid={pid}")
    for path in sorted(files):
        matches: deque[tuple[int, str]] = deque(maxlen=MAX_MATCHED_LOG_LINES)
        tail: deque[tuple[int, str]] = deque(maxlen=MAX_TAIL_LOG_LINES)
        try:
            with path.open(encoding="utf-8", errors="replace") as stream:
                for index, line in enumerate(stream):
                    entry = (index, line.rstrip()[:1600])
                    tail.append(entry)
                    lowered = line.lower()
                    if any(marker in lowered for marker in NATIVE_LOG_MARKERS):
                        matches.append(entry)
        except OSError as exc:
            log(f"could not read {path.name} ({type(exc).__name__})")
            continue
        log(f"pid={pid} CANN log {path.parent.name}/{path.name}")
        for _, line in sorted(set(matches) | set(tail)):
            log(f"{path.name} | {line}")


def run_with_cann_diagnostics(command, cwd, env, log_directory, timeout):
    log_directory.mkdir(parents=True, exist_ok=True)
    child_env = {
        **env,
        "ASCEND_PROCESS_LOG_PATH": str(log_directory),
        "ASCEND_GLOBAL_LOG_LEVEL": "0",
        "ASCEND_SLOG_PRINT_TO_STDOUT": "0",
    }
    proc = subprocess.Popen(command, cwd=cwd, env=child_env)
    log(f"lifecycle child pid={proc.pid}")
    deadline = time.monotonic() + timeout
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                returncode = proc.wait(timeout=min(remaining, PROC_SAMPLE_INTERVAL_SECONDS))
            except subprocess.TimeoutExpired:
                sample_process(proc.pid)
                continue
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            return
        raise subprocess.TimeoutExpired(command, timeout)
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                log(f"child pid={proc.pid} still present after SIGKILL")
        show_native_logs(log_directory, proc.pid)
