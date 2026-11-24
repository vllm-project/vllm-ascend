# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Time the two NPU startup calls in separate processes on the same CI device."""

import argparse
import faulthandler
import os
import platform
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

PROBE_TIMEOUT_SECONDS = 540
PROC_SAMPLE_INTERVAL_SECONDS = 30
TRACEBACK_INTERVAL_SECONDS = 60
KILL_GRACE_SECONDS = 5
MAX_LOG_LINES = 24
OPERATIONS = ("mem_get_info", "set_device")
NATIVE_LOG_MARKERS = (
    "getmeminfo",
    "memgetinfo",
    "setdevice",
    "[error]",
    "[warn]",
    "[warning]",
    "timeout",
    "timed out",
    "ioctl",
)


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} {message}", flush=True)


def run_child(operation):
    faulthandler.dump_traceback_later(TRACEBACK_INTERVAL_SECONDS, repeat=True)
    log(f"{operation} pid={os.getpid()} import started")
    start = time.monotonic()
    import torch
    import torch_npu

    log(
        f"{operation} import finished in {time.monotonic() - start:.3f}s "
        f"torch={torch.__version__} torch_npu={torch_npu.__version__} "
        f"npu_initialized={torch.npu.is_initialized()}"
    )
    log(f"{operation}(0) entered")
    start = time.monotonic()
    result = getattr(torch.npu, operation)(0)
    log(f"{operation}(0) returned in {time.monotonic() - start:.3f}s result={result!r}")


def read_proc(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def sample_process(operation, pid):
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
    log(f"{operation} pid={pid} wchan={read_proc(main_thread / 'wchan')} syscall={number}{detail}")


def show_native_logs(operation, directory, pid):
    # The fallback handles CANN builds that log under the default path.
    files = set(directory.rglob("*.log"))
    default = Path.home() / "ascend" / "log"
    for prefix in ("plog", "device"):
        files.update(default.glob(f"**/{prefix}-{pid}_*.log"))
    if not files:
        log(f"{operation} no per-process CANN log files available")
    for path in sorted(files):
        log(f"{operation} native log {path.parent.name}/{path.name}")
        try:
            with path.open(encoding="utf-8", errors="replace") as stream:
                matches = deque(
                    (line.rstrip() for line in stream if any(marker in line.lower() for marker in NATIVE_LOG_MARKERS)),
                    MAX_LOG_LINES,
                )
        except OSError as exc:
            log(f"{operation} could not read {path.name} ({type(exc).__name__})")
            continue
        for line in matches:
            log(f"{operation} native | {line}")


def run_probe(operation, log_root):
    directory = log_root / operation
    native_directory = directory / "cann"
    native_directory.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "ASCEND_PROCESS_LOG_PATH": str(native_directory),
        "ASCEND_GLOBAL_LOG_LEVEL": "0",
        "ASCEND_SLOG_PRINT_TO_STDOUT": "0",
    }
    start = time.monotonic()
    child_log = directory / "child.log"
    with child_log.open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(
            [sys.executable, "-u", __file__, str(log_root), "--child", operation],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    log(f"{operation} spawned pid={proc.pid}")
    deadline = start + PROBE_TIMEOUT_SECONDS
    while proc.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log(f"{operation} process timed out at {PROBE_TIMEOUT_SECONDS}s")
            proc.kill()
            try:
                proc.wait(timeout=KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                log(f"{operation} pid={proc.pid} still present after SIGKILL")
            break
        try:
            proc.wait(timeout=min(remaining, PROC_SAMPLE_INTERVAL_SECONDS))
        except subprocess.TimeoutExpired:
            sample_process(operation, proc.pid)
    log(f"{operation} process returncode={proc.returncode} elapsed={time.monotonic() - start:.3f}s")
    sys.stdout.write(child_log.read_text(encoding="utf-8", errors="replace"))
    show_native_logs(operation, native_directory, proc.pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--child", choices=OPERATIONS)
    args = parser.parse_args()
    if args.child:
        run_child(args.child)
    else:
        for operation in OPERATIONS:
            run_probe(operation, args.log_root.resolve())
