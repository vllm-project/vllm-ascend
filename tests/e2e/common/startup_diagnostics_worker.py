# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Copied as sitecustomize.py ONLY for opt-in startup diagnostic servers.

Each Python child (including spawn workers) writes its own traceback file.
No signals or ptrace permissions are needed. A stop file cancels the timers
when the server becomes ready, before benchmarks run.
"""

import faulthandler
import importlib.machinery
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path


def _enable():
    root = Path(__file__).resolve().parent
    if (root / "STOP").exists():
        return
    # Keep this handle alive until interpreter shutdown.
    trace_file = (root / f"python-{os.getpid()}.log").open("a", buffering=1)
    trace_file.write(
        json.dumps({"pid": os.getpid(), "ppid": os.getppid(), "argv": sys.argv, "time": time.time()}) + "\n"
    )
    faulthandler.dump_traceback_later(300, repeat=True, file=trace_file)

    def stop_when_ready():
        while not (root / "STOP").exists():
            time.sleep(1)
        faulthandler.cancel_dump_traceback_later()
        trace_file.write(f"Startup diagnostics stopped at {time.time()}\n")

    threading.Thread(target=stop_when_ready, name="startup-trace-stop", daemon=True).start()


try:
    # Preserve any image-provided Python startup customization.
    paths = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", paths)
    if spec is not None and spec.loader is not None:
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    _enable()
except Exception as exc:
    # Diagnostic setup must not prevent model startup.
    print(f"[startup-diagnostics] Python trace setup failed: {exc}", file=sys.stderr, flush=True)
