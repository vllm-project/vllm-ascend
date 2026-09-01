"""Opt-in raw records for G0; changes no scheduling or transfer behavior."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_LOCK = threading.Lock()
_BYTES_IN_FLIGHT = 0


def emit(event: str, **fields: Any) -> None:
    path = os.getenv("VLLM_G0_TELEMETRY_PATH")
    if not path:
        return
    record = {
        "event": event,
        "ts_ns": time.perf_counter_ns(),
        # msServiceProfiler's Chrome trace uses wall-clock microseconds. Keep
        # both clocks so transfer spans can be joined to modelExec spans
        # without changing or synchronizing the runtime fast path.
        "wall_ts_ns": time.time_ns(),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        **fields,
    }
    with _LOCK, open(path, "a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def adjust_bytes_in_flight(delta: int) -> int:
    global _BYTES_IN_FLIGHT
    with _LOCK:
        _BYTES_IN_FLIGHT += delta
        if _BYTES_IN_FLIGHT < 0:
            raise RuntimeError("G0 bytes-in-flight accounting became negative")
        return _BYTES_IN_FLIGHT
