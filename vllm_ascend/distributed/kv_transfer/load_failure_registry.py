"""Process-local registry for requests whose local pool load failed.

The AscendStore pool worker records a request here when its KV load is
cancelled (invalid GVA / lease failure). The SfaRemoteD2H producer send
thread consults the registry before building READ_READY_BATCH items, so a
request that is already doomed is never transferred to D as a valid
success and its chunk-done marker is never sent.

Scope: the registry is per-process BY DESIGN. Recording (pool worker) and
consulting (send thread) run inside the same vLLM worker process; a
scheduler-process discard cannot see these entries and must not be wired
here. Entries are bounded by a TTL instead of a cross-process
notification: request ids are unique per engine process, so a stale entry
can never suppress a different request - unbounded retention would only
leak memory.
"""

from __future__ import annotations

import threading
import time

# How long a failed-load marker stays visible to the send-thread guard. The
# only consumers are the request's own queued send tasks, which drain within
# seconds of the failure; the TTL exists to bound memory.
FAILED_LOAD_TTL_SECONDS = 600.0

_lock = threading.Lock()
_failed_load_req_ids: dict[str, float] = {}


def _prune_locked(now: float) -> None:
    expired = [
        req_id
        for req_id, recorded_at in _failed_load_req_ids.items()
        if now - recorded_at > FAILED_LOAD_TTL_SECONDS
    ]
    for req_id in expired:
        del _failed_load_req_ids[req_id]


def record_failed_load(req_ids) -> None:
    """Record request ids whose local pool load failed and was cancelled."""
    now = time.monotonic()
    with _lock:
        _prune_locked(now)
        for req_id in req_ids:
            _failed_load_req_ids[req_id] = now


def is_failed_load(req_id: str) -> bool:
    now = time.monotonic()
    with _lock:
        _prune_locked(now)
        return req_id in _failed_load_req_ids


def discard_failed_load(req_ids) -> None:
    """Explicit removal for callers running in the recording process."""
    with _lock:
        for req_id in req_ids:
            _failed_load_req_ids.pop(req_id, None)
