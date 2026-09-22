"""Process-local registry for requests whose local KV load failed."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable

FAILED_LOAD_TTL_SECONDS = 600.0

_lock = threading.Lock()
_failed_load_req_ids: dict[str, float] = {}


def _prune_locked(now: float) -> None:
    expired = [
        req_id for req_id, recorded_at in _failed_load_req_ids.items() if now - recorded_at > FAILED_LOAD_TTL_SECONDS
    ]
    for req_id in expired:
        del _failed_load_req_ids[req_id]


def record_failed_load(req_ids: Iterable[str]) -> None:
    """Keep failed requests visible to later layerwise PD send tasks."""
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
