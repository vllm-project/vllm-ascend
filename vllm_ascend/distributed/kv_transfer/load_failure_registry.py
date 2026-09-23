"""Process-local registry for requests whose local KV load failed."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from itertools import count

FAILED_LOAD_TTL_SECONDS = 600.0

_lock = threading.Lock()
_generation_counter = count(1)
_load_generations: dict[str, int] = {}
_failed_load_generations: dict[tuple[str, int], float] = {}


def _prune_locked(now: float) -> None:
    expired = [
        req_key
        for req_key, recorded_at in _failed_load_generations.items()
        if now - recorded_at > FAILED_LOAD_TTL_SECONDS
    ]
    for req_key in expired:
        del _failed_load_generations[req_key]


def begin_load_generation(req_id: str) -> int:
    with _lock:
        generation = next(_generation_counter)
        _load_generations[req_id] = generation
        return generation


def get_load_generation(req_id: str) -> int:
    with _lock:
        return _load_generations.get(req_id, 0)


def finish_load_generation(req_id: str, generation: int) -> None:
    with _lock:
        if _load_generations.get(req_id) == generation:
            del _load_generations[req_id]


def record_failed_load(req_generations: Iterable[tuple[str, int]]) -> None:
    """Keep failed load generations visible to later layerwise PD send tasks."""
    now = time.monotonic()
    with _lock:
        _prune_locked(now)
        for req_key in req_generations:
            _failed_load_generations[req_key] = now


def is_failed_load(req_id: str, generation: int) -> bool:
    now = time.monotonic()
    with _lock:
        _prune_locked(now)
        return (req_id, generation) in _failed_load_generations
