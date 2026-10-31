"""Tests for backend.memcache.fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.backend import (
    MemcacheBackend,
)


def _make_memcache_registration_backend(
    store: MagicMock | None,
    *,
    initialized: bool,
) -> MemcacheBackend:
    backend = MemcacheBackend.__new__(MemcacheBackend)
    backend._store = store
    backend._initialized = initialized
    backend._pending_buffers = None
    backend._pending_registration = None
    backend._registration_error = None
    backend._closed = False
    return backend
