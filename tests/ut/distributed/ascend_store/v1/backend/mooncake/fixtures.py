"""Tests for backend.mooncake.fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake import (
    MooncakeBackend,
)


def _make_mooncake_registration_backend(store: MagicMock) -> MooncakeBackend:
    backend = MooncakeBackend.__new__(MooncakeBackend)
    backend._store = store
    backend._use_store_independent_te = True
    backend._use_fabric_mem = False
    backend._initialized = True
    backend._lazy_init = False
    backend._closed = False
    return backend
