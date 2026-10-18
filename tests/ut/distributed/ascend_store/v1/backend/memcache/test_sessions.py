"""Memcache session lease and region ownership."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    GVARegion,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.sessions import MemcacheSessions


def test_gva_resolves_only_after_lease_and_retains_lease_when_query_raises():
    backend = MagicMock()
    sessions = MemcacheSessions(backend)
    backend.reset_mock()
    backend.add_gva_leases.return_value = (0, -7)
    backend.query_gva_regions.side_effect = RuntimeError("query failed")

    with pytest.raises(RuntimeError, match="query failed"):
        sessions.start_load(["leased", "rejected"], [64, 64])
    assert backend.mock_calls == [call.add_gva_leases(["leased", "rejected"]), call.query_gva_regions(["leased"])]
    assert sessions.has_load_sessions
    sessions.finish_load()
    backend.remove_gva_leases.assert_called_once_with(["leased"])


def test_gva_wrong_size_cleanup_failure_preserves_only_acquired_keys():
    backend = MagicMock()
    backend.add_gva_leases.return_value = (0, 0, -7)
    backend.query_gva_regions.return_value = (GVARegion(1000, 64), GVARegion(2000, 32))
    backend.remove_gva_leases.side_effect = [RuntimeError("release failed"), None]
    sessions = MemcacheSessions(backend)

    with pytest.raises(RuntimeError, match="release failed"):
        sessions.start_load(["valid", "wrong_size", "rejected"], [64, 64, 64])
    assert sessions.load_regions(("valid",), 64) == (GVARegion(1000, 64),)
    with pytest.raises(RuntimeError, match="unavailable"):
        sessions.load_regions(("wrong_size",), 64)
    sessions.finish_load()
    assert backend.remove_gva_leases.call_args_list == [call(["wrong_size"]), call(["valid", "wrong_size"])]
