"""Memcache session lease and region ownership."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    GVARegion,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.sessions import MemcacheSessions


def test_gva_allocation_never_treats_ssd_hit_sentinel_as_a_writable_address():
    backend = MagicMock()
    backend.allocate_gva.return_value = (1000, (1 << 64) - 1)
    sessions = MemcacheSessions(backend)
    assert sessions.start_store(["new", "ssd_hit"], [64, 64]) == (0, -1)
    assert sessions.pending_store_keys == ("new",)
    with pytest.raises(RuntimeError, match="unavailable"):
        sessions.store_regions(("ssd_hit",), 64)
    backend.publish_gva.assert_not_called()
    backend.revoke_gva.assert_not_called()


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


@pytest.mark.parametrize("direction", ("load", "store"))
def test_gva_region_lookup_preserves_heterogeneous_size_order_and_ownership(direction):
    backend = MagicMock()
    regions = (GVARegion(1000, 64), GVARegion(2000, 128))
    backend.allocate_gva.return_value = (1000, 2000)
    backend.add_gva_leases.return_value = (0, 0)
    backend.query_gva_regions.return_value = regions
    sessions = MemcacheSessions(backend)
    start = sessions.start_load if direction == "load" else sessions.start_store
    resolve = sessions.load_regions if direction == "load" else sessions.store_regions
    assert start(["a", "b"], [64, 128]) == (0, 0)
    assert resolve(("b", "a"), (128, 64)) == regions[::-1]
    assert resolve(("a",), 64) == regions[:1]
    with pytest.raises(ValueError, match="sizes must match"):
        resolve(("a", "b"), (64,))
    with pytest.raises(RuntimeError, match="unexpected object size"):
        resolve(("b", "a"), (64, 128))
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve(("unowned",), (64,))
