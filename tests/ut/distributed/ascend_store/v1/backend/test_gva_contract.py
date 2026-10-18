"""Native GVA contracts without Worker batches, mappings, or request ownership."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import GVARegion
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache import backend as memcache_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.backend import (
    GVA_REVOKE_UNCONFIRMED,
    MemcacheBackend,
    MmcDirect,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.results import BackendResultError


@pytest.fixture
def native_backend():
    store = MagicMock()
    store.close.return_value = 0
    with patch.object(MemcacheBackend, "_setup_store", return_value=store):
        backend = MemcacheBackend(SimpleNamespace(data_parallel_size=1), device_id=0, init_bm=False)
    yield backend, store
    backend.close()


def test_memcache_creates_sessions_with_native_lease_and_region_lifetime(native_backend):
    backend, store = native_backend
    store.batch_add_lease.return_value = [0]
    store.batch_remove_lease.return_value = 0
    info = MagicMock()
    info.size.return_value = 64
    info.gva_list.return_value = [1000]
    store.batch_get_key_info.return_value = [info]

    sessions = backend.create_sessions()
    assert sessions.start_load(["key"], [64]) == (0,)
    assert sessions.load_regions(("key",), 64) == (GVARegion(1000, 64),)
    assert store.mock_calls[:2] == [call.batch_add_lease(["key"]), call.batch_get_key_info(["key"], 1)]
    sessions.close()
    store.batch_remove_lease.assert_called_once_with(["key"])
    with pytest.raises(RuntimeError, match="unavailable"):
        sessions.load_regions(("key",), 64)


def test_memcache_session_factory_rejects_missing_native_support(native_backend):
    backend, store = native_backend
    store.batch_add_lease = None
    with pytest.raises(RuntimeError, match="requires native batch_add_lease"):
        backend.create_sessions()
    store.batch_alloc.assert_not_called()


@pytest.mark.parametrize("keys", ([], ["owned"], ["first", "second"], ["duplicate", "duplicate"]))
def test_gva_revoke_is_aligned_repeatable_and_never_calls_unsafe_native_cleanup(native_backend, keys):
    backend, store = native_backend

    for _ in range(2):
        assert backend.revoke_gva(keys) == (GVA_REVOKE_UNCONFIRMED,) * len(keys)

    # No failed-write notification, lease removal, or lookup-based ownership guess.
    assert store.mock_calls == []


@pytest.mark.parametrize("replacement", (False, True))
def test_gva_revoke_preserves_duplicate_or_replacement_writers_object(native_backend, replacement):
    backend, store = native_backend
    objects = {"shared": (1000, 64, True)}
    store.batch_alloc.side_effect = lambda keys, sizes: [objects[key][0] for key in keys]

    def finish(keys, codes):
        for key, code in zip(keys, codes, strict=True):
            if code != 0:
                del objects[key]
        return [0] * len(keys)

    store.batch_write_finish.side_effect = finish
    assert backend.allocate_gva(["shared"], [64]) == (1000,)
    if replacement:
        objects["shared"] = (2000, 64, True)
    original_object = objects["shared"]

    assert backend.revoke_gva(["shared"]) == (GVA_REVOKE_UNCONFIRMED,)

    assert objects["shared"] == original_object
    assert store.mock_calls == [call.batch_alloc(["shared"], [64])]


def test_gva_revoke_does_not_initialize_a_lazy_client():
    with (
        patch.object(memcache_adapter, "_is_device_sdma", return_value=True),
        patch.object(MemcacheBackend, "_setup_store") as initialize,
    ):
        backend = MemcacheBackend(SimpleNamespace(data_parallel_size=1), device_id=0, init_bm=False, lazy_init=True)
        assert backend.revoke_gva(["uncertain"]) == (GVA_REVOKE_UNCONFIRMED,)
        initialize.assert_not_called()
    assert backend._store is None
    backend.close()


@pytest.mark.parametrize("native_result", ([-8], None, [], [True], [0, 0]))
def test_gva_failed_or_malformed_publish_cannot_turn_revoke_into_success(native_backend, native_result):
    backend, store = native_backend
    store.batch_alloc.return_value = [1000]
    store.batch_write_finish.return_value = native_result
    assert backend.allocate_gva(["key"], [64]) == (1000,)

    if native_result == [-8]:
        assert backend.publish_gva(["key"]) == (-8,)
    else:
        with pytest.raises(BackendResultError):
            backend.publish_gva(["key"])
    for _ in range(2):
        assert backend.revoke_gva(["key"]) == (GVA_REVOKE_UNCONFIRMED,)
    store.batch_write_finish.assert_called_once_with(["key"], [0])


def test_gva_publish_exception_keeps_cleanup_unconfirmed(native_backend):
    backend, store = native_backend
    store.batch_write_finish.side_effect = RuntimeError("publication outcome unknown")

    with pytest.raises(RuntimeError, match="outcome unknown"):
        backend.publish_gva(["key"])

    assert backend.revoke_gva(["key"]) == (GVA_REVOKE_UNCONFIRMED,)
    store.batch_write_finish.assert_called_once_with(["key"], [0])


def test_gva_allocation_results_are_addresses_not_success_codes(native_backend):
    backend, store = native_backend
    store.batch_alloc.return_value = [1000, 0, -7]

    assert backend.allocate_gva(["admitted", "rejected", "failed"], [64, 64, 64]) == (1000, 0, -7)
    store.batch_write_finish.assert_not_called()


def test_gva_publish_is_explicit_and_preserves_partial_failure(native_backend):
    backend, store = native_backend
    store.batch_copy.return_value = 0
    store.batch_write_finish.return_value = [0, -8]

    assert backend.copy_to_gva([1000, 2000], [100, 200], [64, 64]) == 0
    store.batch_write_finish.assert_not_called()
    assert backend.publish_gva(["complete", "failed"]) == (0, -8)
    store.batch_write_finish.assert_called_once_with(["complete", "failed"], [0, 0])


@pytest.mark.parametrize("direction", ("from", "to"))
@pytest.mark.parametrize("native_result", (0, -7, None, True, [0], RuntimeError("copy outcome unknown")))
def test_gva_copy_preserves_completed_failures_and_rejects_unknown_results(native_backend, direction, native_result):
    backend, store = native_backend
    copy = backend.copy_from_gva if direction == "from" else backend.copy_to_gva
    native_direction = MmcDirect.COPY_G2L if direction == "from" else MmcDirect.COPY_L2G
    if isinstance(native_result, Exception):
        store.batch_copy.side_effect = native_result
        with pytest.raises(RuntimeError, match="copy outcome unknown"):
            copy([1000], [100], [64])
    else:
        store.batch_copy.return_value = native_result
        if type(native_result) is int:
            assert copy([1000], [100], [64]) == native_result
        else:
            with pytest.raises(BackendResultError):
                copy([1000], [100], [64])
    store.batch_copy.assert_called_once_with([1000], [100], [64], native_direction.value)
    store.batch_write_finish.assert_not_called()


def test_gva_query_is_a_snapshot_and_does_not_acquire_a_lease(native_backend):
    backend, store = native_backend
    info = MagicMock()
    info.size.return_value = 64
    info.gva_list.return_value = [1000]
    store.batch_get_key_info.return_value = [info]

    assert backend.query_gva_regions(["key"]) == (GVARegion(1000, 64),)
    store.batch_add_lease.assert_not_called()
    store.batch_remove_lease.assert_not_called()


def test_gva_read_lease_release_failure_is_retryable_and_not_write_revoke(native_backend):
    backend, store = native_backend
    store.batch_add_lease.return_value = [0, -7]
    store.batch_remove_lease.side_effect = [-8, 0]
    assert backend.add_gva_leases(["leased", "rejected"]) == (0, -7)

    with pytest.raises(RuntimeError, match="failed with result -8"):
        backend.remove_gva_leases(["leased"])
    assert backend.revoke_gva(["write"]) == (GVA_REVOKE_UNCONFIRMED,)
    backend.remove_gva_leases(["leased"])

    assert store.batch_remove_lease.call_args_list == [call(["leased"]), call(["leased"])]
    store.batch_write_finish.assert_not_called()


@pytest.mark.parametrize(
    "operation,args",
    (
        ("set_device", ()),
        ("create_sessions", ()),
        ("exists", (["key"],)),
        ("load", (["key"], [[100]], [[64]])),
        ("store", (["key"], [[100]], [[64]])),
        ("register_buffer", ([100], [64])),
        ("validate_gva_support", ()),
        ("query_gva_regions", ([],)),
        ("add_gva_leases", (["key"],)),
        ("remove_gva_leases", (["key"],)),
        ("allocate_gva", (["key"], [64])),
        ("copy_from_gva", ([1000], [100], [64])),
        ("copy_to_gva", ([1000], [100], [64])),
        ("publish_gva", (["key"],)),
        ("revoke_gva", (["key"],)),
        ("revoke_gva", ([],)),
    ),
)
def test_closed_memcache_rejects_operations_without_reinitializing(native_backend, operation, args):
    backend, store = native_backend
    backend._lazy_init = True
    backend.close()
    store.reset_mock()

    with pytest.raises(RuntimeError, match="Backend is closed"):
        getattr(backend, operation)(*args)

    assert store.mock_calls == []
