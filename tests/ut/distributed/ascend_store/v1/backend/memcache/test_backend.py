"""Native GVA contracts without Worker batches, mappings, or request ownership."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from tests.ut.distributed.ascend_store.v1.backend.memcache.fixtures import _make_memcache_registration_backend
from tests.ut.distributed.ascend_store.v1.infrastructure.resources_fixtures import (
    run_backend_resource_lifecycle,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    BufferRegistrationError,
    GVARegion,
)
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


@pytest.mark.parametrize("result", ([-8], None, [], [True], [0, 0], RuntimeError("publication outcome unknown")))
def test_gva_failed_publish_keeps_cleanup_unconfirmed(native_backend, result):
    backend, store = native_backend
    store.batch_alloc.return_value = [1000]
    assert backend.allocate_gva(["key"], [64]) == (1000,)
    if isinstance(result, Exception):
        store.batch_write_finish.side_effect = result
    else:
        store.batch_write_finish.return_value = result
    if result == [-8]:
        assert backend.publish_gva(["key"]) == (-8,)
    else:
        with pytest.raises(RuntimeError):
            backend.publish_gva(["key"])
    for _ in range(2):
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


def test_named_memcache_worker_unregisters_regions_before_adapter_close() -> None:
    store = MagicMock()
    backend = _make_memcache_registration_backend(store, initialized=True)
    backend_spec = BackendSpec("memcache", MemcacheBackend, None, True)

    run_backend_resource_lifecycle(backend, backend_spec)

    store.register_buffer.assert_called_once()
    address, size = store.register_buffer.call_args.args
    store.unregister_buffer.assert_called_once_with(address, size)
    store.close.assert_called_once_with()
    assert backend._store is None


def test_v1_memcache_adapter_preserves_native_bulk_store_contract() -> None:
    store = MagicMock()
    store.batch_put_from_layers.return_value = [0]
    backend = _make_memcache_registration_backend(store, initialized=True)

    assert backend.store(["key"], [[100]], [[10]]) == (0,)

    store.batch_put_from_layers.assert_called_once_with(["key"], [[100]], [[10]], MmcDirect.COPY_L2G.value)


def test_v1_memcache_adapter_reads_dp_barrier_from_extra_config() -> None:
    parallel_config = SimpleNamespace(data_parallel_size=2)
    with patch.object(MemcacheBackend, "_setup_store", return_value=MagicMock()):
        backend = MemcacheBackend(
            parallel_config,
            device_id=0,
            init_bm=False,
            extra_config={"memcache_dp_init_barrier": False},
        )

    assert not backend._dp_init_barrier
    backend.close()

    with pytest.raises(ValueError, match="must be a boolean"):
        MemcacheBackend(parallel_config, device_id=0, init_bm=False, extra_config={"memcache_dp_init_barrier": "false"})


def test_memcache_eager_registration_failure_rolls_back_owned_prefix() -> None:
    store = MagicMock()
    store.register_buffer.side_effect = [0, -7]
    store.unregister_buffer.return_value = 0
    backend = _make_memcache_registration_backend(store, initialized=True)

    with pytest.raises(RuntimeError, match="registration failed"):
        backend.register_buffer([100, 200], [10, 20])

    assert store.register_buffer.call_args_list == [call(100, 10), call(200, 20)]
    store.unregister_buffer.assert_called_once_with(100, 10)
    assert backend._pending_buffers is None
    assert backend._pending_registration is None


def test_memcache_lazy_registration_can_be_cancelled_before_initialization() -> None:
    backend = _make_memcache_registration_backend(None, initialized=False)

    registration = backend.register_buffer([100], [10])
    registration.close()

    store = MagicMock()
    backend._store = store
    backend._initialized = True
    backend._register_buffers_if_ready()
    store.register_buffer.assert_not_called()
    assert backend._pending_buffers is None
    assert backend._pending_registration is None


def test_memcache_registration_rollback_failure_returns_retryable_owner() -> None:
    store = MagicMock()
    store.register_buffer.side_effect = [0, -7]
    store.unregister_buffer.side_effect = [-9, 0]
    backend = _make_memcache_registration_backend(store, initialized=True)

    with pytest.raises(BufferRegistrationError, match="rollback is incomplete") as error:
        backend.register_buffer([100, 200], [10, 20])

    assert tuple((region.address, region.size) for region in error.value.registration.regions) == ((100, 10),)
    error.value.registration.close()
    assert store.unregister_buffer.call_args_list == [call(100, 10), call(100, 10)]


def test_memcache_close_failure_keeps_client_for_retry() -> None:
    store = MagicMock()
    store.close.side_effect = [-7, 0]
    backend = _make_memcache_registration_backend(store, initialized=True)

    with pytest.raises(RuntimeError, match="close failed"):
        backend.close()

    assert backend._store is store
    assert not backend._closed
    backend.close()
    assert backend._store is None
    assert backend._closed


def test_memcache_post_init_failure_closes_constructed_client() -> None:
    store = MagicMock()
    store.init.return_value = 0
    store.close.return_value = 0
    backend = _make_memcache_registration_backend(None, initialized=False)
    backend.device_id = 0
    backend._init_bm = True
    backend._dp_init_barrier = True

    with (
        patch("memcache_hybrid.DistributedObjectStore", return_value=store, create=True),
        patch.object(MemcacheBackend, "set_device"),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.backend.get_dp_group",
            return_value=SimpleNamespace(cpu_group=object()),
        ),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.backend.torch.distributed.barrier",
            side_effect=RuntimeError("barrier failed"),
        ),
        pytest.raises(RuntimeError, match="barrier failed"),
    ):
        backend._setup_store()

    store.close.assert_called_once_with()
