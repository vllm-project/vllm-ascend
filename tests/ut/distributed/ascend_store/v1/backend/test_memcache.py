"""Memcache native results, lazy registration, initialization, and retryable close."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from tests.ut.distributed.ascend_store.v1.worker.resource_fixtures import (
    run_backend_resource_lifecycle,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    BufferRegistrationError,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache import (
    MemcacheBackend,
    MmcDirect,
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

    store.batch_put_from_layers.assert_called_once_with(
        ["key"],
        [[100]],
        [[10]],
        MmcDirect.COPY_L2G.value,
    )


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
        MemcacheBackend(
            parallel_config,
            device_id=0,
            init_bm=False,
            extra_config={"memcache_dp_init_barrier": "false"},
        )


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
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.get_dp_group",
            return_value=SimpleNamespace(cpu_group=object()),
        ),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache.torch.distributed.barrier",
            side_effect=RuntimeError("barrier failed"),
        ),
        pytest.raises(RuntimeError, match="barrier failed"),
    ):
        backend._setup_store()

    store.close.assert_called_once_with()
