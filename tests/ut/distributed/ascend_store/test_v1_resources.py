"""Registration ownership and teardown order for AscendStore v1 resources."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import vllm_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    BufferRegistration,
    BufferRegistrationError,
    LayerwiseAccessKind,
    rollback_buffer_registration,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    mooncake as mooncake_adapter,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.memcache import (
    MemcacheBackend,
    MmcDirect,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake import (
    MooncakeBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import (
    TokenRange,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    compile_bulk_projection_binder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    KVTransferStep,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import (
    SynchronousBulkWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.resources import (
    KVPoolResources,
)

from .v1.helpers import FakeBackend, FakeEvent, make_backend_spec, make_topology


class RecordingBackend(FakeBackend):
    def __init__(
        self,
        *,
        fail_registration_index: int | None = None,
        fail_unregister_count: int = 0,
        fail_close_count: int = 0,
    ) -> None:
        super().__init__()
        self.fail_registration_index = fail_registration_index
        self.fail_unregister_count = fail_unregister_count
        self.fail_close_count = fail_close_count

    def register_buffer(self, addresses, sizes) -> BufferRegistration:
        registration = BufferRegistration(self._unregister_buffer)
        try:
            for index, (address, size) in enumerate(zip(addresses, sizes, strict=True)):
                self.calls.append(("register_region", address, size))
                if index == self.fail_registration_index:
                    raise RuntimeError(f"registration failed at region {index}")
                registration.acquire(address, size)
        except BaseException as error:
            rollback_buffer_registration(registration, error)
            raise
        return registration

    def _unregister_buffer(self, address: int, size: int) -> None:
        self.calls.append(("unregister_region", address, size))
        if self.fail_unregister_count:
            self.fail_unregister_count -= 1
            raise RuntimeError("unregister failed")

    def close(self) -> None:
        self.calls.append(("backend_close",))
        if self.fail_close_count:
            self.fail_close_count -= 1
            raise RuntimeError("backend close failed")
        self.closed = True


def _make_caches(topology, *, shared_storage: bool = False):
    layer_names = topology.groups[0].layer_names
    if not shared_storage:
        return {layer_name: torch.empty((8, 4), dtype=torch.float32) for layer_name in layer_names}
    storage = torch.empty((8 * len(layer_names), 4), dtype=torch.float32)
    return {layer_name: storage[index * 8 : (index + 1) * 8] for index, layer_name in enumerate(layer_names)}


def _make_resources(backend: RecordingBackend, *, physical_layers=(0, 1)):
    topology = make_topology(physical_layers=physical_layers)
    resources = KVPoolResources(
        backend,
        make_backend_spec(layerwise_access=None),
        8,
        topology.groups,
    )
    return topology, resources


def test_resources_own_exact_shared_storage_region_and_close_in_order() -> None:
    backend = RecordingBackend()
    topology, resources = _make_resources(backend)
    caches = _make_caches(topology, shared_storage=True)
    storage = next(iter(caches.values())).untyped_storage()

    resources.bind_kv_caches(caches)

    assert backend.calls == [("register_region", storage.data_ptr(), storage.nbytes())]
    resources.close()
    resources.close()

    assert backend.calls == [
        ("register_region", storage.data_ptr(), storage.nbytes()),
        ("unregister_region", storage.data_ptr(), storage.nbytes()),
        ("backend_close",),
    ]
    assert resources.kv_caches is None
    assert backend.closed
    with pytest.raises(RuntimeError, match="closed"):
        resources.bind_kv_caches(caches)


def test_registration_failure_rolls_back_successful_prefix_before_backend_close() -> None:
    backend = RecordingBackend(fail_registration_index=1)
    topology, resources = _make_resources(backend)
    caches = _make_caches(topology)

    with pytest.raises(RuntimeError, match="registration failed at region 1"):
        resources.bind_kv_caches(caches)

    assert resources.kv_caches is caches
    assert [call[0] for call in backend.calls] == [
        "register_region",
        "register_region",
        "unregister_region",
    ]
    resources.close()
    assert [call[0] for call in backend.calls][-1] == "backend_close"
    assert resources.kv_caches is None


def test_incomplete_registration_rollback_retains_owner_and_retries_release() -> None:
    backend = RecordingBackend(
        fail_registration_index=1,
        fail_unregister_count=1,
    )
    topology, resources = _make_resources(backend)
    caches = _make_caches(topology)

    with pytest.raises(RuntimeError, match="rollback is incomplete"):
        resources.bind_kv_caches(caches)

    assert resources.kv_caches is caches
    assert "backend_close" not in [call[0] for call in backend.calls]
    resources.close()
    assert [call[0] for call in backend.calls][-2:] == ["unregister_region", "backend_close"]
    assert resources.kv_caches is None


def test_backend_close_failure_retains_cache_owner_until_retry() -> None:
    backend = RecordingBackend(fail_close_count=1)
    topology, resources = _make_resources(backend, physical_layers=(0,))
    caches = _make_caches(topology)
    resources.bind_kv_caches(caches)

    with pytest.raises(RuntimeError, match="backend close failed"):
        resources.close()

    assert resources.kv_caches is caches
    assert [call[0] for call in backend.calls].count("unregister_region") == 1
    resources.close()
    assert [call[0] for call in backend.calls].count("unregister_region") == 1
    assert resources.kv_caches is None


def test_worker_bind_failure_releases_registration_before_backend_close() -> None:
    backend = RecordingBackend()
    topology, resources = _make_resources(backend, physical_layers=(0,))
    caches = _make_caches(topology)
    projection_binder = compile_bulk_projection_binder(topology, 64)

    with patch.object(type(projection_binder), "bind", side_effect=RuntimeError("projection binding failed")):
        worker = SynchronousBulkWorker(topology, projection_binder, resources)
        with pytest.raises(RuntimeError, match="projection binding failed"):
            worker.bind_kv_caches(caches)

    lifecycle = [call[0] for call in backend.calls]
    assert lifecycle == ["register_region", "unregister_region", "backend_close"]
    assert resources.kv_caches is None


def test_worker_factory_closes_backend_when_layout_resolution_fails(monkeypatch) -> None:
    topology = make_topology(physical_layers=(0,))

    class FactoryBackend(RecordingBackend):
        instance = None

        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()
            type(self).instance = self

    backend_spec = BackendSpec(
        "memcache",
        FactoryBackend,
        LayerwiseAccessKind.GVA,
        True,
    )
    route_spec = SimpleNamespace(topology=topology, backend_name="memcache", use_layerwise=True)
    monkeypatch.setattr(vllm_adapter, "resolve_kv_pool_route_spec", lambda *_args: route_spec)
    monkeypatch.setattr(vllm_adapter, "_compile_kv_pool_projection_binder", lambda *_args: object())
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: backend_spec)

    def fail_layout(_parallel_config):
        raise RuntimeError("layout resolution failed")

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={}),
        model_config=SimpleNamespace(get_layers_start_end_indices=fail_layout),
    )

    with pytest.raises(RuntimeError, match="layout resolution failed"):
        vllm_adapter.create_kv_pool_worker(config, SimpleNamespace())

    assert FactoryBackend.instance is not None
    assert FactoryBackend.instance.calls == [("backend_close",)]


def _run_named_backend_resource_lifecycle(backend, backend_spec) -> None:
    topology = make_topology(physical_layers=(0,))
    resources = KVPoolResources(backend, backend_spec, 8, topology.groups)
    projection_binder = compile_bulk_projection_binder(topology, 64)
    worker = SynchronousBulkWorker(
        topology,
        projection_binder,
        resources,
        store_enabled=False,
    )
    worker.bind_kv_caches(_make_caches(topology))
    worker.close()


def _make_mooncake_registration_backend(store: MagicMock) -> MooncakeBackend:
    backend = MooncakeBackend.__new__(MooncakeBackend)
    backend._store = store
    backend._use_store_independent_te = True
    backend._use_fabric_mem = False
    backend._initialized = True
    backend._lazy_init = False
    backend._closed = False
    return backend


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


def test_named_mooncake_worker_unregisters_store_regions_before_client_close() -> None:
    store = MagicMock()
    backend = _make_mooncake_registration_backend(store)
    store.register_buffer.return_value = 0
    store.unregister_buffer.return_value = 0
    store.close.return_value = 0
    lifecycle = MagicMock()
    lifecycle.attach_mock(store.register_buffer, "register")
    lifecycle.attach_mock(store.unregister_buffer, "unregister")
    lifecycle.attach_mock(store.close, "close")
    backend_spec = BackendSpec("mooncake", MooncakeBackend, None, True)

    _run_named_backend_resource_lifecycle(backend, backend_spec)

    assert [entry[0] for entry in lifecycle.mock_calls] == ["register", "unregister", "close"]


def test_v1_mooncake_adapter_preserves_native_bulk_store_contract() -> None:
    store = MagicMock()
    store.batch_put_from_multi_buffers.return_value = [0]
    backend = _make_mooncake_registration_backend(store)
    backend.config = SimpleNamespace(
        preferred_segment=True,
        prefer_alloc_in_same_node=False,
    )
    backend._local_segment = "segment-0"

    assert backend.store(["key"], [[100]], [[10]]) == (0,)

    keys, addresses, sizes, replicate_config = store.batch_put_from_multi_buffers.call_args.args
    assert (keys, addresses, sizes) == (["key"], [[100]], [[10]])
    assert replicate_config.preferred_segment == "segment-0"
    assert not replicate_config.prefer_alloc_in_same_node


def test_mooncake_store_registration_failure_rolls_back_owned_prefix() -> None:
    store = MagicMock()
    store.register_buffer.side_effect = [0, -7]
    store.unregister_buffer.return_value = 0
    backend = _make_mooncake_registration_backend(store)

    with pytest.raises(RuntimeError, match="registration failed"):
        backend.register_buffer([100, 200], [10, 20])

    assert store.register_buffer.call_args_list == [call(100, 10), call(200, 20)]
    store.unregister_buffer.assert_called_once_with(100)


def test_mooncake_registration_rollback_failure_returns_retryable_owner() -> None:
    store = MagicMock()
    store.register_buffer.side_effect = [0, -7]
    store.unregister_buffer.side_effect = [-9, 0]
    backend = _make_mooncake_registration_backend(store)

    with pytest.raises(BufferRegistrationError, match="rollback is incomplete") as error:
        backend.register_buffer([100, 200], [10, 20])

    assert tuple((region.address, region.size) for region in error.value.registration.regions) == ((100, 10),)
    error.value.registration.close()
    assert store.unregister_buffer.call_args_list == [call(100), call(100)]


def test_mooncake_global_registration_preserves_incomplete_shared_owner() -> None:
    class SharedRegistrationError(RuntimeError):
        def __init__(self, operation_error, rollback_error, registration) -> None:
            self.operation_error = operation_error
            self.rollback_error = rollback_error
            self.registration = registration

    store = MagicMock()
    backend = _make_mooncake_registration_backend(store)
    backend._use_store_independent_te = False
    shared_registration = MagicMock()
    operation_error = RuntimeError("registration failed")
    rollback_error = RuntimeError("rollback failed")
    shared_manager = MagicMock()
    shared_manager.acquire_registration.side_effect = SharedRegistrationError(
        operation_error,
        rollback_error,
        shared_registration,
    )

    with (
        patch.object(mooncake_adapter, "global_te", shared_manager),
        patch.object(mooncake_adapter, "RegistrationAcquisitionError", SharedRegistrationError),
        pytest.raises(BufferRegistrationError, match="rollback is incomplete") as error,
    ):
        backend.register_buffer([100], [10])

    assert error.value.registration is shared_registration


def test_mooncake_close_failure_keeps_client_for_retry() -> None:
    store = MagicMock()
    store.close.side_effect = [-7, 0]
    backend = _make_mooncake_registration_backend(store)

    with pytest.raises(RuntimeError, match="close failed"):
        backend.close()

    assert backend._store is store
    assert not backend._closed
    backend.close()
    assert backend._store is None
    assert backend._closed


def test_mooncake_setup_failure_closes_constructed_client() -> None:
    store = MagicMock()
    store.setup.return_value = -7
    store.close.return_value = 0
    backend = _make_mooncake_registration_backend(store)
    backend.parallel_config = SimpleNamespace()
    backend.device_id = 0
    backend._contribute_memory = True
    backend._use_fabric_mem = True
    backend._use_store_independent_te = False
    backend.config = SimpleNamespace(
        metadata_server="metadata",
        global_segment_size=1024,
        local_buffer_size=1024,
        protocol="ascend",
        device_name="",
        master_server_address="master",
        enable_ssd_offload=False,
        ssd_offload_path="",
        tenant_id="default",
    )

    with (
        patch("mooncake.store.MooncakeDistributedStore", return_value=store, create=True),
        patch.object(MooncakeBackend, "set_device"),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend.mooncake.get_ip",
            return_value="127.0.0.1",
        ),
        pytest.raises(RuntimeError, match="initialization failed"),
    ):
        backend._setup_store()

    store.close.assert_called_once_with()


def test_named_memcache_worker_unregisters_regions_before_adapter_close() -> None:
    store = MagicMock()
    backend = _make_memcache_registration_backend(store, initialized=True)
    backend_spec = BackendSpec("memcache", MemcacheBackend, None, True)

    _run_named_backend_resource_lifecycle(backend, backend_spec)

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


def test_unknown_store_source_keeps_real_registration_and_backend_open() -> None:
    topology = make_topology(physical_layers=(0,))
    backend = RecordingBackend()
    backend.put_result = RuntimeError("put failed after address handoff")
    backend_spec = make_backend_spec(layerwise_access=None)
    resources = KVPoolResources(backend, backend_spec, 8, topology.groups)
    projection_binder = compile_bulk_projection_binder(topology, 64)
    worker = SynchronousBulkWorker(
        topology,
        projection_binder,
        resources,
        source_ready_event_factory=FakeEvent,
    )
    caches = _make_caches(topology)
    worker.bind_kv_caches(caches)
    worker.begin_step(
        KVTransferStep(
            store=StoreCommandBatch(
                (
                    RangeStoreCommand(
                        "request",
                        TokenRange(0, 4),
                        ((7,),),
                        (b"a",),
                        4,
                        17,
                    ),
                )
            )
        )
    )
    worker.finish_step()

    with pytest.raises(RuntimeError, match="Store failed"):
        worker.close()

    assert resources.kv_caches is caches
    assert "unregister_region" not in [call[0] for call in backend.calls]
    assert "backend_close" not in [call[0] for call in backend.calls]
