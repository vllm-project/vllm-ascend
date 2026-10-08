"""Registration rollback and Worker teardown preserve native memory lifetime."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeEvent,
    make_backend_spec,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.gva_fixtures import (
    FakeGVABackend,
    make_gva_spec,
)
from tests.ut.distributed.ascend_store.v1.worker.resource_fixtures import (
    RecordingBackend,
    make_caches,
    make_resources,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import vllm_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import (
    TokenRange,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    GVALayerwiseProjectionBinder,
    compile_bulk_projection_binder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    KVTransferStep,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.route import KVPoolRouteSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import (
    AsynchronousBulkWorker,
    SynchronousBulkWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import (
    GVALayerwiseWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.resources import (
    GVAObjectLayout,
    KVPoolResources,
)


def test_resources_own_exact_shared_storage_region_and_close_in_order() -> None:
    backend = RecordingBackend()
    topology, resources = make_resources(backend)
    caches = make_caches(topology, shared_storage=True)
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


@pytest.mark.parametrize("empty_first", (False, True))
@pytest.mark.parametrize("use_gva", (False, True))
def test_resources_skip_empty_rope_views_in_registration_and_projection(empty_first, use_gva) -> None:
    backend = RecordingBackend()
    topology = make_topology(physical_layers=(0,))
    resources = KVPoolResources(
        backend,
        make_backend_spec(layerwise_access=LayerwiseAccessKind.GVA if use_gva else None),
        8,
        topology.transfer_groups,
        gva_layout=GVAObjectLayout(0, 1, 0, 1, 1) if use_gva else None,
    )
    nope_cache = torch.empty((8, 4), dtype=torch.float32)
    rope_cache = torch.empty((8, 4, 0), dtype=torch.float32)
    caches = (rope_cache, nope_cache) if empty_first else (nope_cache, rope_cache)

    registration = resources.bind_kv_caches({topology.groups[0].layer_names[0]: caches})

    address = nope_cache.data_ptr()
    region_size = nope_cache.numel() * nope_cache.element_size()
    block_size_bytes = nope_cache[0].numel() * nope_cache.element_size()
    assert backend.calls == [("register_region", address, region_size)]
    assert registration["base_addresses"] == {0: [address]}
    assert registration["block_lengths"] == {0: [block_size_bytes]}
    assert registration["block_strides"] == {0: [block_size_bytes]}
    assert registration["layer_entry_offsets"] == {0: [0, 1]}
    if use_gva:
        assert registration["object_sizes"] == {0: block_size_bytes}
        assert registration["object_offsets"] == {0: 0}
    resources.close()
    assert backend.calls[-2:] == [("unregister_region", address, region_size), ("backend_close",)]


def test_registration_failure_rolls_back_successful_prefix_before_backend_close() -> None:
    backend = RecordingBackend(fail_registration_index=1)
    topology, resources = make_resources(backend)
    caches = make_caches(topology)

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
    topology, resources = make_resources(backend)
    caches = make_caches(topology)

    with pytest.raises(RuntimeError, match="rollback is incomplete"):
        resources.bind_kv_caches(caches)

    assert resources.kv_caches is caches
    assert "backend_close" not in [call[0] for call in backend.calls]
    resources.close()
    assert [call[0] for call in backend.calls][-2:] == ["unregister_region", "backend_close"]
    assert resources.kv_caches is None


def test_backend_close_failure_retains_cache_owner_until_retry() -> None:
    backend = RecordingBackend(fail_close_count=1)
    topology, resources = make_resources(backend, physical_layers=(0,))
    caches = make_caches(topology)
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
    topology, resources = make_resources(backend, physical_layers=(0,))
    caches = make_caches(topology)
    projection_binder = compile_bulk_projection_binder(topology, 64)

    with patch.object(type(projection_binder), "bind", side_effect=RuntimeError("projection binding failed")):
        worker = SynchronousBulkWorker(topology, projection_binder, resources)
        with pytest.raises(RuntimeError, match="projection binding failed"):
            worker.bind_kv_caches(caches)

    lifecycle = [call[0] for call in backend.calls]
    assert lifecycle == ["register_region", "unregister_region", "backend_close"]
    assert resources.kv_caches is None


@pytest.mark.parametrize("load_async", (False, True))
def test_memcache_bulk_factory_binds_hybrid_buffers_without_gva_constraints(monkeypatch, load_async) -> None:
    topology = make_topology(group_ids=(0, 1), physical_layers=(0,))
    sliding_spec = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=8,
    )
    topology = replace(topology, groups=(topology.groups[0], replace(topology.groups[1], kv_cache_spec=sliding_spec)))
    backend = RecordingBackend()
    backend_spec = BackendSpec("memcache", lambda *_args, **_kwargs: backend, LayerwiseAccessKind.GVA, True)
    route_spec = KVPoolRouteSpec(topology, "memcache", 64, use_layerwise=False)
    projection_binder = compile_bulk_projection_binder(topology, 64)
    monkeypatch.setattr(vllm_adapter, "resolve_kv_pool_route_spec", lambda *_args: route_spec)
    monkeypatch.setattr(vllm_adapter, "_compile_kv_pool_projection_binder", lambda *_args: projection_binder)
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: backend_spec)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer", kv_connector_extra_config={"load_async": load_async}),
        model_config=SimpleNamespace(
            get_layers_start_end_indices=MagicMock(return_value=(0, 1)),
            get_total_num_hidden_layers=MagicMock(return_value=1),
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    cache_config = SimpleNamespace(
        num_blocks=8,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=group.kv_cache_spec) for group in topology.groups],
    )

    worker = vllm_adapter.create_kv_pool_worker(config, cache_config)
    try:
        assert type(worker) is (AsynchronousBulkWorker if load_async else SynchronousBulkWorker)
        config.model_config.get_layers_start_end_indices.assert_not_called()
        config.model_config.get_total_num_hidden_layers.assert_not_called()
        caches = {
            name: torch.empty((8, 4), dtype=torch.float32) for group in topology.groups for name in group.layer_names
        }
        worker.bind_kv_caches(caches)
        registered_regions = [entry for entry in backend.calls if entry[0] == "register_region"]
        assert registered_regions == [
            ("register_region", cache.data_ptr(), cache.numel() * cache.element_size()) for cache in caches.values()
        ]
    finally:
        worker.close()
    assert backend.closed


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
    caches = make_caches(topology)
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

    with pytest.raises(RuntimeError, match="Store source release is unknown"):
        worker.close()

    assert resources.kv_caches is caches
    assert "unregister_region" not in [call[0] for call in backend.calls]
    assert "backend_close" not in [call[0] for call in backend.calls]


def test_gva_worker_close_unregisters_exact_region_before_backend_close() -> None:
    backend = FakeGVABackend()
    topology = make_topology(physical_layers=(0,))
    backend_spec = make_gva_spec()
    resources = KVPoolResources(
        backend,
        backend_spec,
        8,
        topology.groups,
        gva_layout=GVAObjectLayout(0, 1, 0, 1, 1),
    )
    projection_binder = GVALayerwiseProjectionBinder(
        topology,
        64,
        lambda group, value, head, stage: f"g{group}:p{stage}:h{head}:{value}",
    )
    worker = GVALayerwiseWorker(
        topology,
        projection_binder,
        resources,
        source_ready_event_factory=FakeEvent,
    )
    cache = torch.empty((8, 4), dtype=torch.float32)

    worker.bind_kv_caches({topology.groups[0].layer_names[0]: cache})
    worker.close()

    lifecycle = [call for call in backend.calls if call[0] in ("register_buffer", "unregister_buffer", "backend_close")]
    assert lifecycle == [
        ("register_buffer", (cache.data_ptr(),), (cache.nbytes,)),
        ("unregister_buffer", cache.data_ptr(), cache.nbytes),
        ("backend_close",),
    ]
