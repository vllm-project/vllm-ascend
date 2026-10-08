"""Configuration selects the expected Backend access and execution behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeResources,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import (
    FakeBlockPool,
    FakeBlocks,
    FakeRemoteLookup,
    make_output,
    make_request,
    new_request_data,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import backend as backend_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import vllm_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    BulkProjectionBinder,
    GVALayerwiseProjectionBinder,
    KeyRangeLayerwiseProjectionBinder,
    compile_bulk_projection_binder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    StoreSourceReleaseMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.route import (
    KVPoolRouteSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import (
    AsynchronousBulkWorker,
    SynchronousBulkWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import (
    GVALayerwiseWorker,
    KeyRangeLayerwiseWorker,
)


def test_worker_factory_selects_one_concrete_route(monkeypatch) -> None:
    cases = (
        (False, None, False, SynchronousBulkWorker),
        (False, None, True, AsynchronousBulkWorker),
        (True, LayerwiseAccessKind.KEY_RANGE, False, KeyRangeLayerwiseWorker),
        (True, LayerwiseAccessKind.GVA, False, GVALayerwiseWorker),
    )
    topology = make_topology(physical_layers=(0,))
    full_key = lambda group, value, head, stage: f"g{group}:p{stage}:h{head}:{value}"
    monkeypatch.setattr(vllm_adapter, "uses_hybrid_kv_cache", lambda *_args: False)
    cache_config = SimpleNamespace(num_blocks=8, kv_cache_groups=())
    for use_layerwise, layerwise_access, load_async, expected_type in cases:
        backend = FakeBackend()
        backend_spec = SimpleNamespace(
            layerwise_access=layerwise_access,
            requires_exists_before_put=False,
            create=lambda *_args, backend=backend, **_kwargs: backend,
        )
        route_spec = KVPoolRouteSpec(topology, "fake", 64, use_layerwise=use_layerwise)
        if layerwise_access is LayerwiseAccessKind.GVA:
            projection_binder = GVALayerwiseProjectionBinder(topology, 64, full_key)
        elif layerwise_access is LayerwiseAccessKind.KEY_RANGE:
            projection_binder = KeyRangeLayerwiseProjectionBinder(topology, 64, full_key)
        else:
            projection_binder = compile_bulk_projection_binder(topology, 64)

        monkeypatch.setattr(
            vllm_adapter,
            "resolve_kv_pool_route_spec",
            lambda *_args, route_spec=route_spec: route_spec,
        )
        monkeypatch.setattr(
            vllm_adapter,
            "_compile_kv_pool_projection_binder",
            lambda *_args, projection_binder=projection_binder: projection_binder,
        )
        monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name, spec=backend_spec: spec)
        monkeypatch.setattr(
            vllm_adapter,
            "KVPoolResources",
            lambda backend, spec, _num_blocks, _groups, **_kwargs: FakeResources(backend, spec, topology),
        )
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(),
            kv_transfer_config=SimpleNamespace(
                kv_role="kv_both",
                kv_connector_extra_config={"load_async": load_async},
            ),
            model_config=SimpleNamespace(
                get_layers_start_end_indices=lambda _parallel_config: (0, 1),
                get_total_num_hidden_layers=lambda: 1,
            ),
            scheduler_config=SimpleNamespace(),
        )

        worker = vllm_adapter.create_kv_pool_worker(config, cache_config)

        assert type(worker) is expected_type, (use_layerwise, layerwise_access, load_async)
        worker.close()


def test_factory_binds_role_capabilities(monkeypatch) -> None:
    cases: tuple[tuple[str, dict[str, bool], bool, bool], ...] = (
        ("kv_producer", {}, True, True),
        ("kv_both", {}, True, True),
        ("kv_consumer", {}, False, False),
        ("kv_consumer", {"consumer_is_to_load": True}, True, False),
        ("kv_consumer", {"consumer_is_to_put": True}, False, True),
    )
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    cache_config = SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
    )
    for role, extra, load_enabled, store_enabled in cases:
        lookup = FakeRemoteLookup(4)
        monkeypatch.setattr(vllm_adapter, "RemoteLookup", lambda _address, lookup=lookup: lookup)
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role=role, kv_connector_extra_config=extra),
            parallel_config=SimpleNamespace(world_size=1),
            speculative_config=None,
        )
        scheduler = vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
        scheduler.bind_gpu_block_pool(FakeBlockPool())

        lookup_request = make_request("lookup", tokens=9)
        assert scheduler.get_num_new_matched_tokens(lookup_request, 0)[0] == (4 if load_enabled else 0), role
        assert bool(lookup.queries) is load_enabled, role

        store_request = make_request("store")
        scheduler.confirm_allocation(store_request, FakeBlocks([1, 2]), 0)
        step = scheduler.build_step(
            make_output(
                new=(new_request_data("store", ([1, 2],), 0),),
                scheduled_tokens={"store": 4},
            )
        )
        assert bool(step.store.commands) is store_enabled, role
        for command in step.store.commands:
            scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 1}))
        scheduler.close()


def test_scheduler_factory_honors_configured_load_publication_timing(monkeypatch) -> None:
    cases = (
        (False, False, False),
        (False, True, True),
        (True, False, False),
        (True, True, False),
    )
    monkeypatch.setattr(vllm_adapter, "RemoteLookup", lambda _address: FakeRemoteLookup(4))
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    cache_config = SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
    )
    for use_layerwise, load_async, is_deferred in cases:
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_role="kv_consumer",
                kv_connector_extra_config={
                    "use_layerwise": use_layerwise,
                    "load_async": load_async,
                    "consumer_is_to_load": True,
                },
            ),
            parallel_config=SimpleNamespace(world_size=1),
            speculative_config=None,
        )

        scheduler = vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
        request = make_request(prompt_tokens=8)
        assert scheduler.get_num_new_matched_tokens(request, 0) == (4, is_deferred)
        assert scheduler.build_step(make_output()).load.commands == ()
        scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 4)
        if is_deferred:
            published = scheduler.build_step(make_output())
        else:
            assert scheduler.build_step(make_output()).load.commands == ()
            published = scheduler.build_step(
                make_output(
                    new=(new_request_data("request", ([1, 2],), 4),),
                    scheduled_tokens={"request": 0},
                )
            )
        assert len(published.load.commands) == 1
        assert published.load.commands[0].load_range == TokenRange(0, 4)
        scheduler.close()


def test_backend_selection_fixes_access_kind_once(monkeypatch) -> None:
    classes = {class_name: FakeBackend for _, class_name in backend_module.BACKEND_IMPORTS.values()}
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda _: SimpleNamespace(**classes))
    expected = {
        "mooncake": LayerwiseAccessKind.KEY_RANGE,
        "memcache": LayerwiseAccessKind.GVA,
    }
    assert {name: backend_module.resolve_backend_spec(name).layerwise_access for name in expected} == expected


@pytest.mark.parametrize("layout", ("NHD", "HND", "LBHNC"))
def test_bulk_binder_preserves_configured_cache_layout(monkeypatch, layout) -> None:
    monkeypatch.setattr(vllm_adapter.vllm_envs, "VLLM_KV_CACHE_LAYOUT", layout)
    route = KVPoolRouteSpec(make_topology(tp_mismatch=True), "fake", 64)

    binder = vllm_adapter._compile_kv_pool_projection_binder(route, SimpleNamespace(), SimpleNamespace())

    assert isinstance(binder, BulkProjectionBinder)
    assert binder.kv_cache_layout == layout
