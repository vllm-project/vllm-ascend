"""Configuration selects the expected Backend access and execution behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec, SlidingWindowSpec

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeResources,
    make_reachability,
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
    BackendSpec,
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    StoreSourceReleaseMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.route import (
    KVPoolRouteSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import (
    BulkWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import (
    GVALayerwiseWorker,
    KeyRangeLayerwiseWorker,
)


def test_worker_factory_selects_one_concrete_route(monkeypatch) -> None:
    cases = (
        (False, None, False, BulkWorker),
        (False, None, True, BulkWorker),
        (True, LayerwiseAccessKind.KEY_RANGE, False, KeyRangeLayerwiseWorker),
        (True, LayerwiseAccessKind.GVA, False, GVALayerwiseWorker),
    )
    topology = make_topology(physical_layers=(0,))
    cache_config = SimpleNamespace(num_blocks=8, kv_cache_groups=())
    for use_layerwise, layerwise_access, load_async, expected_type in cases:
        backend = FakeBackend()
        backend_spec = SimpleNamespace(
            layerwise_access=layerwise_access,
            requires_exists_before_put=False,
            create=lambda *_args, backend=backend, **_kwargs: backend,
        )
        route_spec = KVPoolRouteSpec(topology, "fake", 64, use_layerwise=use_layerwise)

        monkeypatch.setattr(
            vllm_adapter,
            "resolve_kv_pool_route_spec",
            lambda *_args, route_spec=route_spec: route_spec,
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
        if isinstance(worker, BulkWorker):
            assert (worker._load_executor is not None) is load_async
        worker.close()


@pytest.mark.parametrize("prefetch", (0, -1, True, "invalid"))
def test_invalid_prefetch_is_rejected_before_backend_creation(monkeypatch, prefetch) -> None:
    route = KVPoolRouteSpec(make_topology(), "mooncake", 64, use_layerwise=True)
    create_backend = Mock(side_effect=AssertionError("Invalid configuration must not open a Backend"))
    spec = BackendSpec("mooncake", create_backend, LayerwiseAccessKind.KEY_RANGE, True)
    monkeypatch.setattr(vllm_adapter, "resolve_kv_pool_route_spec", lambda *_args: route)
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: spec)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both", kv_connector_extra_config={"layerwise_prefetch_layers": prefetch}
        ),
    )
    with pytest.raises(ValueError, match="positive integer"):
        vllm_adapter.create_kv_pool_worker(config, SimpleNamespace(num_blocks=8))
    create_backend.assert_not_called()


def test_missing_layerwise_data_plane_is_rejected_before_backend_creation(monkeypatch) -> None:
    route = KVPoolRouteSpec(make_topology(), "mooncake", 64, use_layerwise=True)
    create_backend = Mock(side_effect=AssertionError("Unsupported route must not open a Backend"))
    spec = BackendSpec("mooncake", create_backend, None, True)
    monkeypatch.setattr(vllm_adapter, "resolve_kv_pool_route_spec", lambda *_args: route)
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: spec)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config={}),
    )
    with pytest.raises(TypeError, match="concrete Backend data plane"):
        vllm_adapter.create_kv_pool_worker(config, SimpleNamespace(num_blocks=8))
    create_backend.assert_not_called()


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
        monkeypatch.setattr(vllm_adapter, "RemoteLookup", lambda _address, lookup=lookup, **_kwargs: lookup)
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role=role, kv_connector_extra_config=extra),
            parallel_config=SimpleNamespace(world_size=1, tensor_parallel_size=1),
            model_config=SimpleNamespace(
                use_mla=False,
                get_total_num_kv_heads=lambda: 1,
                get_total_num_hidden_layers=lambda: 2,
                hf_text_config=SimpleNamespace(num_hidden_layers=2),
            ),
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


@pytest.mark.parametrize(
    "backend,use_layerwise,load_async,spec_kind,extra,pins_source",
    (
        ("mooncake", False, False, "full", {}, False),
        ("mooncake", False, True, "mla", {}, False),
        ("mooncake", False, False, "full", {"prefill_tp_size": 1, "decode_tp_size": 2}, False),
        ("mooncake", False, False, "full", {"consumer_is_to_put": True, "prefill_pp_size": 2}, False),
        ("mooncake", False, False, "window", {}, True),
        ("mooncake", False, True, "window", {}, True),
        ("mooncake", True, False, "window", {}, False),
        ("memcache", True, False, "window", {}, False),
    ),
)
def test_factory_selects_source_protection_by_route(
    monkeypatch, backend, use_layerwise, load_async, spec_kind, extra, pins_source
) -> None:
    monkeypatch.setattr(vllm_adapter, "RemoteLookup", lambda *_args, **_kwargs: FakeRemoteLookup())
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    geometry = dict(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    if spec_kind == "window":
        spec = SlidingWindowSpec(**geometry, sliding_window=4)
    elif spec_kind == "mla":
        spec = MLAAttentionSpec(**geometry)
    else:
        spec = FullAttentionSpec(**geometry)
    cache_config = SimpleNamespace(transfer_group_ids=(0,), kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_consumer" if "consumer_is_to_put" in extra else "kv_both",
            kv_connector_extra_config={
                "backend": backend,
                "use_layerwise": use_layerwise,
                "load_async": load_async,
                **extra,
            },
        ),
        parallel_config=SimpleNamespace(world_size=2, tensor_parallel_size=2),
        model_config=SimpleNamespace(
            use_mla=spec_kind == "mla",
            get_total_num_kv_heads=lambda: 2,
            get_total_num_hidden_layers=lambda: 2,
            hf_text_config=SimpleNamespace(num_hidden_layers=2),
        ),
        speculative_config=None,
    )
    scheduler = vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
    pool = FakeBlockPool()
    scheduler.bind_gpu_block_pool(pool)
    scheduler.confirm_allocation(make_request(), FakeBlocks([1, 2, 3]), 0)
    step = scheduler.build_step(
        make_output(new=(new_request_data("request", ([1, 2, 3],), 0),), scheduled_tokens={"request": 8})
    )

    command = step.store.commands[0]
    assert pool.touched == ([(1, 2)] if pins_source else [])
    assert (command.store_job_id is not None) is pins_source
    if command.store_job_id is not None:
        scheduler.accept_worker_metadata(StoreSourceReleaseMetadata({command.store_job_id: 2}))
    scheduler.close()


def test_scheduler_factory_honors_configured_load_publication_timing(monkeypatch) -> None:
    cases = (
        (False, False, False),
        (False, True, True),
        (True, False, False),
        (True, True, False),
    )
    monkeypatch.setattr(vllm_adapter, "RemoteLookup", lambda _address, **_kwargs: FakeRemoteLookup(4))
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
            parallel_config=SimpleNamespace(world_size=1, tensor_parallel_size=1),
            model_config=SimpleNamespace(
                use_mla=False,
                get_total_num_kv_heads=lambda: 1,
                get_total_num_hidden_layers=lambda: 2,
                hf_text_config=SimpleNamespace(num_hidden_layers=2),
            ),
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


@pytest.mark.parametrize(
    "timeout_ms,is_valid",
    (
        (None, True),
        (-1, True),
        (0, True),
        (250, True),
        (2**31 - 1, True),
        (-2, False),
        (True, False),
        (1.5, False),
        ("250", False),
        (2**31, False),
    ),
)
def test_scheduler_factory_configures_lookup_timeout_and_rejects_invalid_values(
    monkeypatch, timeout_ms, is_valid
) -> None:
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_args: (4, 4))
    cache_config = SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
            )
        ],
    )
    extra_config = {} if timeout_ms is None else {"lookup_rpc_timeout_ms": timeout_ms}
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config=extra_config),
        parallel_config=SimpleNamespace(world_size=1, tensor_parallel_size=1),
        model_config=SimpleNamespace(
            use_mla=False,
            get_total_num_kv_heads=lambda: 1,
            get_total_num_hidden_layers=lambda: 2,
            hf_text_config=SimpleNamespace(num_hidden_layers=2),
        ),
        speculative_config=None,
    )
    if not is_valid:
        with pytest.raises(ValueError, match="lookup_rpc_timeout_ms"):
            vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
        return

    observed_timeouts = []
    remote_lookup = vllm_adapter.RemoteLookup

    def create_lookup(address, *, timeout_ms):
        observed_timeouts.append(timeout_ms)
        return remote_lookup(address, timeout_ms=timeout_ms)

    monkeypatch.setattr(vllm_adapter, "RemoteLookup", create_lookup)
    scheduler = vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
    scheduler.close()
    assert observed_timeouts == [-1 if timeout_ms is None else timeout_ms]


@pytest.mark.parametrize("layout", ("NHD", "HND", "LBHNC"))
def test_worker_preserves_configured_cache_layout(layout) -> None:
    topology = make_topology()
    backend = FakeBackend()
    resources = FakeResources(backend, SimpleNamespace(), topology)
    worker = BulkWorker(topology, resources, reachability=make_reachability(topology), kv_cache_layout=layout)
    assert worker._kv_cache_layout == layout
    worker.close()
