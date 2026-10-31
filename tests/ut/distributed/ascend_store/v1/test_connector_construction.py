"""Configuration selects the expected Backend access and execution behavior."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, MLAAttentionSpec, SlidingWindowSpec

from tests.ut.distributed.ascend_store.v1.construction_fixtures import create_scheduler, create_worker
from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeLoadStartGate,
    begin_step,
    make_pool_config,
    make_topology,
    make_worker,
    store_one,
)
from tests.ut.distributed.ascend_store.v1.infrastructure.resources_fixtures import RecordingBackend
from tests.ut.distributed.ascend_store.v1.native_copy_fixtures import ByteCopySDK, native_backend
from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import (
    FakeBlockPool,
    FakeBlocks,
    FakeLookupClient,
    make_output,
    make_request,
    new_request_data,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import backend as backend_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import BackendSpec, LayerwiseAccessKind
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure import configuration
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure import topology as topology_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.mapping import map_keys
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    LoadCommand,
    LookupRequest,
    RangeStoreCommand,
    StoreSourceReleaseMetadata,
)


def test_missing_layerwise_data_plane_is_rejected_before_backend_creation(monkeypatch) -> None:
    topology = make_topology()
    pool_config = make_pool_config(topology, use_layerwise=True)
    create_backend = Mock(side_effect=AssertionError("Unsupported route must not open a Backend"))
    spec = BackendSpec("mooncake", create_backend, None, True)
    monkeypatch.setattr(connector_module, "resolve_kv_pool_config", lambda *_args: pool_config)
    monkeypatch.setattr(connector_module, "resolve_worker_topology", lambda *_args: topology)
    monkeypatch.setattr(connector_module, "resolve_backend_spec", lambda _name: spec)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        model_config=SimpleNamespace(max_model_len=64),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config={}),
    )
    with pytest.raises(TypeError, match="concrete Backend data plane"):
        create_worker(config, SimpleNamespace(num_blocks=8))
    create_backend.assert_not_called()


def test_backend_supplied_naming_is_used_for_transfer_keys(configuration_case, monkeypatch):
    config, cache = configuration_case
    topology = make_topology(physical_layers=(0,))
    backend = FakeBackend()
    builder = Mock(return_value=("custom-wire-", ""))
    builder.lookup_parts = Mock(return_value=(("custom-wire-", ""),))
    make_names = Mock(return_value=builder)
    spec = BackendSpec(
        "custom",
        lambda *_args, **_kwargs: backend,
        LayerwiseAccessKind.KEY_RANGE,
        False,
        layerwise_key_factory=make_names,
    )
    pool = make_pool_config(topology, "custom", use_layerwise=True)
    monkeypatch.setattr(connector_module, "resolve_kv_pool_config", lambda *_: pool)
    monkeypatch.setattr(connector_module, "resolve_worker_topology", lambda *_: topology)
    monkeypatch.setattr(connector_module, "resolve_backend_spec", lambda _: spec)
    worker = create_worker(config, cache)
    try:
        caches = {
            layer.layer_names[0]: (torch.zeros(8, 4), torch.zeros(8, 4)) for layer in topology.transfer_groups[0].layers
        }
        worker.bind_kv_caches(caches)
        assert tuple(map_keys(parts, (b"a",)) for parts in worker.context.key_parts[0].values()) == (
            ("custom-wire-61",),
        )
    finally:
        worker.close()


def test_binding_policy_resolution_failure_does_not_create_backend(configuration_case, monkeypatch):
    config, cache = configuration_case
    topology = make_topology(physical_layers=(0,))
    pool = make_pool_config(topology, "custom", use_layerwise=True)
    create_backend = Mock(side_effect=AssertionError("Backend must not be acquired"))
    fail_policy = Mock(side_effect=RuntimeError("wire policy resolution failed"))
    spec = BackendSpec(
        "custom", create_backend, LayerwiseAccessKind.KEY_RANGE, False, layerwise_key_factory=fail_policy
    )
    monkeypatch.setattr(connector_module, "resolve_kv_pool_config", lambda *_: pool)
    monkeypatch.setattr(connector_module, "resolve_worker_topology", lambda *_: topology)
    monkeypatch.setattr(connector_module, "resolve_backend_spec", lambda _: spec)
    with pytest.raises(RuntimeError, match="wire policy resolution failed"):
        create_worker(config, cache)
    fail_policy.assert_called_once()
    create_backend.assert_not_called()


@pytest.mark.parametrize(
    "role,extra,load,store",
    (
        ("kv_producer", {}, True, True),
        ("kv_both", {}, True, True),
        ("kv_consumer", {}, False, False),
        ("kv_consumer", {"consumer_is_to_load": True}, True, False),
        ("kv_consumer", {"consumer_is_to_put": True}, False, True),
    ),
)
def test_connector_binds_role_capabilities(configuration_case, monkeypatch, role, extra, load, store):
    config, cache = configuration_case
    config.kv_transfer_config.kv_role = role
    config.kv_transfer_config.kv_connector_extra_config.update(extra)
    lookup = FakeLookupClient(4)
    monkeypatch.setattr(connector_module, "LookupClient", lambda *_args, **_kwargs: lookup)
    scheduler = create_scheduler(config, cache, "unused")
    scheduler.bind_gpu_block_pool(FakeBlockPool())
    try:
        assert scheduler.get_num_new_matched_tokens(make_request("lookup", tokens=9), 0)[0] == (4 if load else 0)
        assert bool(lookup.queries) is load
        scheduler.confirm_allocation(make_request("store"), FakeBlocks([1, 2]), 0)
        step = scheduler.build_step(
            make_output(new=(new_request_data("store", ([1, 2],), 0),), scheduled_tokens={"store": 4})
        )
        assert bool(step.store.commands) is store
    finally:
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
def test_connector_selects_source_protection_by_route(
    monkeypatch, backend, use_layerwise, load_async, spec_kind, extra, pins_source
) -> None:
    monkeypatch.setattr(connector_module, "LookupClient", lambda *_args, **_kwargs: FakeLookupClient())
    monkeypatch.setattr(configuration.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    geometry = dict(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    if spec_kind == "window":
        spec = SlidingWindowSpec(**geometry, sliding_window=4)
    elif spec_kind == "mla":
        spec = MLAAttentionSpec(**geometry)
    else:
        spec = FullAttentionSpec(**geometry)
    cache_config = SimpleNamespace(
        transfer_group_ids=(0,),
        prefix_cache_retention_interval=None,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, is_eagle_group=False)],
    )
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
        parallel_config=SimpleNamespace(
            world_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            model="model",
            use_mla=spec_kind == "mla",
            get_total_num_kv_heads=lambda: 2,
            get_total_num_hidden_layers=lambda: 2,
            hf_text_config=SimpleNamespace(num_hidden_layers=2),
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    scheduler = create_scheduler(config, cache_config, "unused")
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


@pytest.mark.parametrize(
    "layerwise,async_load,deferred",
    ((False, False, False), (False, True, True), (True, False, False), (True, True, False)),
)
def test_scheduler_honors_configured_load_publication_timing(
    configuration_case, monkeypatch, layerwise, async_load, deferred
):
    config, cache = configuration_case
    config.kv_transfer_config.kv_connector_extra_config.update(use_layerwise=layerwise, load_async=async_load)
    monkeypatch.setattr(connector_module, "LookupClient", lambda *_args, **_kwargs: FakeLookupClient(4))
    scheduler = create_scheduler(config, cache, "unused")
    try:
        request = make_request(prompt_tokens=8)
        assert scheduler.get_num_new_matched_tokens(request, 0) == (4, deferred)
        assert scheduler.build_step(make_output()).load_commands == ()
        scheduler.confirm_allocation(request, FakeBlocks([1, 2]), 4)
        pending = scheduler.build_step(make_output())
        assert bool(pending.load_commands) is deferred
        published = (
            pending
            if deferred
            else scheduler.build_step(
                make_output(new=(new_request_data("request", ([1, 2],), 4),), scheduled_tokens={"request": 0})
            )
        )
        assert [(command.start_token, command.end_token) for command in published.load_commands] == [(0, 4)]
    finally:
        scheduler.close()


def test_backend_selection_fixes_access_kind_once(monkeypatch) -> None:
    classes = {class_name: FakeBackend for _, class_name in backend_module.BACKEND_IMPORTS.values()}
    monkeypatch.setattr(backend_module.importlib, "import_module", lambda _: SimpleNamespace(**classes))
    expected = {"mooncake": LayerwiseAccessKind.KEY_RANGE, "memcache": LayerwiseAccessKind.GVA}
    assert {name: backend_module.resolve_backend_spec(name).layerwise_access for name in expected} == expected


@pytest.mark.parametrize(
    "timeout_ms,valid",
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
def test_scheduler_configures_lookup_timeout_and_rejects_invalid_values(
    configuration_case, monkeypatch, timeout_ms, valid
):
    config, cache = configuration_case
    if timeout_ms is not None:
        config.kv_transfer_config.kv_connector_extra_config["lookup_rpc_timeout_ms"] = timeout_ms
    create = Mock(wraps=connector_module.LookupClient)
    monkeypatch.setattr(connector_module, "LookupClient", create)
    if not valid:
        with pytest.raises(ValueError, match="lookup_rpc_timeout_ms"):
            create_scheduler(config, cache, "unused")
    else:
        create_scheduler(config, cache, "unused").close()
    create.assert_called_once_with("unused", timeout_ms=-1 if timeout_ms is None else timeout_ms)


@pytest.mark.parametrize("load_async", (False, True))
def test_memcache_bulk_construction_binds_hybrid_buffers_without_gva_constraints(monkeypatch, load_async) -> None:
    topology = make_topology(group_ids=(0, 1), physical_layers=(0,))
    sliding_spec = SlidingWindowSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32, sliding_window=8)
    topology = replace(topology, groups=(topology.groups[0], replace(topology.groups[1], kv_cache_spec=sliding_spec)))
    backend = RecordingBackend()
    backend_spec = BackendSpec("memcache", lambda *_args, **_kwargs: backend, LayerwiseAccessKind.GVA, True)
    pool_config = make_pool_config(topology, "memcache", load_async=load_async, store_enabled=False)
    monkeypatch.setattr(connector_module, "resolve_kv_pool_config", lambda *_args: pool_config)
    monkeypatch.setattr(connector_module, "resolve_worker_topology", lambda *_args: topology)
    monkeypatch.setattr(connector_module, "resolve_backend_spec", lambda _name: backend_spec)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer", kv_connector_extra_config={"load_async": load_async}),
        model_config=SimpleNamespace(
            max_model_len=64,
            get_layers_start_end_indices=MagicMock(return_value=(0, 1)),
            get_total_num_hidden_layers=MagicMock(return_value=1),
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    cache_config = SimpleNamespace(
        num_blocks=8, kv_cache_groups=[SimpleNamespace(kv_cache_spec=group.kv_cache_spec) for group in topology.groups]
    )
    worker = create_worker(config, cache_config)
    try:
        config.model_config.get_layers_start_end_indices.assert_not_called()
        config.model_config.get_total_num_hidden_layers.assert_not_called()
        caches = {
            name: torch.empty((8, 4), dtype=torch.float32)
            for group in topology.groups
            for layer in group.layers
            for name in layer.layer_names
        }
        worker.bind_kv_caches(caches)
        registered_regions = [entry for entry in backend.calls if entry[0] == "register_region"]
        assert registered_regions == [
            ("register_region", cache.data_ptr(), cache.numel() * cache.element_size()) for cache in caches.values()
        ]
    finally:
        worker.close()
    assert backend.closed


def test_connector_closes_backend_when_resource_creation_fails(monkeypatch) -> None:
    topology = make_topology(physical_layers=(0,))

    class FactoryBackend(RecordingBackend):
        instance = None

        def __init__(self, *_args, **_kwargs) -> None:
            super().__init__()
            type(self).instance = self

    backend_spec = BackendSpec("memcache", FactoryBackend, LayerwiseAccessKind.GVA, True)
    pool_config = make_pool_config(topology, "memcache", use_layerwise=True)
    monkeypatch.setattr(connector_module, "resolve_kv_pool_config", lambda *_args: pool_config)
    monkeypatch.setattr(connector_module, "resolve_worker_topology", lambda *_args: topology)
    monkeypatch.setattr(connector_module, "resolve_backend_spec", lambda _name: backend_spec)

    def fail_resources(*_args, **_kwargs):
        raise RuntimeError("resource creation failed")

    monkeypatch.setattr(connector_module, "Resources", fail_resources)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config={}),
        model_config=SimpleNamespace(max_model_len=64),
        scheduler_config=SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match="resource creation failed"):
        create_worker(config, SimpleNamespace(num_blocks=8, kv_cache_groups=topology.groups))
    assert FactoryBackend.instance is not None
    assert FactoryBackend.instance.calls == [("backend_close",)]


def constructed_worker(monkeypatch, name, layerwise, topology, objects):
    sdk = ByteCopySDK(objects)
    backend = native_backend(name, sdk)
    access = LayerwiseAccessKind.KEY_RANGE if name == "mooncake" else LayerwiseAccessKind.GVA
    spec = BackendSpec(name, lambda *_args, **_kwargs: backend, access, True)
    pool_config = make_pool_config(topology, name, use_layerwise=layerwise)
    monkeypatch.setattr(connector_module, "validate_kv_pool_config", lambda *_args: None)
    monkeypatch.setattr(connector_module, "resolve_kv_pool_config", lambda *_args: pool_config)
    monkeypatch.setattr(connector_module, "resolve_worker_topology", lambda *_args: topology)
    monkeypatch.setattr(connector_module, "resolve_backend_spec", lambda _name: spec)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        model_config=SimpleNamespace(max_model_len=64),
        scheduler_config=SimpleNamespace(),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config={}),
    )
    worker = create_worker(config, SimpleNamespace(num_blocks=8, kv_cache_groups=topology.groups))
    worker._source_ready_event_factory = FakeEvent
    if layerwise:
        worker._executor._start_gate_factory = lambda: FakeLoadStartGate(opened=True)
    caches = {}
    for group in topology.transfer_groups:
        for layer in group.layers:
            caches[layer.layer_names[0]] = (
                (torch.arange(8 * 7, dtype=torch.float32).reshape(8, 7) + group.group_id * 100)[:, :4].view(8, 4, 1, 1),
                (torch.arange(8 * 5, dtype=torch.int8).reshape(8, 5) + layer.physical_layer_id * 100)[:, :4].view(
                    8, 4, 1, 1
                ),
            )
    worker.bind_kv_caches(caches)
    assert tuple(worker.context.local) == topology.config.transfer_group_ids
    return (worker, caches, sdk)


@pytest.mark.parametrize(
    "backend_name,pp_rank,dcp_rank,dcp_size",
    (
        ("mooncake", 0, 0, 1),
        ("mooncake", 1, 0, 1),
        ("memcache", 0, 0, 1),
        ("memcache", 1, 0, 1),
        ("memcache", 1, 0, 2),
        ("memcache", 1, 1, 2),
    ),
)
@pytest.mark.parametrize("layerwise_store", (False, True), ids=("bulk_to_layerwise", "layerwise_to_bulk"))
def test_connector_default_keys_share_complete_objects(
    monkeypatch, backend_name, layerwise_store, pp_rank, dcp_rank, dcp_size
):
    topology = make_topology(
        group_ids=(0, 1),
        physical_layers=(2, 5),
        pp_size=pp_rank + 1,
        pp_rank=pp_rank,
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
        tp_rank=dcp_rank,
        tp_size=dcp_size,
        put_step=dcp_size,
    )
    objects: dict[str, ctypes.Array[ctypes.c_char]] = {}
    writer, source, write_sdk = constructed_worker(monkeypatch, backend_name, layerwise_store, topology, objects)
    block_ids = tuple((1,) if group_id in topology.config.transfer_group_ids else () for group_id in range(4))
    command = RangeStoreCommand("request", 0, 4, block_ids, (b"a",), 4, 17)
    expected = {name: tuple(tensor[1].clone() for tensor in tensors) for name, tensors in source.items()}
    try:
        begin_step(writer, store=(command,))
        if layerwise_store:
            for layer_id in (2, 5):
                writer.save_layer(f"layers.{layer_id}.group.0")
        writer.finish_step()
        completions = writer.fence_previous_store()
        assert all(completion.succeeded for completion in completions)
        assert writer.take_released_store_job_ids() == {17}
        writer.end_step()
        keys = {map_keys(next(iter(parts.values())), (b"a",))[0] for parts in writer.context.key_parts.values()}
        assert set(objects) == keys
        assert all(ctypes.sizeof(value) == 40 for value in objects.values())
        assert all(f"@pp_rank:{pp_rank}@" in key for key in keys)
        assert all(f"@dcp:{dcp_rank}@" in key for key in keys)
        for group_id, key_parts in writer.context.key_parts.items():
            expected_bytes = b"".join(
                tensor[1].numpy().tobytes()
                for layer in topology.groups[group_id].layers
                for tensor in source[layer.layer_names[0]]
            )
            assert objects[map_keys(next(iter(key_parts.values())), (b"a",))[0]].raw == expected_bytes
    finally:
        writer.close()
    assert write_sdk.closed
    reader, destination, read_sdk = constructed_worker(
        monkeypatch, backend_name, not layerwise_store, topology, objects
    )
    try:
        for tensors in destination.values():
            for tensor in tensors:
                tensor.fill_(-1)
        assert {map_keys(next(iter(parts.values())), (b"a",))[0] for parts in reader.context.key_parts.values()} == keys
        if pp_rank == 0:
            assert reader.lookup(LookupRequest(0, 4, (0, 1), (b"a",))).available_end_token == 4
        target_ids = tuple((6,) if group_id in topology.config.transfer_group_ids else () for group_id in range(4))
        begin_step(reader, load=(LoadCommand("request", 0, 4, target_ids, (b"a",)),))
        reader.start_load()
        if not layerwise_store:
            for layer_id in (2, 5):
                reader.wait_for_layer_load(f"layers.{layer_id}.group.0")
        assert not reader.collect_load_result().failed_locations
        for name, tensors in destination.items():
            for tensor, expected_tensor in zip(tensors, expected[name], strict=True):
                torch.testing.assert_close(tensor[6], expected_tensor)
                assert torch.all(tensor[5] == -1)
                assert torch.all(tensor[7] == -1)
        reader.end_step()
    finally:
        reader.close()
    assert read_sdk.closed


@pytest.mark.parametrize("layerwise,group_count", ((False, 1), (False, 2), (True, 1), (True, 2)))
@pytest.mark.parametrize("use_eagle", (False, True))
def test_speculative_factories_follow_upstream_eagle_policy(
    configuration_case, monkeypatch, layerwise, group_count, use_eagle
):
    config, cache = configuration_case
    config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = layerwise
    config.speculative_config = SimpleNamespace(use_eagle=lambda: use_eagle, use_eagle_block_drop=lambda: False)
    cache.kv_cache_groups *= group_count
    cache.transfer_group_ids = tuple(range(group_count))
    cache.prefix_cache_retention_interval = 32
    backend = FakeBackend()
    spec = BackendSpec("mooncake", lambda *_args, **_kwargs: backend, LayerwiseAccessKind.KEY_RANGE, True)
    monkeypatch.setattr(connector_module, "resolve_backend_spec", lambda _: spec)
    for name in ("get_tp_group", "get_pp_group"):
        monkeypatch.setattr(topology_module, name, lambda: SimpleNamespace(rank_in_group=0, ranks=[0]))
    monkeypatch.setattr(connector_module, "LookupClient", lambda *_args, **_kwargs: FakeLookupClient(8))
    scheduler = create_scheduler(config, cache, "unused")
    worker = create_worker(config, cache)
    try:
        assert scheduler.get_num_new_matched_tokens(make_request(prompt_tokens=8), 0) == (4 if use_eagle else 7, False)
        assert worker._coordinator.group_ids == tuple(range(group_count))
        assert worker._coordinator.retention_interval == 32
    finally:
        scheduler.close()
        worker.close()
    assert backend.closed


def test_private_groups_are_projected_out_of_scheduler_worker_and_backend(configuration_case, monkeypatch) -> None:
    monkeypatch.setattr(connector_module, "resolve_backend_spec", lambda _name: object())
    monkeypatch.setattr(configuration.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_args: (4, 4))
    monkeypatch.setattr(configuration, "resolve_dcp_kv_cache_spec", lambda spec, _size: spec)
    monkeypatch.setattr(topology_module, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, ranks=[0]))
    monkeypatch.setattr(topology_module, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(
        connector_module, "LookupClient", lambda _address, **_kwargs: SimpleNamespace(close=lambda: None)
    )
    attention = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    private = CircularBufferSpec(block_size=4, num_kv_heads=1, head_size=1, head_size_v=0, dtype=torch.float32)
    cache_config = SimpleNamespace(
        num_blocks=8,
        transfer_group_ids=(0, 1),
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layers.0.attention"], kv_cache_spec=attention, is_eagle_group=False),
            SimpleNamespace(layer_names=["layers.1.private"], kv_cache_spec=private, is_eagle_group=False),
        ],
        prefix_cache_retention_interval=None,
    )
    config, _ = configuration_case
    scheduler = create_scheduler(config, cache_config, "unused")
    assert scheduler._config.transfer_group_ids == (0,)
    assert scheduler._config.has_private_state
    scheduler.close()
    pool_config = configuration.resolve_kv_pool_config(config, cache_config)
    topology = topology_module.resolve_worker_topology(config, cache_config, pool_config)
    assert tuple(group.group_id for group in topology.groups) == (0, 1)
    assert topology.config.transfer_group_ids == (0,)
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=topology)
    completion = store_one(worker, RangeStoreCommand("store", 0, 4, ((7,), (99,)), (b"a",), 4, 17))
    put_call = next(call for call in backend.calls if call[0] == "put")
    assert len(put_call[1]) == 1 and "@group:0@" in put_call[1][0]
    assert put_call[2:] == (((resources.kv_caches["layers.0.attention"][7].data_ptr(),),), ((32,),))
    assert len(completion.keys) == 1 and "group:0" in completion.keys[0]
    assert completion.source_release_confirmed and completion.succeeded
    worker.close()
    assert resources.closed


@pytest.mark.parametrize("invalid", ("boolean", "partial", "unsupported_route", "decode_only_layerwise"))
def test_connector_roles_reject_configuration_before_ranks_and_resource_creation(
    configuration_case, monkeypatch, invalid
):
    config, cache = configuration_case
    if invalid == "boolean":
        config.kv_transfer_config.kv_connector_extra_config["use_layerwise"] = "false"
        message = "use_layerwise must be a boolean"
    elif invalid == "partial":
        config.kv_transfer_config.kv_connector_extra_config["discard_partial_chunks"] = False
        message = "discard_partial_chunks=False"
    elif invalid == "decode_only_layerwise":
        config.kv_transfer_config.kv_role = "kv_consumer"
        config.kv_transfer_config.kv_connector_extra_config.update(use_layerwise=True, save_decode_cache=True)
        message = "Decode-only Store requires Bulk"
    else:
        config.kv_transfer_config.kv_connector_extra_config.update(use_layerwise=True, prefill_tp_size=2)
        config.kv_transfer_config.kv_role = "kv_consumer"
        message = "TP mismatch"
    acquire = Mock(side_effect=AssertionError("Configuration must precede runtime dependencies"))
    for module, names in (
        (topology_module, ("get_tp_group", "get_pp_group")),
        (connector_module, ("LookupClient", "resolve_backend_spec", "Resources")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, acquire)
    messages = []
    for create in (lambda: create_scheduler(config, cache, "unused"), lambda: create_worker(config, cache)):
        with pytest.raises(ValueError, match=message) as error:
            create()
        messages.append(str(error.value))
    assert messages[0] == messages[1]
    acquire.assert_not_called()
