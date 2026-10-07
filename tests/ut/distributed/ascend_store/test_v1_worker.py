"""Public KV Pool Worker routes and lifecycle evidence."""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import vllm_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector import (
    AscendStoreV1Connector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import (
    TokenRange,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    GVALayerwiseProjectionBinder,
    KeyRangeLayerwiseProjectionBinder,
    compile_bulk_projection_binder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupRequest,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    KVTransferStep,
    LoadCommand,
    LoadCommandBatch,
    RangeStoreCommand,
    StoreCommandBatch,
    StoreSourceReleaseMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.route import (
    KVPoolRouteSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import base as worker_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import (
    AsynchronousBulkWorker,
    SynchronousBulkWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise import (
    GVALayerwiseWorker,
    KeyRangeLayerwiseWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.result import (
    LoadResult,
)

from .v1.helpers import FakeBackend, FakeEvent, FakeResources, make_topology, make_worker


def _begin(worker, *, load=(), store=()) -> None:
    worker.begin_step(
        KVTransferStep(
            LoadCommandBatch(tuple(load)),
            StoreCommandBatch(tuple(store)),
        )
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


def test_lookup_exposes_backend_observation_and_preserves_best_effort_miss(caplog, monkeypatch) -> None:
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, store=False)
    request = LookupRequest(TokenRange(0, 8), (0,), (b"a", b"b"))

    assert worker.lookup(request).available_end_token == 8
    exists_call = next(call for call in backend.calls if call[0] == "exists")
    assert len(exists_call[1]) == 2
    assert all("@group:0@" in key for key in exists_call[1])

    def fail_lookup(keys):
        backend.calls.append(("failed_exists", tuple(keys)))
        raise RuntimeError("lookup unavailable")

    monkeypatch.setattr(backend, "exists", fail_lookup)
    assert worker.lookup(request).available_end_token == 0
    assert "Remote Lookup failed" in caplog.text

    worker.close()
    assert resources.closed


def test_synchronous_bulk_load_normalizes_failed_or_malformed_evidence() -> None:
    for native_result in ([-1], None, [0, 0], [True]):
        backend = FakeBackend()
        if native_result is None:

            def return_no_evidence(keys, addresses, sizes, *, backend=backend):
                backend.calls.append(("get", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
                return None

            backend.get = return_no_evidence
        else:
            backend.get_result = native_result
        worker, resources, _ = make_worker(backend, store=False)
        command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
        _begin(worker, load=(command,))

        worker.start_load()
        result = worker.collect_load_result()

        assert [call[0] for call in backend.calls].count("get") == 1, native_result
        assert result.failed_block_ids == {7}, native_result
        assert [(location.group_id, location.block_id) for location in result.failed_locations] == [(0, 7)]
        worker.end_step()
        worker.close()
        assert resources.closed


def test_synchronous_bulk_load_normalizes_backend_exception(monkeypatch) -> None:
    backend = FakeBackend()

    def fail_get(*_args):
        raise RuntimeError("get failed")

    monkeypatch.setattr(backend, "get", fail_get)
    worker, resources, _ = make_worker(backend, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
    _begin(worker, load=(command,))

    worker.start_load()
    result = worker.collect_load_result()

    assert result.failed_block_ids == {7}
    assert [(location.group_id, location.block_id) for location in result.failed_locations] == [(0, 7)]
    worker.end_step()
    worker.close()
    assert resources.closed


def test_layerwise_key_range_load_normalizes_backend_exception() -> None:
    backend = FakeBackend()
    backend.session_copy_result = RuntimeError("range copy failed")
    worker, resources, _ = make_worker(
        backend,
        layerwise=True,
        store=False,
        physical_layers=(0,),
    )
    command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
    _begin(worker, load=(command,))

    worker.start_load()
    worker.wait_for_layer_load("layers.0.group.0")
    result = worker.collect_load_result()

    assert result.failed_block_ids == {7}
    assert [(location.group_id, location.block_id) for location in result.failed_locations] == [(0, 7)]
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    worker.end_step()
    worker.close()
    assert resources.closed


def test_asynchronous_bulk_load_returns_before_copy_and_close_drains_completion(monkeypatch) -> None:
    backend = FakeBackend()
    copy_started = threading.Event()
    allow_copy = threading.Event()

    def blocking_get(keys, addresses, sizes):
        backend.calls.append(("get", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
        copy_started.set()
        assert allow_copy.wait(timeout=5)
        return [0] * len(keys)

    monkeypatch.setattr(backend, "get", blocking_get)
    worker, resources, _ = make_worker(backend, async_load=True, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
    _begin(worker, load=(command,))

    worker.start_load()
    assert copy_started.wait(timeout=2)
    assert not allow_copy.is_set()
    allow_copy.set()
    worker.close()

    result = worker.collect_load_result()
    assert result.completed_request_ids == {"request"}
    assert not result.failed_locations
    assert resources.closed


def test_multi_group_load_failure_reports_original_group_and_block() -> None:
    topology = make_topology(group_ids=(1, 3), physical_layers=(0,))
    backend = FakeBackend()
    backend.get_result = [0, -1]
    worker, resources, _ = make_worker(backend, topology=topology, store=False)
    command = LoadCommand(
        "request",
        TokenRange(0, 4),
        ((), (11,), (), (33,)),
        (b"a",),
    )
    _begin(worker, load=(command,))

    with pytest.raises(RuntimeError, match=r"cache group/block failures \[\(3, 33\)\]"):
        worker.start_load()

    worker.end_step()
    worker.close()
    assert resources.closed


def test_bulk_store_orders_source_ready_copy_and_close_fence() -> None:
    trace: list[str] = []

    class TraceEvent(FakeEvent):
        def record(self) -> None:
            trace.append("record")
            super().record()

        def synchronize(self) -> None:
            trace.append("synchronize")
            super().synchronize()

    class TraceBackend(FakeBackend):
        def put(self, keys, addresses, sizes):
            trace.append("put")
            return super().put(keys, addresses, sizes)

    backend = TraceBackend()
    worker, resources, _ = make_worker(
        backend,
        source_ready_event_factory=TraceEvent,
    )
    command = RangeStoreCommand("request", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    _begin(worker, store=(command,))

    worker.finish_step()
    worker.close()

    assert trace == ["record", "synchronize", "put"]
    assert worker.take_released_store_job_ids() == {17}
    assert resources.closed


@pytest.mark.parametrize("layerwise,async_load", ((False, False), (False, True), (True, False)))
def test_completed_store_failure_logs_releases_sources_and_allows_next_step(
    layerwise: bool, async_load: bool, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(worker_module, "logger", logging.getLogger(__name__))
    backend = FakeBackend()
    backend.presence = [0]
    if layerwise:
        backend.store_session_copy_result = [-9]
    else:
        backend.put_result = [-9]
    worker, resources, _ = make_worker(backend, layerwise=layerwise, async_load=async_load)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    _begin(worker, store=(command,))
    if layerwise:
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
    worker.finish_step()
    worker.fence_previous_store()

    assert "KV cache Store failed for request request (job 17)" in caplog.text
    assert worker.take_released_store_job_ids() == {17}
    assert worker._pending_store_batch is None
    assert worker.lookup(LookupRequest(TokenRange(0, 4), (0,), (b"a",))).available_end_token == 0
    if layerwise:
        assert "batch_commit" not in [call[0] for call in backend.calls]
        assert "batch_revoke" in [call[0] for call in backend.calls]
    worker.end_step()

    backend.put_result = None
    backend.store_session_copy_result = None
    next_command = RangeStoreCommand("next", TokenRange(0, 4), ((9,),), (b"b",), 4, 18)
    _begin(worker, store=(next_command,))
    if layerwise:
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
    worker.finish_step()
    worker.fence_previous_store()
    assert worker.take_released_store_job_ids() == {18}
    worker.close()
    assert resources.closed


@pytest.mark.parametrize("native_result", ([], [True], RuntimeError("put failed after address handoff")))
def test_bulk_store_unknown_evidence_keeps_registered_resources(native_result) -> None:
    backend = FakeBackend()
    backend.put_result = native_result
    worker, resources, _ = make_worker(backend)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    _begin(worker, store=(command,))
    worker.finish_step()

    with pytest.raises(RuntimeError, match="Store source release is unknown"):
        worker.close()

    assert worker.take_released_store_job_ids() == set()
    assert not resources.closed


def test_store_executor_failure_still_blocks_worker_execution(monkeypatch) -> None:
    worker, resources, _ = make_worker()

    def fail_executor(*_args):
        raise RuntimeError("unexpected Store executor failure")

    monkeypatch.setattr(worker._asynchronous_store_timeline, "_operation", fail_executor)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    _begin(worker, store=(command,))
    worker.finish_step()
    with pytest.raises(RuntimeError, match="KVPoolStoreExecutor failed"):
        worker.fence_previous_store()
    worker.end_step()
    with pytest.raises(RuntimeError, match="fatal Store failure"):
        worker.begin_step(KVTransferStep())
    with pytest.raises(RuntimeError, match="fatal Store failure"):
        worker.close()
    assert not resources.closed


def test_cache_binding_closes_partially_started_timelines_and_resources(monkeypatch) -> None:
    backend = FakeBackend()
    initializer_calls = 0

    def fail_load_initializer() -> None:
        nonlocal initializer_calls
        initializer_calls += 1
        backend.calls.append(("set_device",))
        if initializer_calls == 2:
            raise RuntimeError("load thread initialization failed")

    monkeypatch.setattr(backend, "set_device", fail_load_initializer)
    worker, resources, _ = make_worker(
        backend,
        async_load=True,
        bind=False,
    )

    with pytest.raises(RuntimeError, match="terminated during asynchronous Load"):
        worker.bind_kv_caches({"cache": object()})

    assert initializer_calls == 2
    assert worker._asynchronous_store_timeline is not None
    assert worker._asynchronous_store_timeline._executor.closed
    assert not worker._asynchronous_store_timeline._executor.is_alive()
    assert worker._asynchronous_load_timeline is not None
    assert worker._asynchronous_load_timeline._executor.closed
    assert not worker._asynchronous_load_timeline._executor.is_alive()
    assert resources.closed


def test_worker_close_keeps_resources_until_interrupted_join_is_retried(monkeypatch) -> None:
    worker, resources, _ = make_worker(async_load=True)
    assert worker._asynchronous_load_timeline is not None
    executor = worker._asynchronous_load_timeline._executor
    original_join = executor.join

    def interrupt_join() -> None:
        raise KeyboardInterrupt("join interrupted")

    monkeypatch.setattr(executor, "join", interrupt_join)
    with pytest.raises(KeyboardInterrupt, match="join interrupted"):
        worker.close()

    assert executor.closed
    assert not executor.stopped
    assert not resources.closed

    monkeypatch.setattr(executor, "join", original_join)
    worker.close()

    assert executor.stopped
    assert resources.closed


def test_initialization_rejects_active_kvpp_on_both_sides(monkeypatch) -> None:
    monkeypatch.setattr(vllm_adapter, "_kvpp_size", lambda _config: 2)
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: object())
    monkeypatch.setattr(vllm_adapter, "get_layerwise_protocol", lambda _name: None)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config={}),
        parallel_config=SimpleNamespace(world_size=1),
        model_config=SimpleNamespace(max_model_len=64),
        speculative_config=None,
    )
    cache_config = SimpleNamespace(prefix_cache_retention_interval=None)

    for factory in ("scheduler", "worker"):
        with pytest.raises(ValueError, match="does not support active KVPP"):
            if factory == "scheduler":
                vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
            else:
                vllm_adapter.resolve_kv_pool_route_spec(config, cache_config)


def test_initialization_allows_inert_kvpp_size_one(monkeypatch) -> None:
    monkeypatch.setattr(vllm_adapter, "_kvpp_size", lambda _config: 1)
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: object())
    monkeypatch.setattr(vllm_adapter, "get_layerwise_protocol", lambda _name: None)
    monkeypatch.setattr(vllm_adapter.kv_cache_utils, "resolve_kv_cache_block_sizes", lambda *_: (4, 4))
    monkeypatch.setattr(vllm_adapter, "RemoteLookup", lambda _address: SimpleNamespace(close=lambda: None))
    spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32)
    cache_config = SimpleNamespace(
        transfer_group_ids=(0,),
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_both", kv_connector_extra_config={}),
        parallel_config=SimpleNamespace(world_size=1),
        speculative_config=None,
    )

    scheduler = vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
    scheduler.close()


def test_initialization_runs_layerwise_topology_validation_on_both_sides(monkeypatch) -> None:
    monkeypatch.setattr(vllm_adapter, "_kvpp_size", lambda _config: 1)
    monkeypatch.setattr(vllm_adapter, "resolve_backend_spec", lambda _name: object())
    monkeypatch.setattr(vllm_adapter, "get_layerwise_protocol", lambda _name: object())

    def reject_context_parallelism(_protocol, _parallel_config, use_layerwise):
        assert use_layerwise
        raise ValueError("Mooncake block-key layerwise does not support context parallelism")

    monkeypatch.setattr(vllm_adapter, "validate_layerwise_topology", reject_context_parallelism)
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={"use_layerwise": True},
        ),
        parallel_config=SimpleNamespace(world_size=1),
        model_config=SimpleNamespace(max_model_len=64),
        speculative_config=None,
    )
    cache_config = SimpleNamespace(prefix_cache_retention_interval=None)

    for factory in ("scheduler", "worker"):
        with pytest.raises(ValueError, match="does not support context parallelism"):
            if factory == "scheduler":
                vllm_adapter.create_kv_pool_scheduler(config, cache_config, "unused")
            else:
                vllm_adapter.resolve_kv_pool_route_spec(config, cache_config)


@pytest.mark.parametrize("factory", ("scheduler", "worker"))
@pytest.mark.parametrize("kv_role, peer_role", (("kv_producer", "decode"), ("kv_consumer", "prefill")))
@pytest.mark.parametrize("context_axis", ("dcp", "pcp"))
def test_layerwise_factories_reject_peer_context_parallel_mismatch(factory, kv_role, peer_role, context_axis) -> None:
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role=kv_role,
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
                f"{peer_role}_{context_axis}_size": 1 if context_axis == "dcp" else 2,
            },
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2, prefill_context_parallel_size=1),
        speculative_config=None,
    )

    with pytest.raises(ValueError, match="same dcp_size/pcp_size"):
        if factory == "scheduler":
            vllm_adapter.create_kv_pool_scheduler(config, SimpleNamespace(), "unused")
        else:
            vllm_adapter.create_kv_pool_worker(config, SimpleNamespace())


@pytest.mark.parametrize(
    "kv_role, extra_config, use_layerwise",
    (
        ("kv_producer", {"decode_dcp_size": 2, "decode_pcp_size": 1}, True),
        ("kv_consumer", {"prefill_dcp_size": 2, "prefill_pcp_size": 1}, True),
        ("kv_producer", {}, True),
        ("kv_consumer", {}, True),
        ("kv_both", {"prefill_dcp_size": 1, "decode_pcp_size": 2}, True),
        ("kv_producer", {"decode_dcp_size": 1}, False),
        ("kv_consumer", {"prefill_pcp_size": 2}, False),
    ),
)
def test_peer_context_parallel_check_preserves_production_scope(kv_role, extra_config, use_layerwise) -> None:
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role=kv_role, kv_connector_extra_config=extra_config),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2, prefill_context_parallel_size=1),
        speculative_config=None,
    )

    vllm_adapter._validate_kv_pool_preflight(config, "memcache", use_layerwise=use_layerwise)


def test_worker_connector_hooks_delegate_and_drain_worker_results() -> None:
    calls: list[tuple[Any, ...]] = []
    released_store_job_ids = {17}

    def take_released_store_job_ids():
        released = set(released_store_job_ids)
        released_store_job_ids.clear()
        return released

    load_result = LoadResult(frozenset({"request"}), frozenset(), frozenset({7}))
    worker = SimpleNamespace(
        bind_kv_caches=lambda caches: calls.append(("bind", caches)),
        start_load=lambda: calls.append(("start_load",)),
        wait_for_layer_load=lambda layer: calls.append(("wait_layer", layer)),
        save_layer=lambda layer: calls.append(("save_layer", layer)),
        finish_step=lambda: calls.append(("finish",)),
        fence_previous_store=lambda: calls.append(("fence",)),
        take_released_store_job_ids=take_released_store_job_ids,
        collect_load_result=lambda: load_result,
        close=lambda: calls.append(("close",)),
    )
    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector.scheduler = None
    connector.worker = worker
    connector.lookup_server = None
    connector._pending_load_result = None
    connector._released_store_job_ids = set()

    connector.register_kv_caches({"layer": "cache"})
    connector.start_load_kv(None)
    connector.wait_for_layer_load("layer.0")
    connector.save_kv_layer("layer.0", None, None)
    connector.wait_for_save()
    connector.handle_preemptions(KVTransferStep())
    assert connector.get_finished(set()) == (set(), {"request"})
    assert connector.get_block_ids_with_load_errors() == {7}
    metadata = connector.build_connector_worker_meta()
    assert isinstance(metadata, StoreSourceReleaseMetadata)
    assert metadata.released_store_jobs == {17: 1}
    assert connector.build_connector_worker_meta() is None
    connector.shutdown()

    assert calls == [
        ("bind", {"layer": "cache"}),
        ("start_load",),
        ("wait_layer", "layer.0"),
        ("save_layer", "layer.0"),
        ("finish",),
        ("fence",),
        ("close",),
    ]


def test_worker_connector_rolls_back_metadata_when_worker_rejects_step(monkeypatch) -> None:
    calls: list[tuple[Any, ...]] = []
    step = KVTransferStep()

    monkeypatch.setattr(
        KVConnectorBase_V1,
        "bind_connector_metadata",
        lambda _self, metadata: calls.append(("bind", metadata)),
    )
    monkeypatch.setattr(
        KVConnectorBase_V1,
        "clear_connector_metadata",
        lambda _self: calls.append(("clear",)),
        raising=False,
    )

    def reject_step(metadata):
        calls.append(("begin", metadata))
        raise RuntimeError("previous Store failed")

    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector.worker = SimpleNamespace(begin_step=reject_step)

    with pytest.raises(RuntimeError, match="previous Store failed"):
        connector.bind_connector_metadata(step)

    assert calls == [("bind", step), ("begin", step), ("clear",)]
