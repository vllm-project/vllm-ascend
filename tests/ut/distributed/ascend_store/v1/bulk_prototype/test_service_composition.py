"""Run complete services without Worker-side execution or business-state assembly."""

import threading
from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeLoadStartGate,
    FakeResources,
    make_backend_spec,
    make_coordinator,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.layerwise.fixtures import FakeGVABackend, make_gva_spec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import LayerwiseAccessKind
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.bulk_prototype.data.context import bind_context
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.bulk_prototype.service.load import LoadService
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.bulk_prototype.service.lookup import (
    LookupQuery,
    LookupService,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.bulk_prototype.service.store import StoreService
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.bulk_prototype.transfer.bulk import BulkTransfer
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.bulk_prototype.transfer.layerwise import (
    LayerwiseTransfer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.executor import (
    BulkExecutor,
    Executor,
    LayerwiseExecutor,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import LoadCommand, RangeStoreCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.results import DrainResult


@pytest.mark.parametrize("mode", ("bulk", "key_range", "gva"))
def test_services_own_complete_load_and_store_workflows(mode, monkeypatch):
    topology = make_topology(physical_layers=(2, 5, 9))
    coordinator = make_coordinator(topology)
    backend = FakeGVABackend() if mode == "gva" else FakeBackend()
    spec = make_gva_spec() if mode == "gva" else make_backend_spec()
    access = None if mode == "bulk" else LayerwiseAccessKind.GVA if mode == "gva" else LayerwiseAccessKind.KEY_RANGE
    resources = FakeResources(backend, spec, topology)
    load_executor, store_executor = Executor("TestLoad", backend.set_device), Executor("TestStore", backend.set_device)
    resources.bind_executors((store_executor, load_executor))
    backend_sessions = None if access is None else backend.create_sessions()
    if backend_sessions is not None:
        resources.bind_backend_sessions(backend_sessions)
    resources.prepare_kv_caches({})
    context = bind_context(topology, resources.kv_caches, "NHD", layerwise_access=access)
    transfer = (
        BulkTransfer(context, backend, BulkExecutor(load_executor, store_executor))
        if access is None
        else LayerwiseTransfer(
            context,
            backend,
            access,
            backend_sessions,
            LayerwiseExecutor(
                load_executor,
                store_executor,
                prefetch_layers=2,
                start_gate_factory=lambda: FakeLoadStartGate(opened=True),
            ),
        )
    )
    load = LoadService(context, coordinator, transfer)
    store = StoreService(context, coordinator, transfer, requires_exists=False)
    resources.start_executors()
    caller_thread = threading.get_ident()
    write_thread = load_executor.submit(threading.get_ident).result(timeout=2)
    read_thread = store_executor.submit(threading.get_ident).result(timeout=2)
    completions = []
    complete_load, complete_store = load._complete, store._complete

    def finish_load(batch):
        completions.append(("load", threading.get_ident()))
        return complete_load(batch)

    def finish_store(batch):
        completions.append(("store", threading.get_ident()))
        return complete_store(batch)

    monkeypatch.setattr(load, "_complete", finish_load)
    monkeypatch.setattr(store, "_complete", finish_store)
    transfer.begin_step()
    events = []

    def record_source_ready():
        assert threading.get_ident() == caller_thread
        event = FakeEvent()
        event.record()
        events.append(event)
        return event

    recorder = Mock(side_effect=record_source_ready)
    try:
        command = RangeStoreCommand("store", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17)
        store.submit((command,))
        for layer_id in (2, 5, 9):
            transfer.read_layer(layer_id, recorder)
            transfer.read_layer(layer_id, recorder)
        if access is not None:
            with pytest.raises(RuntimeError, match="must be sealed"):
                store.fence(frozenset({"store"}))
        transfer.finish_step(recorder)
        # Layer hooks and sealing stay in the transport; the service interprets completion.
        (result,) = store.fence()
        assert result.succeeded and result.source_release_confirmed and len(result.keys) == 2
        assert store.take_released_jobs() == {17}
        assert store.fence() == () and not store.take_released_jobs()
        assert recorder.call_count == (1 if access is None else 3)
        assert all(event.recorded and event.synchronized for event in events)

        lookup = LookupService(topology, backend, coordinator)
        assert lookup(LookupQuery(8, (0,), (b"a", b"b"))).available_end_token == 8
        load.submit((LoadCommand("load", 4, 8, ((3, 4),), (b"a", b"b")),))
        for layer_id in (2, 5, 9):
            transfer.wait_for_layer(layer_id, recorder)
        load_executor.submit(lambda: None).result(timeout=2)
        result = load.collect()
        assert result.completed_request_ids == {"load"} and not result.failed_locations
        assert not load.collect().completed_request_ids
        assert completions == [("store", read_thread), ("load", write_thread)]
        assert load.transfer is store.transfer is transfer
    finally:
        transfer.end_step(wait=False)
        load_drained, store_drained = load.drain(), store.drain()
        resources.stop_executors()
        if load_drained.safe_to_release and store_drained.safe_to_release:
            resources.close()
    assert load_drained == store_drained == DrainResult(True)
    assert resources.closed
