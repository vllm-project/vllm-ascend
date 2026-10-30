"""Recovery planning, shared selection and real per-layer handoff boundaries."""

import threading
from dataclasses import replace
from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.align_state_fixtures import make_align_state_topology
from tests.ut.distributed.ascend_store.v1.gva_fixtures import FakeGVABackend, make_gva_spec
from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    FakeLoadStartGate,
    FakeResources,
    make_backend_spec,
    make_coordinator,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.worker.fixtures import make_worker as make_bulk
from tests.ut.distributed.ascend_store.v1.worker.fixtures import plan_load
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    LookupRequest,
    RangeStoreCommand,
    StoreCommandBatch,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import Worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.service.lookup import LookupService


def make_worker(*, backend=None, topology=None, event_factory=FakeEvent, gate_factory=None, requires_exists=False):
    backend = backend or FakeBackend()
    topology = topology or make_topology(physical_layers=(2, 5, 9))
    spec = (
        make_gva_spec()
        if isinstance(backend, FakeGVABackend)
        else make_backend_spec(requires_exists_before_put=requires_exists)
    )
    if isinstance(backend, FakeGVABackend):
        backend.exists = lambda keys: tuple(region is not None for region in backend.query_gva_regions(keys))
    resources = FakeResources(backend, spec, topology)
    worker = Worker(
        topology,
        resources,
        coordinator=make_coordinator(topology),
        layerwise=True,
        source_ready_event_factory=event_factory,
        start_gate_factory=gate_factory or (lambda: FakeLoadStartGate(opened=True)),
    )
    worker.bind_kv_caches({})
    return worker, resources, backend


def layer_names(worker):
    return tuple(name for name, _ in sorted(worker.context.layer_ids_by_name.items(), key=lambda item: item[1]))


def test_lookup_keeps_sparse_recovery_points_and_plans_only_the_missing_load_range():
    topology = make_align_state_topology(attention_block_size=16, state_block_size=32, hash_block_size=8, tp_size=1)
    backend = FakeBackend()
    backend.exists = lambda keys: tuple(not key.endswith("@04") for key in keys)
    query = LookupRequest(
        end_token=32,
        transfer_group_ids=(0, 1),
        block_hashes=(b"\x01", b"\x02", b"\x03", b"\x04"),
        local_available_tokens=16,
    )
    result = LookupService(topology, backend, make_coordinator(topology))(query)
    assert result.available_end_token == 24
    assert result.tail_key_boundaries == (TailKeyBoundary(0, 24), TailKeyBoundary(1, 24))
    command = plan_load(
        topology.config, query, result, request_id="r", block_ids_by_group=((1, 2), (3,)), request_token_count=40
    )
    assert command == LoadCommand("r", 16, 24, ((1, 2), (3,)), query.block_hashes, result.tail_key_boundaries)
    # With no valid local residency, the same recovery result requires the earlier data too.
    cold = plan_load(
        topology.config,
        replace(query, local_available_tokens=0),
        result,
        request_id="r",
        block_ids_by_group=((1, 2), (3,)),
        request_token_count=40,
    )
    assert cold.start_token == 0 and cold.end_token == command.end_token


def test_lookup_does_not_require_remote_keys_for_a_valid_local_prefix():
    topology = make_topology(physical_layers=(2,))
    backend = FakeBackend()
    backend.presence = (True, True)
    query = LookupRequest(
        end_token=16, transfer_group_ids=(0,), block_hashes=(b"a", b"b", b"c", b"d"), local_available_tokens=8
    )
    result = LookupService(topology, backend, make_coordinator(topology))(query)
    assert result.available_end_token == 16
    assert [key.rsplit("@", 1)[-1] for key in backend.calls[0][1]] == ["63", "64"]
    command = plan_load(
        topology.config, query, result, request_id="r", block_ids_by_group=((1, 2, 3, 4),), request_token_count=20
    )
    assert command.start_token == 8 and command.end_token == 16
    assert (
        plan_load(
            topology.config,
            replace(query, local_available_tokens=16),
            result,
            request_id="r",
            block_ids_by_group=((1, 2, 3, 4),),
            request_token_count=20,
        )
        is None
    )


def test_bulk_and_layerwise_share_selection_keys_and_source_ownership():
    topology = make_topology(group_ids=(0, 1), physical_layers=(2, 5))
    bulk, _, _ = make_bulk(topology=topology)
    layered, _, backend = make_worker(topology=topology, requires_exists=True)
    command = LoadCommand("r", 4, 12, ((1, 2, 3), (4, 5, 6)), (b"a", b"b", b"c"))
    stores = (RangeStoreCommand("r", 4, 12, command.block_ids_by_group, command.block_hashes, 12, 17),)
    try:
        first = bulk.load._prepare((command,))
        second = layered.load._prepare((command,))
        assert first == second
        assert bulk.store._candidates(stores) == layered.store._candidates(stores)
        backend.presence = (False,) * 4
        layered.begin_step(KVTransferStep(store=StoreCommandBatch(stores)))
        for name in layer_names(layered):
            layered.save_layer(name)
        layered.finish_step()
        layered.end_step()
        assert sum(call[0] == "exists" for call in backend.calls) == 1
        assert layered.take_released_store_job_ids() == {17}
    finally:
        bulk.close()
        layered.close()


def test_prefetch_waits_for_gate_and_completes_request_only_after_all_layers():
    gates = []

    def gate_factory():
        gate = FakeLoadStartGate()
        gates.append(gate)
        return gate

    worker, resources, backend = make_worker(topology=make_topology(physical_layers=(2, 5)), gate_factory=gate_factory)
    worker.begin_step(KVTransferStep((LoadCommand("r", 0, 4, ((1,),), (b"a",)),)))
    worker.start_load()
    first, second = layer_names(worker)
    worker.wait_for_layer_load(first)
    assert not worker.collect_load_result().completed_request_ids
    assert sum(call[0] == "batch_copy_get" for call in backend.calls) == 1
    assert gates and not gates[0]._opened.is_set()
    gates[0].open()
    worker.wait_for_layer_load(second)
    assert worker.collect_load_result().completed_request_ids == {"r"}
    worker.wait_for_layer_load(second)  # An alias/repeated hook cannot copy or report a second time.
    assert not worker.collect_load_result().completed_request_ids
    assert sum(call[0] == "batch_copy_get" for call in backend.calls) == 2
    assert sum(call[0] == "batch_get_start" for call in backend.calls) == 1
    assert sum(call[0] == "batch_get_end" for call in backend.calls) == 1
    worker.end_step()
    worker.close()
    assert resources.closed


def test_load_fans_one_remote_key_out_to_both_destination_owners():
    worker, _, backend = make_worker(topology=make_topology(physical_layers=(2,)))
    worker.begin_step(
        KVTransferStep((LoadCommand("a", 0, 4, ((1,),), (b"a",)), LoadCommand("b", 0, 4, ((3,),), (b"a",))))
    )
    worker.start_load()
    worker.wait_for_layer_load(layer_names(worker)[0])
    copied = next(call for call in backend.calls if call[0] == "batch_copy_get")
    assert len(copied[1]) == 1 and len(copied[2][0]) == 2
    assert worker.collect_load_result().completed_request_ids == {"a", "b"}
    worker.end_step()
    worker.close()


def test_store_records_each_source_on_hook_thread_and_publishes_after_last_layer():
    hooks = []

    class Event(FakeEvent):
        def record(self):
            hooks.append(threading.get_ident())
            super().record()

        def synchronize(self):
            assert self.recorded
            super().synchronize()

    worker, _, backend = make_worker(event_factory=Event)
    command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    names = layer_names(worker)
    worker.save_layer(names[0])
    worker.save_layer(names[0])
    worker.transfer.executor._read.submit(lambda: None).result(timeout=2)
    assert not any(call[0] == "batch_commit" for call in backend.calls)
    for name in names[1:]:
        worker.save_layer(name)
    worker.finish_step()
    assert hooks == [threading.get_ident()] * 3
    operations = [call[0] for call in backend.calls]
    assert operations.count("batch_put_start") == 1
    assert operations.count("batch_copy_put") == 3
    assert operations.index("batch_commit") > max(i for i, op in enumerate(operations) if op == "batch_copy_put")
    assert worker.take_released_store_job_ids() == {17}
    worker.end_step()
    worker.close()


@pytest.mark.parametrize("failure", ("code", "exception", "malformed"))
def test_layer_load_distinguishes_recoverable_codes_from_fatal_or_unknown_writes(failure):
    worker, resources, backend = make_worker(topology=make_topology(physical_layers=(2,)))
    backend.session_copy_result = (-3,) if failure == "code" else RuntimeError("copy uncertain")
    if failure == "malformed":
        worker.transfer._load_ranges = lambda *args: ()
    worker.begin_step(KVTransferStep((LoadCommand("r", 0, 4, ((1,),), (b"a",)),)))
    worker.start_load()
    if failure == "code":
        worker.wait_for_layer_load(layer_names(worker)[0])
        result = worker.collect_load_result()
        assert result.completed_request_ids == {"r"} and result.failed_block_ids == {1}
        worker.end_step()
        worker.close()
        assert resources.closed
    else:
        expected = ValueError if failure == "malformed" else RuntimeError
        with pytest.raises(expected):
            worker.wait_for_layer_load(layer_names(worker)[0])
        with pytest.raises(expected):
            worker.close()
        assert resources.closed == (failure == "malformed")


@pytest.mark.parametrize("failure", ("missing", "code", "exception"))
def test_incomplete_store_is_not_published_and_source_release_is_independent(failure):
    worker, resources, backend = make_worker(topology=make_topology(physical_layers=(2, 5)))
    backend.store_session_copy_result = (-3,) if failure == "code" else None
    if failure == "exception":
        backend.store_session_copy_result = RuntimeError("source uncertain")
    command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    names = layer_names(worker)
    for name in names[:1] if failure == "missing" else names:
        worker.save_layer(name)
    if failure == "exception":
        with pytest.raises(RuntimeError, match="source release is unknown"):
            worker.finish_step()
        assert not worker.take_released_store_job_ids()
        with pytest.raises(RuntimeError):
            worker.close()
        assert not resources.closed
    else:
        worker.finish_step()
        assert not any(call[0] == "batch_commit" for call in backend.calls)
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()
        worker.close()
        assert resources.closed


def test_close_cancels_unopened_gates_and_drains_without_deadlock():
    worker, resources, _ = make_worker(topology=make_topology(physical_layers=(2, 5)), gate_factory=FakeLoadStartGate)
    worker.begin_step(KVTransferStep((LoadCommand("r", 0, 4, ((1,),), (b"a",)),)))
    worker.start_load()
    worker.wait_for_layer_load(layer_names(worker)[0])
    with pytest.raises(RuntimeError, match="did not reach"):
        worker.close()
    assert resources.closed and resources.executors_stopped


def test_gva_uses_the_same_prepared_objects_and_only_publishes_after_all_layers():
    backend = FakeGVABackend()
    worker, resources, _ = make_worker(backend=backend, topology=make_topology(physical_layers=(2, 5)))
    store = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((store,))))
    for name in layer_names(worker):
        worker.save_layer(name)
    worker.finish_step()
    worker.end_step()
    assert worker.take_released_store_job_ids() == {17}
    worker.begin_step(KVTransferStep((LoadCommand("r", 0, 4, ((3,),), (b"a",)),)))
    worker.start_load()
    for name in layer_names(worker):
        worker.wait_for_layer_load(name)
    assert worker.collect_load_result().completed_request_ids == {"r"}
    worker.end_step()
    worker.close()
    assert resources.closed
    operations = [call[0] for call in backend.native_store.calls]
    assert operations.count("copy") == 4
    assert operations.count("publish") == 1
    assert not backend.native_store.leases


def test_terminal_native_failure_stops_later_copies_and_retains_source(monkeypatch):
    worker, resources, backend = make_worker(topology=make_topology(physical_layers=(2, 5)))
    copy = backend.copy_key_range_store
    calls = 0

    def copy_once(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("earlier reads are uncertain")
        return copy(*args)

    monkeypatch.setattr(worker.transfer, "_store_ranges", copy_once)
    command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    for name in layer_names(worker):
        worker.save_layer(name)
    with pytest.raises(RuntimeError, match="source release is unknown"):
        worker.finish_step()
    assert calls == 1 and not worker.take_released_store_job_ids()
    assert not any(call[0] in ("batch_commit", "batch_revoke") for call in backend.calls)
    with pytest.raises(RuntimeError):
        worker.close()
    assert not resources.closed


def test_hybrid_worker_joins_store_to_compute_start_without_load_calling_store():
    gates = []

    def gate_factory():
        gate = FakeLoadStartGate()
        gates.append(gate)
        return gate

    worker, _, backend = make_worker(
        topology=make_topology(group_ids=(0, 1), physical_layers=(2,)), gate_factory=gate_factory
    )
    store = RangeStoreCommand("r", 0, 4, ((1,), (2,)), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((store,))))
    worker.start_load()
    names = layer_names(worker)
    worker.wait_for_layer_load(names[0])
    assert gates and gates[0].on_start is not None
    assert not any(call[0] == "batch_copy_put" for call in backend.calls)
    gates[0].on_start()
    worker.save_layer(names[1])  # Both group names refer to the same physical layer.
    worker.finish_step()
    assert sum(call[0] == "batch_copy_put" for call in backend.calls) == 1
    assert worker.take_released_store_job_ids() == {17}
    worker.end_step()
    assert gates[0].on_start is None
    worker.close()


def test_all_ready_sources_use_one_hook_event_and_need_no_save_hooks():
    events = []

    def event_factory():
        event = FakeEvent()
        events.append(event)
        return event

    worker, _, backend = make_worker(event_factory=event_factory)
    command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch(source_ready_commands=(command,))))
    worker.finish_step()
    assert len(events) == 1 and events[0].recorded and events[0].synchronized
    assert sum(call[0] == "batch_copy_put" for call in backend.calls) == 3
    assert worker.take_released_store_job_ids() == {17}
    worker.end_step()
    worker.close()


@pytest.mark.parametrize("failure", ("rejected", "commit"))
def test_store_session_failures_leave_sources_releasable_without_publishing(failure):
    worker, resources, backend = make_worker(topology=make_topology(physical_layers=(2,)))
    if failure == "rejected":
        backend.store_session_start_result = [-7]
    else:
        backend.store_session_commit_result = [-7]
    command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    for name in layer_names(worker):
        worker.save_layer(name)
    worker.transfer.seal_reads()
    (result,) = worker.store.fence()
    assert not result.succeeded and result.source_release_confirmed
    assert result.result_codes == (-7,)
    assert worker.take_released_store_job_ids() == {17}
    worker.end_step()
    worker.close()
    assert resources.closed


def test_collecting_ready_stores_does_not_seal_an_active_layerwise_producer():
    worker, _, backend = make_worker(topology=make_topology(physical_layers=(2, 5)))
    command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    names = layer_names(worker)
    worker.save_layer(names[0])
    assert worker.fence_previous_store(frozenset()) == ()
    with pytest.raises(RuntimeError, match="must be sealed"):
        worker.fence_previous_store(frozenset({"r"}))
    worker.save_layer(names[1])
    worker.finish_step()
    assert sum(call[0] == "batch_commit" for call in backend.calls) == 1
    assert worker.take_released_store_job_ids() == {17}
    worker.end_step()
    worker.close()


def test_remote_cleanup_failure_does_not_hide_confirmed_source_release(monkeypatch):
    worker, resources, backend = make_worker(topology=make_topology(physical_layers=(2,)))
    backend.store_session_copy_result = (-3,)
    revoke = backend.revoke_key_range_store
    monkeypatch.setattr(backend, "revoke_key_range_store", Mock(side_effect=RuntimeError("cleanup unavailable")))
    command = RangeStoreCommand("r", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    worker.save_layer(layer_names(worker)[0])
    worker.transfer.seal_reads()
    (result,) = worker.store.fence()
    assert not result.succeeded and result.source_release_confirmed
    assert str(result.error) == "cleanup unavailable"
    assert worker.take_released_store_job_ids() == {17}
    worker.end_step()
    with pytest.raises(RuntimeError, match="cleanup unavailable"):
        worker.close()
    assert resources.executors_stopped and not resources.closed
    monkeypatch.setattr(backend, "revoke_key_range_store", revoke)
    worker.close()
    assert resources.closed


def test_close_cancels_write_gates_before_waiting_for_an_inflight_read():
    reading, release_read = threading.Event(), threading.Event()
    closed = threading.Event()
    gates, errors = [], []

    class SourceEvent(FakeEvent):
        def synchronize(self):
            reading.set()
            assert release_read.wait(timeout=5)
            super().synchronize()

    def gate_factory():
        gate = FakeLoadStartGate()
        gates.append(gate)
        return gate

    worker, resources, _ = make_worker(
        topology=make_topology(physical_layers=(2, 5)), event_factory=SourceEvent, gate_factory=gate_factory
    )
    command = RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep((LoadCommand("load", 0, 4, ((2,),), (b"a",)),), StoreCommandBatch((command,))))
    worker.start_load()
    worker.wait_for_layer_load(layer_names(worker)[0])
    worker.save_layer(layer_names(worker)[0])
    assert reading.wait(timeout=2)

    def close():
        try:
            worker.close()
        except BaseException as error:
            errors.append(error)
        finally:
            closed.set()

    thread = threading.Thread(target=close)
    thread.start()
    try:
        assert gates[0]._opened.wait(timeout=2)
        assert gates[0].on_start is None
        assert not closed.is_set() and not resources.closed
    finally:
        release_read.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert len(errors) == 1 and "did not reach physical layers" in str(errors[0])
    assert resources.closed and resources.executors_stopped
    assert worker.take_released_store_job_ids() == {17}
