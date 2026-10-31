"""Bulk reuse, asynchronous independence, early exits and source safety."""

from __future__ import annotations

import threading
from concurrent.futures import BrokenExecutor, CancelledError, Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.align_state_fixtures import TensorBytesBackend
from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeEvent,
    make_backend_spec,
    make_coordinator,
    make_topology,
)
from tests.ut.distributed.ascend_store.v1.scheduler.fixtures import FakeBlocks, make_config, make_output, make_request
from tests.ut.distributed.ascend_store.v1.worker.fixtures import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.resources import Resources
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.messages import (
    KVTransferStep,
    LoadCommand,
    LookupRequest,
    RangeStoreCommand,
    StoreCommandBatch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.results import LoadResult
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.scheduler import AsynchronousScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker import Worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.data.batch import StoreBatch
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.bulk import BulkTransfer


def test_store_lookup_scheduler_async_load_round_trip_preserves_cpu_bytes_and_notifications():
    topology = make_topology()
    backend = TensorBytesBackend()
    caches = {
        layer.layer_names[0]: torch.arange(64, dtype=torch.float32).reshape(8, 4, 2, 1) + index * 1000
        for index, layer in enumerate(topology.groups[0].layers)
    }
    resources = Resources(backend, make_backend_spec(requires_exists_before_put=True), topology.transfer_groups)
    worker = Worker(topology, resources, coordinator=make_coordinator(topology), source_ready_event_factory=FakeEvent)
    worker.bind_kv_caches(caches)
    hashes = (b"a", b"b")
    query = LookupRequest(end_token=8, transfer_group_ids=(0,), block_hashes=hashes, local_available_tokens=0)

    def lookup(request):
        return worker.lookup(request)

    scheduler = AsynchronousScheduler(make_config(), SimpleNamespace(lookup=lookup, close=lambda: None))
    try:
        assert worker.lookup(query).available_end_token == 0
        store = RangeStoreCommand("producer", 0, 8, ((1, 2),), hashes, 8, 17)
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((store,))))
        worker.start_load()
        worker.finish_step()
        (stored,) = worker.fence_previous_store()
        assert stored.succeeded and stored.source_release_confirmed
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()
        assert worker.lookup(query).available_end_token == 8

        expected = {name: cache[[1, 2]].clone() for name, cache in caches.items()}
        for cache in caches.values():
            cache[[5, 6]] = 0
        request = make_request("consumer", prompt_tokens=8, tokens=9)
        request.block_hashes = list(hashes)
        assert scheduler.get_num_new_matched_tokens(request, 0) == (8, True)
        scheduler.confirm_allocation(request, FakeBlocks([5, 6]), 8)
        step = scheduler.build_step(make_output())
        assert tuple(command.request_id for command in step.load_commands) == ("consumer",)
        worker.begin_step(step)
        worker.start_load()
        worker.end_step()

        worker.begin_step(KVTransferStep())
        worker.close()
        result = worker.collect_load_result()
        assert result.completed_request_ids == {"consumer"}
        assert not result.failed_locations
        for name, cache in caches.items():
            torch.testing.assert_close(cache[[5, 6]], expected[name])
        assert not worker.collect_load_result().completed_request_ids
        assert not worker.take_released_store_job_ids()
        worker.end_step()
    finally:
        scheduler.close()
        worker.close()
    assert backend.closed and resources.kv_caches is None


def test_store_prepares_before_source_wait_while_load_and_lookup_make_progress(monkeypatch):
    entered, allow_store = threading.Event(), threading.Event()
    trace = []

    class SourceEvent(FakeEvent):
        def record(self):
            trace.append(("record", threading.get_ident()))
            super().record()

        def synchronize(self):
            trace.append(("wait", threading.get_ident()))
            entered.set()
            assert allow_store.wait(timeout=5)
            super().synchronize()

    event = SourceEvent()
    worker, resources, backend = make_worker(event_factory=lambda: event)
    arguments = BulkTransfer.arguments

    def prepare(transfer, selection):
        trace.append(("prepare", threading.get_ident()))
        return arguments(transfer, selection)

    monkeypatch.setattr(BulkTransfer, "arguments", prepare)
    try:
        store = RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17)
        worker.begin_step(KVTransferStep(store=StoreCommandBatch((store,))))
        worker.finish_step()
        worker.finish_step()
        assert entered.wait(timeout=2)
        assert [item[0] for item in trace] == ["record", "prepare", "wait"]
        assert trace[0][1] == threading.get_ident() and trace[1][1] != threading.get_ident()
        assert not any(call[0] == "put" for call in backend.calls)
        worker.end_step()

        worker.begin_step(KVTransferStep((LoadCommand("load", 0, 4, ((2,),), (b"b",)),)))
        worker.start_load()
        worker.transfer.executor._write.submit(lambda: None).result(timeout=2)
        assert worker.collect_load_result().completed_request_ids == {"load"}
        assert (
            worker.lookup(
                LookupRequest(end_token=4, transfer_group_ids=(0,), block_hashes=(b"b",), local_available_tokens=0)
            ).available_end_token
            == 4
        )
        assert worker.fence_previous_store(frozenset()) == ()
        assert not worker.take_released_store_job_ids()
        allow_store.set()
        assert len(worker.fence_previous_store(frozenset({"store"}))) == 1
        assert worker.take_released_store_job_ids() == {17}
        assert worker.fence_previous_store() == ()
        worker.end_step()
    finally:
        allow_store.set()
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("scenario", ("no_commands", "no_rows", "all_present", "store_disabled"))
def test_empty_store_exits_before_addresses_and_source_wait(scenario, monkeypatch):
    event = Mock(spec=FakeEvent)
    worker, resources, backend = make_worker(
        requires_exists=scenario == "all_present", event_factory=lambda: event, store=scenario != "store_disabled"
    )
    monkeypatch.setattr(BulkTransfer, "arguments", Mock(side_effect=AssertionError("No addresses are needed")))
    command = RangeStoreCommand("store", 0, 0 if scenario == "no_rows" else 4, ((1,),), (b"a",), 4, 17)
    commands = () if scenario == "no_commands" else (command,)
    try:
        worker.begin_step(KVTransferStep(store=StoreCommandBatch(commands)))
        worker.finish_step()
        results = worker.fence_previous_store()
        expected_jobs = {17} if scenario in ("no_rows", "all_present") else set()
        assert worker.take_released_store_job_ids() == expected_jobs
        assert all(result.succeeded and result.source_release_confirmed for result in results)
        event.synchronize.assert_not_called()
        assert not any(call[0] in ("put", "get") for call in backend.calls)
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


def test_empty_load_keeps_completion_delivery_without_submitting_or_materializing(monkeypatch):
    worker, resources, backend = make_worker()
    submit = Mock(side_effect=AssertionError("No transfer task is needed"))
    monkeypatch.setattr(worker.transfer.executor._write, "submit", submit)
    monkeypatch.setattr(BulkTransfer, "arguments", Mock(side_effect=AssertionError("No addresses are needed")))
    try:
        worker.begin_step(KVTransferStep((LoadCommand("empty", 0, 4, ((-1,),), (b"a",)),)))
        worker.start_load()
        with pytest.raises(RuntimeError, match="pending asynchronous Load"):
            worker.start_load()
        assert worker.collect_load_result().completed_request_ids == {"empty"}
        assert not worker.collect_load_result().completed_request_ids
        submit.assert_not_called()
        assert not any(call[0] == "get" for call in backend.calls)
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


def test_partial_store_prepares_final_parameters_and_ownership_once(monkeypatch):
    worker, resources, backend = make_worker(requires_exists=True)
    backend.presence = [0, 1]
    batches = []
    prepare_addresses = BulkTransfer.arguments

    def prepare(transfer, selection):
        arguments = prepare_addresses(transfer, selection)
        batches.append((selection, arguments))
        return arguments

    monkeypatch.setattr(BulkTransfer, "arguments", prepare)
    commands = (
        RangeStoreCommand("first", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17),
        RangeStoreCommand("duplicate", 0, 8, ((3, 4),), (b"a", b"b"), 8, 18),
    )
    try:
        worker.begin_step(KVTransferStep(store=StoreCommandBatch(commands)))
        worker.finish_step()
        first, duplicate = worker.fence_previous_store()
        ((selection, (keys, addresses, sizes)),) = batches
        assert len(selection.keys) == len(selection.sources) == 1
        assert selection.sources[0][2] == 0 and selection.sources[0][1] == 1
        assert first.keys == tuple(keys)
        assert first.result_codes == (0,)
        assert not duplicate.keys and not duplicate.result_codes
        assert worker.take_released_store_job_ids() == {17, 18}
        (copied,) = (call for call in backend.calls if call[0] == "put")
        assert copied[1] == tuple(keys)
        assert copied[2] == tuple(map(tuple, addresses))
        assert worker.fence_previous_store() == ()
        assert len(batches) == 1
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("start,end,hashes", ((0, 0, ()), (0, 8, ()), (4, 4, (b"a",))))
def test_lookup_skips_empty_backend_queries_but_keeps_the_local_prefix(start, end, hashes):
    worker, resources, backend = make_worker()
    try:
        expected = 4 if start == end == 4 else 0
        assert (
            worker.lookup(
                LookupRequest(end_token=end, transfer_group_ids=(0,), block_hashes=hashes, local_available_tokens=start)
            ).available_end_token
            == expected
        )
        assert not any(call[0] == "exists" for call in backend.calls)
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("failure", ("candidate", "arguments", "source_wait", "copy_code", "copy_exception"))
def test_store_failure_keeps_source_safety_and_recovery_boundaries(failure, monkeypatch):
    event = FakeEvent()
    worker, resources, backend = make_worker(event_factory=lambda: event)
    error = RuntimeError(f"failed at {failure}")
    if failure == "candidate":
        monkeypatch.setattr(worker.store, "_candidates", Mock(side_effect=error))
    elif failure == "arguments":
        monkeypatch.setattr(BulkTransfer, "arguments", Mock(side_effect=error))
    elif failure == "source_wait":
        monkeypatch.setattr(event, "synchronize", Mock(side_effect=error))
    elif failure == "copy_code":
        backend.put_result = [-7]
    else:
        backend.put_result = error
    command = RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    worker.finish_step()
    if failure == "copy_exception":
        with pytest.raises(RuntimeError, match="source release is unknown"):
            worker.fence_previous_store()
        assert not worker.take_released_store_job_ids()
        worker.end_step()
        with pytest.raises(RuntimeError, match="fatal Store failure"):
            worker.begin_step(KVTransferStep())
        with pytest.raises(RuntimeError, match="source release is unknown"):
            worker.close()
        assert not resources.closed
    else:
        try:
            (result,) = worker.fence_previous_store()
            assert not result.succeeded and result.source_release_confirmed
            assert worker.take_released_store_job_ids() == {17}
            worker.end_step()
            worker.begin_step(KVTransferStep())
            worker.end_step()
        finally:
            worker.close()
        assert resources.closed


def test_escaped_load_failure_propagates_without_fabricating_a_completion(monkeypatch):
    worker, resources, _ = make_worker()
    error = RuntimeError("escaped Load failure")
    monkeypatch.setattr(worker.transfer, "_write", Mock(side_effect=error))
    interpret = Mock(wraps=worker.load._complete)
    monkeypatch.setattr(worker.load, "_complete", interpret)
    worker.begin_step(KVTransferStep((LoadCommand("load", 0, 4, ((7,),), (b"a",)),)))
    worker.start_load()
    worker.transfer.executor._write.submit(lambda: None).result(timeout=2)
    with pytest.raises(RuntimeError) as raised:
        worker.collect_load_result()
    assert raised.value is error
    interpret.assert_not_called()
    assert len(worker.load._pending) == 1
    worker.end_step()
    with pytest.raises(RuntimeError, match="fatal Load failure"):
        worker.begin_step(KVTransferStep())
    with pytest.raises(RuntimeError) as closed:
        worker.close()
    assert closed.value is error
    assert resources.closed  # The injected task failed before entering Backend I/O.
    interpret.assert_not_called()


@pytest.mark.parametrize("cancelled", (False, True))
def test_load_does_not_downgrade_executor_failure_to_a_cache_miss(cancelled, monkeypatch):
    worker, resources, _ = make_worker()
    future: Future[LoadResult] = Future()
    if cancelled:
        future.cancel()
    else:
        future.set_exception(BrokenExecutor("executor unavailable"))
    monkeypatch.setattr(worker.transfer.executor._write, "submit", lambda *_args: future)
    try:
        worker.begin_step(KVTransferStep((LoadCommand("load", 0, 4, ((7,),), (b"a",)),)))
        worker.start_load()
        with pytest.raises(CancelledError if cancelled else BrokenExecutor):
            worker.collect_load_result()
        worker.end_step()
    finally:
        with pytest.raises(CancelledError if cancelled else BrokenExecutor):
            worker.close()
    assert resources.closed


@pytest.mark.parametrize("failure_stage", ("prepare_submit", "copy_submit", "index"))
def test_store_submission_failure_retains_launched_work_before_resource_release(failure_stage, monkeypatch):
    event = FakeEvent()
    event.record()
    worker, resources, backend = make_worker()
    error = RuntimeError("Store submission bookkeeping failed")
    executor = worker.transfer.executor._read
    if failure_stage == "prepare_submit":
        monkeypatch.setattr(executor, "submit", Mock(side_effect=error))
    elif failure_stage == "copy_submit":
        submit = executor.submit
        calls = 0

        def fail_copy_submission(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise error
            return submit(*args, **kwargs)

        monkeypatch.setattr(executor, "submit", fail_copy_submission)
    else:

        class FailedIndex(dict):
            def update(self, *_args, **_kwargs) -> None:
                raise error

        worker.store._latest = FailedIndex()
        backend.put_result = RuntimeError("copy source release unknown")
    command = RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17)
    with pytest.raises(RuntimeError, match="Store submission bookkeeping failed") as caught:
        worker.store.submit((command,))
    worker.transfer.read_all_layers(lambda: event)
    assert caught.value is error
    assert worker.store.pending == (failure_stage == "index")
    with pytest.raises(RuntimeError, match="Store submission bookkeeping failed"):
        worker.close()
    assert resources.closed == (failure_stage != "index")
    assert not worker.take_released_store_job_ids()
    if failure_stage == "index":
        (pending,) = worker.store._pending
        (result,) = pending.completion.result(timeout=2)
        assert not result.source_release_confirmed
    else:
        assert not any(call[0] == "put" for call in backend.calls)


@pytest.mark.parametrize("failure", ("arguments", "copy_code", "copy_exception"))
def test_filtered_store_jobs_keep_independent_results_and_source_release(failure, monkeypatch):
    worker, resources, backend = make_worker(requires_exists=True)
    backend.presence = [0, 1]
    if failure == "arguments":
        monkeypatch.setattr(BulkTransfer, "arguments", Mock(side_effect=RuntimeError("mapping failed")))
    elif failure == "copy_code":
        backend.put_result = [-7]
    else:
        backend.put_result = RuntimeError("copy release unknown")
    commands = (
        RangeStoreCommand("selected", 0, 8, ((1, 2),), (b"a", b"b"), 8, 17),
        RangeStoreCommand("duplicate", 0, 8, ((3, 4),), (b"a", b"b"), 8, 18),
        RangeStoreCommand("present", 0, 4, ((5,),), (b"b",), 4, 19),
    )
    worker.begin_step(KVTransferStep(store=StoreCommandBatch(commands)))
    worker.finish_step()
    if failure == "copy_exception":
        with pytest.raises(RuntimeError, match="source release is unknown"):
            worker.fence_previous_store()
        assert worker.take_released_store_job_ids() == {18, 19}
        with pytest.raises(RuntimeError, match="source release is unknown"):
            worker.close()
        assert not resources.closed
    else:
        try:
            selected, duplicate, present = worker.fence_previous_store()
            assert not selected.succeeded and selected.source_release_confirmed
            assert duplicate.succeeded and present.succeeded
            assert not duplicate.keys and not present.keys
            assert worker.take_released_store_job_ids() == {17, 18, 19}
        finally:
            worker.close()
        assert resources.closed


@pytest.mark.parametrize("all_present", (False, True))
def test_store_result_interpretation_error_is_not_converted_to_a_write_failure(all_present, monkeypatch):
    worker, resources, backend = make_worker(requires_exists=all_present)
    error = RuntimeError("invalid Store result interpretation")
    interpret = Mock(side_effect=error)
    monkeypatch.setattr(worker.store, "_results", interpret)
    command = RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17)
    event = FakeEvent()
    event.record()
    try:
        prepared = worker.transfer.executor._read.submit(worker.store._prepare, (command,))
        ready: Future[FakeEvent] = Future()
        ready.set_result(event)
        future = worker.transfer.executor._read.submit(
            worker.transfer._read, StoreBatch((command,)), prepared.result, ready.result, worker.store._complete
        )
        with pytest.raises(RuntimeError) as raised:
            future.result(timeout=2)
        assert raised.value is error
        assert any(call[0] == "put" for call in backend.calls) == (not all_present)
    finally:
        worker.close()
    assert resources.closed


def test_nonzero_load_codes_allow_fallback_and_later_submissions():
    worker, resources, backend = make_worker()
    backend.get_result = [-7]
    try:
        worker.begin_step(KVTransferStep((LoadCommand("failed", 0, 4, ((1,),), (b"a",)),)))
        worker.start_load()
        worker.transfer.executor._write.submit(lambda: None).result(timeout=2)
        result = worker.collect_load_result()
        assert result.completed_request_ids == {"failed"} and result.failed_block_ids == {1}
        assert worker.load.safe_to_release
        assert not worker.collect_load_result().completed_request_ids
        worker.end_step()
        backend.get_result = [0]
        worker.begin_step(KVTransferStep((LoadCommand("later", 0, 4, ((2,),), (b"b",)),)))
        worker.start_load()
        worker.close()
        result = worker.collect_load_result()
        assert result.completed_request_ids == {"later"} and not result.failed_locations
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("group_ids", ((0,), (1, 3)))
def test_backend_load_exception_retains_unfenced_destinations_and_blocks_new_work(group_ids, monkeypatch):
    worker, resources, backend = make_worker(topology=make_topology(group_ids=group_ids))
    error = RuntimeError("destination write completion is unknown")
    monkeypatch.setattr(backend, "load", Mock(side_effect=error))
    interpret = Mock(wraps=worker.load._complete)
    monkeypatch.setattr(worker.load, "_complete", interpret)
    blocks = tuple((1,) for _ in range(max(group_ids) + 1))
    worker.begin_step(KVTransferStep((LoadCommand("load", 0, 4, blocks, (b"a",)),)))
    worker.start_load()
    worker.transfer.executor._write.submit(lambda: None).result(timeout=2)
    with pytest.raises(RuntimeError) as raised:
        worker.collect_load_result()
    assert raised.value is error
    assert not worker.load.safe_to_release
    assert len(worker.load._pending) == 1
    interpret.assert_not_called()
    worker.end_step()
    with pytest.raises(RuntimeError, match="fatal Load failure"):
        worker.begin_step(KVTransferStep())
    with pytest.raises(RuntimeError) as closed:
        worker.close()
    assert closed.value is error
    assert resources.executors_stopped and not resources.closed
    assert resources.kv_caches is not None and not backend.closed
    interpret.assert_not_called()


@pytest.mark.parametrize("failure", ("interpretation", "invalid_count"))
def test_load_result_errors_propagate_after_confirmed_writes_without_fallback(failure, monkeypatch):
    worker, resources, backend = make_worker()
    if failure == "interpretation":
        interpret = Mock(side_effect=ValueError("result interpretation failed"))
        monkeypatch.setattr(worker.load, "_complete", interpret)
    else:
        monkeypatch.setattr(backend, "load", lambda *_args: (0, 0))
    worker.begin_step(KVTransferStep((LoadCommand("load", 0, 4, ((1,),), (b"a",)),)))
    worker.start_load()
    worker.transfer.executor._write.submit(lambda: None).result(timeout=2)
    with pytest.raises(ValueError) as raised:
        worker.collect_load_result()
    assert worker.load.safe_to_release
    with pytest.raises(ValueError) as closed:
        worker.close()
    assert closed.value is raised.value
    assert resources.closed
    worker.end_step()


@pytest.mark.parametrize("sources", ("ready", "pending", "mixed"))
def test_worker_records_one_source_boundary_in_the_correct_model_hook(sources):
    trace = []
    hook = "begin"

    class SourceEvent(FakeEvent):
        def record(self):
            trace.append(("record", hook, threading.get_ident()))
            super().record()

        def synchronize(self):
            assert self.recorded
            trace.append(("wait", hook, threading.get_ident()))
            super().synchronize()

    event = SourceEvent()
    factory = Mock(return_value=event)
    worker, resources, _ = make_worker(event_factory=factory)
    command = RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17)
    companion = RangeStoreCommand("companion", 0, 4, ((2,),), (b"b",), 4, 18)
    store = StoreCommandBatch(
        source_pending_commands=(command,) if sources != "ready" else (),
        source_ready_commands=(command,) if sources == "ready" else ((companion,) if sources == "mixed" else ()),
    )
    try:
        worker.begin_step(KVTransferStep(store=store))
        assert factory.call_count == (1 if sources == "ready" else 0)
        hook = "finish"
        worker.finish_step()
        worker.finish_step()
        results = worker.fence_previous_store()
        assert all(result.succeeded and result.source_release_confirmed for result in results)
        assert factory.call_count == 1
        records = [entry for entry in trace if entry[0] == "record"]
        assert records == [("record", "begin" if sources == "ready" else "finish", threading.get_ident())]
        waits = [entry for entry in trace if entry[0] == "wait"]
        assert len(waits) == 1 and waits[0][2] != threading.get_ident()
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


def test_store_only_consumes_the_boundary_supplied_by_its_caller(monkeypatch):
    worker, resources, _ = make_worker()
    event = FakeEvent()
    event.record()
    record = Mock(side_effect=AssertionError("Store must not record the producer stream"))
    monkeypatch.setattr(event, "record", record)
    command = RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17)
    try:
        assert worker.store.submit((command,))
        worker.transfer.read_all_layers(lambda: event)
        (result,) = worker.fence_previous_store()
        assert result.succeeded and result.source_release_confirmed and event.synchronized
        record.assert_not_called()
        assert worker.take_released_store_job_ids() == {17}
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("store_enabled", (False, True))
def test_worker_skips_event_creation_without_a_store_submission(store_enabled):
    factory = Mock(side_effect=AssertionError("No source boundary is needed"))
    worker, resources, _ = make_worker(event_factory=factory, store=store_enabled)
    commands = () if store_enabled else (RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17),)
    try:
        worker.begin_step(KVTransferStep(store=StoreCommandBatch(source_ready_commands=commands)))
        worker.finish_step()
        worker.end_step()
        factory.assert_not_called()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("failure", ("factory", "record"))
@pytest.mark.parametrize("already_ready", (False, True))
def test_worker_owns_source_boundary_failures_and_stops_further_submissions(failure, already_ready, monkeypatch):
    error = RuntimeError("failed to record source boundary")
    event = FakeEvent()
    if failure == "record":
        monkeypatch.setattr(event, "record", Mock(side_effect=error))
    factory = Mock(side_effect=error) if failure == "factory" else Mock(return_value=event)
    worker, resources, backend = make_worker(event_factory=factory)
    command = RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17)
    step = KVTransferStep(
        store=StoreCommandBatch(
            source_ready_commands=(command,) if already_ready else (),
            source_pending_commands=() if already_ready else (command,),
        )
    )
    if not already_ready:
        worker.begin_step(step)
    with pytest.raises(RuntimeError) as raised:
        worker.begin_step(step) if already_ready else worker.finish_step()
    assert raised.value is error
    assert worker.store.pending and not worker.take_released_store_job_ids()
    with pytest.raises(RuntimeError, match="fatal hook failure"):
        worker.start_load()
    with pytest.raises(RuntimeError) as closed:
        worker.close()
    assert closed.value is error
    assert resources.closed
    # Admission preceded the failed producer boundary, and no native read began.
    assert worker.take_released_store_job_ids() == {17}
    assert not any(call[0] in ("put", "get") for call in backend.calls)


def test_bulk_step_end_releases_admitted_sources_without_inventing_a_ready_boundary():
    factory = Mock(side_effect=AssertionError("The unfinished producer has no source boundary"))
    worker, resources, backend = make_worker(event_factory=factory)
    command = RangeStoreCommand("store", 0, 4, ((1,),), (b"a",), 4, 17)
    worker.begin_step(KVTransferStep(store=StoreCommandBatch((command,))))
    worker.end_step()
    worker.close()
    factory.assert_not_called()
    assert resources.closed and resources.executors_stopped
    assert worker.take_released_store_job_ids() == {17}
    assert not any(call[0] == "put" for call in backend.calls)
