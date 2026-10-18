"""Bulk execution modes preserve notification, thread ownership and independent I/O lanes."""

import threading
from concurrent.futures import BrokenExecutor, CancelledError, Future, wait

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import FakeBackend, FakeEvent, begin_step, make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.execution.executor import Executor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import LoadCommand, RangeStoreCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import BulkWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import worker as bulk_worker_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.completion import (
    LoadCompletion,
    StoreCompletion,
    StoreEvidence,
)


@pytest.mark.parametrize("load_async", (False, True), ids=("sync", "async"))
@pytest.mark.parametrize("store_enabled", (False, True), ids=("no_store", "store"))
def test_bulk_constructor_selects_only_the_required_executors(load_async, store_enabled):
    worker, resources, _ = make_worker(async_load=load_async, store=store_enabled, bind=False)
    try:
        assert type(worker) is BulkWorker
        assert isinstance(worker._load_executor, Executor) is load_async
        assert isinstance(worker._store_executor, Executor) is store_enabled
        if load_async and store_enabled:
            assert worker._load_executor is not worker._store_executor
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("load_async", (False, True), ids=("sync", "async"))
def test_bulk_load_uses_the_selected_thread_and_notification_mode(load_async, monkeypatch):
    backend = FakeBackend()
    caller_id = threading.get_ident()
    initialized_on: list[int] = []
    executed_on: list[int] = []
    monkeypatch.setattr(backend, "set_device", lambda: initialized_on.append(threading.get_ident()))

    def get(keys, addresses, sizes):
        executed_on.append(threading.get_ident())
        return [0] * len(keys)

    monkeypatch.setattr(backend, "get", get)
    worker, resources, _ = make_worker(backend, async_load=load_async, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
    try:
        begin_step(worker, load=(command,))
        worker.start_load()
        if load_async:
            assert len(worker._pending_loads) == 1
            worker._pending_loads[0].result(timeout=2)
            assert executed_on == initialized_on
            assert executed_on != [caller_id]
            assert worker._pending_load_request_ids == {"request"}
        else:
            assert executed_on == [caller_id]
            assert not initialized_on
            assert not worker._pending_loads
            assert not worker._pending_load_request_ids
        result = worker.collect_load_result()
        assert result.completed_request_ids == ({"request"} if load_async else set())
        assert not result.failed_locations
        assert not worker.collect_load_result().completed_request_ids
        assert not worker._pending_load_request_ids
    finally:
        worker.close()
    assert resources.closed


def test_bulk_load_can_complete_while_store_is_blocked(monkeypatch):
    backend = FakeBackend()
    store_started = threading.Event()
    release_store = threading.Event()
    store_threads: list[int] = []
    load_threads: list[int] = []

    def put(keys, addresses, sizes):
        store_threads.append(threading.get_ident())
        store_started.set()
        assert release_store.wait(timeout=5)
        return [0] * len(keys)

    def get(keys, addresses, sizes):
        load_threads.append(threading.get_ident())
        return [0] * len(keys)

    monkeypatch.setattr(backend, "put", put)
    monkeypatch.setattr(backend, "get", get)
    worker, resources, _ = make_worker(backend, async_load=True)
    store = RangeStoreCommand("store", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    load = LoadCommand("load", TokenRange(0, 4), ((9,),), (b"b",))
    try:
        begin_step(worker, store=(store,))
        worker.finish_step()
        worker.end_step()
        assert store_started.wait(timeout=2)
        begin_step(worker, load=(load,))
        worker.start_load()
        worker._pending_loads[0].result(timeout=2)
        assert worker.collect_load_result().completed_request_ids == {"load"}
        assert not release_store.is_set()
        assert worker.fence_previous_store(frozenset()) == ()
        assert not worker.take_released_store_job_ids()
        assert store_threads[0] != load_threads[0]
    finally:
        release_store.set()
        worker.close()
    assert resources.closed
    assert worker.take_released_store_job_ids() == {17}


def test_bulk_async_load_submit_failure_rolls_back_request_ownership():
    worker, resources, _ = make_worker(async_load=True, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
    try:
        worker._load_executor.close()
        begin_step(worker, load=(command,))
        with pytest.raises(RuntimeError, match="shutdown"):
            worker.start_load()
        assert not worker._pending_load_request_ids
        assert not worker._pending_loads
        assert not worker._load_batches
    finally:
        worker.close()
    assert resources.closed


def test_bulk_store_keeps_batch_identity_with_the_original_future_result(monkeypatch):
    source_ready = FakeEvent()
    worker, resources, _ = make_worker(source_ready_event_factory=lambda: source_ready)
    first = RangeStoreCommand("request", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    unleased = RangeStoreCommand("unleased", TokenRange(0, 4), ((9,),), (b"b",), 4)
    completions = (StoreCompletion("request", StoreEvidence((), True, True), 17),)
    observed: list[tuple[tuple[RangeStoreCommand, ...], FakeEvent]] = []

    def store(commands, source_ready_event):
        observed.append((commands, source_ready_event))
        return completions

    monkeypatch.setattr(worker, "_execute_bulk_store", store)
    try:
        begin_step(worker, store=(first, first, unleased))
        worker.finish_step()
        pending = worker._pending_store_batch
        assert pending is not None
        assert isinstance(pending.future, Future)
        assert pending.future.result(timeout=2) is completions
        assert pending.request_ids == {"request", "unleased"}
        assert pending.store_job_ids == {17}
        assert observed[0][0] == (first, first, unleased)
        assert observed[0][1] is source_ready
        assert source_ready.recorded
        assert worker.fence_previous_store() == completions
        assert worker.take_released_store_job_ids() == {17}
        assert worker.fence_previous_store() == ()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("observe_before_close", (False, True), ids=("close", "collect_then_close"))
def test_bulk_async_load_business_failure_remains_deliverable_across_shutdown(monkeypatch, observe_before_close):
    worker, resources, _ = make_worker(async_load=True, store=False)
    error = RuntimeError("unexpected Load operation failure")

    def fail(batch):
        raise error

    monkeypatch.setattr(worker, "_execute_bulk_load", fail)
    command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
    begin_step(worker, load=(command,))
    worker.start_load()
    _, unfinished = wait(worker._pending_loads, timeout=2)
    assert not unfinished
    if observe_before_close:
        result = worker.collect_load_result()
        assert result.completed_request_ids == {"request"}
        assert result.failed_block_ids == {7}
        assert not worker._pending_load_request_ids
    worker.close()
    assert worker._load_executor.stopped
    assert resources.closed
    result = worker.collect_load_result()
    assert result.completed_request_ids == (set() if observe_before_close else {"request"})
    assert result.failed_block_ids == (set() if observe_before_close else {7})
    assert not worker._pending_load_request_ids
    assert not worker._pending_loads
    assert not worker._load_batches
    assert not worker.collect_load_result().completed_request_ids


@pytest.mark.parametrize("failed_batch_index", (0, 1, 2), ids=("first", "middle", "last"))
def test_bulk_async_load_mixed_results_are_consumed_independently(monkeypatch, caplog, failed_batch_index):
    worker, resources, _ = make_worker(async_load=True, store=False)
    error = RuntimeError("unexpected Load operation failure")
    executed: list[str] = []
    accepted: list[str] = []
    execute_load = worker._execute_bulk_load
    accept_completion = worker._accept_load_completion
    request_ids = ("first", "middle", "last")

    def load(batch):
        request_id = batch.request_ids[0]
        executed.append(request_id)
        if request_id == request_ids[failed_batch_index]:
            raise error
        return execute_load(batch)

    def accept(completion):
        accepted.append(completion.request_id)
        accept_completion(completion)

    monkeypatch.setattr(worker, "_execute_bulk_load", load)
    monkeypatch.setattr(worker, "_accept_load_completion", accept)
    monkeypatch.setattr(bulk_worker_module.logger, "propagate", True)
    try:
        for index, request_id in enumerate(request_ids):
            command = LoadCommand(request_id, TokenRange(0, 4), ((7 + index,),), (request_id.encode(),))
            begin_step(worker, load=(command,))
            worker.start_load()
            worker.end_step()
        pending = tuple(worker._pending_loads)
        _, unfinished = wait(pending, timeout=2)
        assert not unfinished
        assert executed == list(request_ids)
        for index, future in enumerate(pending):
            if index == failed_batch_index:
                assert future.exception() is error
            else:
                assert tuple(completion.request_id for completion in future.result()) == (request_ids[index],)

        begin_step(worker)
        result = worker.collect_load_result()
        assert result.completed_request_ids == set(request_ids)
        assert result.failed_block_ids == {7 + failed_batch_index}
        assert [(location.group_id, location.block_id) for location in result.failed_locations] == [
            (0, 7 + failed_batch_index)
        ]
        assert accepted == list(request_ids)
        assert not worker._pending_loads
        assert not worker._pending_load_request_ids
        assert not worker._load_batches
        errors = [record for record in caplog.records if record.exc_info and record.exc_info[1] is error]
        assert len(errors) == 1
        assert str((request_ids[failed_batch_index],)) in errors[0].getMessage()
        for _ in range(2):
            result = worker.collect_load_result()
            assert not result.completed_request_ids
            assert not result.failed_block_ids
            assert not result.failed_locations

        # The failed request no longer owns a pending slot and can submit another batch.
        worker.end_step()
        monkeypatch.setattr(worker, "_execute_bulk_load", execute_load)
        retry_id = request_ids[failed_batch_index]
        begin_step(worker, load=(LoadCommand(retry_id, TokenRange(0, 4), ((7,),), (b"retry",)),))
        worker.start_load()
        _, unfinished = wait(worker._pending_loads, timeout=2)
        assert not unfinished
        result = worker.collect_load_result()
        assert result.completed_request_ids == {retry_id}
        assert not result.failed_block_ids
    finally:
        worker.close()
    assert worker._load_executor.stopped
    assert resources.closed
    assert not worker._pending_loads
    assert not worker._pending_load_request_ids
    assert not worker._load_batches


def test_bulk_async_load_business_failure_covers_every_request_in_its_batch(monkeypatch):
    worker, resources, _ = make_worker(async_load=True, store=False)

    def fail(batch):
        raise RuntimeError("business operation failed")

    monkeypatch.setattr(worker, "_execute_bulk_load", fail)
    commands = (
        LoadCommand("first", TokenRange(0, 8), ((1, 2),), (b"a", b"b")),
        LoadCommand("second", TokenRange(0, 4), ((3,),), (b"c",)),
    )
    try:
        begin_step(worker, load=commands)
        worker.start_load()
        _, unfinished = wait(worker._pending_loads, timeout=2)
        assert not unfinished
        result = worker.collect_load_result()
        assert result.completed_request_ids == {"first", "second"}
        assert result.failed_block_ids == {1, 2, 3}
        assert {(location.group_id, location.block_id) for location in result.failed_locations} == {
            (0, 1),
            (0, 2),
            (0, 3),
        }
        assert not worker._pending_load_request_ids
        assert not worker._pending_loads
        assert not worker._load_batches
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("cancelled", (False, True), ids=("broken_executor", "cancelled_future"))
def test_bulk_async_load_execution_error_leaves_later_batches_queued(monkeypatch, cancelled):
    worker, resources, _ = make_worker(async_load=True, store=False)
    first, failed, last = Future(), Future(), Future()
    error = BrokenExecutor("executor unavailable")
    first.set_result((LoadCompletion("first", ()),))
    if cancelled:
        failed.cancel()
    else:
        failed.set_exception(error)
    last.set_result((LoadCompletion("last", ()),))
    submitted = iter((first, failed, last))
    monkeypatch.setattr(worker._load_executor, "submit", lambda *_args: next(submitted))
    try:
        for index, request_id in enumerate(("first", "middle", "last")):
            command = LoadCommand(request_id, TokenRange(0, 4), ((index + 1,),), (request_id.encode(),))
            begin_step(worker, load=(command,))
            worker.start_load()
            worker.end_step()
        begin_step(worker)
        with pytest.raises(CancelledError if cancelled else BrokenExecutor) as raised:
            worker.collect_load_result()
        if not cancelled:
            assert raised.value is error
        assert tuple(worker._pending_loads) == (last,)
        assert tuple(worker._load_batches) == (last,)
        assert worker._load_batches[last].request_ids == ("last",)
        assert "last" in worker._pending_load_request_ids
    finally:
        worker.close()
    assert resources.closed


def test_bulk_async_load_consumes_finished_batches_once_and_preserves_late_notifications(monkeypatch):
    backend = FakeBackend()
    second_started = threading.Event()
    release_second = threading.Event()
    get_calls = 0

    def get(keys, addresses, sizes):
        nonlocal get_calls
        get_calls += 1
        if get_calls == 2:
            second_started.set()
            assert release_second.wait(timeout=5)
        return [0] * len(keys)

    monkeypatch.setattr(backend, "get", get)
    worker, resources, _ = make_worker(backend, async_load=True, store=False)
    first = LoadCommand("first", TokenRange(0, 4), ((7,),), (b"a",))
    companion = LoadCommand("companion", TokenRange(0, 4), ((8,),), (b"b",))
    second = LoadCommand("second", TokenRange(0, 4), ((9,),), (b"c",))
    try:
        begin_step(worker, load=(first, companion))
        worker.start_load()
        worker._pending_loads[0].result(timeout=2)
        worker.end_step()
        begin_step(worker, load=(second,))
        worker.start_load()
        assert second_started.wait(timeout=2)

        completed = worker.collect_load_result()
        assert completed.completed_request_ids == {"first", "companion"}
        assert not completed.failed_locations
        assert worker._pending_load_request_ids == {"second"}
        assert len(worker._pending_loads) == 1
        assert not worker.collect_load_result().completed_request_ids
        worker.end_step()
        begin_step(worker)
        release_second.set()
        worker.close()

        assert resources.closed
        completed = worker.collect_load_result()
        assert completed.completed_request_ids == {"second"}
        assert not completed.failed_locations
        assert not worker.collect_load_result().completed_request_ids
        assert not worker._pending_load_request_ids
        assert not worker._pending_loads
    finally:
        release_second.set()
        worker.close()
