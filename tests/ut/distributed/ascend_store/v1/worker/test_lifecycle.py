"""Worker step overlap, asynchronous completion, executor failure, and retryable close."""

from __future__ import annotations

import threading
from concurrent.futures import BrokenExecutor, wait
from dataclasses import replace

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    begin_step,
    make_worker,
)
from tests.ut.distributed.ascend_store.v1.worker.gva_fixtures import make_gva_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import (
    TokenRange,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    KVTransferStep,
    LoadCommand,
    RangeStoreCommand,
)


class _BlockingStoreBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.copy_started = (threading.Event(), threading.Event())
        self.allow_copy = (threading.Event(), threading.Event())
        self.copied_keys: list[tuple[str, ...]] = []

    def put(self, keys, addresses, sizes):
        index = len(self.copied_keys)
        self.copied_keys.append(tuple(keys))
        self.copy_started[index].set()
        assert self.allow_copy[index].wait(timeout=5)
        return super().put(keys, addresses, sizes)


@pytest.mark.parametrize("route", ("bulk", "async_bulk", "key_range", "gva"))
def test_empty_load_does_not_build_batches_or_submit_background_work(route, monkeypatch) -> None:
    if route == "gva":
        worker, resources, _ = make_gva_worker()
    else:
        worker, resources, _ = make_worker(layerwise=route == "key_range", async_load=route == "async_bulk")

    def unexpected_work(*args, **kwargs):
        pytest.fail("A step without Load commands must not prepare or submit Load work")

    monkeypatch.setattr(worker, "_build_load_batch", unexpected_work)
    monkeypatch.setattr(worker, "_submit_load", unexpected_work)
    try:
        for _ in range(2):
            begin_step(worker)
            worker.start_load()
            assert not worker.collect_load_result().completed_request_ids
            worker.finish_step()
            worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("released_request_ids", (None, frozenset()), ids=("drain", "nonblocking_poll"))
def test_store_executor_failure_still_blocks_worker_execution(monkeypatch, released_request_ids) -> None:
    worker, resources, _ = make_worker()

    def fail_executor(*_args):
        raise RuntimeError("unexpected Store executor failure")

    monkeypatch.setattr(worker, "_execute_bulk_store", fail_executor)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    worker.finish_step()
    pending = worker._pending_store_batch
    assert pending is not None
    _, unfinished = wait((pending.future,), timeout=2)
    assert not unfinished
    with pytest.raises(RuntimeError, match="unexpected Store executor failure"):
        worker.fence_previous_store(released_request_ids)
    assert worker._pending_store_batch is pending
    assert worker.take_released_store_job_ids() == set()
    worker.end_step()
    with pytest.raises(RuntimeError, match="fatal Store failure"):
        worker.begin_step(KVTransferStep())
    with pytest.raises(RuntimeError, match="fatal Store failure"):
        worker.close()
    assert worker._store_executor.stopped
    assert not resources.closed


def test_completed_store_without_job_evidence_retains_sources_and_blocks_future_steps(monkeypatch) -> None:
    worker, resources, _ = make_worker()
    monkeypatch.setattr(worker, "_execute_bulk_store", lambda *_args: ())
    command = RangeStoreCommand("request", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    worker.finish_step()
    pending = worker._pending_store_batch
    assert pending is not None
    assert pending.future.result(timeout=2) == ()

    with pytest.raises(RuntimeError, match="Store source release is missing"):
        worker.fence_previous_store(frozenset())

    assert worker._pending_store_batch is pending
    assert worker.take_released_store_job_ids() == set()
    worker.end_step()
    with pytest.raises(RuntimeError, match="fatal Store failure"):
        worker.begin_step(KVTransferStep())
    with pytest.raises(RuntimeError, match="fatal Store failure"):
        worker.close()
    assert not resources.closed


@pytest.mark.parametrize("leased", (False, True))
def test_bulk_decode_and_other_requests_overlap_store_but_shutdown_retains_sources(leased) -> None:
    backend = _BlockingStoreBackend()
    worker, resources, _ = make_worker(backend)
    first = RangeStoreCommand("first", TokenRange(0, 4), ((7,),), (b"a",), 4, 17 if leased else None)
    second = RangeStoreCommand("second", TokenRange(0, 4), ((9,),), (b"b",), 4, 18 if leased else None)
    fenced = threading.Event()
    closed = threading.Event()
    completions = []
    failures = []
    callers = []

    def fence_first():
        try:
            completions.extend(worker.fence_previous_store(frozenset(("first",))))
        except BaseException as error:
            failures.append(error)
        finally:
            fenced.set()

    def close_worker():
        try:
            worker.close()
        except BaseException as error:
            failures.append(error)
        finally:
            closed.set()

    try:
        begin_step(worker, store=(first,))
        worker.finish_step()
        worker.end_step()
        assert backend.copy_started[0].wait(timeout=2)

        # A decode that writes no source blocks must not wait for the unrelated ongoing copy.
        assert worker.fence_previous_store(frozenset()) == ()
        begin_step(worker)
        worker.finish_step()
        worker.end_step()
        assert worker.take_released_store_job_ids() == set()
        assert not backend.allow_copy[0].is_set()

        begin_step(worker, store=(second,))
        worker.finish_step()
        worker.end_step()
        caller = threading.Thread(target=fence_first)
        callers.append(caller)
        caller.start()
        assert not fenced.wait(timeout=0.05)
        backend.allow_copy[0].set()
        assert backend.copy_started[1].wait(timeout=2)
        assert fenced.wait(timeout=2), "Fencing first must not wait for the later second Store"
        assert not failures
        assert [completion.request_id for completion in completions] == ["first"]
        assert worker.take_released_store_job_ids() == ({17} if leased else set())
        assert worker.fence_previous_store(frozenset()) == ()
        assert worker.take_released_store_job_ids() == set()

        caller = threading.Thread(target=close_worker)
        callers.append(caller)
        caller.start()
        assert not closed.wait(timeout=0.05)
        assert not resources.closed
        backend.allow_copy[1].set()
        assert closed.wait(timeout=2)
        assert not failures
        assert worker.take_released_store_job_ids() == ({18} if leased else set())
        assert worker.take_released_store_job_ids() == set()
        assert len(backend.copied_keys) == 2 and backend.copied_keys[0] != backend.copied_keys[1]
        assert worker._pending_store_batch is None
        assert resources.closed
    finally:
        for allowed in backend.allow_copy:
            allowed.set()
        for caller in callers:
            caller.join(timeout=5)
            assert not caller.is_alive()
        worker.close()


def test_released_request_waits_for_its_latest_store_and_recovers_each_earlier_batch() -> None:
    backend = _BlockingStoreBackend()
    worker, resources, _ = make_worker(backend)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((7, 8),), (b"a", b"b"), 8, 17)
    latest = replace(command, store_range=TokenRange(4, 8), store_job_id=18)
    fenced = threading.Event()
    completions = []
    failures = []

    def fence_request():
        try:
            completions.extend(worker.fence_previous_store(frozenset(("request",))))
        except BaseException as error:
            failures.append(error)
        finally:
            fenced.set()

    caller = threading.Thread(target=fence_request)
    try:
        begin_step(worker, store=(command,))
        worker.finish_step()
        worker.end_step()
        assert backend.copy_started[0].wait(timeout=2)
        begin_step(worker, store=(latest,))
        worker.finish_step()
        worker.end_step()
        caller.start()
        backend.allow_copy[0].set()
        assert backend.copy_started[1].wait(timeout=2)
        assert not fenced.wait(timeout=0.05), "The request still owns its latest, unfinished Store"
        assert not resources.closed
        backend.allow_copy[1].set()
        assert fenced.wait(timeout=2)
        assert not failures
        assert [completion.store_job_id for completion in completions] == [17, 18]
        assert worker.take_released_store_job_ids() == {17, 18}
        assert worker.fence_previous_store(frozenset()) == ()
        assert worker.take_released_store_job_ids() == set()
        assert len(backend.copied_keys) == 2 and backend.copied_keys[0] != backend.copied_keys[1]
    finally:
        for allowed in backend.allow_copy:
            allowed.set()
        if caller.ident is not None:
            caller.join(timeout=5)
            assert not caller.is_alive()
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("failed_initializer", (1, 2), ids=("store", "load"))
def test_cache_binding_closes_partially_started_executors_and_resources(monkeypatch, failed_initializer) -> None:
    backend = FakeBackend()
    initializer_calls = 0

    def fail_initializer() -> None:
        nonlocal initializer_calls
        initializer_calls += 1
        backend.calls.append(("set_device",))
        if initializer_calls == failed_initializer:
            raise RuntimeError("thread initialization failed")

    monkeypatch.setattr(backend, "set_device", fail_initializer)
    worker, resources, _ = make_worker(
        backend,
        async_load=True,
        bind=False,
    )

    with pytest.raises(BrokenExecutor, match="initializer"):
        worker.bind_kv_caches({"cache": object()})

    assert initializer_calls == failed_initializer
    assert worker._store_executor is not None
    assert worker._store_executor.stopped
    assert worker._load_executor is not None
    assert worker._load_executor.stopped
    assert resources.closed


@pytest.mark.parametrize("channel", ("store", "load"))
def test_worker_close_keeps_resources_until_interrupted_shutdown_is_retried(channel, monkeypatch) -> None:
    worker, resources, _ = make_worker(async_load=channel == "load")
    command = RangeStoreCommand("request", TokenRange(0, 4), ((7,),), (b"a",), 4, 17)
    begin_step(worker, store=(command,))
    worker.finish_step()
    worker.end_step()
    executor = worker._store_executor if channel == "store" else worker._load_executor
    assert executor is not None
    pool = executor._pool
    shutdown = pool.shutdown

    def interrupt_shutdown(*args, **kwargs) -> None:
        raise KeyboardInterrupt("shutdown interrupted")

    try:
        monkeypatch.setattr(pool, "shutdown", interrupt_shutdown)
        with pytest.raises(KeyboardInterrupt, match="shutdown interrupted"):
            worker.close()

        assert worker._pending_store_batch is None
        assert worker.take_released_store_job_ids() == {17}
        assert not executor.stopped
        assert not resources.closed
    finally:
        monkeypatch.setattr(pool, "shutdown", shutdown)
        worker.close()

    assert executor.stopped
    assert resources.closed


def test_worker_owns_active_step_and_async_completion_lifecycle() -> None:
    worker, _, _ = make_worker(async_load=True, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))

    with pytest.raises(RuntimeError, match="has not begun"):
        worker.start_load()
    begin_step(worker, load=(command,))
    with pytest.raises(RuntimeError, match="has not ended"):
        worker.begin_step(KVTransferStep())
    worker.start_load()
    worker.end_step()
    begin_step(worker)
    _, unfinished = wait(worker._pending_loads, timeout=2)
    assert not unfinished
    result = worker.collect_load_result()
    assert result.completed_request_ids == {"request"}
    assert worker.collect_load_result().completed_request_ids == set()
    worker.end_step()
    worker.close()


def test_async_load_failure_drains_pending_work_and_rejects_overlap(monkeypatch) -> None:
    backend = FakeBackend()

    def fail_get(*_args):
        raise RuntimeError("backend get failed")

    monkeypatch.setattr(backend, "get", fail_get)
    worker, _, _ = make_worker(backend, async_load=True, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(worker, load=(command,))
    worker.start_load()
    worker.end_step()

    begin_step(worker, load=(replace(command, request_id="other"), command))
    with pytest.raises(RuntimeError, match="already has a pending asynchronous Load"):
        worker.start_load()
    assert worker._pending_load_request_ids == {"request"}
    _, unfinished = wait(worker._pending_loads, timeout=2)
    assert not unfinished
    result = worker.collect_load_result()
    assert result.completed_request_ids == {"request"}
    assert result.failed_block_ids == {1}
    assert [(location.group_id, location.block_id) for location in result.failed_locations] == [(0, 1)]
    worker.close()
