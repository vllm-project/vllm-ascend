"""Executor initialization, serial submission, failure propagation, and resource cleanup."""

import threading
from concurrent.futures import BrokenExecutor, Future

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.execution.executor import Executor


def test_executor_initializes_once_on_its_worker_and_returns_typed_results():
    initialized_on = []
    executor = Executor("test-executor", lambda: initialized_on.append(threading.get_ident()))

    def multiply(value: int, *, factor: int) -> tuple[int, int]:
        return value * factor, threading.get_ident()

    try:
        executor.start()
        executor.start()
        future = executor.submit(multiply, 3, factor=7)
        assert isinstance(future, Future)
        value, worker_id = future.result(timeout=2)
        assert value == 21
        assert initialized_on == [worker_id]
        assert worker_id != threading.get_ident()
    finally:
        executor.close()
    assert executor.stopped


def test_executor_start_waits_for_thread_initialization():
    initializing = threading.Event()
    release = threading.Event()
    started = threading.Event()

    def initialize():
        initializing.set()
        release.wait()

    executor = Executor("test-executor", initialize)

    def start():
        executor.start()
        started.set()

    caller = threading.Thread(target=start)
    try:
        caller.start()
        assert initializing.wait(timeout=2)
        assert not started.wait(timeout=0.05)
        release.set()
        assert started.wait(timeout=2)
    finally:
        release.set()
        caller.join(timeout=2)
        executor.close()


def test_executor_preserves_submission_order_and_close_drains_pending_tasks():
    running = threading.Event()
    release = threading.Event()
    closing = threading.Event()
    closed = threading.Event()
    executed = []
    executor = Executor("test-executor", lambda: None)

    def first():
        running.set()
        release.wait()
        executed.append(1)
        return 1

    def close():
        closing.set()
        executor.close()
        closed.set()

    caller = threading.Thread(target=close)
    try:
        executor.start()
        first_result = executor.submit(first)
        assert running.wait(timeout=2)
        second_result = executor.submit(executed.append, 2)
        third_result = executor.submit(executed.append, 3)
        caller.start()
        assert closing.wait(timeout=2)
        assert not closed.wait(timeout=0.05)
        assert not executor.stopped
        assert not second_result.done()
        release.set()
        assert closed.wait(timeout=2)
        assert first_result.result() == 1
        assert second_result.result() is third_result.result() is None
        assert executed == [1, 2, 3]
        assert executor.stopped
        with pytest.raises(RuntimeError, match="shutdown"):
            executor.submit(lambda: None)
    finally:
        release.set()
        if caller.ident is not None:
            caller.join(timeout=2)
        executor.close()


def test_executor_returns_task_exception_without_deciding_later_task_policy():
    error = RuntimeError("business operation failed")
    executor = Executor("test-executor", lambda: None)

    def fail():
        raise error

    try:
        executor.start()
        failed = executor.submit(fail)
        following = executor.submit(lambda: "next result")
        with pytest.raises(RuntimeError) as raised:
            failed.result(timeout=2)
        assert raised.value is error
        assert following.result(timeout=2) == "next result"
    finally:
        executor.close()


def test_executor_exposes_native_initialization_failure():
    def initialize():
        raise RuntimeError("thread initialization failed")

    executor = Executor("test-executor", initialize)
    try:
        with pytest.raises(BrokenExecutor):
            executor.start()
        with pytest.raises(BrokenExecutor):
            executor.submit(lambda: None)
    finally:
        executor.close()
    assert executor.stopped


def test_executor_close_before_start_is_repeatable():
    initialized = []
    executor = Executor("test-executor", lambda: initialized.append(True))
    executor.close()
    executor.close()
    assert executor.stopped
    assert not initialized


def test_executor_reports_stopped_only_after_shutdown_returns(monkeypatch):
    executor = Executor("test-executor", lambda: None)
    shutdown = executor._pool.shutdown

    def interrupted_shutdown(*args, **kwargs):
        raise KeyboardInterrupt("shutdown interrupted")

    try:
        executor.start()
        monkeypatch.setattr(executor._pool, "shutdown", interrupted_shutdown)
        with pytest.raises(KeyboardInterrupt):
            executor.close()
        assert not executor.stopped
    finally:
        monkeypatch.setattr(executor._pool, "shutdown", shutdown)
        executor.close()
    assert executor.stopped
