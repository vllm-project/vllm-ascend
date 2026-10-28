"""Serial execution and step timing preserve ordering, results, and resource safety."""

import threading
from concurrent.futures import BrokenExecutor, ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import FakeLoadStartGate
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.execution import Executor, Session


def test_executor_initializes_once_on_its_worker_and_returns_typed_results():
    initialized_on = []
    executor = Executor("test-executor", lambda: initialized_on.append(threading.get_ident()))

    def multiply(value: int, *, factor: int) -> tuple[int, int]:
        return value * factor, threading.get_ident()

    try:
        executor.start()
        executor.start()
        future = executor.submit(multiply, 3, factor=7)
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


@pytest.fixture
def queues():
    load = Executor("test-session-load", lambda: None)
    store = Executor("test-session-store", lambda: None)
    load.start()
    store.start()
    try:
        yield load, store
    finally:
        load.close()
        store.close()


@pytest.fixture
def make_session(queues):
    load, store = queues
    sessions = []

    def create(order=(), *, depth=2, gate_factory=FakeLoadStartGate):
        gates = []

        def create_gate():
            gate = gate_factory()
            gates.append(gate)
            return gate

        session: Session[int] = Session(load.submit, store.submit, depth, create_gate)
        session.load_order = order
        sessions.append(session)
        return session, gates

    try:
        yield create
    finally:
        for session in sessions:
            session.end_step(wait=False)


@pytest.mark.parametrize(
    "depth,copied_counts",
    (
        (1, (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)),
        (2, (2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)),
        (8, (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)),
    ),
)
def test_sparse_load_window_preserves_non_load_hooks_aliases_and_tail(depth, copied_counts, make_session, queues):
    session, gates = make_session((2, 5, 9), depth=depth)
    copied = []
    finish = Mock()

    def load(layer_id):
        copied.append(layer_id)
        return layer_id

    steps = (
        (0, None, (2, 5, 9)),
        (0, None, (2, 5, 9)),
        (2, 2, (5, 9)),
        (2, None, (5, 9)),
        (3, None, (5, 9)),
        (3, None, (5, 9)),
        (5, 5, (9,)),
        (5, None, (9,)),
        (8, None, (9,)),
        (9, 9, ()),
        (9, None, ()),
        (12, None, ()),
    )
    for (layer_id, expected, pending), copied_count in zip(steps, copied_counts, strict=True):
        assert session.wait_for_layer(layer_id, load, finish) == expected
        assert session.pending_load_layers == pending
        for gate in gates:
            gate.open()
        queues[0].submit(lambda: None).result(timeout=2)
        assert copied == [2, 5, 9][:copied_count]
    finish.assert_called_once_with()


@pytest.mark.parametrize("failing_layer", (2, 5, 9))
def test_load_error_is_claimed_once_and_abort_prevents_later_io(failing_layer, make_session, queues):
    session, gates = make_session((2, 5, 9))
    error = RuntimeError("layer Load failed")
    attempted = []
    finish = Mock()

    def load(layer_id):
        attempted.append(layer_id)
        if layer_id == failing_layer:
            raise error
        return layer_id

    for layer_id in (2, 5, 9):
        if layer_id == failing_layer:
            break
        assert session.wait_for_layer(layer_id, load, finish) == layer_id
        for gate in gates:
            gate.open()
    with pytest.raises(RuntimeError) as raised:
        session.wait_for_layer(failing_layer, load, finish)
    assert raised.value is error
    assert session.pending_load_layers == tuple(layer for layer in (2, 5, 9) if layer > failing_layer)
    assert session.wait_for_layer(failing_layer, load, finish) is None
    session.abort_load(finish)
    assert queues[0].submit(lambda: "healthy").result(timeout=2) == "healthy"
    assert attempted == [layer for layer in (2, 5, 9) if layer <= failing_layer]
    finish.assert_called_once_with()


def test_hooks_without_load_work_leave_operations_and_cleanup_unused(make_session):
    session, gates = make_session()
    load, finish = Mock(), Mock()
    for layer_id in (2, 2, 5, 9, 9):
        assert session.wait_for_layer(layer_id, load, finish) is None
    assert session.pending_load_layers == ()
    assert not gates
    load.assert_not_called()
    finish.assert_not_called()


def test_abort_cancels_shared_load_gate_once_and_preserves_store_only_callback(make_session, queues):
    class CountingGate(FakeLoadStartGate):
        def __init__(self):
            super().__init__()
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1
            self.on_start = None
            super().cancel()

    session, gates = make_session((2, 5, 9), depth=3, gate_factory=CountingGate)
    release_queue = threading.Event()
    load, finish, store_entry = Mock(), Mock(), Mock()
    queues[0].submit(release_queue.wait, 5)
    try:
        session.wait_for_layer(0, load, finish)
        session.bind_store_to_compute_start(99, store_entry)
        session.abort_load(wait=False)
        # Hold the queue so every shared dependency is still pending during cancellation.
        assert [gate.cancel_calls for gate in gates] == [1, 0]
        assert gates[0].on_start is None
        gates[1].on_start()
        store_entry.assert_called_once_with()
        release_queue.set()
        queues[0].submit(lambda: None).result(timeout=2)
        session.abort_load()
        load.assert_not_called()
        finish.assert_called_once_with()
        assert [gate.cancel_calls for gate in gates] == [1, 0]
    finally:
        release_queue.set()


@pytest.mark.parametrize("first_load_fails", (False, True))
def test_abort_waits_for_all_submitted_loads_without_cleanup(first_load_fails, make_session):
    session, gates = make_session((2, 5, 9))
    load_started, load_finished = threading.Event(), threading.Event()
    release_load, abort_started = threading.Event(), threading.Event()
    attempted = []
    finish = Mock()

    def load(layer_id):
        attempted.append(layer_id)
        if layer_id == 2:
            if first_load_fails:
                raise RuntimeError("first Load failed")
            return layer_id
        load_started.set()
        assert release_load.wait(timeout=5)
        load_finished.set()
        return layer_id

    def run_hooks():
        session.wait_for_layer(0, load, finish)
        gates[0].open()
        assert load_started.wait(timeout=2)
        abort_started.set()
        session.abort_load()

    with ThreadPoolExecutor(max_workers=1) as hooks:
        aborted = hooks.submit(run_hooks)
        try:
            assert abort_started.wait(timeout=2)
            with pytest.raises(TimeoutError):
                aborted.result(timeout=0.05)
            release_load.set()
            aborted.result(timeout=2)
            assert load_finished.is_set()
            assert attempted == [2, 5]
            assert session.pending_load_layers == (2, 5, 9)
            finish.assert_not_called()
        finally:
            release_load.set()


@pytest.mark.parametrize("with_cleanup", (False, True))
@pytest.mark.parametrize("exit_operation", ("abort_load", "end_step"))
def test_nonblocking_exit_keeps_running_load_and_optional_cleanup_asynchronous(
    exit_operation, with_cleanup, make_session, queues
):
    session, gates = make_session((2, 5), depth=1)
    load_started, load_finished, release_load = threading.Event(), threading.Event(), threading.Event()
    finish = Mock()

    def load(layer_id):
        load_started.set()
        assert release_load.wait(timeout=5)
        load_finished.set()
        return layer_id

    try:
        session.wait_for_layer(0, load, finish)
        gates[0].open()
        assert load_started.wait(timeout=2)
        getattr(session, exit_operation)(finish if with_cleanup else None, wait=False)
        assert not load_finished.is_set()
        finish.assert_not_called()
        release_load.set()
        queues[0].submit(lambda: None).result(timeout=2)
        session.wait_load_finished()
        assert load_finished.is_set()
        assert finish.call_count == int(with_cleanup)
    finally:
        release_load.set()


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_abort_waits_for_cleanup_and_propagates_its_error(cleanup_fails, make_session):
    session, _ = make_session((2, 5), depth=1)
    cleanup_started, release_cleanup = threading.Event(), threading.Event()
    error = RuntimeError("cleanup failed")
    trace: list[int] = []

    def finish():
        assert trace == [2]
        cleanup_started.set()
        assert release_cleanup.wait(timeout=5)
        if cleanup_fails:
            raise error

    def run_hooks():
        session.wait_for_layer(2, trace.append, finish)
        session.abort_load(finish)

    with ThreadPoolExecutor(max_workers=1) as hooks:
        aborted = hooks.submit(run_hooks)
        try:
            assert cleanup_started.wait(timeout=2)
            assert not aborted.done()
            release_cleanup.set()
            if cleanup_fails:
                with pytest.raises(RuntimeError) as raised:
                    aborted.result(timeout=2)
                assert raised.value is error
            else:
                aborted.result(timeout=2)
        finally:
            release_cleanup.set()


def test_end_step_detaches_callbacks_and_drains_load_without_waiting_for_store(make_session, queues):
    session, gates = make_session((2, 5))
    store_started, store_finished, release_store = threading.Event(), threading.Event(), threading.Event()
    entry, finish = Mock(), Mock()
    copied: list[int] = []

    def store(source_ready, layer_id):
        store_started.set()
        assert release_store.wait(timeout=5)
        store_finished.set()

    opening = queues[1].submit(lambda: None)
    try:
        session.wait_for_layer(2, copied.append, finish)
        session.bind_store_to_compute_start(2, entry)
        session.bind_store_to_compute_start(99, entry)
        session.store_layer(9, opening, object, store)
        assert store_started.wait(timeout=2)
        session.end_step(finish)
        session.end_step(finish)
        assert copied == [2]
        assert all(gate.on_start is None for gate in gates)
        finish.assert_called_once_with()
        entry.assert_not_called()
        assert not store_finished.is_set()
        closing = session.finish_store(opening, lambda _opening, layers: tuple(layers))
        assert not closing.done()
        release_store.set()
        assert closing.result(timeout=2) == (9,)
        assert store_finished.is_set()
    finally:
        release_store.set()


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_last_load_waits_for_cleanup_and_preserves_result_or_error(cleanup_fails, make_session):
    session, _ = make_session((2, 5), depth=1)
    result, error = object(), RuntimeError("cleanup failed")
    cleanup_started, release_cleanup = threading.Event(), threading.Event()
    trace = []

    def load(layer_id):
        trace.append(layer_id)
        return result

    def cleanup():
        cleanup_started.set()
        assert release_cleanup.wait(timeout=5)
        trace.append("finished")
        if cleanup_fails:
            raise error

    finish = Mock(side_effect=cleanup)
    assert session.wait_for_layer(2, load, finish) is result
    with ThreadPoolExecutor(max_workers=1) as hooks:
        waiting = hooks.submit(session.wait_for_layer, 5, load, finish)
        try:
            assert cleanup_started.wait(timeout=2)
            assert not waiting.done()
            release_cleanup.set()
            if cleanup_fails:
                with pytest.raises(RuntimeError) as raised:
                    waiting.result(timeout=2)
                assert raised.value is error
                with pytest.raises(RuntimeError):
                    session.abort_load(finish)
            else:
                assert waiting.result(timeout=2) is result
                session.abort_load(finish)
            assert session.wait_for_layer(5, load, finish) is None
            assert trace == [2, 5, "finished"]
            finish.assert_called_once_with()
        finally:
            release_cleanup.set()


@pytest.mark.parametrize("layer_ids", ((), (2, 5)))
def test_store_orders_preparation_layers_and_finish_without_blocking_hooks(layer_ids, make_session, queues):
    session, _ = make_session()
    preparation_started, release_preparation = threading.Event(), threading.Event()
    trace = []
    state, result = object(), object()
    caller_id = threading.get_ident()

    def prepare():
        preparation_started.set()
        assert release_preparation.wait(timeout=5)
        trace.append("prepared")

    def record_source():
        assert threading.get_ident() == caller_id
        trace.append("recorded")
        return state

    def copy(source_ready, layer_id, supplied_state):
        assert source_ready is supplied_state is state
        trace.append(layer_id)

    def finish(opening, layers, supplied_state):
        assert supplied_state is state and opening.done()
        assert tuple(layers) == layer_ids
        assert all(future.done() for future in layers.values())
        trace.append("finished")
        return result

    opening = queues[1].submit(prepare)
    try:
        assert preparation_started.wait(timeout=2)
        for layer_id in layer_ids:
            session.store_layer(layer_id, opening, record_source, copy, state)
            session.store_layer(layer_id, opening, record_source, copy, state)
        closing = session.finish_store(opening, finish, state)
        assert not closing.done()
        assert trace == ["recorded"] * len(layer_ids)
        assert session.finish_store(opening, finish, state) is closing
        session.store_layer(9, opening, record_source, copy, state)
        release_preparation.set()
        assert closing.result(timeout=2) is result
        assert trace == [*["recorded"] * len(layer_ids), "prepared", *layer_ids, "finished"]
    finally:
        release_preparation.set()


@pytest.mark.parametrize("stage", ("preparation", "layer"))
def test_store_finish_receives_failed_futures_without_interpreting_them(stage, make_session, queues):
    session, _ = make_session()
    error = RuntimeError(stage)
    copy = Mock(side_effect=error if stage == "layer" else None)

    def prepare():
        if stage == "preparation":
            raise error

    def finish(opening, layers):
        assert opening.done() and layers[2].done()
        assert opening.exception() is (error if stage == "preparation" else None)
        assert layers[2].exception() is error
        return error

    opening = queues[1].submit(prepare)
    session.store_layer(2, opening, object, copy)
    assert session.finish_store(opening, finish).result(timeout=2) is error
    assert copy.call_count == int(stage == "layer")


@pytest.mark.parametrize("bind_store_first", (False, True), ids=("load_first", "store_first"))
def test_one_compute_boundary_releases_load_and_submits_store_once(bind_store_first, make_session, queues):
    session, gates = make_session((2, 5))
    opening = queues[1].submit(lambda: None)
    source, copy, finish = Mock(side_effect=object), Mock(), Mock()

    def save_layer(name):
        assert name == "layer.2"
        session.store_layer(2, opening, source, copy)

    entry = Mock(side_effect=save_layer)
    if bind_store_first:
        session.bind_store_to_compute_start(2, entry, "layer.2")
    assert session.wait_for_layer(2, int, finish) == 2
    if not bind_store_first:
        session.bind_store_to_compute_start(2, entry, "layer.2")
    source.assert_not_called()
    gates[0].on_start()
    gates[0].open()
    queues[0].submit(lambda: None).result(timeout=2)
    assert session.wait_for_layer(5, int, finish) == 5
    session.store_layer(2, opening, source, copy)
    session.bind_store_to_compute_start(2, entry, "alias.2")
    session.finish_store(opening, lambda *_: None).result(timeout=2)
    session.bind_store_to_compute_start(9, entry, "layer.9")
    entry.assert_called_once_with("layer.2")
    source.assert_called_once_with()
    copy.assert_called_once()
    session.end_step()
    assert all(gate.on_start is None for gate in gates)
