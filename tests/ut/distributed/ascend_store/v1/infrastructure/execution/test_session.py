"""Load and Store ordering with opaque operations supplied at each call."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import FakeLoadStartGate
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.execution.executor import Executor
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.execution.session import Session


@pytest.fixture
def session():
    load = Executor("test-session-load", lambda: None)
    store = Executor("test-session-store", lambda: None)
    load.start()
    store.start()
    current: Session[Any] = Session(load.submit, store.submit, 2, FakeLoadStartGate)
    try:
        yield current
    finally:
        current.abort_load(wait=False)
        load.close()
        store.close()


@pytest.fixture
def finish():
    return Mock(return_value=None)


def test_layer_hooks_do_not_rescan_existing_load_jobs(session, finish):
    class NoScanList(list):
        def __iter__(self):
            pytest.fail("A layer hook must not scan all existing jobs")

    class NoScanOrder(tuple):
        def __iter__(self):
            pytest.fail("A layer hook must not scan the Load order")

        def index(self, *args):
            pytest.fail("A layer hook must not search for the current layer")

    session._start_gate_factory = lambda: FakeLoadStartGate(opened=True)
    session._load_futures = NoScanList()
    session.load_order = NoScanOrder(range(36))
    try:
        for layer_id in range(36):
            assert session.wait_for_layer(layer_id, int, finish) == layer_id
            assert session.wait_for_layer(layer_id, int, finish) is None
        assert len(session._load_futures) == 36
    finally:
        session._load_futures = session._load_futures[:]


@pytest.mark.parametrize(
    ("depth", "submitted_counts"),
    (
        (1, (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)),
        (2, (2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)),
        (8, (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3)),
    ),
)
def test_sparse_load_window_preserves_non_load_hooks_aliases_and_tail(depth, submitted_counts, session, finish):
    copied = []

    def load(layer_id):
        copied.append(layer_id)
        return layer_id

    session._prefetch_layers = depth
    session.load_order = (2, 5, 9)
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
    for (layer_id, expected, pending), submitted in zip(steps, submitted_counts, strict=True):
        previous_gate = session._start_gates.get(layer_id)
        assert session.wait_for_layer(layer_id, load, finish) == expected
        assert session.pending_load_layers == pending
        assert len(session._load_futures) == submitted
        gate = session._start_gates.get(layer_id)
        if previous_gate is not None:
            assert gate is previous_gate
        if gate is not None:
            gate.open()
    assert copied == [2, 5, 9]
    assert session.loads_submitted and session.loads_claimed
    assert 12 not in session._start_gates
    finish.assert_called_once_with()


@pytest.mark.parametrize("failing_layer", (2, 5, 9))
def test_sparse_load_error_is_claimed_once_and_abort_wakes_pending_prefetch(failing_layer, session, finish):
    error = RuntimeError("layer Load failed")
    attempted = []
    session.load_order = (2, 5, 9)

    def load(layer_id):
        attempted.append(layer_id)
        if layer_id == failing_layer:
            raise error
        return layer_id

    for layer_id in session.load_order:
        if layer_id == failing_layer:
            break
        assert session.wait_for_layer(layer_id, load, finish) == layer_id
        gate = session._start_gates.get(layer_id)
        if gate is not None:
            gate.open()
    with pytest.raises(RuntimeError) as raised:
        session.wait_for_layer(failing_layer, load, finish)
    assert raised.value is error
    assert session.pending_load_layers == tuple(layer for layer in session.load_order if layer > failing_layer)
    submitted = len(session._load_futures)
    assert session.wait_for_layer(failing_layer, load, finish) is None
    assert len(session._load_futures) == submitted
    session.abort_load(finish)
    assert session._load_submit(lambda: "healthy").result(timeout=5) == "healthy"
    assert attempted == [layer for layer in session.load_order if layer <= failing_layer]
    finish.assert_called_once_with()


def test_hooks_without_load_work_do_not_create_gates_or_submit_jobs(session, finish):
    copied: list[int] = []
    for layer_id in (2, 2, 5, 9, 9):
        assert session.wait_for_layer(layer_id, copied.append, finish) is None
    assert not session._load_futures
    assert not session._start_gates
    assert session.pending_load_layers == ()
    assert copied == []
    finish.assert_not_called()
    assert session._load_finished is None


def test_prefetch_binds_each_layer_and_gate_before_background_execution(session):
    release_queue = threading.Event()
    caller = threading.get_ident()
    copied = []
    session.load_order = (2, 5)

    def load(layer_id):
        assert threading.get_ident() != caller
        copied.append(layer_id)
        return layer_id

    session._load_submit(release_queue.wait, 5)
    try:
        session.prefetch_loads(2, load)
        assert len(session._load_futures) == 2
        assert not any(future.done() for future in session._load_futures)
        release_queue.set()
        assert session._load_futures[0].result(timeout=5) == 2
        assert not session._load_futures[1].done()
        assert copied == [2]
        session._load_gates[5].open()
        assert session._load_futures[1].result(timeout=5) == 5
        assert copied == [2, 5]
    finally:
        release_queue.set()


def test_abort_wakes_prefetch_without_starting_cancelled_io(session, finish, monkeypatch):
    copied: list[int] = []
    waiting = threading.Event()
    gate = FakeLoadStartGate()
    wait, cancel = gate.wait, gate.cancel

    def wait_at_gate(timeout):
        waiting.set()
        return wait(timeout)

    def cancel_after_abort():
        assert session._load_aborted.is_set()
        cancel()

    monkeypatch.setattr(gate, "wait", wait_at_gate)
    monkeypatch.setattr(gate, "cancel", cancel_after_abort)
    session._start_gate_factory = lambda: gate
    session.load_order = (0, 1, 2)
    session.wait_for_layer(0, copied.append, finish)
    assert waiting.wait(timeout=2)
    session.abort_load(finish)
    assert session._load_futures[1].result(timeout=5) is None
    assert session._load_submit(lambda: "healthy").result(timeout=5) == "healthy"
    assert copied == [0]
    finish.assert_called_once_with()


@pytest.mark.parametrize("first_load_fails", (False, True))
def test_abort_waits_for_all_submitted_loads_without_cleanup(first_load_fails, session):
    session.load_order = (0, 1)
    session._start_gate_factory = lambda: FakeLoadStartGate(opened=True)
    load_started = threading.Event()
    release_load = threading.Event()
    abort_started = threading.Event()
    error = RuntimeError("first Load failed")

    def load(layer_id):
        if layer_id == 0:
            if first_load_fails:
                raise error
            return layer_id
        load_started.set()
        assert release_load.wait(timeout=5)
        return layer_id

    def run_hooks():
        session.prefetch_loads(0, load)
        assert load_started.wait(timeout=2)
        abort_started.set()
        session.abort_load(wait=True)

    with ThreadPoolExecutor(max_workers=1) as hooks:
        aborted = hooks.submit(run_hooks)
        try:
            assert abort_started.wait(timeout=2)
            with pytest.raises(TimeoutError):
                aborted.result(timeout=0.05)
            release_load.set()
            aborted.result(timeout=5)
            assert all(future.done() for future in session._load_futures)
            assert session._load_futures[0].exception() is (error if first_load_fails else None)
        finally:
            release_load.set()


@pytest.mark.parametrize("with_cleanup", (False, True))
@pytest.mark.parametrize("exit_operation", ("abort_load", "end_step"))
def test_nonblocking_exit_keeps_running_load_and_optional_cleanup_asynchronous(
    exit_operation, with_cleanup, session, finish
):
    session.load_order = (0,)
    load_started = threading.Event()
    release_load = threading.Event()

    def load(layer_id):
        load_started.set()
        assert release_load.wait(timeout=5)
        return layer_id

    try:
        session.prefetch_loads(0, load)
        assert load_started.wait(timeout=2)
        getattr(session, exit_operation)(finish if with_cleanup else None, wait=False)
        assert not session._load_futures[0].done()
        finish.assert_not_called()
        release_load.set()
        assert session._load_futures[0].result(timeout=5) == 0
        session.wait_load_finished()
        assert finish.call_count == int(with_cleanup)
    finally:
        release_load.set()


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_abort_waits_for_cleanup_and_propagates_its_error(cleanup_fails, session):
    session.load_order = (0,)
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    error = RuntimeError("cleanup failed")

    def finish():
        assert session._load_futures[0].done()
        cleanup_started.set()
        assert release_cleanup.wait(timeout=5)
        if cleanup_fails:
            raise error

    def run_hooks():
        session.prefetch_loads(0, int)
        session.abort_load(finish)

    with ThreadPoolExecutor(max_workers=1) as hooks:
        aborted = hooks.submit(run_hooks)
        try:
            assert cleanup_started.wait(timeout=2)
            assert not aborted.done()
            release_cleanup.set()
            if cleanup_fails:
                with pytest.raises(RuntimeError) as raised:
                    aborted.result(timeout=5)
                assert raised.value is error
            else:
                aborted.result(timeout=5)
        finally:
            release_cleanup.set()


def test_end_step_detaches_callbacks_before_abort_and_leaves_store_running(session, finish, monkeypatch):
    session.load_order = (0, 1)
    gate = FakeLoadStartGate()
    gate_waiting = threading.Event()
    store_started = threading.Event()
    release_store = threading.Event()
    entry = Mock()
    copied: list[int] = []
    wait, cancel = gate.wait, gate.cancel

    def wait_at_gate(timeout):
        gate_waiting.set()
        return wait(timeout)

    def cancel_after_detaching():
        assert gate.on_start is None
        cancel()

    def store(source_ready, layer_id):
        store_started.set()
        assert release_store.wait(timeout=5)

    monkeypatch.setattr(gate, "wait", wait_at_gate)
    monkeypatch.setattr(gate, "cancel", cancel_after_detaching)
    session._start_gate_factory = lambda: gate
    opening = session._store_submit(lambda: None)
    try:
        session.prefetch_loads(0, copied.append)
        assert gate_waiting.wait(timeout=2)
        session.bind_store_to_compute_start(0, entry)
        session.store_layer(9, opening, object, store)
        assert store_started.wait(timeout=2)
        session.end_step(finish)
        session.end_step(finish)
        assert copied == [0]
        assert all(future.done() for future in session._load_futures)
        finish.assert_called_once_with()
        entry.assert_not_called()
        assert gate.on_start is None
        assert not session._store_futures[9].done()
        store_finish = Mock()
        closing = session.finish_store(opening, store_finish)
        assert not closing.done()
        store_finish.assert_not_called()
        release_store.set()
        closing.result(timeout=5)
        store_finish.assert_called_once_with(opening, session._store_futures)
    finally:
        gate.on_start = None
        release_store.set()


def test_last_load_waits_for_cleanup_and_preserves_the_opaque_result(session):
    session.load_order = (2,)
    result = object()
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    trace = []

    def load(layer_id):
        trace.append(layer_id)
        return result

    def finish():
        cleanup_started.set()
        assert release_cleanup.wait(timeout=5)
        trace.append("finished")

    with ThreadPoolExecutor(max_workers=1) as hooks:
        waiting = hooks.submit(session.wait_for_layer, 2, load, finish)
        try:
            assert cleanup_started.wait(timeout=2)
            assert not waiting.done()
            release_cleanup.set()
            assert waiting.result(timeout=5) is result
            assert session.wait_for_layer(2, load, finish) is None
            assert trace == [2, "finished"]
        finally:
            release_cleanup.set()


def test_load_cleanup_is_queued_once_after_the_last_submitted_layer(session):
    session.load_order = (2, 5)
    trace = []
    finish = Mock(side_effect=lambda: trace.append("finished"))
    session.wait_for_layer(2, trace.append, finish)
    assert session._load_finished is not None and not session._load_finished.done()
    assert trace == [2]
    session._get_or_create_compute_start_gate(2).open()
    session.wait_for_layer(5, trace.append, finish)
    session.abort_load(finish)
    assert trace == [2, 5, "finished"]
    finish.assert_called_once_with()


def test_load_cleanup_error_is_propagated_without_interpretation(session):
    session.load_order = (2,)
    error = RuntimeError("cleanup failed")
    finish = Mock(side_effect=error)
    with pytest.raises(RuntimeError) as raised:
        session.wait_for_layer(2, int, finish)
    assert raised.value is error
    with pytest.raises(RuntimeError) as raised:
        session.wait_load_finished()
    assert raised.value is error
    finish.assert_called_once_with()


def test_store_orders_preparation_layers_and_finish_without_blocking_the_hook(session):
    preparation_started = threading.Event()
    release_preparation = threading.Event()
    caller = threading.get_ident()
    trace = []
    state = object()
    result = object()
    source = object()

    def prepare():
        preparation_started.set()
        assert release_preparation.wait(timeout=5)
        trace.append("prepared")

    def record_source():
        assert threading.get_ident() == caller
        trace.append("recorded")
        return source

    def copy(event, layer_id, argument):
        assert argument is state and event is source
        assert threading.get_ident() != caller
        trace.append(layer_id)

    def finish(opening, layers, argument):
        assert argument is state and opening.done()
        assert tuple(layers) == (2, 5)
        assert all(future.done() for future in layers.values())
        trace.append("finished")
        return result

    opening = session._store_submit(prepare)
    try:
        assert preparation_started.wait(timeout=2)
        for layer_id in (2, 2, 5, 5):
            session.store_layer(layer_id, opening, record_source, copy, state)
        closing = session.finish_store(opening, finish, state)
        assert not closing.done()
        assert trace == ["recorded", "recorded"]
        assert session.finish_store(opening, finish, state) is closing
        session.store_layer(9, opening, record_source, copy, state)
        release_preparation.set()
        assert closing.result(timeout=5) is result
        assert trace == ["recorded", "recorded", "prepared", 2, 5, "finished"]
    finally:
        release_preparation.set()


@pytest.mark.parametrize("stage", ("preparation", "layer"))
def test_store_finish_receives_failed_futures_without_session_interpreting_them(stage, session):
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

    opening = session._store_submit(prepare)
    session.store_layer(2, opening, object, copy)
    assert session.finish_store(opening, finish).result(timeout=5) is error
    assert copy.call_count == int(stage == "layer")


def test_store_finish_with_no_layers_still_runs_after_preparation(session):
    trace: list[str] = []
    opening = session._store_submit(trace.append, "prepared")

    def finish(prepared, layers):
        assert prepared.done() and layers == {}
        trace.append("finished")

    session.finish_store(opening, finish).result(timeout=5)
    assert trace == ["prepared", "finished"]
    assert not session._start_gates


def test_store_compute_boundary_invokes_the_supplied_entry_point_and_deduplicates_layers(session):
    opening: Future[None] = Future()
    opening.set_result(None)
    source = Mock(side_effect=object)
    copy = Mock()
    layer_name = "layer alias"

    def save_layer(name):
        assert name == layer_name
        session.store_layer(2, opening, source, copy)

    entry = Mock(side_effect=save_layer)
    session.bind_store_to_compute_start(2, entry, layer_name)
    gate = session._get_or_create_compute_start_gate(2)
    source.assert_not_called()
    entry.assert_not_called()
    gate.on_start()
    session.store_layer(2, opening, source, copy)
    callback = gate.on_start
    session.bind_store_to_compute_start(2, entry, layer_name)
    assert gate.on_start is callback
    session.finish_store(opening, lambda *_: None).result(timeout=5)
    session.bind_store_to_compute_start(5, entry, layer_name)
    assert 5 not in session._start_gates
    entry.assert_called_once_with(layer_name)
    source.assert_called_once_with()
    copy.assert_called_once()
    session.clear_callbacks()
    assert gate.on_start is None


@pytest.mark.parametrize("bind_store_first", (False, True), ids=("load_first", "store_first"))
def test_load_prefetch_and_store_binding_share_the_same_layer_gate(bind_store_first, session, finish):
    factory = Mock(side_effect=FakeLoadStartGate)
    session._start_gate_factory = factory
    session.load_order = (2, 5)
    entry = Mock()
    if bind_store_first:
        session.bind_store_to_compute_start(2, entry, "layer.2")
    assert session.wait_for_layer(2, int, finish) == 2
    gate = session._load_gates[5]
    if not bind_store_first:
        session.bind_store_to_compute_start(2, entry, "layer.2")
    assert session._start_gates[2] is gate
    factory.assert_called_once_with()
    entry.assert_not_called()
    gate.on_start()
    entry.assert_called_once_with("layer.2")
    gate.open()
    assert session.wait_for_layer(5, int, finish) == 5
