"""Layer scheduling owns task termination while caller operations own IO outcomes."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import FakeLoadStartGate as FakeGate
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.infrastructure.executor import (
    Executor,
    LayerwiseExecutor,
)


@pytest.fixture
def queues():
    write, read = Executor("write-test", lambda: None), Executor("read-test", lambda: None)
    write.start()
    read.start()
    try:
        yield write, read
    finally:
        write.close()
        read.close()


@pytest.fixture
def make_execution(queues):
    instances = []

    def create(depth=2, gate_type=FakeGate):
        gates: dict[int, FakeGate] = {}

        def gate():
            value = gate_type()
            gates[len(gates)] = value
            return value

        execution = LayerwiseExecutor(*queues, prefetch_layers=depth, start_gate_factory=gate)
        execution.begin_step()
        instances.append(execution)
        return execution, gates

    yield create
    for execution in instances:
        execution.end_step(wait=False)


@pytest.mark.parametrize(
    "depth,counts",
    ((1, (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3)), (2, (2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)), (8, (3,) * 12)),
)
def test_sparse_prefetch_window_and_aliases(depth, counts, make_execution, queues):
    execution, gates = make_execution(depth)
    copied: list[int] = []
    finish = Mock()
    execution.start_write(lambda: (2, 5, 9), copied.append, finish)
    for layer, count in zip((0, 0, 2, 2, 3, 3, 5, 5, 8, 9, 9, 12), counts, strict=True):
        execution.wait_for_layer(layer, object, read_at_compute_start=False)
        for gate in gates.values():
            gate.open()
        queues[0].submit(lambda: None).result(timeout=2)
        assert copied == [2, 5, 9][:count]
    finish.assert_called_once_with(None)


@pytest.mark.parametrize("failing_layer", (2, 5, 9))
def test_write_error_claimed_once_and_abort_blocks_later_io(failing_layer, make_execution, queues):
    execution, gates = make_execution()
    error, attempted, finish = RuntimeError("copy failed"), [], Mock()

    def write(layer):
        attempted.append(layer)
        if layer == failing_layer:
            raise error

    execution.start_write(lambda: (2, 5, 9), write, finish)
    for layer in (2, 5, 9):
        if layer == failing_layer:
            with pytest.raises(RuntimeError) as caught:
                execution.wait_for_layer(layer, object, read_at_compute_start=False)
            assert caught.value is error
            execution.wait_for_layer(layer, object, read_at_compute_start=False)
            break
        execution.wait_for_layer(layer, object, read_at_compute_start=False)
        for gate in gates.values():
            gate.open()
    execution.end_step(wait=True)
    assert attempted == [layer for layer in (2, 5, 9) if layer <= failing_layer]
    finish.assert_called_once_with(error)
    assert queues[0].submit(lambda: "alive").result(timeout=2) == "alive"


def test_idle_hooks_create_no_gates_or_tasks(make_execution):
    execution, gates = make_execution()
    record = Mock()
    for layer in (2, 2, 5, 9):
        execution.wait_for_layer(layer, record, read_at_compute_start=True)
        execution.read_layer(layer, record)
    execution.end_step(wait=True)
    assert not gates
    record.assert_not_called()


def test_abort_cancels_shared_gate_once_and_detaches_callbacks(make_execution, queues):
    class CountingGate(FakeGate):
        def __init__(self):
            super().__init__()
            self.cancels = 0

        def cancel(self):
            self.cancels += 1
            super().cancel()

    execution, gates = make_execution(3, CountingGate)
    write, finish, read = Mock(), Mock(), Mock()
    release = threading.Event()
    execution.start_write(lambda: (2, 5, 9), write, finish)
    queues[0].submit(release.wait, 5)
    execution.start_read(lambda: None, read, lambda _: True, lambda: (0, 99), lambda *_: None)
    try:
        execution.wait_for_layer(0, object, read_at_compute_start=True)
        execution.wait_for_layer(99, object, read_at_compute_start=True)
        execution.end_step(wait=False)
        assert [gate.cancels for gate in gates.values()] == [1, 0]
        assert all(gate.on_start is None for gate in gates.values())
        release.set()
        queues[0].submit(lambda: None).result(timeout=2)
        write.assert_not_called()
        read.assert_not_called()
        finish.assert_called_once_with(None)
    finally:
        release.set()


@pytest.mark.parametrize("wait", (False, True))
def test_end_step_preserves_running_write_and_orders_cleanup(wait, make_execution, queues):
    execution, gates = make_execution()
    entered, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    trace = []

    def write(_layer):
        entered.set()
        assert release.wait(5)
        trace.append("write")

    def finish(error):
        assert error is None
        trace.append("cleanup")
        cleaned.set()

    execution.start_write(lambda: (2,), write, finish)
    execution.wait_for_layer(0, object, read_at_compute_start=False)
    gates[0].open()
    assert entered.wait(2)
    with ThreadPoolExecutor(1) as hooks:
        end_entered = threading.Event()

        def end():
            end_entered.set()
            execution.end_step(wait=wait)

        ended = hooks.submit(end)
        try:
            assert end_entered.wait(2)
            if wait:
                assert not ended.done() and not cleaned.is_set()
            else:
                ended.result(timeout=2)
                assert not cleaned.is_set()
            release.set()
            ended.result(timeout=2)
            assert cleaned.wait(2)
            assert trace == ["write", "cleanup"]
        finally:
            release.set()


@pytest.mark.parametrize("stage", ("prepare", "last_layer", "empty"))
def test_write_finalizer_runs_once_and_propagates_cleanup_failure(stage, make_execution):
    execution, _ = make_execution(1)
    opening_error, cleanup_error = RuntimeError("opening"), RuntimeError("cleanup")
    finish = Mock(side_effect=cleanup_error)

    def prepare():
        if stage == "prepare":
            raise opening_error
        return () if stage == "empty" else (2,)

    try:
        with pytest.raises(RuntimeError) as caught:
            execution.start_write(prepare, lambda _: None, finish)
            execution.wait_for_layer(2, object, read_at_compute_start=False)
        assert caught.value is cleanup_error
        with pytest.raises(RuntimeError):
            execution.end_step(wait=True)
        finish.assert_called_once_with(opening_error if stage == "prepare" else None)
    finally:
        execution.end_step(wait=False)


@pytest.mark.parametrize("layer_ids", ((), (2,), (2, 5, 9)))
def test_reads_record_on_hook_thread_and_seal_after_open_and_layers(layer_ids, make_execution, queues):
    execution, _ = make_execution()
    hook_thread = threading.get_ident()
    release, opened = threading.Event(), threading.Event()
    trace, recorded, copied, terminal = [], [], [], []

    def prepare():
        assert release.wait(5)
        trace.append("open")
        opened.set()

    def record():
        assert threading.get_ident() == hook_thread
        event = object()
        recorded.append(event)
        return event

    def read(event, layer):
        assert opened.is_set() and event in recorded
        trace.append(layer)
        copied.append(event)

    execution.start_read(prepare, read, lambda _: True, lambda: layer_ids, lambda *args: terminal.append(args))
    try:
        for layer in layer_ids:
            execution.read_layer(layer, record)
            execution.read_layer(layer, record)
        execution.seal_reads()
        execution.seal_reads()
        execution.read_layer(99, record)
        assert not terminal
        release.set()
        queues[1].submit(lambda: None).result(timeout=2)
        assert trace == ["open", *layer_ids]
        assert recorded == copied and len(recorded) == len(layer_ids)
        assert terminal == [(None, {}, frozenset(layer_ids))]
    finally:
        release.set()


@pytest.mark.parametrize("stage", ("open", "layer", "none"))
def test_read_finalizer_receives_execution_facts_even_on_failure(stage, make_execution, queues):
    execution, _ = make_execution()
    error, reports = RuntimeError("failed"), []
    release = threading.Event()

    def prepare():
        assert release.wait(5)
        if stage == "open":
            raise error

    def read(_event, _layer):
        if stage == "layer":
            raise error

    execution.start_read(prepare, read, lambda _: True, lambda: (2,), lambda *args: reports.append(args))
    try:
        execution.read_layer(2, object)
        execution.seal_reads()
        release.set()
        queues[1].submit(lambda: None).result(timeout=2)
        assert reports == [(error if stage == "open" else None, {} if stage == "none" else {2: error}, frozenset({2}))]
    finally:
        release.set()


@pytest.mark.parametrize("read_first", (False, True))
def test_compute_boundary_releases_prefetch_and_submits_one_read(read_first, make_execution, queues):
    execution, gates = make_execution()
    copied: list[int] = []
    read, record, finish = Mock(), Mock(return_value=object()), Mock()
    execution.start_write(lambda: (2, 5), copied.append, finish)
    execution.start_read(lambda: None, read, lambda _: True, lambda: (2,), lambda *_: None)
    if read_first:
        execution.read_layer(2, record)
    execution.wait_for_layer(2, record, read_at_compute_start=True)
    gates[0].on_start() if gates[0].on_start is not None else None
    gates[0].open()
    execution.wait_for_layer(5, record, read_at_compute_start=False)
    execution.read_layer(2, record)
    execution.seal_reads()
    queues[1].submit(lambda: None).result(timeout=2)
    assert copied == [2, 5]
    record.assert_called_once()
    read.assert_called_once_with(record.return_value, 2)
    finish.assert_called_once_with(None)


def test_detached_step_finishes_its_own_reads_after_next_step_starts(make_execution, queues):
    execution, _ = make_execution()
    release, entered = threading.Event(), threading.Event()
    completed = []

    def old_read(_event, _layer):
        entered.set()
        assert release.wait(5)

    execution.start_read(
        lambda: None, old_read, lambda _: True, lambda: (2,), lambda *facts: completed.append(("old", facts))
    )
    execution.read_layer(2, object)
    assert entered.wait(2)
    try:
        execution.end_step(wait=True)
        assert not completed
        execution.begin_step()
        execution.start_read(
            lambda: None, lambda *_: None, lambda _: True, lambda: (9,), lambda *facts: completed.append(("new", facts))
        )
        execution.read_layer(9, object)
        execution.end_step(wait=False)
        release.set()
        queues[1].submit(lambda: None).result(timeout=2)
        assert completed == [("old", (None, {}, frozenset({2}))), ("new", (None, {}, frozenset({9})))]
    finally:
        release.set()


def test_read_all_layers_records_one_boundary_and_skips_known_empty(make_execution, queues):
    execution, _ = make_execution()
    record, read = Mock(return_value=object()), Mock()
    execution.start_read(lambda: None, read, lambda _: False, lambda: (2, 5), lambda *_: None)
    queues[1].submit(lambda: None).result(timeout=2)
    execution.read_all_layers(record)
    queues[1].submit(lambda: None).result(timeout=2)
    record.assert_not_called()
    execution.end_step(wait=True)
    execution.begin_step()
    execution.start_read(lambda: None, read, lambda _: True, lambda: (2, 5), lambda *_: None)
    execution.read_all_layers(record)
    queues[1].submit(lambda: None).result(timeout=2)
    record.assert_called_once()
    assert [call.args for call in read.call_args_list] == [(record.return_value, 2), (record.return_value, 5)]


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_last_write_hook_waits_for_cleanup_before_returning(cleanup_fails, make_execution):
    execution, _ = make_execution(1)
    entered, release, hook_entered = threading.Event(), threading.Event(), threading.Event()
    error = RuntimeError("cleanup failed")

    def finish(_error):
        entered.set()
        assert release.wait(5)
        if cleanup_fails:
            raise error

    execution.start_write(lambda: (2,), lambda _: None, finish)
    with ThreadPoolExecutor(1) as hooks:

        def wait():
            hook_entered.set()
            execution.wait_for_layer(2, object, read_at_compute_start=False)

        waiting = hooks.submit(wait)
        try:
            assert hook_entered.wait(2) and entered.wait(2)
            assert not waiting.done()
            release.set()
            if cleanup_fails:
                with pytest.raises(RuntimeError) as caught:
                    waiting.result(timeout=2)
                assert caught.value is error
            else:
                waiting.result(timeout=2)
            execution.wait_for_layer(2, object, read_at_compute_start=False)
        finally:
            release.set()
            execution.end_step(wait=False)


@pytest.mark.parametrize("wait", (False, True))
def test_failed_gate_cancel_still_seals_reads_and_stops_prefetch(make_execution, queues, wait):
    error = RuntimeError("gate cancellation failed")

    class BrokenCancelGate(FakeGate):
        def cancel(self):
            raise error

    execution, gates = make_execution(3, BrokenCancelGate)
    copied: list[int] = []
    writes_finished, reads_finished = Mock(), Mock()
    execution.start_write(lambda: (2, 5, 9), copied.append, writes_finished)
    execution.start_read(lambda: None, lambda *_: None, lambda _: False, lambda: (), reads_finished)
    execution.wait_for_layer(0, object, read_at_compute_start=False)
    with pytest.raises(RuntimeError) as caught:
        execution.end_step(wait=wait)
    assert caught.value is error and not execution.active
    # The abort flag terminates gate polling even when cancellation did not wake it.
    queues[0].submit(lambda: None).result(timeout=2)
    queues[1].submit(lambda: None).result(timeout=2)
    assert not copied
    writes_finished.assert_called_once()
    reads_finished.assert_called_once()
    assert all(gate.on_start is None for gate in gates.values())
