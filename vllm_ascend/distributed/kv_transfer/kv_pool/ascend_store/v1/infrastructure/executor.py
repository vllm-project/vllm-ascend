"""Worker queues and ordered whole-batch or per-layer execution over local memory."""

from __future__ import annotations

import threading
from collections.abc import Callable, Collection
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, field
from functools import partial
from typing import Any, ParamSpec, TypeVar

from vllm.logger import logger

_COMPUTE_START_LOG_INTERVAL_S = 10
Parameters = ParamSpec("Parameters")
ResultT = TypeVar("ResultT")
PreparedT = TypeVar("PreparedT")


class Executor:
    """Own one initialized worker thread; task results and exceptions belong to callers."""

    def __init__(self, name: str, initializer: Callable[[], None]) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name, initializer=initializer)
        self._stopped = False

    def start(self) -> None:
        # The pool starts lazily; this barrier confirms its thread initializer has finished.
        self._pool.submit(lambda: None).result()

    def submit(
        self, fn: Callable[Parameters, ResultT], /, *args: Parameters.args, **kwargs: Parameters.kwargs
    ) -> Future[ResultT]:
        return self._pool.submit(fn, *args, **kwargs)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._stopped = True

    @property
    def stopped(self) -> bool:
        """Whether close has finished draining tasks and joining the worker thread."""
        return self._stopped


class BulkExecutor:
    """Sequence preparation and IO on borrowed queues; own the calling thread's read boundary."""

    def __init__(self, write: Executor, read: Executor | None) -> None:
        self._write, self._read = write, read
        self._source_ready: Future[Any] | None = None

    @property
    def read_enabled(self) -> bool:
        return self._read is not None

    def submit_write(self, operation: Callable[[], ResultT]) -> Future[ResultT]:
        return self._write.submit(operation)

    def submit_read(
        self,
        prepare: Callable[[], PreparedT],
        operation: Callable[[Callable[[], PreparedT], Callable[[], Any]], ResultT],
    ) -> Future[ResultT]:
        assert self._read is not None
        ready: Future[Any] = Future()
        prepared = self._read.submit(prepare)
        self._source_ready = ready
        try:
            return self._read.submit(operation, prepared.result, ready.result)
        except BaseException:
            self._source_ready = None
            raise

    def record_read_ready(self, needed: bool, record: Callable[[], Any]) -> None:
        ready = self._source_ready
        if ready is not None and not ready.done() and needed:
            try:
                ready.set_result(record())
            except BaseException as error:
                ready.set_exception(error)
                raise

    def check_read_ready(self, needed: bool) -> None:
        if self._source_ready is not None and not self._source_ready.done() and needed:
            raise RuntimeError("The source boundary must be ready before releasing its source")

    def end_step(self) -> None:
        if self._source_ready is not None and not self._source_ready.done():
            self._source_ready.set_exception(RuntimeError("The step ended before its source was marked ready"))
        self._source_ready = None


@dataclass(slots=True)
class _LayerwiseStep:
    """Execution control only; queued closures retain this record after the step detaches."""

    write_order: tuple[int, ...] = ()
    write_pos: int = 0
    write_futures: list[Future[None]] = field(default_factory=list)
    write_gates: list[Any] = field(default_factory=list)
    write_aborted: threading.Event = field(default_factory=threading.Event)
    write_finished: Future[None] | None = None
    write_error: BaseException | None = None
    write: Callable[[int], None] | None = None
    finish_write: Callable[[BaseException | None], None] | None = None
    start_gates: dict[int, Any] = field(default_factory=dict)
    read_opening: Future[None] | None = None
    read_futures: dict[int, Future[None]] = field(default_factory=dict)
    read_finished: Future[None] | None = None
    read: Callable[[Any, int], None] | None = None
    needs_read: Callable[[int], bool] | None = None
    read_layers: Callable[[], Collection[int]] | None = None
    finish_read: Callable[[BaseException | None, dict[int, BaseException], frozenset[int]], None] | None = None


class LayerwiseExecutor:
    """Compose separate write/read queues and own each step's gates, tasks and finalization.

    Queue lifetimes belong to Resources. Directions refer to caller-owned memory;
    operations interpret Backend results, while this executor interprets task termination.
    Hooks use one calling thread. No abort interrupts an operation already in progress.
    """

    def __init__(
        self, write: Executor, read: Executor | None, *, prefetch_layers: int, start_gate_factory: Callable[[], Any]
    ) -> None:
        if prefetch_layers <= 0:
            raise ValueError("Write prefetch depth must be positive")
        self._write, self._read = write, read
        self._prefetch_layers, self._start_gate_factory = prefetch_layers, start_gate_factory
        self._step: _LayerwiseStep | None = None

    @property
    def read_enabled(self) -> bool:
        return self._read is not None

    @property
    def active(self) -> bool:
        return self._step is not None

    @property
    def reads_sealed(self) -> bool:
        step = self._step
        return step is None or step.read_opening is None or step.read_finished is not None

    def begin_step(self) -> None:
        if self._step is not None:
            raise RuntimeError("Previous Layerwise execution step has not ended")
        self._step = _LayerwiseStep()

    def _active_step(self) -> _LayerwiseStep:
        if self._step is None:
            raise RuntimeError("Layerwise execution step has not begun")
        return self._step

    # ================================================
    # Write preparation, bounded prefetch and cleanup
    # ================================================

    def start_write(
        self,
        prepare: Callable[[], tuple[int, ...]],
        operation: Callable[[int], None],
        finish: Callable[[BaseException | None], None],
    ) -> None:
        step = self._active_step()
        if step.finish_write is not None:
            raise RuntimeError("Layerwise write already started for this step")
        step.write, step.finish_write = operation, finish
        try:
            step.write_order = self._write.submit(prepare).result()
        except BaseException as error:
            step.write_error = error
            self._finish_writes(step).result()
            raise
        if not step.write_order:
            self._finish_writes(step).result()

    def wait_for_layer(
        self, layer_id: int, record_read_ready: Callable[[], Any], *, read_at_compute_start: bool
    ) -> None:
        step = self._step
        if step is None:
            return
        if step.write_order:
            position = step.write_pos
            repeated = position > 0 and layer_id == step.write_order[position - 1]
            if not repeated:
                end = min(position + self._prefetch_layers, len(step.write_order))
                for index in range(len(step.write_futures), end):
                    next_layer = step.write_order[index]
                    gate = None if next_layer == layer_id else self._gate(step, layer_id)
                    step.write_gates.append(gate)
                    step.write_futures.append(self._write.submit(self._write_layer, step, next_layer, gate))
                if len(step.write_futures) == len(step.write_order):
                    self._finish_writes(step)
                if position < len(step.write_order) and layer_id == step.write_order[position]:
                    step.write_pos += 1
                    step.write_futures[position].result()
                    if step.write_pos == len(step.write_order):
                        assert step.write_finished is not None
                        step.write_finished.result()
        if read_at_compute_start and self._needs_read_layer(step, layer_id):
            self._gate(step, layer_id).on_start = partial(self._submit_read_layer, step, layer_id, record_read_ready)

    @staticmethod
    def _write_layer(step: _LayerwiseStep, layer_id: int, gate: Any) -> None:
        if gate is not None:
            while not gate.wait(timeout=_COMPUTE_START_LOG_INTERVAL_S):
                logger.info("Layerwise %d write waits for compute start", layer_id)
        if not step.write_aborted.is_set():
            assert step.write is not None
            step.write(layer_id)

    def _finish_writes(self, step: _LayerwiseStep) -> Future[None]:
        if step.write_finished is None:

            def finish() -> None:
                error = step.write_error
                for future in step.write_futures:
                    try:
                        future.result()
                    except BaseException as caught:
                        error = error or caught
                assert step.finish_write is not None
                step.finish_write(error)

            step.write_finished = self._write.submit(finish)
        return step.write_finished

    # ================================================
    # Read preparation, producer boundaries and sealing
    # ================================================

    def start_read(
        self,
        prepare: Callable[[], None],
        operation: Callable[[Any, int], None],
        needs_layer: Callable[[int], bool],
        layers: Callable[[], Collection[int]],
        finish: Callable[[BaseException | None, dict[int, BaseException], frozenset[int]], None],
    ) -> None:
        step = self._active_step()
        if step.read_opening is not None:
            raise RuntimeError("Layerwise read already started for this step")
        assert self._read is not None
        step.read, step.needs_read, step.read_layers, step.finish_read = operation, needs_layer, layers, finish
        step.read_opening = self._read.submit(prepare)

    @staticmethod
    def _needs_read_layer(step: _LayerwiseStep, layer_id: int) -> bool:
        opening = step.read_opening
        if opening is None or step.read_finished is not None or layer_id in step.read_futures:
            return False
        if not opening.done():
            return True
        if opening.cancelled() or opening.exception() is not None:
            return False
        assert step.needs_read is not None
        return step.needs_read(layer_id)

    def read_layer(self, layer_id: int, record_source_ready: Callable[[], Any]) -> None:
        if self._step is not None:
            self._submit_read_layer(self._step, layer_id, record_source_ready)

    def _submit_read_layer(self, step: _LayerwiseStep, layer_id: int, record_source_ready: Callable[[], Any]) -> None:
        if not self._needs_read_layer(step, layer_id):
            return
        assert self._read is not None and step.read_opening is not None
        source_ready = record_source_ready()

        def run() -> None:
            assert step.read_opening is not None and step.read is not None
            step.read_opening.result()
            step.read(source_ready, layer_id)

        step.read_futures[layer_id] = self._read.submit(run)

    def read_all_layers(self, record_source_ready: Callable[[], Any]) -> None:
        step = self._step
        if step is None or step.read_opening is None or step.read_finished is not None:
            return
        assert step.read_layers is not None
        ready = None

        def recorded_ready() -> Any:
            return ready

        for layer_id in step.read_layers():
            if self._needs_read_layer(step, layer_id):
                if ready is None:
                    ready = record_source_ready()
                self._submit_read_layer(step, layer_id, recorded_ready)
        self.seal_reads()

    def seal_reads(self) -> None:
        step = self._step
        if step is None or step.read_opening is None or step.read_finished is not None:
            return
        assert self._read is not None

        def finish() -> None:
            assert step.read_opening is not None and step.finish_read is not None
            opening_error = None
            try:
                step.read_opening.result()
            except BaseException as error:
                opening_error = error
            errors = {}
            for layer_id, future in step.read_futures.items():
                try:
                    future.result()
                except BaseException as error:
                    errors[layer_id] = error
            step.finish_read(opening_error, errors, frozenset(step.read_futures))

        step.read_finished = self._read.submit(finish)

    # ================================================
    # Step teardown without reclaiming borrowed queues
    # ================================================

    def _gate(self, step: _LayerwiseStep, layer_id: int) -> Any:
        gate = step.start_gates.get(layer_id)
        if gate is None:
            gate = step.start_gates[layer_id] = self._start_gate_factory()
        return gate

    def end_step(self, *, wait: bool) -> None:
        step = self._step
        if step is None:
            return
        for gate in step.start_gates.values():
            gate.on_start = None
        step.write_aborted.set()
        cancelled: set[int] = set()
        for future, gate in zip(step.write_futures, step.write_gates, strict=True):
            if gate is not None and not future.done() and id(gate) not in cancelled:
                gate.cancel()
                cancelled.add(id(gate))
        error = None
        try:
            if step.finish_write is not None:
                self._finish_writes(step)
            if wait:
                wait_futures(step.write_futures)
                if step.write_finished is not None:
                    step.write_finished.result()
        except BaseException as caught:
            error = caught
        try:
            self.seal_reads()
        except BaseException as caught:
            error = error or caught
        if error is not None:
            raise error
        self._step = None
