"""Coordinate one model step's interleaved layer transfers without interpreting operation results."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from typing import Any, Generic, TypeVar

from vllm.logger import logger

_COMPUTE_START_LOG_INTERVAL_S = 10

LoadResult = TypeVar("LoadResult")
StoreResult = TypeVar("StoreResult")


class Session(Generic[LoadResult]):
    """Order one step's Load and Store calls without interpreting their business results.

    Model hooks use one calling thread. The supplied submit functions run separate
    serial Load and Store queues; preparation uses those same queues. Operations
    and their arguments are supplied at each call, not registered for later dispatch.
    Abort unblocks pending Load gates but never interrupts an active operation.
    """

    def __init__(
        self,
        load_submit: Callable[..., Future[Any]],
        store_submit: Callable[..., Future[Any]] | None,
        prefetch_layers: int,
        start_gate_factory: Callable[[], Any],
    ) -> None:
        if prefetch_layers <= 0:
            raise ValueError("Layerwise Load prefetch depth must be positive")
        self._load_submit = load_submit
        self._store_submit = store_submit
        self._prefetch_layers = prefetch_layers
        self._start_gate_factory = start_gate_factory
        self._start_gates: dict[int, Any] = {}
        self.load_order: tuple[int, ...] = ()
        self._load_pos = 0
        self._load_futures: list[Future[LoadResult | None]] = []
        self._load_gates: dict[int, Any] = {}
        self._load_aborted = threading.Event()
        self._load_finished: Future[None] | None = None
        self._store_futures: dict[int, Future[None]] = {}
        self._store_finished: Future[Any] | None = None

    def wait_for_layer(
        self, layer_id: int, load: Callable[[int], LoadResult], finish: Callable[[], None]
    ) -> LoadResult | None:
        """Advance the Load window, then wait for this layer and the final cleanup when due."""
        if not self.load_order:
            return None
        self.prefetch_loads(layer_id, load)
        if self.loads_submitted:
            self.finish_load(finish)
        position = self._load_pos
        if position == len(self.load_order) or layer_id != self.load_order[position]:
            return None
        # Claim before waiting so an exception is delivered once, just like a result.
        self._load_pos += 1
        result = self._load_futures[position].result()
        if self.loads_claimed:
            self.wait_load_finished()
        return result

    def prefetch_loads(self, layer_id: int, load: Callable[[int], LoadResult]) -> None:
        """Submit only the newly exposed part of the Load window.

        Hooks reach every required physical layer in order; aliases may repeat.
        Positions index load_order, not physical layer IDs. load_pos is the
        first unclaimed position; len(_load_futures) is the sole submit cursor.
        With depth 2, load_pos=1 and submit_pos=2, only Load 9 is newly submitted:

            boundary:      0         1         2         3         4
                           +---------+---------+---------+---------+
            load_order:    |    2    |    5    |    9    |    12   |
                           +---------+---------+---------+---------+
                           | claimed | pending |   new   | unsent  |
                                     ^         ^         ^
                                  load_pos submit_pos   end

        depth is _prefetch_layers; count is len(load_order).
        The window is [load_pos, end), where end=min(load_pos + depth, count).
        Only [submit_pos, end) needs submission. A hook without a Load can fill
        this window, but does not advance load_pos; a repeated claimed hook does neither.
        """
        load_pos = self._load_pos
        if load_pos > 0 and layer_id == self.load_order[load_pos - 1]:
            return
        submit_pos = len(self._load_futures)
        end = min(load_pos + self._prefetch_layers, len(self.load_order))

        def run_load_layer(layer_id: int, gate: Any) -> LoadResult | None:
            if gate is not None:
                while not gate.wait(timeout=_COMPUTE_START_LOG_INTERVAL_S):
                    logger.info("Layerwise %d load waits for compute start", layer_id)
            if self._load_aborted.is_set():
                return None
            return load(layer_id)

        for position in range(submit_pos, end):
            next_layer = self.load_order[position]
            # Allocate a gate only for work waiting on this computation, never an empty window.
            dependency = None if next_layer == layer_id else self._get_or_create_compute_start_gate(layer_id)
            self._load_gates[next_layer] = dependency
            self._load_futures.append(self._load_submit(run_load_layer, next_layer, dependency))

    def finish_load(self, finish: Callable[[], None]) -> Future[None]:
        if self._load_finished is None:
            self._load_finished = self._load_submit(finish)
        return self._load_finished

    def wait_load_finished(self) -> None:
        if self._load_finished is not None:
            self._load_finished.result()

    def store_layer(
        self,
        layer_id: int,
        opening: Future[None],
        record_source_ready: Callable[[], Any],
        operation: Callable[..., None],
        *args: Any,
    ) -> None:
        """Submit a physical layer once, recording its source boundary on the calling thread.

        operation receives (source_ready, layer_id, *args). Its IO boundary still
        owns the source-ready wait, so host argument preparation can precede it.
        """
        if self._store_finished is not None or layer_id in self._store_futures:
            return
        assert self._store_submit is not None
        source_ready = record_source_ready()

        def run_store_layer() -> None:
            opening.result()
            operation(source_ready, layer_id, *args)

        self._store_futures[layer_id] = self._store_submit(run_store_layer)

    def bind_store_to_compute_start(self, layer_id: int, submit_store: Callable[..., None], *args: Any) -> None:
        """Bind the caller's Store entry point to this layer's producer boundary."""
        if self._store_finished is None and layer_id not in self._store_futures:
            self._get_or_create_compute_start_gate(layer_id).on_start = partial(submit_store, *args)

    def finish_store(
        self, opening: Future[None], finish: Callable[..., StoreResult], *args: Any
    ) -> Future[StoreResult]:
        """Seal layer submission and invoke finish(opening, layers, *args) after the queued work.

        FIFO makes every supplied Future terminal before finish runs, including
        failed tasks. The caller's finish operation interprets their results.
        """
        if self._store_finished is None:
            assert self._store_submit is not None
            self._store_finished = self._store_submit(finish, opening, self._store_futures, *args)
        return self._store_finished

    def _get_or_create_compute_start_gate(self, layer_id: int) -> Any:
        gate = self._start_gates.get(layer_id)
        if gate is None:
            gate = self._start_gate_factory()
            self._start_gates[layer_id] = gate
        return gate

    @property
    def loads_submitted(self) -> bool:
        return len(self._load_futures) == len(self.load_order)

    @property
    def loads_claimed(self) -> bool:
        return self._load_pos == len(self.load_order)

    @property
    def pending_load_layers(self) -> tuple[int, ...]:
        return self.load_order[self._load_pos :]

    def abort_load(self, finish: Callable[[], None] | None = None, *, wait: bool = True) -> None:
        self._load_aborted.set()
        for position, future in enumerate(self._load_futures):
            gate = self._load_gates[self.load_order[position]]
            if gate is not None and not future.done():
                gate.cancel()
        if finish is not None:
            self.finish_load(finish)
        if wait:
            self.wait_load_finished()

    def clear_callbacks(self) -> None:
        for gate in self._start_gates.values():
            gate.on_start = None
