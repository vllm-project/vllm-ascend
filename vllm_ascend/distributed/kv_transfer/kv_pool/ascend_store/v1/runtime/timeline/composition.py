"""Bind a compiled KV Pool schedule to runtime timeline state."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from ...program.invocation import LoadCompletion, LoadTransfer, StoreCompletion, StoreTransfer
from ...program.spec.schedule import KVPoolSchedule, LoadScheduleKind, StoreScheduleKind
from ...program.spec.topology import KVPoolTopology
from . import LoadTimelineProtocol, StoreBatch, StoreTimelineProtocol
from .bulk import AsyncLoadTimeline, LoadTimeline, StoreTimeline
from .layerwise import (
    LayerwiseBackendOperations,
    LayerwiseLoadTimeline,
    LayerwiseLoadTimelineProtocol,
    LayerwiseStoreTimeline,
    LayerwiseStoreTimelineProtocol,
)

LoadOperation = Callable[[LoadTransfer], LoadCompletion]
StoreOperation = Callable[[StoreTransfer, Any], StoreCompletion]
StoreAdmission = Callable[[list[StoreTransfer]], list[StoreTransfer]]


class KVPoolTimelineRuntime:
    """Own runtime timelines selected by one compiled Load/Store schedule."""

    def __init__(self, schedule: KVPoolSchedule, topology: KVPoolTopology) -> None:
        self._schedule = schedule
        self._topology = topology
        self._load: LoadTimelineProtocol | None = None
        self._store: StoreTimelineProtocol | LayerwiseStoreTimelineProtocol | None = None
        self._source_ready_event_factory: Callable[[], Any] | None = None

    @property
    def store_enabled(self) -> bool:
        return self._schedule.store_enabled

    @property
    def fences_store_on_finish(self) -> bool:
        return self._schedule.fences_store_on_finish

    @property
    def collects_load_completions(self) -> bool:
        return self.load.collects_completions

    @property
    def load(self) -> LoadTimelineProtocol:
        if self._load is None:
            raise RuntimeError("KV Pool timeline resources have not been bound")
        return self._load

    @property
    def store(self) -> StoreTimelineProtocol | LayerwiseStoreTimelineProtocol | None:
        return self._store

    def bind_resources(
        self,
        thread_initializer: Callable[[], None],
        source_ready_event_factory: Callable[[], Any],
        load_operation: LoadOperation,
        store_operation: StoreOperation,
        store_transfer_operation: StoreOperation,
        store_admission: StoreAdmission,
        layerwise_backend: LayerwiseBackendOperations | None,
        start_gate_factory: Callable[[], Any],
    ) -> None:
        if self._load is not None:
            raise RuntimeError("KV Pool timeline resources are already bound")
        if self._schedule.requires_layerwise_backend and layerwise_backend is None:
            raise TypeError("Layerwise timeline requires layerwise Backend operations")

        if self._schedule.load_kind is LoadScheduleKind.LAYERWISE:
            assert layerwise_backend is not None
            self._load = LayerwiseLoadTimeline(
                self._topology,
                layerwise_backend,
                self._schedule.layerwise_prefetch_layers,
                thread_initializer,
                start_gate_factory,
            )
        elif self._schedule.load_kind is LoadScheduleKind.ASYNC:
            self._load = AsyncLoadTimeline(thread_initializer)
        else:
            self._load = LoadTimeline()
        self._load.bind_operation(load_operation)

        if self._schedule.store_kind is StoreScheduleKind.LAYERWISE:
            assert layerwise_backend is not None
            layerwise_store = LayerwiseStoreTimeline(self._topology, layerwise_backend, thread_initializer)
            layerwise_store.bind_admission(store_admission)
            layerwise_store.bind_operation(store_transfer_operation)
            self._store = layerwise_store
        elif self._schedule.store_kind is StoreScheduleKind.ASYNC:
            store = StoreTimeline(thread_initializer)
            store.bind_operation(store_operation)
            self._store = store
        self._source_ready_event_factory = source_ready_event_factory

    def start(self) -> None:
        if self._store is not None:
            self._store.start()
        self.load.start()

    def submit_load(self, transfers: list[LoadTransfer]) -> tuple[LoadCompletion, ...]:
        return tuple(self.load.submit(transfers))

    def collect_load(self) -> tuple[LoadCompletion, ...]:
        return tuple(self.load.collect())

    def wait_for_load_layer(self, layer_name: str) -> tuple[LoadCompletion, ...]:
        if self._schedule.load_kind is not LoadScheduleKind.LAYERWISE:
            return ()
        return tuple(cast(LayerwiseLoadTimelineProtocol, self.load).wait_for_layer(layer_name))

    def abort_load(self) -> None:
        self.load.abort()

    def prepare_store(self, transfers: list[StoreTransfer]) -> None:
        if self._schedule.store_kind is StoreScheduleKind.LAYERWISE:
            cast(LayerwiseStoreTimelineProtocol, self._store).prepare(transfers)

    def submit_store_layer(self, layer_name: str) -> None:
        if self._schedule.store_kind is not StoreScheduleKind.LAYERWISE:
            return
        cast(LayerwiseStoreTimelineProtocol, self._store).submit_layer(layer_name, self._record_source_ready())

    def finish_store(self, transfers: list[StoreTransfer]) -> StoreBatch:
        if self._store is None:
            raise RuntimeError("KV Pool program has no Store timeline")
        if self._schedule.store_kind is StoreScheduleKind.LAYERWISE:
            return cast(LayerwiseStoreTimelineProtocol, self._store).finalize()
        return cast(StoreTimelineProtocol, self._store).submit(transfers, self._record_source_ready())

    def wait_store(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]:
        if self._store is None:
            return ()
        return self._store.wait(batch)

    def close(self) -> None:
        close_error: BaseException | None = None
        if self._store is not None:
            try:
                self._store.close()
            except BaseException as error:
                close_error = error
        try:
            self.load.close()
        except BaseException as error:
            if close_error is None:
                close_error = error
        if close_error is not None:
            raise close_error

    def _record_source_ready(self) -> Any:
        if self._source_ready_event_factory is None:
            raise RuntimeError("KV Pool timeline resources have not been bound")
        source_ready_event = self._source_ready_event_factory()
        source_ready_event.record()
        return source_ready_event
