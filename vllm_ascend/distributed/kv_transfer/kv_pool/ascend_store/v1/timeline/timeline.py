"""Select and bind the state machines of one KV Pool timeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..protocol.transfer import StoreCommand
from ..runtime.batch import KVTransferBatch
from ..runtime.evidence import LoadCompletion, StoreCompletion
from ..topology import KVPoolTopology
from . import (
    BulkStoreOperation,
    LayerwiseStoreOperation,
    LayerwiseStorePreparation,
    LoadOperation,
    LoadTimelineProtocol,
    StoreBatch,
    StoreTimelineProtocol,
)
from .bulk import AsyncLoadTimeline, LoadTimeline, StoreTimeline
from .layerwise import (
    LayerwiseBackendOperations,
    LayerwiseLoadTimeline,
    LayerwiseLoadTimelineProtocol,
    LayerwiseStoreTimeline,
    LayerwiseStoreTimelineProtocol,
)
from .schedule import KVPoolSchedule, LoadScheduleKind, StoreScheduleKind


class KVPoolTimeline:
    """Own the state machines selected by one Load/Store schedule."""

    def __init__(self, schedule: KVPoolSchedule, topology: KVPoolTopology) -> None:
        self._schedule = schedule
        self._topology = topology
        self._load: LoadTimelineProtocol | None = None
        self._store: StoreTimelineProtocol | LayerwiseStoreTimelineProtocol | None = None
        self._layerwise_load: LayerwiseLoadTimelineProtocol | None = None
        self._bulk_store: StoreTimelineProtocol | None = None
        self._layerwise_store: LayerwiseStoreTimelineProtocol | None = None
        self._source_ready_event_factory: Callable[[], Any] | None = None

    @property
    def store_enabled(self) -> bool:
        return self._schedule.store_enabled

    @property
    def fences_store_on_finish(self) -> bool:
        return self._schedule.fences_store_on_finish

    @property
    def prepares_store_by_layer(self) -> bool:
        return self._schedule.store_kind is StoreScheduleKind.LAYERWISE

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
        bulk_store_operation: BulkStoreOperation,
        layerwise_store_operation: LayerwiseStoreOperation,
        layerwise_store_preparation: LayerwiseStorePreparation,
        layerwise_backend: LayerwiseBackendOperations | None,
        start_gate_factory: Callable[[], Any],
    ) -> None:
        if self._load is not None:
            raise RuntimeError("KV Pool timeline resources are already bound")
        if self._schedule.requires_layerwise_backend and layerwise_backend is None:
            raise TypeError("Layerwise timeline requires layerwise Backend operations")

        if self._schedule.load_kind is LoadScheduleKind.LAYERWISE:
            assert layerwise_backend is not None
            layerwise_load = LayerwiseLoadTimeline(
                self._topology,
                layerwise_backend,
                self._schedule.layerwise_prefetch_layers,
                thread_initializer,
                start_gate_factory,
            )
            self._layerwise_load = layerwise_load
            load: LoadTimelineProtocol = layerwise_load
        elif self._schedule.load_kind is LoadScheduleKind.ASYNC:
            load = AsyncLoadTimeline(thread_initializer)
        else:
            load = LoadTimeline()
        load.bind_operation(load_operation)
        self._load = load

        if self._schedule.store_kind is StoreScheduleKind.LAYERWISE:
            assert layerwise_backend is not None
            layerwise_store = LayerwiseStoreTimeline(self._topology, layerwise_backend, thread_initializer)
            layerwise_store.bind_preparation(layerwise_store_preparation)
            layerwise_store.bind_operation(layerwise_store_operation)
            self._layerwise_store = layerwise_store
            self._store = layerwise_store
        elif self._schedule.store_kind is StoreScheduleKind.ASYNC:
            bulk_store = StoreTimeline(thread_initializer)
            bulk_store.bind_operation(bulk_store_operation)
            self._bulk_store = bulk_store
            self._store = bulk_store
        self._source_ready_event_factory = source_ready_event_factory

    def start(self) -> None:
        if self._store is not None:
            self._store.start()
        self.load.start()

    def submit_load(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        return tuple(self.load.submit(batch))

    def collect_load(self) -> tuple[LoadCompletion, ...]:
        return tuple(self.load.collect())

    def wait_for_load_layer(self, layer_name: str) -> tuple[LoadCompletion, ...]:
        if self._layerwise_load is None:
            return ()
        return tuple(self._layerwise_load.wait_for_layer(layer_name))

    def abort_load(self) -> None:
        self.load.abort()

    def prepare_store(self, commands: tuple[StoreCommand, ...]) -> None:
        if self._layerwise_store is not None:
            self._layerwise_store.prepare(commands)

    def submit_store_layer(self, layer_name: str) -> None:
        if self._layerwise_store is not None:
            self._layerwise_store.submit_layer(layer_name, self._record_source_ready)

    def finish_store(self, commands: tuple[StoreCommand, ...]) -> StoreBatch:
        if self._store is None:
            raise RuntimeError("KV Pool runtime has no Store timeline")
        if self._layerwise_store is not None:
            return self._layerwise_store.finalize()
        assert self._bulk_store is not None
        return self._bulk_store.submit(commands, self._record_source_ready())

    def wait_store(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]:
        if self._store is None:
            return ()
        return self._store.wait(batch)

    def prepare_store_close(self) -> StoreBatch | None:
        if self._layerwise_store is None:
            return None
        return self._layerwise_store.prepare_close()

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
