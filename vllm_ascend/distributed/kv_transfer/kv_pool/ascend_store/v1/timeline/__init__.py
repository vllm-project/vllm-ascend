"""Shared contracts and completion fences for KV Pool timelines."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..protocol.transfer import StoreCommand
from ..runtime.batch import KVTransferBatch
from ..runtime.evidence import LayerStoreResult, LoadCompletion, StoreCompletion

LoadOperation = Callable[[KVTransferBatch, int | None], tuple[LoadCompletion, ...]]
BulkStoreOperation = Callable[[tuple[StoreCommand, ...], Any], tuple[StoreCompletion, ...]]
LayerwiseStoreOperation = Callable[[Any, int], LayerStoreResult]
LayerwiseStorePreparation = Callable[
    [tuple[StoreCommand, ...]], tuple[KVTransferBatch | None, tuple[StoreCompletion, ...]]
]


class LoadTimelineProtocol(Protocol):
    collects_completions: bool

    def bind_operation(self, operation: LoadOperation) -> None: ...

    def start(self) -> None: ...

    def submit(self, batch: KVTransferBatch) -> Iterable[LoadCompletion]: ...

    def collect(self) -> Iterable[LoadCompletion]: ...

    def abort(self) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class StoreBatch:
    """The exact completion fence for one submitted Store invocation."""

    completed: threading.Event = field(default_factory=threading.Event)
    completions: list[StoreCompletion] = field(default_factory=list)


class StoreTimelineProtocol(Protocol):
    def bind_operation(self, operation: BulkStoreOperation) -> None: ...

    def start(self) -> None: ...

    def submit(self, commands: tuple[StoreCommand, ...], source_ready_event: Any) -> StoreBatch: ...

    def wait(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]: ...

    def close(self) -> None: ...
