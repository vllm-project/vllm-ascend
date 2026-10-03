"""Shared contracts and completion fences for KV Pool timelines."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..batch import KVTransferBatch
from ..evidence import LoadCompletion, StoreCompletion


class LoadTimelineProtocol(Protocol):
    collects_completions: bool

    def bind_operation(
        self,
        operation: Callable[[KVTransferBatch, int | None], tuple[LoadCompletion, ...]],
    ) -> None: ...

    def start(self) -> None: ...

    def submit(self, batch: KVTransferBatch) -> Iterable[LoadCompletion]: ...

    def collect(self) -> Iterable[LoadCompletion]: ...

    def abort(self) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class StoreBatch:
    """One accepted Store batch and its exact completion fence."""

    transfer: KVTransferBatch
    completed: threading.Event = field(default_factory=threading.Event)
    completions: list[StoreCompletion] = field(default_factory=list)


class StoreTimelineProtocol(Protocol):
    def bind_operation(
        self,
        operation: Callable[[KVTransferBatch, Any, int | None], tuple[StoreCompletion, ...]],
    ) -> None: ...

    def start(self) -> None: ...

    def submit(self, batch: KVTransferBatch, source_ready_event: Any) -> StoreBatch: ...

    def wait(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]: ...

    def close(self) -> None: ...
