"""Shared contracts and values for KV Pool runtime timelines."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ...program.invocation import LoadCompletion, LoadTransfer, StoreCompletion, StoreTransfer


class LoadTimelineProtocol(Protocol):
    """Control when fixed Load work is executed and made visible."""

    collects_completions: bool

    def bind_operation(self, operation: Callable[[LoadTransfer], LoadCompletion]) -> None: ...

    def start(self) -> None: ...

    def submit(self, transfers: list[LoadTransfer]) -> Iterable[LoadCompletion]: ...

    def collect(self) -> Iterable[LoadCompletion]: ...

    def abort(self) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class StoreBatch:
    """One accepted step batch and its exact completion fence."""

    transfers: tuple[StoreTransfer, ...]
    completed: threading.Event = field(default_factory=threading.Event)
    completions: list[StoreCompletion] = field(default_factory=list)


class StoreTimelineProtocol(Protocol):
    """Submit whole-step Store work and expose its completion fence."""

    def bind_operation(self, operation: Callable[[StoreTransfer, Any], StoreCompletion]) -> None: ...

    def start(self) -> None: ...

    def submit(self, transfers: list[StoreTransfer], source_ready_event: Any) -> StoreBatch: ...

    def wait(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]: ...

    def close(self) -> None: ...
