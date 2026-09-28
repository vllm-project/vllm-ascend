"""Shared contracts and values for KV Pool execution timelines."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ...graph.elements import BindingBatch, KVBinding
from ..io import BindingEvidence, StoreEvidence


@dataclass(frozen=True, slots=True)
class LoadTransfer:
    """One request's immutable Load work after spatial projection."""

    request_id: str
    traversal: tuple[KVBinding, ...]


@dataclass(frozen=True, slots=True)
class LoadCompletion:
    """Load evidence produced when one transfer reaches its timeline completion point."""

    request_id: str
    binding_evidence: tuple[BindingEvidence, ...]


class LoadTimelineProtocol(Protocol):
    """Control when fixed Load work is executed and made visible."""

    collects_completions: bool

    def attach_operation(self, operation: Callable[[LoadTransfer], LoadCompletion]) -> None: ...

    def start(self) -> None: ...

    def submit(self, transfers: list[LoadTransfer]) -> Iterable[LoadCompletion]: ...

    def collect(self) -> Iterable[LoadCompletion]: ...

    def abort(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StoreTransfer:
    """One request's immutable Store work after spatial projection."""

    request_id: str
    batches: tuple[BindingBatch, ...]


@dataclass(frozen=True, slots=True)
class StoreCompletion:
    """Store completion with transfer and source-release facts kept separate."""

    request_id: str
    evidence: StoreEvidence


@dataclass(slots=True)
class StoreBatch:
    """One accepted step batch and its exact completion fence."""

    transfers: tuple[StoreTransfer, ...]
    completed: threading.Event = field(default_factory=threading.Event)
    completions: list[StoreCompletion] = field(default_factory=list)


class StoreTimelineProtocol(Protocol):
    """Submit whole-step Store work and expose its completion fence."""

    def attach_operation(self, operation: Callable[[StoreTransfer, Any], StoreCompletion]) -> None: ...

    def start(self) -> None: ...

    def submit(self, transfers: list[StoreTransfer], source_ready_event: Any) -> StoreBatch: ...

    def wait(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]: ...

    def close(self) -> None: ...
