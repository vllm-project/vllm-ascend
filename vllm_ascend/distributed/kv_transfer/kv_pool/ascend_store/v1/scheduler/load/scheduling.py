"""Scheduler-side timing variants for Load requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .service import LoadCandidate


class LoadScheduling(Protocol):
    """Own confirmed candidates according to one Load scheduling mode."""

    is_deferred: bool

    def confirm(self, request_id: str, candidate: LoadCandidate) -> LoadCandidate | None: ...

    def take_for_transfer(self, request_id: str) -> LoadCandidate | None: ...

    def take_ready_for_transfer(self) -> list[tuple[str, LoadCandidate]]: ...

    def discard(self, request_id: str) -> LoadCandidate | None: ...


class ImmediateLoadScheduling:
    """Retain confirmed candidates until vLLM schedules their requests."""

    is_deferred = False

    def __init__(self) -> None:
        self._confirmed_candidates: dict[str, LoadCandidate] = {}

    def confirm(self, request_id: str, candidate: LoadCandidate) -> None:
        self._confirmed_candidates[request_id] = candidate

    def take_for_transfer(self, request_id: str) -> LoadCandidate | None:
        return self._confirmed_candidates.pop(request_id, None)

    def take_ready_for_transfer(self) -> list[tuple[str, LoadCandidate]]:
        return []

    def discard(self, request_id: str) -> LoadCandidate | None:
        return self._confirmed_candidates.pop(request_id, None)


class DeferredLoadScheduling:
    """Publish confirmed candidates immediately after block allocation."""

    is_deferred = True

    def __init__(self) -> None:
        self._ready_candidates: dict[str, LoadCandidate] = {}

    def confirm(self, request_id: str, candidate: LoadCandidate) -> LoadCandidate:
        self._ready_candidates[request_id] = candidate
        return candidate

    def take_for_transfer(self, request_id: str) -> None:
        return None

    def take_ready_for_transfer(self) -> list[tuple[str, LoadCandidate]]:
        ready_candidates = list(self._ready_candidates.items())
        self._ready_candidates.clear()
        return ready_candidates

    def discard(self, request_id: str) -> LoadCandidate | None:
        return self._ready_candidates.pop(request_id, None)
