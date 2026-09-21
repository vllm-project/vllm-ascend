"""Scheduler-side components for the AscendStore v1 classic path."""

from __future__ import annotations

from .communication import LookupChannel
from .protocol import LookupRequest, LookupResponse


class SchedulerAdapter:
    """Boundary component for vLLM scheduler hooks."""

    def __init__(self, service: SchedulerService) -> None:
        self._service = service


class SchedulerService:
    """Scheduler-side AscendStore service boundary."""

    def __init__(self, lookup_channel: LookupChannel) -> None:
        self._lookup_channel = lookup_channel

    def lookup(self, request: LookupRequest) -> LookupResponse:
        return self._lookup_channel.lookup(request)
