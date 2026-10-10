"""Bulk Scheduler route with allocation-ready asynchronous Load publication."""

from __future__ import annotations

from ..protocol.transfer import LoadCommand
from .base import KVPoolScheduler
from .state import LoadCandidate


class AsynchronousBulkScheduler(KVPoolScheduler):
    """Publish an allocated Bulk Load in the next metadata envelope."""

    load_is_deferred = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ready_loads: dict[str, LoadCandidate] = {}

    def _confirm_load_candidate(
        self,
        request_id: str,
        candidate: LoadCandidate,
        *,
        selected_for_load: bool,
    ) -> None:
        if selected_for_load:
            self._ready_loads[request_id] = candidate

    def _take_scheduled_load(self, request_id: str) -> LoadCommand | None:
        candidate = self._ready_loads.pop(request_id, None)
        return None if candidate is None else self._make_load_command(request_id, candidate)

    def _take_allocation_ready_loads(self) -> list[LoadCommand]:
        ready = self._ready_loads
        self._ready_loads = {}
        commands = []
        for request_id, candidate in ready.items():
            command = self._make_load_command(request_id, candidate)
            if command is not None:
                commands.append(command)
        return commands

    def _discard_confirmed_load(self, request_id: str) -> None:
        self._ready_loads.pop(request_id, None)
