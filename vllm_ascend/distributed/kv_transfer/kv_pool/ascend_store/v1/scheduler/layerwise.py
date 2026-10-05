"""Layerwise Scheduler route with allocation-confirmed scheduled Load."""

from __future__ import annotations

from ..protocol.transfer import LoadCommand
from .base import KVPoolScheduler
from .state import LoadCandidate


class LayerwiseScheduler(KVPoolScheduler):
    """Load per-layer data independently of the local HBM prefix allocation."""

    store_with_scheduled_load = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._scheduled_loads: dict[str, LoadCandidate] = {}

    def _lookup_start_token(self, local_cached_tokens: int, query_end: int) -> int:
        del local_cached_tokens, query_end
        return 0

    def _confirm_load_candidate(
        self,
        request_id: str,
        candidate: LoadCandidate,
        *,
        selected_for_load: bool,
    ) -> None:
        if selected_for_load or candidate.num_new_matched_tokens == 0:
            self._scheduled_loads[request_id] = candidate

    def _take_scheduled_load(self, request_id: str) -> LoadCommand | None:
        candidate = self._scheduled_loads.pop(request_id, None)
        return None if candidate is None else self._make_load_command(request_id, candidate)

    def _take_allocation_ready_loads(self) -> list[LoadCommand]:
        return []

    def _discard_confirmed_load(self, request_id: str) -> None:
        self._scheduled_loads.pop(request_id, None)
