"""Business entry point for Scheduler Load."""

from __future__ import annotations

from dataclasses import dataclass

from ...metadata import LoadRequest
from ..request_tracker import RequestTracker
from .scheduling import LoadScheduling


@dataclass(frozen=True, slots=True)
class LoadCandidate:
    """A Lookup hit awaiting vLLM block-allocation confirmation."""

    local_cached_tokens: int
    kv_pool_cached_tokens: int


class LoadService:
    """Own Load candidates from Lookup through transfer request publication."""

    def __init__(self, scheduling: LoadScheduling) -> None:
        self._scheduling = scheduling
        self._pending_candidates: dict[str, LoadCandidate] = {}

    @property
    def is_deferred(self) -> bool:
        return self._scheduling.is_deferred

    def record_candidate(self, request_id: str, candidate: LoadCandidate) -> None:
        self._pending_candidates[request_id] = candidate

    def confirm_allocation(self, request_id: str, allocated_external_tokens: int) -> LoadCandidate | None:
        candidate = self._pending_candidates.get(request_id)
        if candidate is None:
            return None
        if allocated_external_tokens == 0:
            return None

        expected_tokens = candidate.kv_pool_cached_tokens - candidate.local_cached_tokens
        assert allocated_external_tokens == expected_tokens, (
            f"Mismatch in number of tokens: {allocated_external_tokens} vs "
            f"{candidate.kv_pool_cached_tokens} - {candidate.local_cached_tokens} for request {request_id}"
        )
        self._pending_candidates.pop(request_id)
        return self._scheduling.confirm(request_id, candidate)

    def take_for_transfer(self, request_id: str) -> LoadCandidate | None:
        self._pending_candidates.pop(request_id, None)
        return self._scheduling.take_for_transfer(request_id)

    def take_ready_for_transfer(self) -> list[tuple[str, LoadCandidate]]:
        return self._scheduling.take_ready_for_transfer()

    def schedule_request(
        self,
        tracker: RequestTracker,
        transfer_end_token: int,
        candidate: LoadCandidate,
    ) -> LoadRequest:
        return LoadRequest(
            request_id=tracker.request_id,
            transfer_end_token=transfer_end_token,
            block_ids_by_group=tuple(tuple(block_ids) for block_ids in tracker.block_ids_by_group),
            block_hashes=tuple(tracker.block_hashes),
            local_cached_tokens=candidate.local_cached_tokens,
            kv_pool_cached_tokens=candidate.kv_pool_cached_tokens,
        )

    def discard_transfer(self, request_id: str) -> None:
        candidate = self._scheduling.discard(request_id)
        if candidate is not None:
            self._pending_candidates[request_id] = candidate
