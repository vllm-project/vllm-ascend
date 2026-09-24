"""Business entry point for classic Scheduler Load."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LoadCandidate:
    """A Lookup hit awaiting vLLM block-allocation confirmation."""

    vllm_cached_tokens: int
    kvpool_cached_tokens: int
    allocation_confirmed: bool = False


class LoadService:
    """Own Load candidates from Lookup until their scheduled transfer."""

    def __init__(self) -> None:
        self._pending_candidates: dict[str, LoadCandidate] = {}

    def record_candidate(self, request_id: str, candidate: LoadCandidate) -> None:
        self._pending_candidates[request_id] = candidate

    def confirm_allocation(self, request_id: str, num_external_tokens: int) -> None:
        candidate = self._pending_candidates.get(request_id)
        if candidate is None:
            return
        if num_external_tokens == 0:
            candidate.allocation_confirmed = False
            return

        expected_tokens = candidate.kvpool_cached_tokens - candidate.vllm_cached_tokens
        assert num_external_tokens == expected_tokens, (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{candidate.kvpool_cached_tokens} - {candidate.vllm_cached_tokens} for request {request_id}"
        )
        candidate.allocation_confirmed = True

    def take_for_transfer(self, request_id: str) -> LoadCandidate | None:
        return self._pending_candidates.pop(request_id, None)
