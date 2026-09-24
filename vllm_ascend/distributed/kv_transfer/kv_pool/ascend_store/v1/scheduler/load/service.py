"""Business entry point for classic Scheduler Load."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoadCandidate:
    """A Lookup hit awaiting vLLM block-allocation confirmation."""

    vllm_cached_tokens: int
    kvpool_cached_tokens: int


class LoadService:
    """Own Load candidates from Lookup through asynchronous completion."""

    def __init__(self, load_async: bool = False) -> None:
        self._load_async = load_async
        self._pending_candidates: dict[str, LoadCandidate] = {}
        self._confirmed_candidates: dict[str, LoadCandidate] = {}
        self._ready_candidates: dict[str, LoadCandidate] = {}
        self._inflight_request_ids: set[str] = set()

    def executes_asynchronously(self) -> bool:
        return self._load_async

    def record_candidate(self, request_id: str, candidate: LoadCandidate) -> None:
        self._pending_candidates[request_id] = candidate

    def confirm_allocation(self, request_id: str, num_external_tokens: int) -> LoadCandidate | None:
        candidate = self._pending_candidates.get(request_id)
        if candidate is None:
            return None
        if num_external_tokens == 0:
            return None

        expected_tokens = candidate.kvpool_cached_tokens - candidate.vllm_cached_tokens
        assert num_external_tokens == expected_tokens, (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{candidate.kvpool_cached_tokens} - {candidate.vllm_cached_tokens} for request {request_id}"
        )
        self._pending_candidates.pop(request_id)
        if self._load_async:
            self._ready_candidates[request_id] = candidate
        else:
            self._confirmed_candidates[request_id] = candidate
        return candidate

    def take_for_transfer(self, request_id: str) -> LoadCandidate | None:
        self._pending_candidates.pop(request_id, None)
        return self._confirmed_candidates.pop(request_id, None)

    def take_ready_for_transfer(self) -> list[tuple[str, LoadCandidate]]:
        ready_candidates = list(self._ready_candidates.items())
        self._ready_candidates.clear()
        return ready_candidates

    def record_inflight(self, request_id: str) -> None:
        self._inflight_request_ids.add(request_id)

    def finish(self, request_ids: set[str] | None) -> None:
        if request_ids:
            self._inflight_request_ids.difference_update(request_ids)

    def discard_transfer(self, request_id: str) -> None:
        candidate = self._ready_candidates.pop(request_id, None)
        if candidate is None:
            candidate = self._confirmed_candidates.pop(request_id, None)
        if candidate is not None:
            self._pending_candidates[request_id] = candidate
        self._inflight_request_ids.discard(request_id)

    def inflight_request_ids(self) -> set[str]:
        return self._inflight_request_ids.copy()
