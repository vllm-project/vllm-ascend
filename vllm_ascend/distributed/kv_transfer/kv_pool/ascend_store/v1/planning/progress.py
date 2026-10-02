"""Scheduler-owned request progress and Load publication timing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from vllm.v1.core.kv_cache_utils import BlockHash

from ..coordinates import TokenRange
from ..protocol.lookup import TailKeyBoundary


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    """Immutable request position used to derive Load and Store work."""

    request_id: str
    request_token_len: int
    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[BlockHash, ...]
    num_prompt_tokens: int
    published_store_end_token: int = 0
    committable_end_token: int | None = None

    def with_position(
        self,
        request_token_len: int,
        new_block_ids: tuple[tuple[int, ...], ...] | None,
        block_hashes: Sequence[BlockHash],
        committable_end_token: int,
    ) -> RequestSnapshot:
        """Replace optimistic progress with vLLM's current authoritative position."""

        block_ids_by_group = self.block_ids_by_group
        if new_block_ids:
            block_ids_by_group = tuple(
                block_ids + tuple(new_ids)
                for block_ids, new_ids in zip(self.block_ids_by_group, new_block_ids, strict=True)
            )
        return replace(
            self,
            request_token_len=request_token_len,
            block_ids_by_group=block_ids_by_group,
            block_hashes=tuple(block_hashes),
            committable_end_token=committable_end_token,
        )

    @property
    def store_end_token(self) -> int:
        if self.committable_end_token is None:
            return self.request_token_len
        return min(self.request_token_len, self.committable_end_token)

    def with_published_store_end(self, end_token: int) -> RequestSnapshot:
        if end_token < self.published_store_end_token:
            raise ValueError(
                f"Store publication for request {self.request_id} retreats from "
                f"{self.published_store_end_token} to {end_token}"
            )
        return replace(self, published_store_end_token=end_token)


@dataclass(frozen=True, slots=True)
class LoadCandidate:
    """An available remote range awaiting local allocation confirmation."""

    load_range: TokenRange
    matched_end_token: int
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()


class LoadPublication(Protocol):
    """Control when an allocation-confirmed Load enters a Worker step."""

    is_deferred: bool

    def confirm(self, request_id: str, candidate: LoadCandidate) -> LoadCandidate | None: ...

    def take_scheduled(self, request_id: str) -> LoadCandidate | None: ...

    def take_ready(self) -> list[tuple[str, LoadCandidate]]: ...

    def discard(self, request_id: str) -> LoadCandidate | None: ...


class ScheduledLoadPublication:
    """Publish a confirmed Load when vLLM next schedules its request."""

    is_deferred = False

    def __init__(self) -> None:
        self._confirmed: dict[str, LoadCandidate] = {}

    def confirm(self, request_id: str, candidate: LoadCandidate) -> None:
        self._confirmed[request_id] = candidate

    def take_scheduled(self, request_id: str) -> LoadCandidate | None:
        return self._confirmed.pop(request_id, None)

    def take_ready(self) -> list[tuple[str, LoadCandidate]]:
        return []

    def discard(self, request_id: str) -> LoadCandidate | None:
        return self._confirmed.pop(request_id, None)


class AllocationLoadPublication:
    """Publish a confirmed asynchronous Load in the next metadata envelope."""

    is_deferred = True

    def __init__(self) -> None:
        self._ready: dict[str, LoadCandidate] = {}

    def confirm(self, request_id: str, candidate: LoadCandidate) -> LoadCandidate:
        self._ready[request_id] = candidate
        return candidate

    def take_scheduled(self, request_id: str) -> None:
        return None

    def take_ready(self) -> list[tuple[str, LoadCandidate]]:
        ready = list(self._ready.items())
        self._ready.clear()
        return ready

    def discard(self, request_id: str) -> LoadCandidate | None:
        return self._ready.pop(request_id, None)
