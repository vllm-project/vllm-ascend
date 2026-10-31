"""Lookup facts and Scheduler request progress."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import BlockHash

from ..protocol.messages import TailKeyBoundary

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass(frozen=True, slots=True)
class LoadCandidate:
    """Load boundaries and Store-skip progress waiting for allocation confirmation."""

    local_cached_tokens: int
    store_skip_end_token: int
    safe_load_end_token: int
    load_start_token: int
    load_end_token: int
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()

    @property
    def num_new_matched_tokens(self) -> int:
        return max(0, self.safe_load_end_token - self.local_cached_tokens)


@dataclass(frozen=True, slots=True)
class RequestAllocation:
    """vLLM allocation confirmation retained while a request is active."""

    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[BlockHash, ...]
    store_skip_end_token: int = 0


@dataclass(frozen=True, slots=True)
class RequestProgress:
    """Request position and Store planning progress for the current allocation."""

    request_id: str
    request_token_len: int
    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[BlockHash, ...]
    prefill_end_token: int
    # Starts at Lookup's known remote frontier; advances when Store commands are issued.
    store_planned_end_token: int = 0
    committable_end_token: int | None = None

    def advance(
        self,
        request_token_len: int,
        new_block_ids: Sequence[Sequence[int]] | None,
        block_hashes: tuple[BlockHash, ...],
        committable_end_token: int,
    ) -> RequestProgress:
        """Replace request position and append only vLLM-declared block suffixes."""

        block_ids_by_group = self.block_ids_by_group
        if new_block_ids is not None:
            if len(new_block_ids) != len(block_ids_by_group):
                raise ValueError(
                    f"Request {self.request_id} appended {len(new_block_ids)} cache groups to "
                    f"a {len(block_ids_by_group)}-group allocation"
                )
            if any(new_block_ids):
                block_ids_by_group = tuple(
                    block_ids + tuple(appended_ids) if appended_ids else block_ids
                    for block_ids, appended_ids in zip(block_ids_by_group, new_block_ids, strict=True)
                )
        return replace(
            self,
            request_token_len=request_token_len,
            block_ids_by_group=block_ids_by_group,
            block_hashes=block_hashes,
            committable_end_token=committable_end_token,
        )

    @property
    def store_end_token(self) -> int:
        if self.committable_end_token is None:
            return self.request_token_len
        return min(self.request_token_len, self.committable_end_token)

    def with_store_planned_end(self, end_token: int) -> RequestProgress:
        if end_token < self.store_planned_end_token:
            raise ValueError(
                f"Store planning for request {self.request_id} retreats from "
                f"{self.store_planned_end_token} to {end_token}"
            )
        return replace(self, store_planned_end_token=end_token)


@dataclass(slots=True)
class RequestState:
    """Own one request's candidates, confirmed allocation, and scheduled progress.

    Allocation and progress snapshots describe different points in time. Commands
    retain those immutable snapshots when the request advances or is discarded.
    Store source leases outlive this state and belong to the Scheduler's job ledger.
    """

    request: Request
    pending_lookup: LoadCandidate | None = None
    allocation: RequestAllocation | None = None
    pending_load: LoadCandidate | None = None
    progress: RequestProgress | None = None
    _hash_source: Sequence[BlockHash] | None = field(default=None, init=False, repr=False, compare=False)
    _hash_snapshot: tuple[BlockHash, ...] = field(default=(), init=False, repr=False, compare=False)

    def snapshot_hashes(self, hashes: Sequence[BlockHash]) -> tuple[BlockHash, ...]:
        """Reuse one immutable snapshot of vLLM's append-only confirmed hashes.

        A different source, including a replacement with the same length, starts
        a new snapshot. Old commands keep their original tuple when hashes grow.
        """
        if hashes is not self._hash_source or len(hashes) != len(self._hash_snapshot):
            self._hash_snapshot = tuple(hashes)
            self._hash_source = hashes
        return self._hash_snapshot
