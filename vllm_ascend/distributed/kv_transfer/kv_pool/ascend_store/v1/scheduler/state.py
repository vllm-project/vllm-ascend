"""Immutable Scheduler configuration, Lookup facts, and request progress."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from vllm.v1.core.kv_cache_utils import BlockHash

from ..coordinates import TokenRange
from ..protocol.lookup import TailKeyBoundary


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Static facts that select and constrain one Scheduler route."""

    cache_transfer_granularity: int
    hash_block_size: int
    transfer_group_ids: tuple[int, ...]
    align_state_group_ids: frozenset[int]
    load_enabled: bool
    store_enabled: bool
    save_decode_cache: bool
    discard_partial_chunks: bool
    use_eagle_block_drop: bool
    has_private_state: bool
    expected_worker_count: int


@dataclass(frozen=True, slots=True)
class LoadCandidate:
    """Three remote prefix frontiers waiting for allocation confirmation."""

    local_cached_tokens: int
    readable_end_token: int
    store_skip_end_token: int
    safe_load_end_token: int
    load_range: TokenRange
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
    """Authoritative Scheduler position used for monotonic Store publication."""

    request_id: str
    request_token_len: int
    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[BlockHash, ...]
    prefill_end_token: int
    store_skip_end_token: int = 0
    committable_end_token: int | None = None

    def advance(
        self,
        request_token_len: int,
        new_block_ids: tuple[tuple[int, ...], ...] | None,
        block_hashes: Sequence[BlockHash],
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
            block_ids_by_group = tuple(
                block_ids + appended_ids
                for block_ids, appended_ids in zip(block_ids_by_group, new_block_ids, strict=True)
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

    def with_store_skip_end(self, end_token: int) -> RequestProgress:
        if end_token < self.store_skip_end_token:
            raise ValueError(
                f"Store publication for request {self.request_id} retreats from "
                f"{self.store_skip_end_token} to {end_token}"
            )
        return replace(self, store_skip_end_token=end_token)
