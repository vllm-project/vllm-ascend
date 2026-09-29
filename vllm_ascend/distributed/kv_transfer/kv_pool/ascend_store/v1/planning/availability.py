"""Translate vLLM prefix queries into remote KV availability."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash

from ..coordinates import TokenRange
from ..protocol.lookup import LookupRequest, TailKeyBoundary
from ..protocol.rpc import LookupClient


@dataclass(frozen=True, slots=True)
class LookupQuery:
    """vLLM request facts required to query remote KV availability."""

    request_id: str
    prompt_token_len: int
    request_token_len: int
    block_hashes: list[BlockHash]
    local_cached_tokens: int


@dataclass(frozen=True, slots=True)
class RemoteAvailability:
    """Remote Load range and the prefix endpoint vLLM may accept."""

    load_range: TokenRange
    matched_end_token: int
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()


@dataclass(frozen=True, slots=True)
class ExternalPrefixPlan:
    """External prefix accepted by vLLM and the timing of its Load publication."""

    num_new_matched_tokens: int
    load_is_deferred: bool


class RemoteAvailabilityProbe:
    """Query the Worker and derive vLLM's allocatable external prefix."""

    def __init__(
        self,
        address: str,
        *,
        group_ids: tuple[int, ...],
        transfer_granularity: int,
        discard_partial_chunks: bool,
        enabled: bool,
    ) -> None:
        self.address = address
        self.group_ids = group_ids
        self.transfer_granularity = transfer_granularity
        self.discard_partial_chunks = discard_partial_chunks
        self.enabled = enabled
        self.client: LookupClient | None = None

    def query(self, query: LookupQuery) -> RemoteAvailability | None:
        if not self.enabled:
            return None
        query_end = query.prompt_token_len
        if self.discard_partial_chunks:
            query_end -= query_end % self.transfer_granularity
        if query_end < self.transfer_granularity or query.local_cached_tokens >= query_end:
            return None
        if self.client is None:
            self.client = LookupClient(self.address)
        lookup_result = self.client.lookup(
            LookupRequest(
                TokenRange(query.local_cached_tokens, query_end),
                self.group_ids,
                tuple(query.block_hashes),
            )
        )
        if lookup_result.available_end_token <= query.local_cached_tokens:
            return None

        matched_end = lookup_result.available_end_token
        if matched_end == query.request_token_len:
            matched_end -= 1
        if matched_end <= query.local_cached_tokens:
            return None
        allocated_end = cdiv(matched_end, self.transfer_granularity) * self.transfer_granularity
        return RemoteAvailability(
            TokenRange(query.local_cached_tokens, min(lookup_result.available_end_token, allocated_end)),
            matched_end,
            lookup_result.tail_key_boundaries,
        )

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
