"""Scheduler-owned remote Lookup and prefix-frontier derivation."""

from __future__ import annotations

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash

from ..coordinates import TokenRange
from ..protocol.lookup import LookupRequest, LookupResult
from ..protocol.rpc import LookupClient
from .state import LoadCandidate, SchedulerConfig


class RemoteLookup:
    """Own the lazy Scheduler-side Lookup RPC client."""

    def __init__(self, address: str) -> None:
        self.address = address
        self._client: LookupClient | None = None

    def query(
        self,
        query_range: TokenRange,
        transfer_group_ids: tuple[int, ...],
        block_hashes: tuple[BlockHash, ...],
    ) -> LookupResult:
        if self._client is None:
            self._client = LookupClient(self.address)
        return self._client.lookup(LookupRequest(query_range, transfer_group_ids, block_hashes))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def resolve_load_candidate(
    config: SchedulerConfig,
    lookup_result: LookupResult,
    *,
    query_range: TokenRange,
    local_cached_tokens: int,
    request_token_count: int,
) -> LoadCandidate | None:
    """Derive readable, Store-skip, and Scheduler-safe prefix frontiers."""

    readable_end = lookup_result.available_end_token
    if readable_end == 0 or readable_end <= query_range.start_token:
        return None
    if readable_end > query_range.end_token:
        raise ValueError(f"Lookup returned token {readable_end} beyond query end {query_range.end_token}")

    store_skip_end = readable_end
    safe_load_end = readable_end
    granularity = config.cache_transfer_granularity
    if config.use_eagle_block_drop and request_token_count > 0:
        final_block_start = (request_token_count - 1) // granularity * granularity
        if safe_load_end > final_block_start:
            safe_load_end = max(local_cached_tokens, safe_load_end - granularity)
    if safe_load_end == request_token_count:
        safe_load_end -= 1
    if config.has_private_state:
        safe_load_end = safe_load_end // granularity * granularity
    safe_load_end = max(local_cached_tokens, safe_load_end)

    physical_load_end = min(readable_end, cdiv(safe_load_end, granularity) * granularity)
    load_start = 0 if query_range.start_token == 0 else local_cached_tokens
    if physical_load_end < load_start:
        physical_load_end = load_start
    tail_boundaries = tuple(
        boundary for boundary in lookup_result.tail_key_boundaries if boundary.boundary_token <= physical_load_end
    )
    return LoadCandidate(
        local_cached_tokens,
        readable_end,
        store_skip_end,
        safe_load_end,
        TokenRange(load_start, physical_load_end),
        tail_boundaries,
    )
