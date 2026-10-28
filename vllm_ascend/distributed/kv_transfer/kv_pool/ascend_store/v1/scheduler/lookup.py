"""Derive Scheduler prefix frontiers from remote Lookup results."""

from __future__ import annotations

from vllm.utils.math_utils import cdiv

from ..infrastructure.configuration import KVPoolConfig
from ..infrastructure.protocol.messages import LookupResult
from .state import LoadCandidate


def resolve_load_candidate(
    config: KVPoolConfig,
    lookup_result: LookupResult,
    *,
    query_start_token: int,
    query_end_token: int,
    local_cached_tokens: int,
    request_token_count: int,
) -> LoadCandidate | None:
    """Derive readable, Store-skip, and Scheduler-safe prefix frontiers."""

    readable_end = lookup_result.available_end_token
    if readable_end == 0 or readable_end <= query_start_token:
        return None
    if readable_end > query_end_token:
        raise ValueError(f"Lookup returned token {readable_end} beyond query end {query_end_token}")

    store_skip_end = readable_end
    safe_load_end = readable_end
    granularity = config.cache_transfer_granularity
    if config.use_eagle and request_token_count > 0:
        final_block_start = (request_token_count - 1) // granularity * granularity
        if safe_load_end > final_block_start:
            safe_load_end = max(local_cached_tokens, safe_load_end - granularity)
    if safe_load_end == request_token_count:
        safe_load_end -= 1
    if config.has_private_state:
        safe_load_end = safe_load_end // granularity * granularity
    safe_load_end = max(local_cached_tokens, safe_load_end)

    physical_load_end = min(readable_end, cdiv(safe_load_end, granularity) * granularity)
    load_start = 0 if query_start_token == 0 else local_cached_tokens
    if physical_load_end < load_start:
        physical_load_end = load_start
    # Lookup may select a longer Attention object in the same physical tail page.
    # Its identity remains valid unless the physical Load frontier changes.
    tail_boundaries = lookup_result.tail_key_boundaries
    if physical_load_end != readable_end:
        tail_boundaries = tuple(
            boundary for boundary in tail_boundaries if boundary.boundary_token <= physical_load_end
        )
    return LoadCandidate(
        local_cached_tokens,
        store_skip_end,
        safe_load_end,
        load_start,
        physical_load_end,
        tail_boundaries,
    )
