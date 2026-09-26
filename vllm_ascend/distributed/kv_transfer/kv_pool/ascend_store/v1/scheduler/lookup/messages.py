"""Messages crossing Scheduler-side Lookup boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(frozen=True, slots=True)
class SchedulerLookupRequest:
    """Request facts needed to plan a Scheduler Lookup."""

    request_id: str
    prompt_token_len: int
    request_token_len: int
    block_hashes: list[BlockHash]
    local_cached_tokens: int


@dataclass(frozen=True, slots=True)
class SchedulerLookupResult:
    """Lookup allocation returned to the Connector."""

    num_new_matched_tokens: int
    load_is_deferred: bool
