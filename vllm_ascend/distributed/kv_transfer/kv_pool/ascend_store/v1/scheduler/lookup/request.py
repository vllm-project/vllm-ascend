"""Input and output values for classic Scheduler Lookup."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(frozen=True, slots=True)
class SchedulerLookupRequest:
    """The request facts needed to plan a classic Scheduler Lookup."""

    req_id: str
    prompt_token_len: int
    num_tokens: int
    block_hashes: list[BlockHash]
    num_computed_tokens: int


@dataclass(frozen=True, slots=True)
class SchedulerLookupResult:
    """Allocation and KV-pool hit facts produced by Scheduler Lookup."""

    num_new_matched_tokens: int
    kvpool_cached_tokens: int | None
