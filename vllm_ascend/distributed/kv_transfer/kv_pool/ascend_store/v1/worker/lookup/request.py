"""Worker Lookup input decoded from the Scheduler RPC."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(frozen=True, slots=True)
class WorkerLookupRequest:
    """Content hashes and cache groups participating in one Lookup."""

    lookup_end_token: int
    transfer_group_ids: tuple[int, ...]
    local_cached_tokens: int
    block_hashes: tuple[BlockHash | str, ...]
