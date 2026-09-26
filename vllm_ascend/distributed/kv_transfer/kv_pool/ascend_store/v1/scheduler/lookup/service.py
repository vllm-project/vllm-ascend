"""Business entry point for Scheduler Lookup."""

from __future__ import annotations

from .client import LookupClient
from .request import SchedulerLookupRequest, SchedulerLookupResult


class LookupService:
    """Query the Worker and decide which external prefix can be loaded."""

    def __init__(
        self,
        lookup_address: str,
        *,
        transfer_group_ids: tuple[int, ...],
        cache_transfer_granularity: int,
        discard_partial_chunks: bool,
        enabled: bool,
    ) -> None:
        self.lookup_address = lookup_address
        self.transfer_group_ids = transfer_group_ids
        self.cache_transfer_granularity = cache_transfer_granularity
        self.discard_partial_chunks = discard_partial_chunks
        self.enabled = enabled
        self.client: LookupClient | None = None

    def lookup(self, request: SchedulerLookupRequest) -> SchedulerLookupResult:
        if not self.enabled:
            return SchedulerLookupResult(0, None)

        lookup_end_token = request.prompt_token_len
        if self.discard_partial_chunks:
            lookup_end_token -= lookup_end_token % self.cache_transfer_granularity
        if lookup_end_token < self.cache_transfer_granularity or request.local_cached_tokens >= lookup_end_token:
            return SchedulerLookupResult(0, None)

        if self.client is None:
            self.client = LookupClient(self.lookup_address)
        kv_pool_cached_tokens = self.client.lookup(
            lookup_end_token,
            self.transfer_group_ids,
            request.block_hashes,
            request.local_cached_tokens,
        )
        if kv_pool_cached_tokens == 0:
            return SchedulerLookupResult(0, None)

        if kv_pool_cached_tokens == request.request_token_len:
            kv_pool_cached_tokens -= 1
        need_to_allocate = max(kv_pool_cached_tokens - request.local_cached_tokens, 0)
        if need_to_allocate == 0:
            return SchedulerLookupResult(0, None)

        return SchedulerLookupResult(need_to_allocate, kv_pool_cached_tokens)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
