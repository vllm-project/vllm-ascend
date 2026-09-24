"""Business entry point for classic Scheduler Lookup."""

from __future__ import annotations

from .client import LookupKeyClient
from .request import SchedulerLookupRequest, SchedulerLookupResult


class LookupService:
    """Query the Worker and decide which external prefix can be loaded."""

    def __init__(
        self,
        lookup_address: str,
        *,
        cache_transfer_granularity: int,
        discard_partial_chunks: bool,
        kv_role: str,
        consumer_is_to_load: bool,
    ) -> None:
        self.lookup_address = lookup_address
        self.cache_transfer_granularity = cache_transfer_granularity
        self.discard_partial_chunks = discard_partial_chunks
        self.kv_role = kv_role
        self.consumer_is_to_load = consumer_is_to_load
        self.client: LookupKeyClient | None = None

    def lookup(self, request: SchedulerLookupRequest) -> SchedulerLookupResult:
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_load:
            return SchedulerLookupResult(0, None)

        token_len = request.prompt_token_len
        if self.discard_partial_chunks:
            token_len -= token_len % self.cache_transfer_granularity
        if token_len < self.cache_transfer_granularity or request.num_computed_tokens >= token_len:
            return SchedulerLookupResult(0, None)

        if self.client is None:
            self.client = LookupKeyClient(self.lookup_address)
        num_external_hit_tokens = self.client.lookup(token_len, request.block_hashes, request.num_computed_tokens)
        if num_external_hit_tokens == 0:
            return SchedulerLookupResult(0, None)

        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1
        need_to_allocate = max(num_external_hit_tokens - request.num_computed_tokens, 0)
        if need_to_allocate == 0:
            return SchedulerLookupResult(0, None)

        return SchedulerLookupResult(need_to_allocate, num_external_hit_tokens)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
