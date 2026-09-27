"""Business entry point for Scheduler Lookup."""

from __future__ import annotations

from vllm.utils.math_utils import cdiv

from ...protocol.coordinates import TokenRange
from ...protocol.lookup import LookupRequest
from .client import LookupClient
from .messages import LookupAvailability, SchedulerLookupRequest


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

    def lookup(self, request: SchedulerLookupRequest) -> LookupAvailability | None:
        if not self.enabled:
            return None

        lookup_end_token = request.prompt_token_len
        if self.discard_partial_chunks:
            lookup_end_token -= lookup_end_token % self.cache_transfer_granularity
        if lookup_end_token < self.cache_transfer_granularity or request.local_cached_tokens >= lookup_end_token:
            return None

        if self.client is None:
            self.client = LookupClient(self.lookup_address)
        lookup_result = self.client.lookup(
            LookupRequest(
                TokenRange(request.local_cached_tokens, lookup_end_token),
                self.transfer_group_ids,
                tuple(request.block_hashes),
            )
        )
        available_end_token = lookup_result.available_end_token
        if available_end_token <= request.local_cached_tokens:
            return None

        matched_end_token = available_end_token
        if matched_end_token == request.request_token_len:
            matched_end_token -= 1
        if matched_end_token <= request.local_cached_tokens:
            return None

        # Keep one token for vLLM computation on a full hit, but load its KV when the containing chunk is allocated.
        allocated_end_token = cdiv(matched_end_token, self.cache_transfer_granularity)
        allocated_end_token *= self.cache_transfer_granularity
        load_end_token = min(available_end_token, allocated_end_token)
        return LookupAvailability(
            TokenRange(request.local_cached_tokens, load_end_token),
            matched_end_token,
        )

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
