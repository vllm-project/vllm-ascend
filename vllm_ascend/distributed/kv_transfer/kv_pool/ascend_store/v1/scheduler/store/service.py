"""Business entry point for Scheduler Store."""

from __future__ import annotations

from vllm.utils.math_utils import cdiv

from ...metadata import StoreRequest
from ..request_tracker import RequestTracker


class StoreService:
    """Decide Store work and own each request's scheduled Store progress."""

    def __init__(
        self,
        *,
        cache_transfer_granularity: int,
        discard_partial_chunks: bool,
        save_decode_cache: bool,
        enabled: bool,
    ) -> None:
        self._cache_transfer_granularity = cache_transfer_granularity
        self._discard_partial_chunks = discard_partial_chunks
        self._save_decode_cache = save_decode_cache
        self._enabled = enabled
        self._scheduled_tokens: dict[str, int] = {}

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def accepts_cached_request(self, *, is_decoding: bool) -> bool:
        return not is_decoding or self._save_decode_cache

    def schedule_request(
        self,
        tracker: RequestTracker,
        transfer_end_token: int,
    ) -> StoreRequest | None:
        if not self._should_store(tracker.request_id, transfer_end_token):
            return None

        request = StoreRequest(
            request_id=tracker.request_id,
            store_end_token=transfer_end_token,
            block_ids_by_group=tuple(tuple(block_ids) for block_ids in tracker.block_ids_by_group),
            block_hashes=tuple(tracker.block_hashes),
            num_prompt_tokens=tracker.num_prompt_tokens,
        )
        # This progress records published Store work, not a confirmed Backend write.
        previous_saved_tokens = self._scheduled_tokens.get(tracker.request_id, 0)
        self._scheduled_tokens[tracker.request_id] = max(previous_saved_tokens, transfer_end_token)
        return request

    def _should_store(self, request_id: str, transfer_end_token: int) -> bool:
        previous_saved_tokens = self._scheduled_tokens.get(request_id, 0)
        chunk_boundary = (
            cdiv(previous_saved_tokens + 1, self._cache_transfer_granularity) * self._cache_transfer_granularity
            if self._discard_partial_chunks
            else 0
        )
        return self._enabled and transfer_end_token >= chunk_boundary

    def discard(self, request_id: str) -> None:
        self._scheduled_tokens.pop(request_id, None)
