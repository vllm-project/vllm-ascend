"""Business entry point for classic Scheduler Store."""

from __future__ import annotations

from vllm.utils.math_utils import cdiv


class StoreService:
    """Decide Store work and own each request's scheduled Store progress."""

    def __init__(
        self,
        *,
        cache_transfer_granularity: int,
        discard_partial_chunks: bool,
        save_decode_cache: bool,
        kv_role: str,
        consumer_is_to_put: bool,
    ) -> None:
        self._cache_transfer_granularity = cache_transfer_granularity
        self._discard_partial_chunks = discard_partial_chunks
        self._save_decode_cache = save_decode_cache
        self._can_store = kv_role != "kv_consumer" or consumer_is_to_put
        self._scheduled_tokens: dict[str, int] = {}

    @property
    def can_store(self) -> bool:
        return self._can_store

    def accepts_cached_request(self, is_decoding: bool) -> bool:
        return not is_decoding or self._save_decode_cache

    def should_store(self, request_id: str, transfer_end_token: int) -> bool:
        previous_saved_tokens = self._scheduled_tokens.get(request_id, 0)
        chunk_boundary = (
            cdiv(previous_saved_tokens + 1, self._cache_transfer_granularity) * self._cache_transfer_granularity
            if self._discard_partial_chunks
            else 0
        )
        return self._can_store and transfer_end_token >= chunk_boundary

    def record_scheduled(self, request_id: str, save_end_token: int) -> None:
        previous_saved_tokens = self._scheduled_tokens.get(request_id, 0)
        self._scheduled_tokens[request_id] = max(previous_saved_tokens, save_end_token)

    def discard(self, request_id: str) -> None:
        self._scheduled_tokens.pop(request_id, None)
