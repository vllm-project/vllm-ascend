"""Scheduler-owned request facts shared by operation services."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class RequestTracker:
    request_id: str
    token_len: int
    block_ids: list[int]
    block_hashes: list[BlockHash]
    num_prompt_tokens: int

    def advance(
        self,
        num_tokens: int,
        new_block_ids: tuple[list[int], ...] | None,
        block_hashes: list[BlockHash],
    ) -> None:
        self.token_len += num_tokens
        if new_block_ids is not None:
            self.block_ids.extend(new_block_ids[0])
        self.block_hashes = block_hashes
