"""Scheduler-owned request facts shared by operation services."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class RequestTracker:
    request_id: str
    request_token_len: int
    block_ids_by_group: list[list[int]]
    block_hashes: list[BlockHash]
    num_prompt_tokens: int

    def advance(
        self,
        num_tokens: int,
        new_block_ids: tuple[list[int], ...] | None,
        block_hashes: list[BlockHash],
    ) -> None:
        self.request_token_len += num_tokens
        if new_block_ids:
            for block_ids, new_ids in zip(self.block_ids_by_group, new_block_ids, strict=True):
                block_ids.extend(new_ids)
        self.block_hashes = block_hashes
