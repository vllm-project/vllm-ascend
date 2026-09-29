"""Immutable work selected before KV Pool Backend I/O begins."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash

from ...coordinates import TokenRange
from .representation import BindingBatch, KVBinding

ChunkMask = tuple[bool, ...] | None


@dataclass(frozen=True, slots=True)
class GroupSelection:
    """Logical chunks selected for one original vLLM cache group."""

    group_id: int
    chunk_mask: ChunkMask

    def includes(self, start_token: int, block_size: int) -> bool:
        chunk_index = start_token // block_size
        return self.chunk_mask is None or (chunk_index < len(self.chunk_mask) and self.chunk_mask[chunk_index])


@dataclass(frozen=True, slots=True)
class KVSelection:
    """Content-identified semantic KV selected on the token axis."""

    token_range: TokenRange
    block_hashes: tuple[BlockHash | str, ...]
    groups: tuple[GroupSelection, ...]


@dataclass(frozen=True, slots=True)
class LoadTransfer:
    """One request's selected Load work after spatial projection."""

    request_id: str
    traversal: tuple[KVBinding, ...]


@dataclass(frozen=True, slots=True)
class StoreTransfer:
    """One request's selected Store work after spatial projection."""

    request_id: str
    batches: tuple[BindingBatch, ...]
    store_job_id: int | None = None
