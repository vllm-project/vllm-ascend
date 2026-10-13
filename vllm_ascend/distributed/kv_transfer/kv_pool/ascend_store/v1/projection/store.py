"""Select Store source rows independently of Worker addresses and writer ranks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import get_block_hashes

StoreCandidateRows: TypeAlias = tuple[list[int], list[BlockHash | str], list[int]]


def store_candidate_rows(
    end_token: int,
    block_hashes: Sequence[BlockHash | str],
    block_ids: Sequence[int],
    *,
    block_size: int,
    hash_block_size: int,
    start_token: int = 0,
    mask: Sequence[bool] | None = None,
    minimum_block_id: int = 0,
) -> StoreCandidateRows:
    """Select Store rows before Backend admission or writer partitioning."""

    if not block_hashes or end_token <= 0:
        return [], [], []
    grouped_hashes = get_block_hashes(block_hashes, block_size, hash_block_size)
    indices = store_block_range(start_token, end_token, block_size, len(grouped_hashes), len(block_ids))
    block_offset = max(indices.stop - len(block_ids), 0)
    counts: list[int] = []
    hashes: list[BlockHash | str] = []
    selected_block_ids: list[int] = []
    for block_index in indices:
        start = block_index * block_size
        end = min(start + block_size, end_token)
        local_index = block_index - block_offset
        if mask is not None and (block_index >= len(mask) or not mask[block_index]):
            continue
        block_id = block_ids[local_index]
        if block_id < minimum_block_id:
            continue
        counts.append(end - start)
        hashes.append(grouped_hashes[block_index])
        selected_block_ids.append(block_id)
    return counts, hashes, selected_block_ids


def store_block_range(start_token: int, end_token: int, block_size: int, hash_count: int, block_count: int) -> range:
    """Intersect the Store interval, confirmed hashes, and retained block-table suffix."""

    logical_count = min(hash_count, -(-end_token // block_size))
    block_offset = max(logical_count - block_count, 0)
    return range(max(start_token // block_size, block_offset), logical_count)


def checkpoint_companion_indices(
    boundary_token: int, published_store_end_token: int, block_size: int, block_count: int
) -> range:
    """Locate the unpublished attention pages and the checkpoint's boundary page."""

    logical_count = -(-boundary_token // block_size)
    block_offset = max(logical_count - block_count, 0)
    first_block = max(min(published_store_end_token // block_size, logical_count - 1), block_offset)
    return range(first_block - block_offset, logical_count - block_offset)
