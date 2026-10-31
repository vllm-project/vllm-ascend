# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Physical cache page inspection shared by P2P and pool transfers."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm_ascend.distributed.kv_transfer.utils.utils import tensor_storage_key

if TYPE_CHECKING:
    import torch


def kv_head_interval(tp_rank: int, tp_size: int, total_num_kv_heads: int) -> tuple[int, int]:
    """Existing V2 head partition/replication rule for a validated topology."""
    if total_num_kv_heads >= tp_size:
        heads_per_rank = total_num_kv_heads // tp_size
        start = tp_rank * heads_per_rank
        return start, start + heads_per_rank
    replication_size = tp_size // total_num_kv_heads
    head_index = tp_rank // replication_size
    return head_index, head_index + 1


def expand_cache_block_ids(block_ids: list[int], scale: int) -> list[int]:
    """Existing V2 scheduler-block to kernel-block numbering rule."""
    if scale == 1:
        return block_ids
    return [block_id * scale + offset for block_id in block_ids for offset in range(scale)]


@dataclass(frozen=True)
class KernelCachePage:
    physical_block_id: int
    token_start: int
    token_end: int
    logical_block_index: int


def kernel_cache_pages(
    block_ids: list[int],
    *,
    logical_block_size: int,
    physical_scale: int,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    first_logical_block: int = 0,
) -> list[KernelCachePage]:
    """Place attention kernel pages in the V2 virtual token sequence.

    This is the local-side formula used in V2's heterogeneous DCP block mapping:
    logical index * (block_size * DCP) + DCP rank * block_size + kernel offset.
    Replicated components (e.g. replicated indexer/SWA) pass effective DCP=1.
    This does not interpret recurrent state or a circular request-lifetime page.
    """
    for name, value in (("block size", logical_block_size), ("physical scale", physical_scale), ("DCP size", dcp_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"Cache {name} must be a positive integer")
    if logical_block_size % physical_scale:
        raise ValueError("Logical cache block size must be divisible by its physical scale")
    if type(dcp_rank) is not int or not 0 <= dcp_rank < dcp_size:
        raise ValueError("DCP rank is outside its effective cache group")
    if type(first_logical_block) is not int or first_logical_block < 0:
        raise ValueError("First logical cache block must be nonnegative")
    if any(type(block) is not int or block < 0 for block in block_ids):
        raise ValueError("Physical block table contains invalid IDs")
    kernel_size = logical_block_size // physical_scale
    result = []
    for index, block_id in enumerate(block_ids, start=first_logical_block):
        local_start = index * logical_block_size * dcp_size + dcp_rank * logical_block_size
        for offset in range(physical_scale):
            start = local_start + offset * kernel_size
            result.append(KernelCachePage(block_id * physical_scale + offset, start, start + kernel_size, index))
    return result


def describe_shared_page(
    caches: tuple["torch.Tensor", ...], num_blocks: int
) -> tuple[int, int, tuple[int, ...], int] | None:
    """Describe views packed into one physical page allocation.

    Sharing a storage alone is insufficient: conventional planar K/V
    views can share one allocation while their first blocks are many pages
    apart. A packed layout must place every view's first block inside one
    common page stride and use the same physical block count and stride.
    Their contiguous block spans must cover that entire stride.
    """
    if not caches or len({tensor_storage_key(cache) for cache in caches}) != 1:
        return None

    tensor_num_blocks = {cache.shape[0] for cache in caches}
    page_strides = {cache.stride(0) * cache.element_size() for cache in caches}
    if len(tensor_num_blocks) != 1 or len(page_strides) != 1:
        return None

    num_tensor_blocks = next(iter(tensor_num_blocks))
    page_stride = next(iter(page_strides))
    if num_tensor_blocks <= 0 or page_stride <= 0 or num_tensor_blocks % num_blocks:
        return None

    page_base_addr = min(cache.data_ptr() for cache in caches)
    block_lens = [math.prod(cache.shape[1:]) * cache.element_size() for cache in caches]
    # Storage capacity does not prove that this layer owns a whole page:
    # its views may occupy only a subregion of a larger shared allocation.
    # Merge only internally contiguous blocks whose union covers the full
    # stride. Overlapping aliases are allowed, but holes and padding are not.
    if any(block_len == 0 or not cache[0].is_contiguous() for cache, block_len in zip(caches, block_lens)):
        return None
    block_ranges = sorted(
        (cache.data_ptr() - page_base_addr, cache.data_ptr() - page_base_addr + block_len)
        for cache, block_len in zip(caches, block_lens)
    )
    covered_end = 0
    for start, end in block_ranges:
        if start > covered_end or end > page_stride:
            return None
        covered_end = max(covered_end, end)
    if covered_end != page_stride:
        return None

    storage = caches[0].untyped_storage()
    storage_base_addr = storage.data_ptr()
    storage_end_addr = storage_base_addr + storage.nbytes()
    if page_base_addr < storage_base_addr or page_base_addr + num_tensor_blocks * page_stride > storage_end_addr:
        return None

    selected_cache = max(
        zip(caches, block_lens),
        key=lambda cache_and_len: cache_and_len[1],
    )[0]
    return (
        page_base_addr,
        page_stride,
        tuple(selected_cache.shape[1:]),
        num_tensor_blocks // num_blocks,
    )
