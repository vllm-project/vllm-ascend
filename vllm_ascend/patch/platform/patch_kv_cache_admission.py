# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from bisect import bisect_left
from collections.abc import Sequence

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)

_ORIGINAL_GET_NUM_BLOCKS_TO_ALLOCATE = SingleTypeKVCacheManager.get_num_blocks_to_allocate


def get_num_blocks_to_allocate(
    self: SingleTypeKVCacheManager,
    request_id: str,
    num_tokens: int,
    new_computed_blocks: Sequence[KVCacheBlock],
    total_computed_tokens: int,
    num_local_computed_tokens: int,
    num_tokens_main_model: int,
    apply_admission_cap: bool = False,
    prefill_end: int = 0,
) -> int:
    """Apply recycling caps to physical blocks, excluding the null prefix."""
    cap = self.max_admission_blocks_per_request
    if apply_admission_cap and cap is not None:
        if request_id in self.num_cached_block:
            blocks = self.req_to_blocks.get(request_id, ())
            # SWA/chunked-local tables contain a null prefix followed by real
            # blocks. Find that boundary in O(log(context)) without scanning
            # or copying the long logical block table.
            num_skipped_blocks = bisect_left(blocks, True, key=lambda block: not block.is_null)
            # Use blocks actually freed, not the optimistic computed-token
            # count: async steps may still be reading the retained prefix.
        else:
            num_skipped_blocks = self.get_num_skipped_tokens(total_computed_tokens) // self.block_size

        # Upstream caps the logical end at `cap`, then subtracts the logical
        # table length. After a long cache hit this incorrectly reserves zero
        # blocks for the next chunk. Shift the cap past the skipped prefix and
        # let upstream retain its prefix-hit/eviction/CoW accounting.
        num_tokens = min(num_tokens, (num_skipped_blocks + cap) * self.block_size)
        apply_admission_cap = False

    return _ORIGINAL_GET_NUM_BLOCKS_TO_ALLOCATE(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=apply_admission_cap,
        prefill_end=prefill_end,
    )


SlidingWindowManager.get_num_blocks_to_allocate = get_num_blocks_to_allocate
ChunkedLocalAttentionManager.get_num_blocks_to_allocate = get_num_blocks_to_allocate
