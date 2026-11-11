# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    """Reserve capacity using only held blocks that this request can recycle."""
    cap = self.max_admission_blocks_per_request
    remaining_admission_blocks = None
    if apply_admission_cap and cap is not None:
        blocks = self.req_to_blocks.get(request_id, ())
        if blocks:
            # Requests can hold blocks without a num_cached_block entry when
            # prefix caching is disabled. Use the actual allocation state.
            # Releasing a shared block only drops this request's reference;
            # it does not return capacity to the pool. Scan the live suffix,
            # stopping before the potentially long NULL prefix.
            num_recyclable_blocks = 0
            for block in reversed(blocks):
                if block.is_null:
                    break
                num_recyclable_blocks += block.ref_cnt == 1
            remaining_admission_blocks = max(0, cap - num_recyclable_blocks)
        else:
            # First-time admission has no held table. Shift the cap past the
            # skipped prefix so long local/remote hits still reserve capacity.
            num_skipped_blocks = self.get_num_skipped_tokens(total_computed_tokens) // self.block_size
            num_tokens = min(num_tokens, (num_skipped_blocks + cap) * self.block_size)

        apply_admission_cap = False

    # Keep prefix-hit, eviction, CoW, and speculative-token accounting. For
    # existing requests the uncapped need also bounds the remaining tail.
    num_blocks = _ORIGINAL_GET_NUM_BLOCKS_TO_ALLOCATE(
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
    if remaining_admission_blocks is not None:
        return min(num_blocks, remaining_admission_blocks)
    return num_blocks


SlidingWindowManager.get_num_blocks_to_allocate = get_num_blocks_to_allocate
ChunkedLocalAttentionManager.get_num_blocks_to_allocate = get_num_blocks_to_allocate
