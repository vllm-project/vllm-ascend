# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager, spec_manager_map)
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_spec import (Compress4AttentionSpec,
                                            Compress128AttentionSpec,
                                            CompressAttentionSpec)


class CompressAttentionManager(SingleTypeKVCacheManager):

    def __init__(self, kv_cache_spec: CompressAttentionSpec,
                 block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.compress_ratio = kv_cache_spec.compress_ratio
        self._null_block = block_pool.null_block

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
    ) -> int:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        assert isinstance(self.kv_cache_spec, CompressAttentionSpec)

        num_tokens //= self.compress_ratio

        return super().get_num_blocks_to_allocate(request_id, num_tokens,
                                                  new_computed_blocks)

    def allocate_new_blocks(self, request_id: str,
                            num_tokens: int) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).

        Returns:
            The new allocated blocks.
        """
        num_tokens //= self.compress_ratio

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(
                num_new_blocks, self.kv_cache_group_id)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
        """
        num_tokens //= self.compress_ratio

        return super().cache_blocks(request, num_tokens)

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, CompressAttentionSpec), (
            "SFACompressRatio4Manager can only be used for C128")
        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids)))
        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by mamba
        """
        return 0

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks, self.kv_cache_group_id)
        self.num_cached_block.pop(request_id, None)


def get_manager_for_kv_cache_spec(kv_cache_spec: KVCacheSpec,
                                  **kwargs) -> SingleTypeKVCacheManager:
    spec_manager_map.update({
        Compress4AttentionSpec: CompressAttentionManager,
        Compress128AttentionSpec: CompressAttentionManager,
    })
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
