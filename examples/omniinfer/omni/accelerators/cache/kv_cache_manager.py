# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from typing_extensions import override
from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.utils import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (FullAttentionManager,
                                                       SingleTypeKVCacheManager
                                                       )
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

from .kv_cache_interface import OmniAttentionSpec, OmniKVCacheConfig

logger = init_logger("vllm.v1.omni")


@dataclass
class OmniKVCacheBlocks:
    blocks: list[list[KVCacheBlock]]

    def __add__(self, other: "OmniKVCacheBlocks") -> "OmniKVCacheBlocks":
        if len(self.blocks) == 0:
            return OmniKVCacheBlocks(other.blocks)
        if len(other.blocks) == 0:
            return OmniKVCacheBlocks(self.blocks)
        return OmniKVCacheBlocks(
            [b1 + b2 for b1, b2 in zip(self.blocks, other.blocks)])

    @classmethod
    def create_empty(cls) -> "OmniKVCacheBlocks":
        """Creates a new KVCacheBlocks instance with no blocks."""
        return cls([])

    def get_block_ids(self) -> list[list[int]]:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            list[list[int]]: A two-level list where
            * the outer list corresponds to KV cache groups
            * each inner list contains the block_ids of the blocks in that group
        """
        return [[block.block_id for block in group] for group in self.blocks]

    def get_unhashed_block_ids(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from OmniKVCacheBlocks instance."""
        return [[
            block.block_id for block in group if block.block_hash is None
        ] for group in self.blocks]


class OmniKVCacheManager(KVCacheManager):

    def __init__(
        self,
        kv_cache_config: OmniKVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        if len(kv_cache_config.kv_cache_groups) != 2:
            raise ValueError(
                "OmniKVCacheManager does not support hybrid models with more than 2 "
                "kv cache groups")
        if enable_caching:
            enable_caching = False
            logger.warning(
                "OmniKVCacheManager does not support prefix caching yet, enable_caching is set to False"
            )

        if enable_kv_cache_events:
            raise ValueError(
                "OmniKVCacheManager does not support cache events yet")
        # `block_size` of all groups are assumed to be the same
        self.block_size = kv_cache_config.kv_cache_groups[
            0].kv_cache_spec.block_size
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        full_attn_pool, full_attn_mgr = None, None
        self.block_pools: list[BlockPool] = []
        self.hybrid_managers: list[SingleTypeKVCacheManager] = []
        for group in kv_cache_config.kv_cache_groups:
            mgr_kwargs = dict(
                kv_cache_spec=group.kv_cache_spec,
                use_eagle=self.use_eagle,
                num_kv_cache_groups=1,
                caching_hash_fn=self.caching_hash_fn,
            )
            if isinstance(group.kv_cache_spec, FullAttentionSpec):
                full_attn_pool = BlockPool(self.num_gpu_blocks, enable_caching,
                                           enable_kv_cache_events)
                full_attn_mgr = get_manager_for_kv_cache_spec(
                    block_pool=full_attn_pool,
                    **mgr_kwargs,
                )
            else:
                num_blocks = kv_cache_config.num_blocks_per_group[type(
                    group.kv_cache_spec)]
                bp = BlockPool(num_blocks, enable_caching,
                               enable_kv_cache_events)
                self.block_pools.append(bp)
                self.hybrid_managers.append(
                    get_manager_for_kv_cache_spec(
                        block_pool=bp,
                        **mgr_kwargs,
                    ))
        # put full attention at the first position
        if full_attn_pool is None:
            raise RuntimeError("No FullAttentionSpec is found.")
        if len(self.block_pools) == 0:
            raise RuntimeError("No other AttentionSpec is found.")
        self.block_pools = [full_attn_pool] + self.block_pools
        self.hybrid_managers = [full_attn_mgr] + self.hybrid_managers

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHashType]] = defaultdict(list)
        logger.warning(
            f"OmniKVCacheManager is being used with {len(self.hybrid_managers)} KV cache groups."
        )

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return max(bp.get_usage() for bp in self.block_pools)

    @property
    def single_type_manager(self) -> FullAttentionManager:
        """For compatibility, we create a pseudo `single_type_manager` which
        is actually the full attention manager.
        """
        return self.hybrid_managers[0]

    @property
    def block_pool(self) -> BlockPool:
        """For compatibility, we create a pseudo `block_pool` which
        is actually the full attention manager.
        """
        return self.block_pools[0]

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[OmniKVCacheBlocks] = None,
        num_draft_tokens: int = 0,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        is_swap: bool = False,
    ) -> Optional[OmniKVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_new_tokens == 0 and not is_swap:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = []

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        for mgr in self.hybrid_managers:
            mgr.remove_skipped_blocks(request.request_id,
                                      request.num_computed_tokens)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len)
        num_blocks_to_allocate: list[int] = [
            mgr.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=num_tokens_need_slot,
                new_computed_blocks=new_computed_block_list,
            ) for mgr in self.hybrid_managers
        ]

        free_blocks = [bp.get_num_free_blocks() for bp in self.block_pools]
        if any(need > free
               for need, free in zip(num_blocks_to_allocate, free_blocks)):
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not new_computed_block_list, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.single_type_manager.save_new_computed_blocks(
            request.request_id, new_computed_block_list)

        # outer list is group
        # inner list is blocks of each group
        new_blocks: list[list[KVCacheBlock]] = []
        for mgr in self.hybrid_managers:
            new_blocks.append(
                mgr.allocate_new_blocks(request.request_id,
                                        num_tokens_need_slot))

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return OmniKVCacheBlocks(new_blocks)

        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        # NOTE: call `cache_blocks` only on full attention layers
        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens,
                                  request.num_tokens)
        self.single_type_manager.cache_blocks(
            request, self.req_to_block_hashes[request.request_id],
            num_tokens_to_cache)

        return OmniKVCacheBlocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that he tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        for mgr in self.hybrid_managers:
            mgr.free(request.request_id)

    def get_block_ids(self, request_id: str) -> list[list[int]]:
        """Get the block ids of a request.

        Returns:
            A list of lists of integers. The outer list corresponds to KV Cache groups,
            where the first is full attention group.
        """
        group_block_ids: list[list[int]] = []
        for mgr in self.hybrid_managers:
            if request_id in mgr.req_to_blocks:
                blocks = mgr.req_to_blocks[request_id]
                block_ids = [blk.block_id for blk in blocks]
                group_block_ids.append(block_ids)
            else:
                group_block_ids.append([])
        return group_block_ids


class OmniAttentionManager(SingleTypeKVCacheManager):

    def __init__(self, kv_cache_spec: OmniAttentionSpec, *args, **kwargs):
        super().__init__(kv_cache_spec, *args, **kwargs)
        self.max_tokens = self.kv_cache_spec.max_compressed_len
        self.max_num_blocks = self.kv_cache_spec.max_num_blocks

    @override
    def get_num_blocks_to_allocate(
            self, request_id: str, num_tokens: int,
            new_computed_blocks: list[KVCacheBlock]) -> int:
        if len(self.req_to_blocks[request_id]) > 0:
            return 0
        else:
            # allocate max blocks for each new request, so that no more allocation is done afterwards
            return self.max_num_blocks

    @override
    def save_new_computed_blocks(
            self, request_id: str,
            new_computed_blocks: list[KVCacheBlock]) -> None:
        return

    @override
    def allocate_new_blocks(self, request_id: str,
                            num_tokens: int) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = self.max_num_blocks
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            if num_new_blocks != self.max_num_blocks:
                raise RuntimeError(
                    f"{num_new_blocks=}, while {self.max_num_blocks=}")
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    @override
    def cache_blocks(self, request: Request, block_hashes: list[BlockHashType],
                     num_tokens: int) -> None:
        return

    def find_longest_cache_hit(self, block_hashes: list[BlockHashType],
                               max_length: int) -> list[KVCacheBlock]:
        raise NotImplementedError(
            "Method find_longest_cache_hit is not implemented yet for OmniAttentionManager"
        )

    def remove_skipped_blocks(self, request_id: str,
                              num_computed_tokens: int) -> None:
        pass

    def get_num_common_prefix_blocks(self, request_id: str,
                                     num_running_requests: int) -> int:
        return 0


def get_manager_for_kv_cache_spec(kv_cache_spec: KVCacheSpec,
                                  **kwargs) -> SingleTypeKVCacheManager:
    # from .nsa_kv_manager import ConvolutionCompressSpec, ConvolutionCompressKVManager
    spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = \
    {
        FullAttentionSpec: FullAttentionManager,
        OmniAttentionSpec: OmniAttentionManager,
        # ConvolutionCompressSpec: ConvolutionCompressKVManager,
    }
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
