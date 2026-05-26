# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


class CompressAttentionManager(FullAttentionManager):
    def __init__(self, kv_cache_spec: MLAAttentionSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.compress_ratio = kv_cache_spec.compress_ratio
        self._null_block = block_pool.null_block

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        # assert isinstance(self.kv_cache_spec, (CompressAttentionSpec, C4IndexerSpec))

        num_tokens //= self.compress_ratio
        num_tokens_main_model //= self.compress_ratio

        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. This involves three steps:
        1. Touch the computed blocks to make sure they won't be evicted.
        1.5. (Optional) For sliding window, skip blocks are padded with null blocks.
        2. Add the remaining computed blocks.
        3. (Optional) For KV connectors, allocate new blocks for external computed
            tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            # It should not have any new computed blocks.
            assert len(new_computed_blocks) == 0
            return

        # A new request.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = num_local_computed_tokens + num_external_computed_tokens
        num_total_computed_tokens //= self.compress_ratio
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), "Computed blocks should be empty when prefix caching is disabled"

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)

        if num_external_computed_tokens > 0:
            # Allocate new blocks for external computed tokens.
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
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
        ## TODO: check spec decode
        num_tokens_main_model //= self.compress_ratio

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
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
        # assert isinstance(
        #     kv_cache_spec, Compress4AttentionSpec | Compress128AttentionSpec | C4IndexerSpec
        # ), (
        #     "CompressAttentionManager can only be used for compressor attention groups"
        # )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple([] for _ in range(len(kv_cache_group_ids)))
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            # Need to drop the last matched block if eagle is enabled.
            for computed in computed_blocks:
                computed.pop()

        # NOTE: Div the compress ratio when finding the longest cache hit token length.
        alignment_tokens = cdiv(alignment_tokens, kv_cache_spec.compress_ratio)
        while (
            block_size != alignment_tokens  # Faster for common case.
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_num_batched_tokens: int | None = None,
    max_model_len: int | None = None,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """Build the per-spec KV cache manager.

    For DSv4 / DSA path (``MLAAttentionSpec`` with ``compress_ratio>1``), align
    the runtime admission gate with the startup pool-sizing bound the same way
    vLLM PR #40946 does for ``SlidingWindowSpec`` / ``ChunkedLocalAttentionSpec``.
    Without this cap, an admitted request can demand more blocks than the pool
    was sized to back, and ``allocate_slots`` silently returns ``None`` from
    the ``full_sequence_must_fit`` branch, leaving long-input requests stuck
    in the waiting queue (see vLLM issue #40863, observed on DSv4 + MTP with
    cc>=1 and prompt>=32K).

    The compressed-MLA peak per request is bounded by
    ``cdiv(max_model_len // compress_ratio, block_size)`` (it does not shrink
    via recycling like SWA, but neither does it ever exceed this). Capping at
    this value matches the pool sizer and makes admission consistent with the
    block budget actually held.
    """
    manager_class = spec_manager_map[type(kv_cache_spec)]
    if isinstance(kv_cache_spec, MLAAttentionSpec) and kv_cache_spec.compress_ratio > 1:
        manager_class = CompressAttentionManager
        if max_model_len is not None:
            # Compressed-MLA peak in blocks: ceil(max_model_len/compress/block).
            compress_ratio = kv_cache_spec.compress_ratio
            block_size = kv_cache_spec.block_size
            max_compressed_tokens = max_model_len // compress_ratio
            kwargs["max_admission_blocks_per_request"] = cdiv(max_compressed_tokens, block_size) + 1
    elif isinstance(kv_cache_spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec)):
        # Replicate the upstream PR #40946 cap setting for recycling specs.
        # We override the vLLM factory above, so the upstream block that does
        # this lives in dead code (never reached); without re-applying it here
        # SlidingWindowMLASpec / ChunkedLocalAttentionSpec groups have no cap
        # and ``full_sequence_must_fit`` admission reserves the full
        # ``max_model_len`` worth of blocks per request, exhausting the pool
        # at cc>=2 on DSv4 (see vLLM issue #40863).
        if max_num_batched_tokens is not None and max_model_len is not None:
            kwargs["max_admission_blocks_per_request"] = kv_cache_spec.max_admission_blocks_per_request(
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
