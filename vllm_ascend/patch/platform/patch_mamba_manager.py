# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
from collections import Counter, OrderedDict
from collections.abc import Sequence

import vllm.v1.core.single_type_kv_cache_manager as single_type_kv_cache_manager
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashWithGroupId
from vllm.v1.core.single_type_kv_cache_manager import (
    BlockHashList,
    BlockPool,
    KVCacheBlock,
    KVCacheSpec,
    MambaManager,
)
from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position
from vllm.v1.request import Request


class GDNCheckpointCandidates:
    """Bound complete request hash chains and index their reusable prefixes."""

    MAX_CHAINS = 4096
    MAX_HASHES = 262144

    def __init__(self) -> None:
        self.chains: OrderedDict[BlockHashWithGroupId, tuple[str, tuple[BlockHash, ...]]] = OrderedDict()
        self.anchors_by_request: dict[str, BlockHashWithGroupId] = {}
        self.counts: Counter[BlockHash] = Counter()
        self.num_hashes = 0

    def remember(self, anchor: BlockHashWithGroupId, request_id: str, hashes: Sequence[BlockHash], length: int) -> None:
        length = min(length, len(hashes))
        if anchor in self.chains or not 0 < length <= self.MAX_HASHES:
            return
        previous = self.anchors_by_request.get(request_id)
        if previous is not None:
            if len(self.chains[previous][1]) >= length:
                return
            self.discard(previous)
        chain = tuple(hashes[:length])
        self.chains[anchor] = (request_id, chain)
        self.anchors_by_request[request_id] = anchor
        self.counts.update(chain)
        self.num_hashes += len(chain)
        while len(self.chains) > self.MAX_CHAINS or self.num_hashes > self.MAX_HASHES:
            self.discard(next(iter(self.chains)))

    def discard(self, anchor: BlockHashWithGroupId) -> None:
        record = self.chains.pop(anchor, None)
        if record is None:
            return
        request_id, chain = record
        del self.anchors_by_request[request_id]
        self.num_hashes -= len(chain)
        for key in chain:
            self.counts[key] -= 1
            if not self.counts[key]:
                del self.counts[key]

    def repeated(self, request_id: str, key: BlockHash, index: int) -> bool:
        anchor = self.anchors_by_request.get(request_id)
        own = self.chains[anchor][1] if anchor is not None else ()
        return self.counts[key] > (index < len(own) and own[index] == key)

    def clear(self) -> None:
        self.chains.clear()
        self.anchors_by_request.clear()
        self.counts.clear()
        self.num_hashes = 0


class AscendCheckpointBlockPool(BlockPool):
    checkpoint_candidates: GDNCheckpointCandidates | None = None

    def _emit_block_removed_events(self, block_hashes: list[BlockHashWithGroupId]) -> None:
        # This path observes logical removals even when KV events are disabled.
        # Physical hash migration does not call it.
        candidates = self.checkpoint_candidates
        if candidates is not None:
            for key in block_hashes:
                if key in candidates.chains and self.cached_block_hash_to_block.get_one_block(key) is None:
                    candidates.discard(key)
        super()._emit_block_removed_events(block_hashes)


class AscendMambaManager(MambaManager):
    enable_checkpoint_repeat_admission = False
    checkpoint_candidates: GDNCheckpointCandidates | None = None

    def checkpoint_candidate_key(self, request: Request, position: int) -> BlockHash | None:
        unit = self.block_pool.hash_block_size
        if (
            not self.enable_checkpoint_repeat_admission
            or self.kv_cache_spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN
            or not 0 < position < request.num_prompt_tokens
            or position % unit
            or position % self.block_size == 0
        ):
            return None
        index = position // unit - 1
        return request.block_hashes[index] if index < len(request.block_hashes) else None

    def has_repeated_checkpoint(self, request: Request, position: int) -> bool:
        key = self.checkpoint_candidate_key(request, position)
        candidates = self.checkpoint_candidates
        return (
            key is not None
            and candidates is not None
            and candidates.repeated(request.request_id, key, position // self.block_pool.hash_block_size - 1)
        )

    def _cache_partial_tail_block(
        self, request: Request, num_tokens: int, retention_interval: int | None
    ) -> BlockHashWithGroupId | None:
        unit = self.block_pool.hash_block_size
        if (
            not self.enable_checkpoint_repeat_admission
            or self.kv_cache_spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN
            or request.request_id in self._checkpoints
            or (
                self.shared_prefix_checkpoint
                and num_tokens == request.shared_prefix_boundary
                and request.num_computed_tokens < num_tokens <= request.num_prompt_tokens
            )
        ):
            return super()._cache_partial_tail_block(request, num_tokens, retention_interval)

        tail = get_mamba_prefill_checkpoint_position(
            request.num_prompt_tokens,
            unit,
            self.drop_eagle_checkpoint_block,
            num_hash_lookahead_tokens=self.block_pool.num_hash_lookahead_tokens,
        )
        candidates = self.checkpoint_candidates
        assert candidates is not None
        if num_tokens == request.num_prompt_tokens:
            # The coordinator records the chain after FA publishes its key.
            return None
        key = self.checkpoint_candidate_key(request, num_tokens)
        if key is None:
            return super()._cache_partial_tail_block(request, num_tokens, retention_interval)
        promoted = getattr(request, "_ascend_gdn_promotion", 0) == num_tokens
        if num_tokens != tail and not promoted:
            return None

        block_index = (num_tokens - 1) // self.block_size
        blocks = self.req_to_blocks[request.request_id]
        if block_index >= len(blocks) or blocks[block_index].is_null:
            return None
        if blocks[block_index].block_hash_num_tokens == num_tokens:
            return None

        if not candidates.repeated(request.request_id, key, num_tokens // unit - 1):
            return None

        if num_tokens == tail:
            return super()._cache_partial_tail_block(request, num_tokens, retention_interval)
        # Use upstream junction publication to preserve recurrent-state CoW
        # bookkeeping at a boundary selected from the candidate hash chain.
        old_boundary = request.shared_prefix_boundary
        old_enabled = self.shared_prefix_checkpoint
        request.shared_prefix_boundary = num_tokens
        self.shared_prefix_checkpoint = True
        try:
            return super()._cache_partial_tail_block(request, num_tokens, retention_interval)
        finally:
            request.shared_prefix_boundary = old_boundary
            self.shared_prefix_checkpoint = old_enabled

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        drop_eagle_block: bool = False,
    ) -> tuple[list[KVCacheBlock], ...] | tuple[tuple[list[KVCacheBlock], ...], int]:
        # Mamba recurrent states are replicated across DCP ranks, so
        # their prefix-cache lookup keeps the ordinary logical
        # block layout. Only attention groups shard their KV cache for DCP.
        return super().find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_length,
            kv_cache_group_ids=kv_cache_group_ids,
            block_pool=block_pool,
            kv_cache_spec=kv_cache_spec,
            alignment_tokens=alignment_tokens,
            dcp_world_size=1,
            pcp_world_size=pcp_world_size,
            drop_eagle_block=drop_eagle_block,
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int | None = None,
        num_tokens_main_model: int | None = None,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        if num_tokens_main_model is None:
            assert num_local_computed_tokens is not None
            num_tokens_main_model = num_local_computed_tokens
        # Include tokens this request already computed locally. New hit blocks
        # alone omit running-request state and round partial hits up to a full
        # block, either inventing an external load or hiding a real one.
        local_hit_tokens = (
            num_local_computed_tokens
            if num_local_computed_tokens is not None
            else len(new_computed_blocks) * self.block_size
        )
        num_new_blocks = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
            prefill_end=prefill_end,
        )
        # When external KV cache is loaded synchronously with new
        # tokens, allocate_new_computed_blocks() allocates one
        # extra block to hold the external cache content. Account
        # for it here so the free-capacity check is accurate.
        # (External tokens exist when total_computed_tokens exceeds
        # what local prefix-cache hits cover; sync loading when
        # num_tokens_main_model exceeds total_computed_tokens.)
        has_external_tokens = total_computed_tokens > local_hit_tokens
        has_new_scheduled_tokens = num_tokens_main_model > total_computed_tokens
        if has_external_tokens and has_new_scheduled_tokens:
            # one more block for external computed tokens
            num_new_blocks += 1
        return num_new_blocks


single_type_kv_cache_manager.MambaManager = AscendMambaManager
