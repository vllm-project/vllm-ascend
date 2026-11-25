# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
from collections import OrderedDict
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


class AscendMambaManager(MambaManager):
    MAX_CHECKPOINT_CANDIDATES = 4096
    _checkpoint_candidates: OrderedDict[BlockHash, str] | None = None

    def _cache_partial_tail_block(
        self, request: Request, num_tokens: int, retention_interval: int | None
    ) -> BlockHashWithGroupId | None:
        unit = self.block_pool.hash_block_size
        if (
            self.kv_cache_spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN
            or request.request_id in self._checkpoints
            or not 0 < num_tokens < request.num_prompt_tokens
            or num_tokens % unit
            or num_tokens % self.block_size == 0
        ):
            return super()._cache_partial_tail_block(request, num_tokens, retention_interval)

        tail = get_mamba_prefill_checkpoint_position(
            request.num_prompt_tokens,
            unit,
            self.drop_eagle_checkpoint_block,
            num_hash_lookahead_tokens=self.block_pool.num_hash_lookahead_tokens,
        )
        is_junction = (
            self.shared_prefix_checkpoint
            and num_tokens == request.shared_prefix_boundary
            and request.num_computed_tokens < num_tokens
        )
        if num_tokens != tail and not is_junction:
            return None

        block_index = (num_tokens - 1) // self.block_size
        blocks = self.req_to_blocks[request.request_id]
        if block_index >= len(blocks) or blocks[block_index].is_null:
            return None
        if blocks[block_index].block_hash_num_tokens == num_tokens:
            return None

        hash_index = num_tokens // unit - 1
        if hash_index >= len(request.block_hashes):
            return None
        key = request.block_hashes[hash_index]
        if self._checkpoint_candidates is None:
            self._checkpoint_candidates = OrderedDict()
        candidates = self._checkpoint_candidates
        first_request = candidates.get(key)
        if first_request is None:
            candidates[key] = request.request_id
            if len(candidates) > self.MAX_CHECKPOINT_CANDIDATES:
                candidates.popitem(last=False)
            return None
        if first_request == request.request_id:
            return None

        published = super()._cache_partial_tail_block(request, num_tokens, retention_interval)
        if published is not None:
            del candidates[key]
        return published

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
