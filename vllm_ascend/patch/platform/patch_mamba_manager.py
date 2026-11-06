# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
import functools
from collections.abc import Sequence

import vllm.v1.core.single_type_kv_cache_manager as single_type_kv_cache_manager
from vllm.v1.core.single_type_kv_cache_manager import (
    BlockHashList,
    BlockPool,
    KVCacheBlock,
    KVCacheSpec,
    MambaManager,
    MambaSpec,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, get_mamba_prefill_checkpoint_position
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry


class AscendMambaManager(MambaManager):
    def __init__(self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.block_size = kv_cache_spec.block_size

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
        # Mamba recurrent states are replicated across DCP ranks (see
        # __init__), so their prefix-cache lookup keeps the ordinary logical
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
        local_hit_tokens = len(new_computed_blocks) * self.block_size
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


# Exact-PMU fallback: keep native allocation, CoW and handoff lifecycle.
# Remove when upstream materializes the common reusable boundary itself.
# v0.30 resolves managers through the registry, not the module's class alias.
# Look up the registered class without constructing a device-specific spec.
_registered_mamba_manager = KVCacheSpecRegistry.get_manager_class(object.__new__(MambaSpec))
assert _registered_mamba_manager is not None
_cache_tail = _registered_mamba_manager._cache_partial_tail_block
_registered_attention_manager = KVCacheSpecRegistry.get_manager_class(object.__new__(FullAttentionSpec))
assert _registered_attention_manager is not None
_attention_tail = _registered_attention_manager._cache_partial_tail_block


@functools.wraps(_cache_tail)
def cache_tail(self, request, num_tokens, retention_interval):
    result = _cache_tail(self, request, num_tokens, retention_interval)
    if result is not None or request.request_id in self._checkpoints:
        return result
    unit = self.block_pool.hash_block_size
    if (
        self.mamba_cache_mode != "align"
        or self.use_eagle
        or unit >= self.block_size
        or retention_interval is not None
        or request.num_prompt_tokens % unit
        or num_tokens != get_mamba_prefill_checkpoint_position(request.num_prompt_tokens, unit, False)
        or num_tokens <= 0
        or num_tokens % self.block_size == 0
    ):
        return result
    # Same source slot and lifecycle records as the upstream fallback tail.
    block_idx = num_tokens // self.block_size
    blocks = self.req_to_blocks[request.request_id]
    if block_idx >= len(blocks) or blocks[block_idx].is_null:
        return result
    source_block = blocks[block_idx]
    partial_hash = self.block_pool.cache_partial_block(
        request=request,
        block=source_block,
        num_tokens=num_tokens,
        kv_cache_group_id=self.kv_cache_group_id,
        block_size=self.block_size,
    )
    if partial_hash is not None:
        self._partial_hit_reqs[request.request_id] = (block_idx, source_block)
        self.num_cached_block[request.request_id] = block_idx
        self._producer_partial_tail_reqs[request.request_id] = (source_block, num_tokens)
    return partial_hash


_registered_mamba_manager._cache_partial_tail_block = cache_tail


@functools.wraps(_attention_tail)
def attention_tail(self, request, num_tokens):
    _attention_tail(self, request, num_tokens)
    unit = self.block_pool.hash_block_size
    if unit >= self.block_size or request.num_prompt_tokens % unit:
        return
    boundary = get_mamba_prefill_checkpoint_position(request.num_prompt_tokens, unit, False)
    if boundary <= 0 or boundary > num_tokens or boundary % self.block_size == 0:
        return
    blocks = self.req_to_blocks[request.request_id]
    block_idx = boundary // self.block_size
    if block_idx >= len(blocks) or blocks[block_idx].is_null:
        return
    # Attention KV is append-only. A later tail registration can replace its
    # primary hash; restore the reusable alias *after* the upstream method.
    self.block_pool.cache_partial_block(
        request=request,
        block=blocks[block_idx],
        num_tokens=boundary,
        kv_cache_group_id=self.kv_cache_group_id,
        block_size=self.block_size,
    )


_registered_attention_manager._cache_partial_tail_block = attention_tail
