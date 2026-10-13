# SPDX-License-Identifier: Apache-2.0
"""Materialize the last *reusable* PMU boundary for exact-length prefills.

Keep upstream allocation, CoW and connector handoff unchanged. This adapter
only fills the fallback chunk-end path when no internal checkpoint was emitted.

Related work: vllm-ascend RFC #16451. Remove this adapter when upstream's
fallback materializes the common reusable boundary for exact-length prompts.
"""

import functools

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _skips_eagle_block_drop
from vllm_ascend.patch.platform.patch_mamba_config import _get_sparse_index_kpool

_split = Scheduler._mamba_block_aligned_split
# v0.30 resolves managers through the registry, not the module's class alias.
# Look up the registered class without constructing a device-specific spec.
MambaManager = KVCacheSpecRegistry.get_manager_class(object.__new__(MambaSpec))
assert MambaManager is not None
_cache_tail = MambaManager._cache_partial_tail_block
FullAttentionManager = KVCacheSpecRegistry.get_manager_class(object.__new__(FullAttentionSpec))
assert FullAttentionManager is not None
_attention_tail = FullAttentionManager._cache_partial_tail_block


@functools.wraps(_split)
def split(self, request, num_new_tokens, num_new_local_computed_tokens=0, num_external_computed_tokens=0):
    result = _split(self, request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens)
    if not self.mamba_partial_cache_hit:
        return result
    if not _skips_eagle_block_drop(self.vllm_config.kv_transfer_config):
        return result
    if _get_sparse_index_kpool(self.vllm_config.model_config) is not None:
        return result
    unit = self.hash_block_size
    if request.num_prompt_tokens % unit:
        return result
    boundary = get_mamba_prefill_checkpoint_position(request.num_prompt_tokens, unit, False)
    start = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    # Do not alter decode/verifier windows or replay of generated tokens.
    if request.num_tokens != request.num_prompt_tokens:
        return result
    # A backend-exported in-query checkpoint needs no extra forward pass.
    if (
        self.mamba_has_prefill_checkpoint_blocks
        and start + num_new_tokens >= request.num_prompt_tokens
        and is_mamba_prefill_checkpoint_valid(
            query_start=start,
            query_end=start + num_new_tokens,
            checkpoint_position=boundary,
            hash_block_size=unit,
            mamba_block_size=self.cache_config.block_size,
            checkpoint_alignment=self.mamba_prefill_checkpoint_alignment,
        )
    ):
        return result
    if start < boundary < start + result:
        return boundary - start
    return result


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


Scheduler._mamba_block_aligned_split = split
MambaManager._cache_partial_tail_block = cache_tail


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


FullAttentionManager._cache_partial_tail_block = attention_tail
