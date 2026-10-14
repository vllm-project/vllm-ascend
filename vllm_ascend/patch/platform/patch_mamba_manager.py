# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
import inspect
from collections.abc import Sequence

import vllm.v1.core.single_type_kv_cache_manager as single_type_kv_cache_manager
from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId
from vllm.v1.core.single_type_kv_cache_manager import (
    BlockHashList,
    BlockPool,
    KVCacheBlock,
    KVCacheSpec,
    MambaManager,
    MambaSpec,
)
from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position
from vllm.v1.request import Request

_SUPPORTS_PREFILL_END = "prefill_end" in inspect.signature(MambaManager.get_num_blocks_to_allocate).parameters


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
        admission_kwargs: dict[str, int | bool] = {"apply_admission_cap": apply_admission_cap}
        if _SUPPORTS_PREFILL_END:
            # Newer sparse-checkpoint admission uses the same prefill end as
            # the scheduler to avoid reserving transient mid-prompt exports.
            admission_kwargs["prefill_end"] = prefill_end
        num_new_blocks = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            **admission_kwargs,
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

    def _cache_partial_tail_block(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None,
    ) -> BlockHashWithGroupId | None:
        if not getattr(self, "skips_eagle_block_drop", False):
            return super()._cache_partial_tail_block(request, num_tokens, retention_interval)

        hash_block_size = self.block_pool.hash_block_size
        checkpoint = self._checkpoints.get(request.request_id)
        if checkpoint is not None:
            # This reserved block holds the worker's separately exported state,
            # not the running tail at ``num_tokens``. Preserve the prompt's
            # resend checkpoint even when the coordinator rounds num_tokens
            # below P; sparse retention should discard only transient snapshots.
            prompt_checkpoint = get_mamba_prefill_checkpoint_position(
                request.num_prompt_tokens,
                hash_block_size,
                drop_eagle_block=False,
            )
            if checkpoint[0] == prompt_checkpoint:
                retention_interval = None
            return super()._cache_partial_tail_block(request, num_tokens, retention_interval)

        if self.block_size == hash_block_size or num_tokens % self.block_size == 0:
            return None
        if num_tokens % hash_block_size != 0:
            return None

        alignment = self.cache_hit_alignment_tokens or hash_block_size
        resend_boundary = max(request.num_prompt_tokens - 1, 0) // alignment * alignment
        extension_boundary = request.num_prompt_tokens // hash_block_size * hash_block_size
        if num_tokens not in (resend_boundary, extension_boundary) and not (
            getattr(self, "shared_prefix_checkpoint", getattr(self, "fine_grained_prefix_cache", False))
            and num_tokens == request.shared_prefix_boundary
            and request.num_computed_tokens < num_tokens <= request.num_prompt_tokens
        ):
            return None

        blocks = self.req_to_blocks[request.request_id]
        block_idx = num_tokens // self.block_size
        if block_idx >= len(blocks) or blocks[block_idx].is_null:
            return None
        source_block = blocks[block_idx]
        # Recurrent state is not append-only. Only key the actual chunk-end
        # state; never alias a later P state under the earlier resend boundary.
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


single_type_kv_cache_manager.MambaManager = AscendMambaManager
