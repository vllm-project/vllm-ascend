# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
from collections.abc import Sequence

import vllm.v1.core.sched.scheduler as scheduler_module
import vllm.v1.core.single_type_kv_cache_manager as single_type_kv_cache_manager
import vllm.v1.kv_cache_interface as kv_cache_interface
from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import (
    BlockHashList,
    BlockPool,
    KVCacheBlock,
    KVCacheSpec,
    MambaManager,
)
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
)


def _is_mamba_prefill_checkpoint_valid(
    query_start: int,
    query_end: int,
    checkpoint_position: int,
    hash_block_size: int,
    mamba_block_size: int,
    checkpoint_alignment: int | None,
) -> bool:
    """Allow internal checkpoint export for same-block fine-PMU hits.

    Upstream requires ``checkpoint_col > initial_state_col``, which rejects
    same-block prefix hits when ``hash_block_size < mamba_block_size`` (e.g.
    PMU=128 with block_size=2048) even though ``checkpoint_col = r - 1 >= 0``
    is a ``NULL_BLOCK`` slot distinct from ``initial_state_col = r``.
    """
    if checkpoint_alignment is None:
        return False
    assert checkpoint_alignment > 0

    initial_state_col = (query_start - 1) // mamba_block_size
    checkpoint_col = cdiv(query_end, mamba_block_size) - 2
    return (
        query_start % hash_block_size == 0
        and checkpoint_col >= 0
        and checkpoint_col != initial_state_col
        and query_start + hash_block_size <= checkpoint_position
        and query_start < checkpoint_position < query_end
        and (checkpoint_position - query_start) % checkpoint_alignment == 0
    )


def _install_checkpoint_validity_patch() -> None:
    if not hasattr(kv_cache_interface, "is_mamba_prefill_checkpoint_valid"):
        return
    kv_cache_interface.is_mamba_prefill_checkpoint_valid = _is_mamba_prefill_checkpoint_valid
    if hasattr(scheduler_module, "is_mamba_prefill_checkpoint_valid"):
        scheduler_module.is_mamba_prefill_checkpoint_valid = _is_mamba_prefill_checkpoint_valid
    if hasattr(single_type_kv_cache_manager, "is_mamba_prefill_checkpoint_valid"):
        single_type_kv_cache_manager.is_mamba_prefill_checkpoint_valid = _is_mamba_prefill_checkpoint_valid
    try:
        import vllm.model_executor.layers.mamba.checkpoint as mamba_checkpoint

        if hasattr(mamba_checkpoint, "is_mamba_prefill_checkpoint_valid"):
            mamba_checkpoint.is_mamba_prefill_checkpoint_valid = _is_mamba_prefill_checkpoint_valid
    except ImportError:
        pass


_install_checkpoint_validity_patch()


class AscendMambaManager(MambaManager):
    def _needs_internal_checkpoint(
        self,
        request_id: str,
        query_start: int,
        query_end: int,
        checkpoint_position: int,
        prefill_end: int,
    ) -> bool:
        if not super()._needs_internal_checkpoint(
            request_id,
            query_start,
            query_end,
            checkpoint_position,
            prefill_end,
        ):
            return False
        # Running requests have append-only worker block tables: never fill an
        # earlier NULL column in-place once blocks have been allocated.
        checkpoint_idx = cdiv(query_end, self.block_size) - 2
        blocks = self.req_to_blocks[request_id]
        return not (
            request_id in self._allocated_block_reqs and checkpoint_idx < len(blocks) - self.num_speculative_blocks
        )

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
        # Same-block prefix hits place the internal checkpoint into an existing
        # NULL column (< len(new_computed_blocks)), which upstream's append-only
        # block count does not include. Reserve 1 block for it.
        if (
            self.mamba_cache_mode == "align"
            and isinstance(self.kv_cache_spec, MambaSpec)
            and request_id not in self._allocated_block_reqs
            and num_new_blocks <= self.block_pool.num_gpu_blocks
        ):
            checkpoint_idx = cdiv(num_tokens_main_model, self.block_size) - 2
            existing_prefix_len = len(new_computed_blocks) + len(self.req_to_blocks[request_id])
            if 0 <= checkpoint_idx < existing_prefix_len:
                checkpoint_position = get_mamba_prefill_checkpoint_position(
                    num_tokens_main_model,
                    self.cache_hit_alignment_tokens,
                    self.drop_eagle_checkpoint_block,
                    num_hash_lookahead_tokens=self.block_pool.num_hash_lookahead_tokens,
                )
                if self._needs_internal_checkpoint(
                    request_id,
                    total_computed_tokens,
                    num_tokens_main_model,
                    checkpoint_position,
                    prefill_end,
                ):
                    num_new_blocks += 1
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

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        checkpoint = self._checkpoints.get(request_id) if self.mamba_cache_mode == "align" else None
        needs_in_place_checkpoint = (
            checkpoint is not None
            and request_id not in self._allocated_block_reqs
            and 0 <= checkpoint[1] < len(req_blocks)
            and req_blocks[checkpoint[1]].is_null
        )
        returned_blocks = super().allocate_new_blocks(request_id, num_tokens, num_tokens_main_model)
        if needs_in_place_checkpoint:
            assert checkpoint is not None
            ckpt_block = self.block_pool.get_new_blocks(1)[0]
            req_blocks[checkpoint[1]] = ckpt_block
            returned_blocks = [ckpt_block, *returned_blocks]
        return returned_blocks


single_type_kv_cache_manager.MambaManager = AscendMambaManager
