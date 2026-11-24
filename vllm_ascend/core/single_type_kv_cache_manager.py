# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend KV-cache manager specializations."""

from collections.abc import Sequence

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager, MambaManager
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec
from vllm.v1.request import Request


class AscendCircularBufferManager(CircularBufferManager):
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        needs_kv_cache_zeroing: bool = False,
        max_admission_blocks_per_request: int | None = None,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            scheduler_block_size,
            dcp_world_size,
            pcp_world_size,
            needs_kv_cache_zeroing,
            max_admission_blocks_per_request,
        )
        # Ascend zeroes every physical backing of a recycled block ID. Ring
        # allocations must report IDs too, even when the last owner was an
        # attention group. The upstream ring manager excludes these IDs.
        self._record_new_block_ids = needs_kv_cache_zeroing


class AscendGDNLocalHitManager(MambaManager):
    """Track local warm prefills through budgeted chunks and preemption."""

    def __init__(self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self._local_prefix_hits: set[str] = set()

    def has_local_prefix_hit(self, request_id: str) -> bool:
        return request_id in self._local_prefix_hits

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        super().add_local_computed_blocks(
            request_id, new_computed_blocks, num_local_computed_tokens, num_external_computed_tokens
        )
        if (
            num_local_computed_tokens > 0
            and num_external_computed_tokens == 0
            and new_computed_blocks
            and not new_computed_blocks[-1].is_null
        ):
            self._local_prefix_hits.add(request_id)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._local_prefix_hits.discard(request_id)
        return super().pop_blocks_for_free(request_id)


class AscendGDNPrefixOnlyManager(AscendGDNLocalHitManager):
    """Keep the admitted local prefix without publishing the suffix."""

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
        *,
        replay_boundaries: Sequence[int],
    ) -> None:
        # Keep the matched state reusable. The inherited allocation and CoW
        # paths still provide a writable state for prefill and decode.
        if not self.has_local_prefix_hit(request.request_id):
            super().cache_blocks(
                request, num_tokens, retention_interval=retention_interval, replay_boundaries=replay_boundaries
            )


class AscendGDNCheckpointManager(AscendGDNLocalHitManager):
    """Reserve warm checkpoints outside the positional state table."""

    def __init__(self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.checkpoint_plans: dict[str, tuple[int, int]] = {}
        self.checkpoint_pages: dict[str, tuple[int, KVCacheBlock]] = {}

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
        prefill_end: int = 0,
    ) -> int:
        blocks = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
            prefill_end=prefill_end,
        )
        plan = self.checkpoint_plans.get(request_id)
        return blocks + int(plan is not None and plan[1] == num_tokens_main_model)

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        blocks = super().allocate_new_blocks(request_id, num_tokens, num_tokens_main_model)
        plan = self.checkpoint_plans.pop(request_id, None)
        if plan is not None and plan[1] == num_tokens_main_model:
            # The exporter overwrites both states. The page never appears in a
            # request row or in the CoW source/destination list.
            checkpoint = self.block_pool.get_new_blocks(1)[0]
            self.checkpoint_pages[request_id] = (plan[0], checkpoint)
        return blocks

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        # A page that has already been handed to the worker stays pinned by
        # its scheduler step until that step's feedback arrives.
        self.checkpoint_plans.pop(request_id, None)
        page = self.checkpoint_pages.pop(request_id, None)
        blocks = super().pop_blocks_for_free(request_id)
        if page is not None:
            blocks.append(page[1])
        return blocks
