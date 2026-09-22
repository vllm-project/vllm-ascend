# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common KV/SWA/tail recovery for the synchronous compressor scheduler."""

from collections.abc import Sequence

from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock

from vllm_ascend.core.compressor_checkpoint import CompressorCheckpointPool
from vllm_ascend.core.single_type_kv_cache_manager import CompressorTailManager
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator

# Snapshots borrow at most this fraction of the existing global page pool.
# They are reclaimable on ordinary request admission, unlike running rings.
CHECKPOINT_POOL_DIVISOR = 16


class CompressorCheckpointCoordinator(AscendHybridKVCacheCoordinator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_managers = {
            group_id: manager
            for group_id, manager in enumerate(self.single_type_managers)
            if isinstance(manager, CompressorTailManager)
        }
        self.checkpoints = CompressorCheckpointPool(
            self.block_pool,
            {group_id: manager.ring_blocks_per_request for group_id, manager in self.tail_managers.items()},
            self.kv_cache_config.num_blocks // CHECKPOINT_POOL_DIVISOR,
        )
        self.pending_restores: dict[str, int] = {}
        self.admission_watermark = 0
        # Both SWA retention and the write path must use the same grid as
        # the complete compressed-page lookup, not the physical state page.
        self.scheduler_block_size = self.lcm_block_size

    @staticmethod
    def _is_tail_spec(spec) -> bool:
        if hasattr(spec, "ring_blocks_per_request"):
            return True
        nested = getattr(spec, "kv_cache_specs", {})
        return bool(nested) and all(hasattr(item, "ring_blocks_per_request") for item in nested.values())

    def _get_effective_block_size(self, kv_cache_spec):
        if self._is_tail_spec(kv_cache_spec):
            # Storage page widths do not impose prefix-hash alignment.
            # The ring snapshot is keyed at the other groups' common grid.
            return self.hash_block_size
        return super()._get_effective_block_size(kv_cache_spec)

    def verify_and_split_kv_cache_groups(self):
        super().verify_and_split_kv_cache_groups()
        self.attention_groups = [group for group in self.attention_groups if not self._is_tail_spec(group[0])]
        # Speculative decoding is excluded from this first implementation.
        assert not self.eagle_group_ids

    def find_longest_cache_hit(self, block_hashes: list[BlockHash], max_cache_hit_length: int):
        candidate_limit = max_cache_hit_length
        while candidate_limit > 0:
            candidate = self.checkpoints.find_candidate(
                block_hashes, candidate_limit, self.lcm_block_size, self.hash_block_size
            )
            if candidate is None:
                break
            handle, position = candidate
            # Re-run every cache group when either the tail or another sparse
            # group moves the boundary. A later SWA window is not an earlier one.
            blocks, hit_length, _ = super().find_longest_cache_hit(block_hashes, position)
            if hit_length == position:
                result = list(blocks)
                for group_id, sources in self.checkpoints.snapshot_blocks(handle).items():
                    result[group_id] = list(sources)
                return tuple(result), position, 0
            candidate_limit = hit_length
        return tuple([] for _ in self.single_type_managers), 0, 0

    def _without_tail_sources(self, blocks):
        return tuple(() if group_id in self.tail_managers else group for group_id, group in enumerate(blocks))

    def _restore_handle(self, blocks) -> int | None:
        sources = {group_id: blocks[group_id] for group_id in self.tail_managers}
        if not any(sources.values()):
            return None
        return self.checkpoints.handle_for_blocks(sources)

    def get_num_blocks_to_allocate(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        num_encoder_tokens,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=False,
    ):
        handle = self._restore_handle(new_computed_blocks)
        if handle is not None:
            self.checkpoints.acquire(handle)
        try:
            needed = super().get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                self._without_tail_sources(new_computed_blocks),
                num_encoder_tokens,
                total_computed_tokens,
                num_local_computed_tokens,
                num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
            # Do not preempt a running request merely to retain idle snapshots.
            self.checkpoints.reclaim(needed + self.admission_watermark)
            return needed
        finally:
            if handle is not None:
                self.checkpoints.release(handle)

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert num_external_computed_tokens == 0
        handle = self._restore_handle(new_computed_blocks)
        if handle is not None:
            self.checkpoints.acquire(handle)
        try:
            super().allocate_new_computed_blocks(
                request_id,
                self._without_tail_sources(new_computed_blocks),
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
        except Exception:
            if handle is not None:
                self.checkpoints.release(handle)
            raise
        if handle is not None:
            self.pending_restores[request_id] = handle

    def pop_blocks_for_free(self, request_id: str):
        # A not-yet-dispatched recovery can be preempted during scheduling.
        # Dispatched handles are removed from this map and owned by the step.
        if (handle := self.pending_restores.pop(request_id, None)) is not None:
            self.checkpoints.release(handle)
        return super().pop_blocks_for_free(request_id)

    def free(self, request_id: str) -> None:
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))

    def cache_blocks(self, request, num_computed_tokens):
        # allocate_slots calls this before forward. Publishing ordinary KV
        # here could combine an old READY tail with newly allocated but still
        # unwritten SWA/KV pages from another request in the same batch.
        # The synchronous scheduler publishes all groups after worker completion.
        pass

    def cache_completed_blocks(self, request, num_computed_tokens):
        super().cache_blocks(request, num_computed_tokens)
