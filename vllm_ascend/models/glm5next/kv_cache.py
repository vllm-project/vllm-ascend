# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache spec and manager for the GLM-5.3-Flash kpool indexer tail.

The kpool indexer compresses whole pools of ``index_kpool`` tokens into one
cached vector. The incomplete pool at the end of a sequence has no compressed
form yet, so its raw K plus gate score lives in a separate one-block scratch
cache. That block is overwritten in place by ``pos % kpool``, which makes it
per-request transient state rather than a shareable prefix.

Neither the spec nor the manager exists upstream while the GLM-5.3-Flash
architecture lives downstream, so both are defined here. They are registered
from ``vllm_ascend.core.kv_cache_interface.register_ascend_kv_cache_specs``,
which vLLM invokes through the ``register_custom_kv_cache_specs`` platform hook
after the built-in specs are in place.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from vllm.config import VllmConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec
from vllm.v1.request import Request


class KpoolTailManager(FullAttentionManager):
    """Fixed 1-block-per-request circular buffer for ``KpoolTailSpec``.

    The tail cache holds the incomplete pool's raw K + gate score: exactly one
    block of ``kpool`` slots per request, overwritten in place by ``pos % kpool``
    as decode/spec-decode advances. Prefill seeds it; the connector transfers it
    across PD; decode reads it to compress the boundary pool correctly.

    This manager allocates that single block on first admission and reuses it
    for the request's whole lifetime. It never skips, never prunes, never
    prefix-caches. The no-prune guarantee is load-bearing:
    ``SlidingWindowManager.remove_skipped_blocks`` would evict the in-progress
    pool's earlier tokens mid-pool, before completion and before PD transfer,
    which is fatal. Because the block is circularly reused, allocation is
    independent of sequence length and of MTP size (MTP > kpool still fits in
    one block, since completed pools flush mid-step).
    """

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        # Tail state is per-request transient (circularly overwritten), so it is
        # neither shareable nor a stable function of a shareable prefix.
        return tuple([] for _ in range(len(kv_cache_group_ids))), 0

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        # Never hash tail blocks into the prefix cache.
        return

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        # The single block holds the in-progress pool for the whole request; no
        # token is ever out of window.
        return 0

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        # Never prune mid-request; the block is freed on request completion.
        return

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        # Exactly one block per request, reused circularly; never grow.
        return max(1 - len(self.req_to_blocks.get(request_id, ())), 0)

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int) -> list[KVCacheBlock]:
        # Cap at one block regardless of num_tokens; the kernel reuses its slots
        # via pos % kpool. No partial-hit CoW path (find_longest_cache_hit never
        # hits, so _partial_hit_reqs is always empty).
        req_blocks = self.req_to_blocks[request_id]
        if len(req_blocks) >= 1:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in new_blocks)
        return new_blocks

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # The tail never has local prefix-cache hits (find_longest_cache_hit
        # returns none); external (PD-transferred) tokens are handled by
        # allocate_external_computed_blocks below.
        return

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # The tail is a fixed 1-block circular buffer; PD-transferred (external)
        # tokens do not grow it -- the kernel reuses the single block's slots via
        # pos % kpool. The base FullAttention path would allocate
        # cdiv(num_external, block_size) blocks (one per kpool tokens), which
        # both wastes blocks and mismatches the producer's 1-block transfer,
        # tripping the NIXL reconcile block-count assert. Cap at one block,
        # matching allocate_new_blocks and the producer.
        req_blocks = self.req_to_blocks[request_id]
        if len(req_blocks) >= 1:
            return
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in new_blocks)


@dataclass(frozen=True, kw_only=True)
class KpoolTailSpec(SlidingWindowSpec):
    """One-block circular scratch cache for a kpool indexer's raw tail."""

    def max_admission_blocks_per_request(self, max_in_flight_tokens: int, max_model_len: int) -> int:
        return 1

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        return 1

    def is_uniform_with_collection(self, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        return all(isinstance(spec, KpoolTailSpec) for spec in kv_cache_specs.values())

    @property
    def participates_in_prefix_caching(self) -> bool:
        return False
