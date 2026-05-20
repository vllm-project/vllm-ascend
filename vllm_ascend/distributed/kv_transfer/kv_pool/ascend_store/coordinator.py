# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Adapted from vllm-project/vllm:
#   vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py
"""External-store cache-hit coordinator for the Ascend (Mooncake) store connector.

Mirrors ``HybridKVCacheCoordinator.find_longest_cache_hit`` against an
external pool, so the worker side can ask "for these N hashes, how many
*lcm-aligned* prefix tokens are present across **all** KV-cache groups
simultaneously?" — exactly the question the engine's hybrid prefix-cache
allocator asks against the in-process block pool.

Without this layer, a hybrid model (e.g. full-attn + SWA / mamba) would
either:

* over-report cache hits (counting groups individually instead of the
  intersection), causing the recv-side to attempt loads for chunks that
  no group will actually use → wasted DMA + data corruption risk; or
* under-report (assuming a single group), shrinking effective hit rate
  on multi-group models.

``ExternalCachedBlockPool`` is a duck-typed ``BlockPool`` whose
``get_cached_block`` answers from a ``(group_id, hash)`` exists-set.
That lets the unmodified ``SingleTypeKVCacheManager.find_longest_cache_hit``
work over the remote store's existence info.
"""

from __future__ import annotations

from typing import cast

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    SingleTypeKVCacheManager,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

# Dummy placeholder hash for store_mask's template computation.
_DUMMY_BLOCK_HASH = BlockHash(b"\x00" * 32)


class ExternalCachedBlockPool:
    """Duck-typed ``BlockPool`` backed by a ``(group_id, hash)`` exists set.

    Two construction modes:

    * ``exists`` is a set: send/lookup side — answer ``get_cached_block``
      from the membership set.
    * ``exists is None``: recv side — every chunk is treated as "present
      across all queried groups". Used to derive per-group **load masks**
      (which group populates which chunk locally, e.g. SWA tail vs Mamba
      tail) when the hit-length is already known from the producer.
    """

    def __init__(self, exists: set[tuple[int, bytes]] | None = None) -> None:
        self._exists = exists
        self.null_block = KVCacheBlock(block_id=0)
        # Dummy block id 1 for "present"; we never index into KV memory
        # via this block — it's purely a sentinel for the manager's
        # ``find_longest_cache_hit`` loop to count hits.
        self._present_block = KVCacheBlock(block_id=1)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        group_ids: list[int],
    ) -> list[KVCacheBlock] | None:
        # Mirrors BlockPool.get_cached_block: a hit only when *every*
        # group_id in ``group_ids`` (groups sharing one spec) has the hash.
        if self._exists is None:
            return [self._present_block] * len(group_ids)
        h = bytes(block_hash)
        if all((g, h) in self._exists for g in group_ids):
            return [self._present_block] * len(group_ids)
        return None


class MooncakeStoreCoordinator:
    """Per-group cache-hit oracle for the Ascend store connector.

    Built once at worker init from the engine's ``kv_cache_groups``; used
    by both send and recv paths to ask:

    * ``find_longest_cache_hit(hashes, max_len, pool)`` — longest prefix
      where every group in the topology can serve from ``pool``.
    * ``load_mask(hashes, token_len)`` — for the recv side: per-group
      boolean per-chunk mask (which chunks the consumer's group spec
      actually populates locally, e.g. SWA pre-window False).
    * ``store_mask(aligned_token_len)`` — for the send side: per-group
      template mask of chunks worth putting (so we don't store
      pre-window chunks that no future cache hit will ever read).
    """

    def __init__(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        scheduler_block_size: int,
        hash_block_size: int,
        use_eagle: bool = False,
    ) -> None:
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0 for g in kv_cache_groups
        ), "every group's block_size must be a multiple of hash_block_size"
        assert scheduler_block_size % hash_block_size == 0, (
            f"scheduler_block_size ({scheduler_block_size}) must be a multiple of "
            f"hash_block_size ({hash_block_size})"
        )
        assert all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_groups
        ), "scheduler_block_size must be a multiple of each group's block_size"
        self.kv_cache_groups = kv_cache_groups
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        self.use_eagle = use_eagle
        self._verify_and_split_kv_cache_groups()

    def _verify_and_split_kv_cache_groups(self) -> None:
        """Group ``kv_cache_groups`` by spec, sorted with full-attn first.

        Mirrors ``KVCacheCoordinator.verify_and_split_kv_cache_groups`` but
        only collects ``(spec, group_ids, manager_cls)`` triples — we
        never allocate the actual ``SingleTypeKVCacheManager`` since this
        coordinator never owns KV memory.
        """
        attention_groups: list[
            tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]
        ] = []
        for i, g in enumerate(self.kv_cache_groups):
            spec = _unwrap_spec(g.kv_cache_spec)
            manager_cls = spec_manager_map[type(spec)]
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))
        # Full-attention first matches upstream convergence ordering.
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )
        self.eagle_attn_group_indices: set[int] = {
            i
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(self.kv_cache_groups[gid].is_eagle_group for gid in group_ids)
        }
        if self.use_eagle and not self.eagle_attn_group_indices:
            # Eagle declared at the connector level but the kv-cache config
            # didn't mark any group as eagle — treat all groups as eagle.
            self.eagle_attn_group_indices = set(range(len(self.attention_groups)))

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
    ) -> tuple[tuple[list[bool], ...], int]:
        """Per-group load masks + lcm-aligned global hit length."""
        blocks_per_group, hit_length = self._find_hit_blocks(
            block_hashes, max_length, cached_block_pool
        )
        masks = tuple(
            [blk is not cached_block_pool.null_block for blk in blocks]
            for blocks in blocks_per_group
        )
        return masks, hit_length

    def load_mask(
        self,
        block_hashes: list[BlockHash],
        token_len: int,
    ) -> tuple[list[bool], ...]:
        """Recv-side: which chunks this consumer's spec populates locally."""
        masks, _ = self.find_longest_cache_hit(
            block_hashes, token_len, ExternalCachedBlockPool()
        )
        return masks

    def store_mask(self, aligned_token_len: int) -> tuple[list[bool], ...]:
        """Send-side: per-group template mask covering all lcm segments
        up to ``aligned_token_len``.
        """
        assert aligned_token_len % self.lcm_block_size == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of "
            f"lcm_block_size ({self.lcm_block_size})"
        )
        if aligned_token_len == 0:
            return tuple([] for _ in self.kv_cache_groups)

        num_chunks_per_group = [
            aligned_token_len // g.kv_cache_spec.block_size
            for g in self.kv_cache_groups
        ]
        # Fast path: every group is full-attn or its block_size == lcm.
        if all(
            isinstance(spec, FullAttentionSpec)
            or spec.block_size == self.lcm_block_size
            for spec, _, _ in self.attention_groups
        ):
            return tuple([True] * n for n in num_chunks_per_group)

        # Generic path: build one lcm-segment template via the same
        # manager logic; tile by n_segments.
        n_segments = aligned_token_len // self.lcm_block_size
        dummy_hashes: list[BlockHash] = [_DUMMY_BLOCK_HASH] * (
            self.lcm_block_size // self.hash_block_size
        )
        template_masks, _ = self.find_longest_cache_hit(
            dummy_hashes,
            max_length=self.lcm_block_size,
            cached_block_pool=ExternalCachedBlockPool(),
        )
        return tuple(
            list(template_masks[g]) * n_segments
            for g in range(len(self.kv_cache_groups))
        )

    def block_hashes_for_spec(
        self, block_hashes: list[BlockHash], spec: KVCacheSpec
    ) -> BlockHashList:
        """Re-bucket ``block_hashes`` from ``hash_block_size`` to the
        group's ``spec.block_size`` (e.g. mamba groups using larger blocks)."""
        if spec.block_size == self.hash_block_size:
            return block_hashes
        return BlockHashListWithBlockSize(
            block_hashes, self.hash_block_size, spec.block_size
        )

    def _find_hit_blocks(
        self,
        block_hashes: list[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """Mirrors ``HybridKVCacheCoordinator.find_longest_cache_hit`` but
        dispatches via ``spec_manager_map`` (no manager instances)."""
        if len(self.attention_groups) == 1:
            # Single spec across all groups: defer to the manager class.
            spec, group_ids, manager_cls = self.attention_groups[0]
            hashes = self.block_hashes_for_spec(block_hashes, spec)
            hit_blocks = manager_cls.find_longest_cache_hit(
                block_hashes=hashes,
                max_length=max_length,
                kv_cache_group_ids=group_ids,
                block_pool=cast(BlockPool, cached_block_pool),
                kv_cache_spec=spec,
                use_eagle=(0 in self.eagle_attn_group_indices),
                alignment_tokens=spec.block_size,
            )
            num_groups = len(self.kv_cache_groups)
            blocks_by_group: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
            for gid, blks in zip(group_ids, hit_blocks, strict=True):
                blocks_by_group[gid] = blks
            return tuple(blocks_by_group), len(hit_blocks[0]) * spec.block_size

        # Multi-spec hybrid: iterate the "shrink-until-stable" loop.
        num_groups = len(self.kv_cache_groups)
        hit_length = max_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        # Fast-converge if we're full-attn + 1 other spec.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length
            for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                cached = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached is not None:
                    # Full-attn truncates to curr_hit_length-aligned value.
                    curr_hit_length = (
                        curr_hit_length // spec.block_size * spec.block_size
                    )
                    continue
                use_eagle = (
                    idx in self.eagle_attn_group_indices and idx not in eagle_verified
                )
                _max_length = curr_hit_length
                if use_eagle:
                    # Eagle peeks one extra block past the current hit
                    # length to verify alignment.
                    _max_length = min(curr_hit_length + spec.block_size, max_length)
                hashes = self.block_hashes_for_spec(block_hashes, spec)
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=hashes,
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=cast(BlockPool, cached_block_pool),
                    kv_cache_spec=spec,
                    use_eagle=use_eagle,
                    alignment_tokens=self.lcm_block_size,
                )
                _new_hit_length = len(hit_blocks[0]) * spec.block_size
                if use_eagle:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # Convergence shrank → re-verify eagle groups.
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for gid, blocks in zip(group_ids, hit_blocks, strict=True):
                    hit_blocks_by_group[gid] = blocks
            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full-attn blocks to final converged length; other
        # spec classes already trim themselves inside their hit logic.
        spec0, group_ids0, _ = self.attention_groups[0]
        if isinstance(spec0, FullAttentionSpec):
            num_blocks = hit_length // spec0.block_size
            for gid in group_ids0:
                full_blks = hit_blocks_by_group[gid]
                assert full_blks is not None
                del full_blks[num_blocks:]

        return (
            tuple(blks if blks is not None else [] for blks in hit_blocks_by_group),
            hit_length,
        )


def _unwrap_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec
