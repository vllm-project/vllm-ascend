# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Per-group KV block pools for window-capped hybrid attention on NPU.

The NPU materializes private contiguous per-layer KV buffers (LBNHC), so the
physical cost of a KV cache config is the sum over groups of
(group layers x group blocks x page bytes) rather than a single shared
backing. For hybrid full+sliding models (e.g. gemma4: 5 sliding + 1 full
equal-page group) this makes the generic 1/n budget scaling shrink every
group's pool, collapsing the token capacity of the prefix cache.

``PerGroupBlockPools`` keeps each group's allocation independent. Full
attention defines the scheduler's scalar capacity unit. The coordinator
predicts pure demand vectors; the manager checks every pool at admission.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.logger import logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs


def get_full_attention_group_ids(kv_cache_groups) -> tuple[int, ...]:
    full_group_ids = []
    for group_id, group in enumerate(kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = next(iter(spec.kv_cache_specs.values()))
        if isinstance(spec, FullAttentionSpec):
            full_group_ids.append(group_id)
    return tuple(full_group_ids)


@dataclass(frozen=True)
class GroupBlockReservation:
    """Remaining in-flight demand, in each group's own block units."""

    blocks: tuple[int, ...]


class PerGroupBlockPools:
    """Independent block pools with a full-attention capacity interface.

    The first full-attention group defines scalar capacity and reservation
    units. Admission checks every pool; unused sliding blocks cannot satisfy
    a full-attention allocation. Usage reports the most pressured full group.
    """

    def __init__(self, pools: list[BlockPool], full_group_ids: tuple[int, ...]):
        self.pools = list(pools)
        if not self.pools:
            raise ValueError("PerGroupBlockPools requires at least one pool")
        if (
            not full_group_ids
            or len(set(full_group_ids)) != len(full_group_ids)
            or any(group_id < 0 or group_id >= len(self.pools) for group_id in full_group_ids)
        ):
            raise ValueError("PerGroupBlockPools requires valid full-attention group IDs")
        self.full_group_ids = tuple(full_group_ids)
        self.primary_group_id = self.full_group_ids[0]
        self.primary_pool = self.pools[self.primary_group_id]
        logger.info(
            "Window-capped per-group KV block pools: %d pools with %s blocks "
            "(group-local ID spaces); full groups=%s, primary=%d.",
            len(self.pools),
            [pool.num_gpu_blocks for pool in self.pools],
            self.full_group_ids,
            self.primary_group_id,
        )

    @property
    def num_gpu_blocks(self) -> int:
        return self.primary_pool.num_gpu_blocks

    @property
    def null_block(self) -> KVCacheBlock:
        # Every pool's null block has ID 0; group-local block tables use 0 as
        # the "no block" placeholder, so any pool's null block works.
        return self.pools[0].null_block

    @property
    def hash_block_size(self) -> int:
        # All pools are built with the coordinator's single hash_block_size.
        return self.pools[0].hash_block_size

    def get_cached_block(self, block_hash, kv_cache_group_ids: list[int]) -> list[KVCacheBlock] | None:
        # Cache lookups for a SpecGroup span several groups; in per-group mode
        # pool i owns group i, so look each group up in its own pool (the
        # cached-block hash is group-aware via BlockHashWithGroupId). Misses
        # for any group, matching the shared-pool semantics.
        cached_blocks: list[KVCacheBlock] = []
        for group_id in kv_cache_group_ids:
            pool = self.pools[group_id] if group_id < len(self.pools) else self.pools[0]
            blocks = pool.get_cached_block(block_hash, [group_id])
            if not blocks:
                return None
            cached_blocks.append(blocks[0])
        return cached_blocks

    def get_num_free_blocks(self) -> int:
        return self.primary_pool.get_num_free_blocks()

    def get_usage(self) -> float:
        # Delegate null-block accounting to the real pools.
        return max(self.pools[group_id].get_usage() for group_id in self.full_group_ids)

    def can_allocate(
        self,
        group_demands: Sequence[int],
        reserved_blocks: int | GroupBlockReservation = 0,
        watermark_blocks: int = 0,
    ) -> bool:
        if len(group_demands) != len(self.pools):
            raise ValueError("Block demand must contain every KV cache group")
        if isinstance(reserved_blocks, GroupBlockReservation):
            reservations = reserved_blocks.blocks
        elif reserved_blocks == 0:
            reservations = (0,) * len(self.pools)
        else:
            raise ValueError("Independent pools require a per-group reservation, not a nonzero scalar")
        if len(reservations) != len(self.pools):
            raise ValueError("Block reservation must contain every KV cache group")
        if watermark_blocks < 0 or any(n < 0 for n in (*group_demands, *reservations)):
            raise ValueError("Block demand, reservation and watermark must be non-negative")
        return all(
            demand + reserved + (watermark_blocks if group_id == self.primary_group_id else 0)
            <= pool.get_num_free_blocks()
            for group_id, (pool, demand, reserved) in enumerate(zip(self.pools, group_demands, reservations))
        )

    def free_blocks(self, ordered_blocks) -> None:
        # Group by owning pool, preserving per-pool relative order (free
        # queues rely on ordering for eviction LRU).
        by_pool: dict[BlockPool, list[KVCacheBlock]] = {}
        for block in ordered_blocks:
            pool = block.pool if block.pool is not None else self.pools[0]
            by_pool.setdefault(pool, []).append(block)
        for pool, blocks in by_pool.items():
            pool.free_blocks(blocks)

    def evict_blocks(self, block_ids: set[int]) -> None:
        for pool in self.pools:
            # Block IDs are group-local; only evict IDs this pool owns.
            valid = {bid for bid in block_ids if bid < len(pool.blocks)}
            if valid:
                pool.evict_blocks(valid)

    def take_events(self) -> list:
        events: list = []
        for pool in self.pools:
            events.extend(pool.take_events())
        return events

    def reset_prefix_cache(self) -> bool:
        return all(pool.reset_prefix_cache() for pool in self.pools)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        # Pools own disjoint, group-local ID spaces, so a group-agnostic
        # caller cannot be routed safely: handing out pool 0's (sliding) IDs
        # for a full-group block table would address the wrong ID space.
        # Every legitimate allocation goes through the per-group manager
        # wired in the coordinator (each manager holds its own pool), so a
        # call here means an unexpected shared caller — fail loud instead of
        # silently allocating from the wrong pool.
        raise RuntimeError(
            "PerGroupBlockPools.get_new_blocks: group-agnostic allocation is "
            f"not supported (requested {num_blocks} blocks); allocation must "
            "go through the per-group KV cache managers."
        )


def is_window_capped_hybrid(vllm_config, kv_cache_groups) -> bool:
    """Whether the NPU window-capped per-group pool layout (Plan B) applies:
    >=2 KV cache groups, all attention (no mamba), at least one sliding-window
    group, on a per-layer (layer- and block-compact) layout. gemma4 (5 sliding
    + 1 full equal-page groups) qualifies; DSV4/GLM5/K3/mamba/single-group
    models never do.

    Kill switch: set ``VLLM_ASCEND_DISABLE_WINDOW_CAPPED_POOLS=1`` to force the
    stock shared-pool path. AscendStoreConnector additionally requires the
    per-group tensor-registration support from companion PR #19249."""
    import os as _os

    if _os.environ.get("VLLM_ASCEND_DISABLE_WINDOW_CAPPED_POOLS", "0") == "1":
        return False
    if len(kv_cache_groups) < 2:
        return False
    try:
        layout = vllm_config.cache_config.get_resolved_kv_cache_layout()
        if not (layout.is_layer_compact and layout.is_block_compact):
            return False
    except Exception:
        return False
    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )

    has_full = False
    has_sliding = False
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        inner = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        for layer_spec in inner:
            if isinstance(layer_spec, MambaSpec):
                return False
            if not isinstance(layer_spec, AttentionSpec):
                return False
            if isinstance(layer_spec, FullAttentionSpec):
                has_full = True
            if isinstance(layer_spec, SlidingWindowSpec):
                has_sliding = True
    return has_full and has_sliding
