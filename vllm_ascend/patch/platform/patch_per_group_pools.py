# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Per-group KV block pools for window-capped hybrid attention on NPU.

The NPU materializes private contiguous per-layer KV buffers (LBNHC), so the
physical cost of a KV cache config is the sum over groups of
(group layers x group blocks x page bytes) rather than a single shared
backing. For hybrid full+sliding models (e.g. gemma4: 5 sliding + 1 full
equal-page group) this makes the generic 1/n budget scaling shrink every
group's pool, collapsing the token capacity of the prefix cache.

``PerGroupBlockPools`` presents one ``BlockPool`` per KV cache group (sliding
groups sized by their window-aware admission bound, the full group sized by
the full-sequence bound) as a single pool interface, so the scheduler-side
call sites that assume one shared pool keep working unchanged.
"""
from __future__ import annotations

from vllm.logger import logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock


class PerGroupBlockPools:
    """Aggregate of per-group :class:`BlockPool` instances behind one-pool API.

    Block IDs are group-local: each group's manager draws from its own pool,
    so a sliding group's block tables only ever contain IDs in its own
    (smaller) ID space. The aggregate exposes the conservative semantics the
    scheduler needs:

    - ``get_num_free_blocks``: total free across pools, because the
      top-level admission gate compares a request's total (summed over
      groups) block need against it; per-group shortfalls then surface as
      per-group allocation backpressure, not as over-rejection;
    - ``free_blocks``: routed to each block's owning pool (``KVCacheBlock.pool``);
    - ``evict_blocks``: each pool evicts the IDs in its own range (IDs beyond
      a pool's range belong to another group; skipping is correct);
    - ``null_block``: every pool's null block has ID 0, which is always the
      "no block" sentinel inside any group-local block table.
    """

    def __init__(self, pools: list[BlockPool]):
        self.pools = list(pools)
        if not self.pools:
            raise ValueError("PerGroupBlockPools requires at least one pool")
        logger.info(
            "Window-capped per-group KV block pools: %d pools with %s blocks "
            "(group-local ID spaces).",
            len(self.pools),
            [pool.num_gpu_blocks for pool in self.pools],
        )

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
        # Total free across pools. The top-level admission gate
        # (KVCacheManager.allocate_slots) compares a request's TOTAL
        # across-groups block need against this value, so the total free is
        # the correct pre-screen. A per-group shortfall (one pool tight while
        # others have room) then surfaces as per-group allocation
        # backpressure — the request simply waits — instead of min(), which
        # rejected requests that fit per-group whenever the single tightest
        # pool was below the total need (e.g. full pool near capacity capped
        # every new request even though the sliding pools were mostly free).
        return sum(pool.get_num_free_blocks() for pool in self.pools)

    def get_usage(self) -> float:
        total = sum(pool.num_gpu_blocks for pool in self.pools)
        if total == 0:
            return 0.0
        free = sum(pool.get_num_free_blocks() for pool in self.pools)
        return (total - free) / total

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
    models never do."""
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
        MambaSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )

    has_sliding = False
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        inner = (
            spec.kv_cache_specs.values()
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )
        for layer_spec in inner:
            if isinstance(layer_spec, MambaSpec):
                return False
            if not isinstance(layer_spec, AttentionSpec):
                return False
            if isinstance(layer_spec, SlidingWindowSpec):
                has_sliding = True
    return has_sliding
