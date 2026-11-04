"""Select request-visible KV regions and reduce them to one reachable prefix.

Selection returns cache-group-ordered masks derived from static cache geometry.
Resolution combines the corresponding chunk rows and Backend availability with
the locally available prefix, preserving any remote tail identities needed by
Load. Request hashes, token ranges, and observations remain call-time inputs;
Backend queries, object keys, memory projection, and transfer state stay outside
this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import SpecGroup
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    block_hash_to_bytes,
    get_block_hashes,
)

from ..protocol.messages import TailKeyBoundary
from .topology import KVPoolGroupTopology

BlockHashes = Sequence[BlockHash | str]
LookupChunkRanges: TypeAlias = tuple[
    Sequence[int] | NDArray[np.uint64], Sequence[int] | NDArray[np.uint64], tuple[BlockHash | str, ...]
]
ChunkMask: TypeAlias = tuple[bool, ...] | None
ChunkMasks: TypeAlias = tuple[ChunkMask, ...]
LookupObservation: TypeAlias = tuple[LookupChunkRanges, tuple[bool, ...]]


# =============================================================================
# Recovery Result and Cache View
# =============================================================================
#
# Represent the common-prefix result and adapt external key observations to the
# BlockPool surface expected by vLLM's cache-manager algorithms.


@dataclass(frozen=True, slots=True)
class ReachablePrefix:
    """Common reachable frontier and the remote identities needed by its tail."""

    end_token: int
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()


@dataclass(frozen=True, slots=True)
class ContiguousObservation:
    """Ordered full blocks; positions are derived instead of materialized per candidate."""

    start_token: int
    block_size: int
    available: tuple[bool, ...]


class ExternalCachedBlockPool:
    """Duck-typed BlockPool backed by external AscendStore key existence."""

    def __init__(self, hash_block_size: int, cached_hashes: set[tuple[int, bytes]] | None = None) -> None:
        # Load uses cached_hashes=None to select required blocks at an already confirmed endpoint.
        self._cached_hashes = cached_hashes
        self.hash_block_size = hash_block_size
        self.null_block = KVCacheBlock(block_id=0)
        self._present_block = KVCacheBlock(block_id=1)

    def get_cached_block(self, block_hash: BlockHash, group_indices: list[int]) -> list[KVCacheBlock] | None:
        if self._cached_hashes is None:
            return [self._present_block] * len(group_indices)
        h = block_hash_to_bytes(block_hash)
        if all((group_index, h) in self._cached_hashes for group_index in group_indices):
            return [self._present_block] * len(group_indices)
        return None

    def contains(self, group_index: int, block_hash: BlockHash | str) -> bool:
        if self._cached_hashes is None:
            return True
        return (group_index, block_hash_to_bytes(block_hash)) in self._cached_hashes


# =============================================================================
# Cache-Group Coordination
# =============================================================================
#
# Bind each cache type's Manager once, then reconcile its results through a shared cache view.


class Coordinator:
    """Select KV through registered Managers and resolve a common prefix across transfer groups."""

    def __init__(
        self,
        groups: tuple[KVPoolGroupTopology, ...],
        scheduler_block_size: int,
        hash_block_size: int,
        max_model_len: int,
        use_eagle: bool = False,
        retention_interval: int | None = None,
    ) -> None:
        self.groups = groups
        self.group_ids = tuple(group.group_id for group in groups)
        self.hash_block_size = hash_block_size
        self.scheduler_block_size = scheduler_block_size
        self.max_model_len = max_model_len
        self.retention_interval = retention_interval
        self.manager_groups = self._bind_managers(use_eagle)
        self.partial_hash_hits = any(
            isinstance(_unwrap_spec(group.kv_cache_spec), MambaSpec) and group.block_size > hash_block_size
            for group in groups
        )
        # Ordered observations already encode standard Full Attention's contiguous-prefix rule.
        self._contiguous_full_attention = bool(groups) and all(group.kv_cache_spec.prefix_cacheable for group in groups)
        self._contiguous_full_attention &= all(
            isinstance(binding.spec, FullAttentionSpec)
            and binding.manager_cls is FullAttentionManager
            and binding.spec.block_size == scheduler_block_size == hash_block_size
            for binding in self.manager_groups
        )

    def _bind_managers(self, use_eagle: bool) -> tuple[SpecGroup, ...]:
        bindings: list[SpecGroup] = []
        for group_index, group in enumerate(self.groups):
            if not group.kv_cache_spec.prefix_cacheable:
                continue
            spec = _unwrap_spec(group.kv_cache_spec)
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            if manager_cls is None:
                raise ValueError(f"No vLLM KV cache manager is registered for {type(spec).__name__}")
            for index, binding in enumerate(bindings):
                if binding.spec == spec:
                    assert manager_cls is binding.manager_cls, "Expected same manager for identical KV cache specs."
                    binding.group_ids.append(group_index)
                    if group.is_eagle_group and not binding.use_eagle:
                        bindings[index] = binding._replace(use_eagle=True)
                    break
            else:
                bindings.append(SpecGroup(spec, [group_index], manager_cls, group.is_eagle_group))
        if use_eagle and not any(binding.use_eagle for binding in bindings):
            bindings = [binding._replace(use_eagle=True) for binding in bindings]
        # Find Attention coverage before a checkpoint can move the endpoint inside its last object.
        return tuple(sorted(bindings, key=lambda binding: not isinstance(binding.spec, FullAttentionSpec)))

    @property
    def contiguous_lookup(self) -> bool:
        return self._contiguous_full_attention

    def select_for_lookup(self, end_token: int) -> ChunkMasks:
        aligned_token_len = (
            cdiv(min(end_token, self.max_model_len), self.scheduler_block_size) * self.scheduler_block_size
        )
        masks: list[ChunkMask] = [()] * len(self.groups)
        for binding in self.manager_groups:
            num_chunks = aligned_token_len // binding.spec.block_size
            # Lookup includes readable windows regardless of Store retention; Scheduler applies EAGLE trimming.
            mask = binding.manager_cls.reachable_block_mask(
                start_block=0,
                end_block=num_chunks,
                alignment_tokens=self.scheduler_block_size,
                kv_cache_spec=binding.spec,
                use_eagle=False,
                retention_interval=None,
                reachable_boundaries=(),
                dcp_world_size=1,
            )
            if mask is not None:
                assert len(mask) == num_chunks
            selected = None if mask is None or all(mask) else tuple(mask)
            for group_index in binding.group_ids:
                masks[group_index] = selected
        return tuple(masks)

    def resolve_available_end(
        self,
        start_token: int,
        end_token: int,
        block_hashes: BlockHashes,
        observations: Sequence[LookupObservation | ContiguousObservation],
    ) -> ReachablePrefix:
        if len(observations) != len(self.group_ids):
            raise ValueError(f"Lookup observations do not match configured groups {self.group_ids}")

        max_hit_length = min(end_token, self.max_model_len)
        if self._contiguous_full_attention:
            return self._resolve_contiguous_prefix(start_token, max_hit_length, block_hashes, observations)
        local_hit_length = min(start_token, max_hit_length)
        cached_hashes: set[tuple[int, bytes]] = set()
        for group_index, (group, observation) in enumerate(zip(self.groups, observations, strict=True)):
            group_block_size = group.block_size
            group_block_hashes = (
                block_hashes
                if self.partial_hash_hits
                else get_block_hashes(block_hashes, group_block_size, self.hash_block_size)
            )
            local_hit_count = local_hit_length // (self.hash_block_size if self.partial_hash_hits else group_block_size)
            cached_hashes.update(
                (group_index, block_hash_to_bytes(group_block_hashes[index]))
                for index in range(min(local_hit_count, len(group_block_hashes)))
            )
            if isinstance(observation, ContiguousObservation):
                raise ValueError("Compact Lookup observations require contiguous Full Attention")
            (_, _, observed_hashes), available = observation
            if len(observed_hashes) != len(available):
                raise ValueError("Lookup chunks and Backend availability have different lengths")
            cached_hashes.update(
                (group_index, block_hash_to_bytes(block_hash))
                for block_hash, hit in zip(observed_hashes, available, strict=True)
                if hit
            )

        if not cached_hashes:
            return ReachablePrefix(0)
        cached_block_pool = ExternalCachedBlockPool(self.hash_block_size, cached_hashes)
        hit_length = self._find_available_end(block_hashes, max_hit_length, cached_block_pool)
        return ReachablePrefix(hit_length, self._tail_key_boundaries(block_hashes, hit_length, cached_block_pool))

    def _resolve_contiguous_prefix(
        self,
        start_token: int,
        end_token: int,
        block_hashes: BlockHashes,
        observations: Sequence[LookupObservation | ContiguousObservation],
    ) -> ReachablePrefix:
        # Skip the trusted local prefix instead of rebuilding and scanning it through a virtual BlockPool.
        block_size = self.hash_block_size
        hit_length = min(end_token, len(block_hashes) * block_size)
        hit_length -= hit_length % block_size
        for observation in observations:
            if isinstance(observation, ContiguousObservation):
                if observation.block_size != block_size or observation.start_token % block_size:
                    raise ValueError("Compact Lookup observations must use aligned hash-sized blocks")
                group_end = min(start_token, hit_length)
                group_end -= group_end % block_size
                for position, hit in enumerate(observation.available):
                    chunk_start = observation.start_token + position * block_size
                    chunk_end = chunk_start + block_size
                    if chunk_end <= group_end:
                        continue
                    if chunk_start > group_end or chunk_end > hit_length or not hit:
                        break
                    group_end = chunk_end
                hit_length = group_end
                continue
            (starts, counts, observed_hashes), available = observation
            if not len(starts) == len(counts) == len(observed_hashes) == len(available):
                raise ValueError("Lookup chunks and Backend availability have different lengths")
            group_end = min(start_token, hit_length)
            group_end -= group_end % block_size
            for start, count, hit in zip(starts, counts, available, strict=True):
                chunk_end = int(start) + int(count)
                if chunk_end <= group_end:
                    continue
                if int(start) > group_end or chunk_end > hit_length or not hit:
                    break
                group_end = chunk_end
            hit_length = group_end
        return ReachablePrefix(hit_length)

    def _tail_key_boundaries(
        self, block_hashes: BlockHashes, hit_length: int, cached_block_pool: ExternalCachedBlockPool
    ) -> tuple[TailKeyBoundary, ...]:
        if not self.partial_hash_hits or hit_length <= 0:
            return ()

        hit_hash_index = hit_length // self.hash_block_size - 1
        boundaries = []
        for group_index, group in enumerate(self.groups):
            if not group.kv_cache_spec.prefix_cacheable:
                continue
            boundary_token = hit_length
            if not cached_block_pool.contains(group_index, block_hashes[hit_hash_index]):
                next_block_hash_index = min(
                    cdiv(hit_length, group.block_size) * group.block_size // self.hash_block_size, len(block_hashes)
                )
                for hash_index in range(hit_hash_index + 1, next_block_hash_index):
                    if cached_block_pool.contains(group_index, block_hashes[hash_index]):
                        boundary_token = (hash_index + 1) * self.hash_block_size
                        break
                else:
                    raise AssertionError(f"No remote tail key found for cache group {group.group_id} at {hit_length}")
            boundaries.append(TailKeyBoundary(group.group_id, boundary_token))
        return tuple(boundaries)

    def select_for_load(self, block_hashes: BlockHashes, end_token: int) -> ChunkMasks:
        if self._contiguous_full_attention:
            return (None,) * len(self.groups)
        cached_block_pool = ExternalCachedBlockPool(self.hash_block_size)
        # A confirmed Load still needs the physical block containing its partial tail.
        alignment = self.hash_block_size if self.partial_hash_hits else self.scheduler_block_size
        load_end = cdiv(end_token, alignment) * alignment
        masks: list[ChunkMask] = [()] * len(self.groups)
        for binding in self.manager_groups:
            if isinstance(binding.spec, FullAttentionSpec) and binding.manager_cls is FullAttentionManager:
                for group_index in binding.group_ids:
                    masks[group_index] = None
                continue
            blocks_by_group, hit_length = binding.manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=load_end,
                kv_cache_group_ids=binding.group_ids,
                block_pool=cast(BlockPool, cached_block_pool),
                kv_cache_spec=binding.spec,
                drop_eagle_block=False,
                alignment_tokens=alignment,
            )
            if hit_length < end_token:
                raise ValueError(f"{binding.manager_cls.__name__} cannot cover the confirmed Load endpoint {end_token}")
            for group_index, blocks in zip(binding.group_ids, blocks_by_group, strict=True):
                masks[group_index] = tuple(block is not cached_block_pool.null_block for block in blocks)
        return tuple(masks)

    def select_for_store(self, start_token: int, end_token: int, num_prompt_tokens: int) -> ChunkMasks:
        """Return per-group masks relative to the block containing start_token."""

        masks: list[ChunkMask] = [()] * len(self.groups)
        if end_token <= start_token:
            return tuple(masks)
        for binding in self.manager_groups:
            mask = store_group_mask(
                binding.spec,
                binding.manager_cls,
                end_token,
                num_prompt_tokens,
                alignment_tokens=self.scheduler_block_size,
                use_eagle=binding.use_eagle,
                retention_interval=self.retention_interval,
                start_block=max(start_token // binding.spec.block_size, 0),
            )
            for group_index in binding.group_ids:
                masks[group_index] = mask
        return tuple(masks)

    def _find_available_end(
        self, block_hashes: BlockHashes, max_length: int, cached_block_pool: ExternalCachedBlockPool
    ) -> int:
        alignment_tokens = self.hash_block_size if self.partial_hash_hits else self.scheduler_block_size
        if not self.manager_groups:
            return 0
        hit_length = max_length
        first_pass = True

        while True:
            previous_length = hit_length
            for binding in self.manager_groups:
                # The frontier only shrinks: Attention coverage already proved in the first pass remains valid.
                if not first_pass and isinstance(binding.spec, FullAttentionSpec):
                    continue
                _, hit_length = binding.manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=hit_length,
                    kv_cache_group_ids=binding.group_ids,
                    block_pool=cast(BlockPool, cached_block_pool),
                    kv_cache_spec=binding.spec,
                    drop_eagle_block=False,
                    alignment_tokens=alignment_tokens,
                )
                if hit_length == 0:
                    return 0
            if len(self.manager_groups) == 1 or hit_length == previous_length:
                return hit_length
            first_pass = False


# =============================================================================
# Store Source Selection
# =============================================================================
#
# Scheduler leases must cover the pages that Worker selection may read. Share
# interval, checkpoint-companion, and retention rules before either side applies
# its ownership or address-materialization steps.


def store_block_range(start_token: int, end_token: int, block_size: int, hash_count: int, block_count: int) -> range:
    """Intersect the Store interval, confirmed hashes, and retained block-table suffix."""

    logical_count = min(hash_count, -(-end_token // block_size))
    block_offset = max(logical_count - block_count, 0)
    return range(max(start_token // block_size, block_offset), logical_count)


def checkpoint_companion_indices(
    boundary_token: int, store_planned_end_token: int, block_size: int, block_count: int
) -> range:
    """Locate attention pages not covered by earlier commands and the checkpoint's boundary page."""

    logical_count = -(-boundary_token // block_size)
    block_offset = max(logical_count - block_count, 0)
    first_block = max(min(store_planned_end_token // block_size, logical_count - 1), block_offset)
    return range(first_block - block_offset, logical_count - block_offset)


def store_group_mask(
    spec: KVCacheSpec,
    manager_cls: type[SingleTypeKVCacheManager],
    end_token: int,
    num_prompt_tokens: int,
    *,
    alignment_tokens: int,
    use_eagle: bool,
    retention_interval: int | None,
    start_block: int = 0,
) -> ChunkMask:
    """Share upstream Store retention between source leases and Worker selection."""

    if not spec.prefix_cacheable or isinstance(spec, MambaSpec):
        return ()
    if end_token % alignment_tokens:
        return None
    mask = manager_cls.reachable_block_mask(
        start_block=start_block,
        end_block=end_token // spec.block_size,
        alignment_tokens=alignment_tokens,
        kv_cache_spec=spec,
        use_eagle=use_eagle,
        retention_interval=retention_interval,
        reachable_boundaries=(num_prompt_tokens - 1,),
        dcp_world_size=1,
    )
    return None if mask is None else tuple(mask)


def _unwrap_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec
