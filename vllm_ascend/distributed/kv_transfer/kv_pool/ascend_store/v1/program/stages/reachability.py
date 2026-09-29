from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
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

from ...coordinates import TokenRange
from ...protocol.lookup import TailKeyBoundary
from ..spec.topology import KVPoolGroupTopology

ChunkMask = tuple[bool, ...] | None
BlockHashes = Sequence[BlockHash | str]


@dataclass(frozen=True, slots=True)
class GroupSelection:
    """Logical chunks selected for one original vLLM cache group."""

    group_id: int
    chunk_mask: ChunkMask

    def includes(self, start_token: int, block_size: int) -> bool:
        chunk_index = start_token // block_size
        return self.chunk_mask is None or (chunk_index < len(self.chunk_mask) and self.chunk_mask[chunk_index])


@dataclass(frozen=True, slots=True)
class KVSelection:
    """Content-identified semantic KV selected on the token axis."""

    token_range: TokenRange
    block_hashes: tuple[BlockHash | str, ...]
    groups: tuple[GroupSelection, ...]


@dataclass(frozen=True, slots=True)
class ChunkAvailability:
    """Backend availability observed for one semantic KV chunk."""

    token_range: TokenRange
    content_hash: BlockHash | str
    available: bool


@dataclass(frozen=True, slots=True)
class GroupAvailability:
    """Semantic chunk observations for one original vLLM cache group."""

    group_id: int
    chunks: tuple[ChunkAvailability, ...]


@dataclass(frozen=True, slots=True)
class ReachablePrefix:
    """Common reachable frontier and the remote identities needed to load its tail."""

    end_token: int
    tail_key_boundaries: tuple[TailKeyBoundary, ...] = ()


class ReachableRegionSelection(Protocol):
    """Select reachable logical regions and reduce their observed common frontier."""

    group_ids: tuple[int, ...]

    def select_for_lookup(self, block_hashes: BlockHashes, query_range: TokenRange) -> KVSelection: ...

    def resolve_available_end(
        self, selection: KVSelection, availability: Sequence[GroupAvailability]
    ) -> ReachablePrefix: ...

    def select_for_load(self, block_hashes: BlockHashes, load_range: TokenRange) -> KVSelection: ...

    def select_for_store(
        self, block_hashes: BlockHashes, store_range: TokenRange, num_prompt_tokens: int
    ) -> KVSelection: ...


class ExternalCachedBlockPool:
    """Duck-typed BlockPool backed by external AscendStore key existence."""

    def __init__(self, hash_block_size: int, cached_hashes: set[tuple[int, bytes]] | None = None) -> None:
        # cached_hashes=None is used for Load/Store masks where hit length has already
        # been decided and each manager only needs to apply its own reachability.
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


class UnitaryReachability:
    """Select reachable KV for one transferable cache group."""

    def __init__(
        self,
        group_id: int,
        max_model_len: int,
        cache_transfer_granularity: int,
    ) -> None:
        self.group_ids: tuple[int, ...] = (group_id,)
        self._max_model_len = max_model_len
        self._cache_transfer_granularity = cache_transfer_granularity

    def select_for_lookup(self, block_hashes: BlockHashes, query_range: TokenRange) -> KVSelection:
        return KVSelection(query_range, tuple(block_hashes), (GroupSelection(self.group_ids[0], None),))

    def resolve_available_end(
        self, selection: KVSelection, availability: Sequence[GroupAvailability]
    ) -> ReachablePrefix:
        if len(availability) != 1 or availability[0].group_id != self.group_ids[0]:
            raise ValueError(f"Expected one Lookup observation for group {self.group_ids[0]}")

        query_range = selection.token_range
        max_hit_length = min(query_range.end_token, self._max_model_len)
        hit_end = min(query_range.start_token, max_hit_length)
        hit_end -= hit_end % self._cache_transfer_granularity
        observation = availability[0]
        for chunk in observation.chunks:
            if chunk.token_range.end_token > max_hit_length or not chunk.available:
                break
            if chunk.token_range.end_token % self._cache_transfer_granularity == 0:
                hit_end = chunk.token_range.end_token
        return ReachablePrefix(hit_end)

    def select_for_load(self, block_hashes: BlockHashes, load_range: TokenRange) -> KVSelection:
        return KVSelection(load_range, tuple(block_hashes), (GroupSelection(self.group_ids[0], None),))

    def select_for_store(
        self, block_hashes: BlockHashes, store_range: TokenRange, num_prompt_tokens: int
    ) -> KVSelection:
        return KVSelection(store_range, tuple(block_hashes), (GroupSelection(self.group_ids[0], None),))


class HybridReachability:
    """Select reachable KV across heterogeneous transfer groups.

    This mirrors vLLM's external KV reachability rules but uses AscendStore's external
    key granularity. Compressed specs already expose raw-token block sizes,
    while transfer addresses remain in cache-domain blocks.
    """

    def __init__(
        self,
        groups: tuple[KVPoolGroupTopology, ...],
        scheduler_block_size: int,
        hash_block_size: int,
        max_model_len: int,
        use_eagle: bool = False,
        retention_interval: int | None = None,
    ) -> None:
        assert scheduler_block_size % hash_block_size == 0, (
            f"scheduler_block_size ({scheduler_block_size}) must be a multiple of hash_block_size ({hash_block_size})"
        )

        self.groups = groups
        self.group_ids = tuple(group.group_id for group in groups)
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        self.max_model_len = max_model_len
        self.use_eagle = use_eagle
        self.retention_interval = retention_interval
        self.effective_block_sizes = [group.kv_cache_spec.block_size for group in groups]
        for effective_block_size in self.effective_block_sizes:
            assert effective_block_size % hash_block_size == 0, "block_size must be divisible by hash_block_size"
            assert scheduler_block_size % effective_block_size == 0, (
                "scheduler_block_size must be a multiple of each group's effective block_size"
            )

        self.eagle_group_indices = {index for index, group in enumerate(groups) if group.is_eagle_group}
        if use_eagle and not self.eagle_group_indices:
            self.eagle_group_indices = set(range(len(groups)))

        self._verify_and_split_kv_cache_groups()
        self.mamba_group_indices = {
            index for index, spec in enumerate(self.effective_specs) if isinstance(spec, MambaSpec)
        }
        self.partial_hash_hits = any(
            index in self.mamba_group_indices and block_size > hash_block_size
            for index, block_size in enumerate(self.effective_block_sizes)
        )

    def _verify_and_split_kv_cache_groups(self) -> None:
        spec_groups: list[tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]] = []
        self.effective_specs: list[KVCacheSpec] = []
        self.manager_classes: list[type[SingleTypeKVCacheManager] | None] = []

        for group_index, group in enumerate(self.groups):
            spec = _unwrap_spec(group.kv_cache_spec)
            self.effective_specs.append(spec)
            if not group.kv_cache_spec.prefix_cacheable:
                self.manager_classes.append(None)
                continue
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            if manager_cls is None:
                raise ValueError(f"No vLLM KV cache manager is registered for {type(spec).__name__}")
            self.manager_classes.append(manager_cls)

            for existing_spec, group_indices, existing_cls in spec_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, "Expected same manager class for identical KV cache specs."
                    group_indices.append(group_index)
                    break
            else:
                spec_groups.append((spec, [group_index], manager_cls))

        self.spec_groups = sorted(
            spec_groups,
            key=lambda item: not isinstance(item[0], FullAttentionSpec),
        )
        self.eagle_spec_group_indices: set[int] = {
            index
            for index, (_, group_indices, _) in enumerate(self.spec_groups)
            if any(group_index in self.eagle_group_indices for group_index in group_indices)
        }
        if self.use_eagle and not self.eagle_spec_group_indices:
            self.eagle_spec_group_indices = set(range(len(self.spec_groups)))
        self.eagle_reachable_group_indices: set[int] = {
            group_index
            for spec_group_index in self.eagle_spec_group_indices
            for group_index in self.spec_groups[spec_group_index][1]
        }

    def select_for_lookup(self, block_hashes: BlockHashes, query_range: TokenRange) -> KVSelection:
        aligned_token_len = cdiv(min(query_range.end_token, self.max_model_len), self.lcm_block_size)
        aligned_token_len *= self.lcm_block_size
        lookup_masks = self.lookup_mask(aligned_token_len)
        chunk_selections = tuple(
            GroupSelection(group_id, None if mask is None else tuple(mask))
            for group_id, mask in zip(self.group_ids, lookup_masks, strict=True)
        )
        return KVSelection(query_range, tuple(block_hashes), chunk_selections)

    def resolve_available_end(
        self, selection: KVSelection, availability: Sequence[GroupAvailability]
    ) -> ReachablePrefix:
        observations_by_group = {observation.group_id: observation for observation in availability}
        if set(observations_by_group) != set(self.group_ids):
            raise ValueError(f"Lookup observations do not match configured groups {self.group_ids}")

        query_range = selection.token_range
        block_hashes = selection.block_hashes
        max_hit_length = min(query_range.end_token, self.max_model_len)
        block_hashes_to_check = block_hashes[: max_hit_length // self.hash_block_size]
        cached_hashes: set[tuple[int, bytes]] = set()
        for group_index, (group_id, group_block_size) in enumerate(
            zip(self.group_ids, self.effective_block_sizes, strict=True)
        ):
            group_block_hashes = (
                block_hashes_to_check
                if self.partial_hash_hits
                else get_block_hashes(block_hashes_to_check, group_block_size, self.hash_block_size)
            )
            local_hit_count = query_range.start_token // (
                self.hash_block_size if self.partial_hash_hits else group_block_size
            )
            cached_hashes.update(
                (group_index, block_hash_to_bytes(block_hash)) for block_hash in group_block_hashes[:local_hit_count]
            )
            observation = observations_by_group[group_id]
            cached_hashes.update(
                (group_index, block_hash_to_bytes(chunk.content_hash))
                for chunk in observation.chunks
                if chunk.available
            )

        if not cached_hashes:
            return ReachablePrefix(0)
        cached_block_pool = ExternalCachedBlockPool(self.hash_block_size, cached_hashes)
        _, hit_length = self.find_longest_cache_hit(
            block_hashes,
            max_hit_length,
            cached_block_pool,
        )
        return ReachablePrefix(
            hit_length,
            self._tail_key_boundaries(block_hashes, hit_length, cached_block_pool),
        )

    def _tail_key_boundaries(
        self,
        block_hashes: BlockHashes,
        hit_length: int,
        cached_block_pool: ExternalCachedBlockPool,
    ) -> tuple[TailKeyBoundary, ...]:
        if not self.partial_hash_hits or hit_length <= 0:
            return ()

        hit_hash_index = hit_length // self.hash_block_size - 1
        boundaries = []
        for group_index, (group_id, spec, block_size) in enumerate(
            zip(self.group_ids, self.effective_specs, self.effective_block_sizes, strict=True)
        ):
            if not spec.prefix_cacheable:
                continue
            boundary_token = hit_length
            if not cached_block_pool.contains(group_index, block_hashes[hit_hash_index]):
                next_block_hash_index = min(
                    cdiv(hit_length, block_size) * block_size // self.hash_block_size,
                    len(block_hashes),
                )
                for hash_index in range(hit_hash_index + 1, next_block_hash_index):
                    if cached_block_pool.contains(group_index, block_hashes[hash_index]):
                        boundary_token = (hash_index + 1) * self.hash_block_size
                        break
                else:
                    raise AssertionError(f"No remote tail key found for cache group {group_id} at {hit_length}")
            boundaries.append(TailKeyBoundary(group_id, boundary_token))
        return tuple(boundaries)

    def select_for_load(self, block_hashes: BlockHashes, load_range: TokenRange) -> KVSelection:
        selections = self._group_selections(self.load_mask(block_hashes, load_range.end_token))
        return KVSelection(load_range, tuple(block_hashes), selections)

    def select_for_store(
        self, block_hashes: BlockHashes, store_range: TokenRange, num_prompt_tokens: int
    ) -> KVSelection:
        if store_range.end_token % self.lcm_block_size == 0:
            selections = self._group_selections(self.store_mask(store_range.end_token, num_prompt_tokens))
        else:
            logger.debug("Use unfiltered Store chunks for unaligned end token %d", store_range.end_token)
            selections = tuple(
                GroupSelection(group_id, () if group_index in self.mamba_group_indices else None)
                for group_index, group_id in enumerate(self.group_ids)
            )
        return KVSelection(store_range, tuple(block_hashes), selections)

    def _group_selections(self, masks: Sequence[Sequence[bool] | None]) -> tuple[GroupSelection, ...]:
        return tuple(
            GroupSelection(group_id, None if mask is None else tuple(mask))
            for group_id, mask in zip(self.group_ids, masks, strict=True)
        )

    def find_longest_cache_hit(
        self,
        block_hashes: BlockHashes,
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
        *,
        apply_eagle: bool = True,
    ) -> tuple[tuple[list[bool], ...], int]:
        blocks_by_group_index, hit_length = self._find_hit_blocks(
            block_hashes,
            max_length,
            cached_block_pool,
            apply_eagle=apply_eagle,
        )
        masks = tuple(
            [block is not cached_block_pool.null_block for block in blocks] for blocks in blocks_by_group_index
        )
        return masks, hit_length

    def load_mask(self, block_hashes: BlockHashes, load_end_token: int) -> tuple[list[bool], ...]:
        masks, _ = self.find_longest_cache_hit(
            block_hashes,
            load_end_token,
            ExternalCachedBlockPool(self.hash_block_size),
            apply_eagle=False,
        )
        return masks

    def _reachable_masks(
        self,
        aligned_token_len: int,
        retention_interval: int | None,
        num_prompt_tokens: int | None,
    ) -> list[tuple[int, list[bool] | None]]:
        assert aligned_token_len % self.lcm_block_size == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of lcm_block_size ({self.lcm_block_size})"
        )
        masks: list[tuple[int, list[bool] | None]] = []
        for group_index, (spec, manager_cls) in enumerate(zip(self.effective_specs, self.manager_classes, strict=True)):
            num_chunks = aligned_token_len // self.effective_block_sizes[group_index]
            group = self.groups[group_index]
            if not group.kv_cache_spec.prefix_cacheable:
                masks.append((num_chunks, [False] * num_chunks))
                continue
            if isinstance(spec, MambaSpec) and num_prompt_tokens is not None:
                masks.append((num_chunks, [False] * num_chunks))
                continue
            assert manager_cls is not None
            mask = manager_cls.reachable_block_mask(
                start_block=0,
                end_block=num_chunks,
                alignment_tokens=self.lcm_block_size,
                kv_cache_spec=spec,
                use_eagle=group_index in self.eagle_reachable_group_indices,
                retention_interval=retention_interval,
                reachable_boundaries=() if num_prompt_tokens is None else (num_prompt_tokens - 1,),
                dcp_world_size=1,
            )
            masks.append((num_chunks, mask))
        return masks

    def store_mask(self, aligned_token_len: int, num_prompt_tokens: int | None = None) -> tuple[list[bool], ...]:
        masks = self._reachable_masks(aligned_token_len, self.retention_interval, num_prompt_tokens)
        return tuple([True] * num_chunks if mask is None else mask for num_chunks, mask in masks)

    def lookup_mask(self, aligned_token_len: int) -> tuple[list[bool] | None, ...]:
        masks = self._reachable_masks(aligned_token_len, self.retention_interval, None)
        for num_chunks, mask in masks:
            if mask is not None:
                assert len(mask) == num_chunks
        return tuple(None if mask is None or all(mask) else mask for _, mask in masks)

    def _find_hit_blocks(
        self,
        block_hashes: BlockHashes,
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
        *,
        apply_eagle: bool = True,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        eagle_spec_group_indices = self.eagle_spec_group_indices if apply_eagle else set()
        alignment_tokens = self.hash_block_size if self.partial_hash_hits else self.lcm_block_size
        if not self.spec_groups:
            return tuple([] for _ in self.groups), 0
        if len(self.spec_groups) == 1:
            spec, group_indices, manager_cls = self.spec_groups[0]
            hit_blocks, hit_length = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_length,
                kv_cache_group_ids=group_indices,
                block_pool=cast(BlockPool, cached_block_pool),
                kv_cache_spec=spec,
                drop_eagle_block=0 in eagle_spec_group_indices,
                alignment_tokens=alignment_tokens,
            )
            blocks_by_group_index: list[list[KVCacheBlock]] = [[] for _ in range(len(self.groups))]
            for group_index, blocks in zip(group_indices, hit_blocks, strict=True):
                blocks_by_group_index[group_index] = blocks
            return tuple(blocks_by_group_index), hit_length

        hit_length = max_length
        hit_blocks_by_group_index: list[list[KVCacheBlock] | None] = [None] * len(self.groups)
        hit_lengths_by_group_index: list[int] = [0] * len(self.groups)
        is_simple_hybrid = len(self.spec_groups) == 2 and isinstance(self.spec_groups[0][0], FullAttentionSpec)
        verified_eagle_spec_groups: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for spec_group_index, (spec, group_indices, manager_cls) in enumerate(self.spec_groups):
                first_group_index = group_indices[0]
                cached = hit_blocks_by_group_index[first_group_index]
                if isinstance(spec, FullAttentionSpec) and cached is not None:
                    curr_hit_length = min(curr_hit_length, hit_lengths_by_group_index[first_group_index])
                    continue

                is_unverified_eagle_group = spec_group_index not in verified_eagle_spec_groups
                drop_eagle_block = spec_group_index in eagle_spec_group_indices and is_unverified_eagle_group
                max_spec_group_length = curr_hit_length
                if drop_eagle_block and not isinstance(spec, MambaSpec):
                    eagle_margin = (
                        self.hash_block_size
                        if self.partial_hash_hits
                        and getattr(manager_cls, "supports_fine_grained_hash_lookup", False)
                        and spec.block_size > self.hash_block_size
                        else spec.block_size
                    )
                    max_spec_group_length = min(curr_hit_length + eagle_margin, max_length)
                hit_blocks, new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=max_spec_group_length,
                    kv_cache_group_ids=group_indices,
                    block_pool=cast(BlockPool, cached_block_pool),
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=alignment_tokens,
                )
                if drop_eagle_block:
                    verified_eagle_spec_groups.add(spec_group_index)
                elif new_hit_length < curr_hit_length:
                    verified_eagle_spec_groups.clear()
                curr_hit_length = new_hit_length
                for group_index, blocks in zip(group_indices, hit_blocks, strict=True):
                    hit_blocks_by_group_index[group_index] = blocks
                    hit_lengths_by_group_index[group_index] = new_hit_length

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        for spec, group_indices, _ in self.spec_groups:
            if not isinstance(spec, FullAttentionSpec):
                continue
            num_blocks = cdiv(hit_length, spec.block_size)
            for group_index in group_indices:
                full_blocks = hit_blocks_by_group_index[group_index]
                assert full_blocks is not None
                del full_blocks[num_blocks:]
                hit_lengths_by_group_index[group_index] = hit_length

        return (
            tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group_index),
            hit_length,
        )


def _unwrap_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec
