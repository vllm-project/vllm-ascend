from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    block_hash_to_bytes,
    get_block_hashes,
)

_CACHE_MISSING = object()
_MANAGER_CLASS_CACHE_ATTR = "_manager_class_cache"

# kwargs every reachable_block_mask implementation must accept. Used when the
# manager's signature cannot be introspected.
_REACHABLE_MASK_BASE_KWARGS = frozenset(("start_block", "end_block", "alignment_tokens", "kv_cache_spec", "use_eagle"))
_REACHABLE_MASK_OPTIONAL_KWARGS = frozenset(("retention_interval", "reachable_boundaries", "dcp_world_size"))
# manager class -> accepted reachable_block_mask parameter names.
_REACHABLE_MASK_KWARGS_CACHE: dict[type[SingleTypeKVCacheManager], frozenset[str]] = {}

ChunkMask = tuple[bool, ...] | None
BlockHashes = Sequence[BlockHash | str]


@dataclass(frozen=True, slots=True)
class ChunkSelection:
    """Logical chunks selected for one original vLLM cache group."""

    group_id: int
    chunk_mask: ChunkMask

    def includes(self, start_token: int, block_size: int) -> bool:
        chunk_index = start_token // block_size
        return self.chunk_mask is None or (chunk_index < len(self.chunk_mask) and self.chunk_mask[chunk_index])


@dataclass(frozen=True, slots=True)
class LookupChunkSelection(ChunkSelection):
    """One group's Lookup selection after the local cache prefix."""

    query_start_token: int


@dataclass(frozen=True, slots=True)
class LookupObservation:
    """Per-chunk Backend observations produced by one Lookup task."""

    group_id: int
    chunk_ends: tuple[int, ...]
    chunk_hashes: tuple[BlockHash | str, ...]
    chunk_presence: tuple[bool, ...]


class KVTransferCoordinator(Protocol):
    """Select logical KV chunks without owning task construction or I/O."""

    group_ids: tuple[int, ...]

    def select_lookup(self, lookup_end_token: int, local_cached_tokens: int) -> tuple[LookupChunkSelection, ...]: ...

    def resolve_lookup(
        self,
        block_hashes: BlockHashes,
        lookup_end_token: int,
        local_cached_tokens: int,
        observations: Sequence[LookupObservation],
    ) -> int: ...

    def select_load(self, block_hashes: BlockHashes, load_end_token: int) -> tuple[ChunkSelection, ...]: ...

    def select_store(self, store_end_token: int, num_prompt_tokens: int) -> tuple[ChunkSelection, ...]: ...


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


class UnitaryKVTransferCoordinator:
    """Coordinate one transferable cache group without cross-group convergence."""

    def __init__(
        self,
        group_id: int,
        group_block_size: int,
        max_model_len: int,
        cache_transfer_granularity: int,
    ) -> None:
        self.group_ids = (group_id,)
        self._group_block_size = group_block_size
        self._max_model_len = max_model_len
        self._cache_transfer_granularity = cache_transfer_granularity

    def select_lookup(self, lookup_end_token: int, local_cached_tokens: int) -> tuple[LookupChunkSelection, ...]:
        query_start_token = local_cached_tokens // self._group_block_size * self._group_block_size
        return (LookupChunkSelection(self.group_ids[0], None, query_start_token),)

    def resolve_lookup(
        self,
        block_hashes: BlockHashes,
        lookup_end_token: int,
        local_cached_tokens: int,
        observations: Sequence[LookupObservation],
    ) -> int:
        if len(observations) != 1 or observations[0].group_id != self.group_ids[0]:
            raise ValueError(f"Expected one Lookup observation for group {self.group_ids[0]}")

        max_hit_length = min(lookup_end_token, self._max_model_len)
        hit_end = min(local_cached_tokens, max_hit_length)
        hit_end -= hit_end % self._cache_transfer_granularity
        observation = observations[0]
        for end, is_present in zip(observation.chunk_ends, observation.chunk_presence, strict=True):
            if end > max_hit_length or not is_present:
                break
            if end % self._cache_transfer_granularity == 0:
                hit_end = end
        return hit_end

    def select_load(self, block_hashes: BlockHashes, load_end_token: int) -> tuple[ChunkSelection, ...]:
        return (ChunkSelection(self.group_ids[0], None),)

    def select_store(self, store_end_token: int, num_prompt_tokens: int) -> tuple[ChunkSelection, ...]:
        return (ChunkSelection(self.group_ids[0], None),)


class HybridKVTransferCoordinator:
    """Coordinate cache reachability across heterogeneous transfer groups.

    This mirrors vLLM's external KV coordinator but uses AscendStore's external
    key granularity. Compressed specs already expose raw-token block sizes,
    while transfer addresses remain in cache-domain blocks.
    """

    def __init__(
        self,
        group_ids: tuple[int, ...],
        kv_cache_groups: list[KVCacheGroupSpec],
        scheduler_block_size: int,
        hash_block_size: int,
        max_model_len: int,
        use_eagle: bool = False,
        retention_interval: int | None = None,
    ) -> None:
        assert len(group_ids) == len(kv_cache_groups)
        assert scheduler_block_size % hash_block_size == 0, (
            f"scheduler_block_size ({scheduler_block_size}) must be a multiple of hash_block_size ({hash_block_size})"
        )

        self.group_ids = group_ids
        self.kv_cache_groups = kv_cache_groups
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        self.max_model_len = max_model_len
        self.use_eagle = use_eagle
        self.retention_interval = retention_interval
        self.effective_block_sizes = [group.kv_cache_spec.block_size for group in kv_cache_groups]
        for effective_block_size in self.effective_block_sizes:
            assert effective_block_size % hash_block_size == 0, "block_size must be divisible by hash_block_size"
            assert scheduler_block_size % effective_block_size == 0, (
                "scheduler_block_size must be a multiple of each group's effective block_size"
            )

        self.eagle_group_indices = {index for index, group in enumerate(kv_cache_groups) if group.is_eagle_group}
        if use_eagle and not self.eagle_group_indices:
            self.eagle_group_indices = set(range(len(kv_cache_groups)))

        self._verify_and_split_kv_cache_groups()

    def _verify_and_split_kv_cache_groups(self) -> None:
        spec_groups: list[tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]] = []
        self.effective_specs: list[KVCacheSpec] = []

        for group_index, group in enumerate(self.kv_cache_groups):
            spec = _unwrap_spec(group.kv_cache_spec)
            self.effective_specs.append(spec)
            if not group.kv_cache_spec.prefix_cacheable:
                continue
            manager_cls = _get_manager_class(spec)

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

    def select_lookup(self, lookup_end_token: int, local_cached_tokens: int) -> tuple[LookupChunkSelection, ...]:
        aligned_token_len = cdiv(min(lookup_end_token, self.max_model_len), self.lcm_block_size) * self.lcm_block_size
        lookup_masks = self.lookup_mask(aligned_token_len)
        return tuple(
            LookupChunkSelection(
                group_id,
                None if mask is None else tuple(mask),
                local_cached_tokens // group_block_size * group_block_size,
            )
            for group_id, group_block_size, mask in zip(
                self.group_ids,
                self.effective_block_sizes,
                lookup_masks,
                strict=True,
            )
        )

    def resolve_lookup(
        self,
        block_hashes: BlockHashes,
        lookup_end_token: int,
        local_cached_tokens: int,
        observations: Sequence[LookupObservation],
    ) -> int:
        observations_by_group = {observation.group_id: observation for observation in observations}
        if set(observations_by_group) != set(self.group_ids):
            raise ValueError(f"Lookup observations do not match configured groups {self.group_ids}")

        max_hit_length = min(lookup_end_token, self.max_model_len)
        block_hashes_to_check = block_hashes[: max_hit_length // self.hash_block_size]
        cached_hashes: set[tuple[int, bytes]] = set()
        for group_index, (group_id, group_block_size) in enumerate(
            zip(self.group_ids, self.effective_block_sizes, strict=True)
        ):
            group_block_hashes = get_block_hashes(block_hashes_to_check, group_block_size, self.hash_block_size)
            local_hit_count = local_cached_tokens // group_block_size
            cached_hashes.update(
                (group_index, block_hash_to_bytes(block_hash)) for block_hash in group_block_hashes[:local_hit_count]
            )
            observation = observations_by_group[group_id]
            cached_hashes.update(
                (group_index, block_hash_to_bytes(block_hash))
                for block_hash, is_present in zip(
                    observation.chunk_hashes,
                    observation.chunk_presence,
                    strict=True,
                )
                if is_present
            )

        if not cached_hashes:
            return 0
        _, hit_length = self.find_longest_cache_hit(
            block_hashes,
            max_hit_length,
            ExternalCachedBlockPool(self.hash_block_size, cached_hashes),
            apply_eagle=False,
        )
        return hit_length

    def select_load(self, block_hashes: BlockHashes, load_end_token: int) -> tuple[ChunkSelection, ...]:
        return self._group_selections(self.load_mask(block_hashes, load_end_token))

    def select_store(self, store_end_token: int, num_prompt_tokens: int) -> tuple[ChunkSelection, ...]:
        return self._group_selections(self.store_mask(store_end_token, num_prompt_tokens))

    def _group_selections(self, masks: Sequence[Sequence[bool] | None]) -> tuple[ChunkSelection, ...]:
        return tuple(
            ChunkSelection(group_id, None if mask is None else tuple(mask))
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
        for group_index, spec in enumerate(self.effective_specs):
            num_chunks = aligned_token_len // self.effective_block_sizes[group_index]
            group = self.kv_cache_groups[group_index]
            if not group.kv_cache_spec.prefix_cacheable:
                masks.append((num_chunks, [False] * num_chunks))
                continue
            if isinstance(spec, MambaSpec) and num_prompt_tokens is not None:
                masks.append((num_chunks, [False] * num_chunks))
                continue
            manager_cls = _get_manager_class(spec)
            mask = _reachable_block_mask(
                manager_cls,
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
        masks = self._reachable_masks(aligned_token_len, None, None)
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
        if not self.spec_groups:
            return tuple([] for _ in self.kv_cache_groups), 0
        if len(self.spec_groups) == 1:
            spec, group_indices, manager_cls = self.spec_groups[0]
            hit_blocks, hit_length = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_length,
                kv_cache_group_ids=group_indices,
                block_pool=cast(BlockPool, cached_block_pool),
                kv_cache_spec=spec,
                drop_eagle_block=0 in eagle_spec_group_indices,
                alignment_tokens=spec.block_size,
            )
            blocks_by_group_index: list[list[KVCacheBlock]] = [[] for _ in range(len(self.kv_cache_groups))]
            for group_index, blocks in zip(group_indices, hit_blocks, strict=True):
                blocks_by_group_index[group_index] = blocks
            return tuple(blocks_by_group_index), hit_length

        hit_length = max_length
        hit_blocks_by_group_index: list[list[KVCacheBlock] | None] = [None] * len(self.kv_cache_groups)
        hit_lengths_by_group_index: list[int] = [0] * len(self.kv_cache_groups)
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
                    max_spec_group_length = min(curr_hit_length + spec.block_size, max_length)
                hit_blocks, new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=max_spec_group_length,
                    kv_cache_group_ids=group_indices,
                    block_pool=cast(BlockPool, cached_block_pool),
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self.lcm_block_size,
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


def _get_manager_class_cache() -> dict[str, Any]:
    cache = getattr(_get_manager_class, _MANAGER_CLASS_CACHE_ATTR, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(_get_manager_class, _MANAGER_CLASS_CACHE_ATTR, cache)
    return cast(dict[str, Any], cache)


def _get_manager_class(spec: KVCacheSpec) -> type[SingleTypeKVCacheManager]:
    cache = _get_manager_class_cache()
    registry = cache.get("registry", _CACHE_MISSING)
    if registry is _CACHE_MISSING:
        try:
            registry_module = import_module("vllm.v1.kv_cache_spec_registry")
            registry = getattr(registry_module, "KVCacheSpecRegistry", None)
        except ImportError:
            registry = None
        cache["registry"] = registry

    if registry is not None:
        manager_cls = registry.get_manager_class(spec)
        if manager_cls is not None:
            return manager_cls

    spec_manager_map = cache.get("spec_manager_map", _CACHE_MISSING)
    if spec_manager_map is _CACHE_MISSING:
        try:
            manager_module = import_module("vllm.v1.core.single_type_kv_cache_manager")
            spec_manager_map = vars(manager_module)["spec_manager_map"]
        except Exception as exc:
            raise AssertionError(f"No manager registered for KVCacheSpec {type(spec)}") from exc
        cache["spec_manager_map"] = spec_manager_map

    try:
        manager_cls = spec_manager_map[type(spec)]
    except Exception as exc:
        raise AssertionError(f"No manager registered for KVCacheSpec {type(spec)}") from exc
    return manager_cls


def _reachable_mask_accepted_kwargs(
    manager_cls: type[SingleTypeKVCacheManager], reachable_block_mask: Any
) -> frozenset[str]:
    """Parameter names a manager's ``reachable_block_mask`` accepts.

    Falls back to the base contract when introspection is unavailable or the
    implementation swallows everything via ``**kwargs``.
    """
    cached = _REACHABLE_MASK_KWARGS_CACHE.get(manager_cls, _CACHE_MISSING)
    if cached is not _CACHE_MISSING:
        return cast(frozenset[str], cached)

    accepted = _REACHABLE_MASK_BASE_KWARGS
    try:
        parameters = inspect.signature(reachable_block_mask).parameters
    except (TypeError, ValueError):
        pass
    else:
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            accepted = _REACHABLE_MASK_BASE_KWARGS | _REACHABLE_MASK_OPTIONAL_KWARGS
        else:
            accepted = frozenset(parameters)
    _REACHABLE_MASK_KWARGS_CACHE[manager_cls] = accepted
    return accepted


def _reachable_block_mask(manager_cls: type[SingleTypeKVCacheManager], **kwargs: Any) -> list[bool] | None:
    reachable_block_mask = getattr(manager_cls, "reachable_block_mask", None)
    if reachable_block_mask is None:
        return None
    # Filter by signature instead of sniffing TypeError text: the old fallback
    # keyed on a keyword no manager accepts and therefore dropped
    # retention_interval on every call, silently degrading every sparse-retention
    # mask to the dense default.
    accepted = _reachable_mask_accepted_kwargs(manager_cls, reachable_block_mask)
    unsupported = set(kwargs) - accepted
    if unsupported:
        logger.debug(
            "KV cache manager %s does not accept reachable_block_mask kwargs %s; ignoring them.",
            manager_cls.__name__,
            sorted(unsupported),
        )
    return reachable_block_mask(**{name: value for name, value in kwargs.items() if name in accepted})
