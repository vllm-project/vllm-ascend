from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from typing import Any, cast

from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    _block_hash_to_bytes,
    get_block_hashes,
)
from vllm_ascend.utils import vllm_version_is

_CACHE_MISSING = object()
_MANAGER_CLASS_CACHE_ATTR = "_manager_class_cache"


class ExternalCachedBlockPool:
    """Duck-typed BlockPool backed by external AscendStore key existence."""

    def __init__(
        self,
        hash_block_size: int,
        exists: set[tuple[int, bytes]] | None = None,
    ) -> None:
        # exists=None is used for load/store masks where hit length has already
        # been decided and each manager only needs to apply its own reachability.
        self._exists = exists
        self.hash_block_size = hash_block_size
        self.null_block = KVCacheBlock(block_id=0)
        self._present_block = KVCacheBlock(block_id=1)

    def get_cached_block(
        self,
        block_hash: BlockHash,
        group_ids: list[int],
    ) -> list[KVCacheBlock] | None:
        if self._exists is None:
            return [self._present_block] * len(group_ids)
        h = _block_hash_to_bytes(block_hash)
        if all((group_id, h) in self._exists for group_id in group_ids):
            return [self._present_block] * len(group_ids)
        return None


class AscendStoreCoordinator:
    """Hybrid cache-hit/mask coordinator for AscendStore external KV Pool.

    This mirrors vLLM MooncakeStoreCoordinator but uses AscendStore's external
    key granularity. For DSV4 compressed groups, keys are generated over the
    raw-token span ``group_block_size * compress_ratio`` while transfer
    addresses remain in cache-domain blocks.
    """

    def __init__(
        self,
        kv_cache_groups: list[KVCacheGroupSpec],
        scheduler_block_size: int,
        hash_block_size: int,
        group_block_sizes: list[int],
        group_cache_families: list[str],
        use_eagle: bool = False,
        retention_interval: int | None = None,
    ) -> None:
        assert len(kv_cache_groups) == len(group_block_sizes)
        assert len(kv_cache_groups) == len(group_cache_families)
        assert scheduler_block_size % hash_block_size == 0, (
            f"scheduler_block_size ({scheduler_block_size}) must be a multiple of hash_block_size ({hash_block_size})"
        )

        self.kv_cache_groups = kv_cache_groups
        self.hash_block_size = hash_block_size
        self.lcm_block_size = scheduler_block_size
        self.use_eagle = use_eagle
        self.retention_interval = retention_interval
        self.group_block_sizes = group_block_sizes
        self.group_cache_families = group_cache_families
        self.group_effective_block_sizes = [
            _cache_family_granularity(block_size, family)
            for block_size, family in zip(group_block_sizes, group_cache_families, strict=True)
        ]
        for effective_block_size in self.group_effective_block_sizes:
            assert effective_block_size % hash_block_size == 0, "block_size must be divisible by hash_block_size"
            assert scheduler_block_size % effective_block_size == 0, (
                "scheduler_block_size must be a multiple of each group's effective block_size"
            )

        self.eagle_group_ids = {i for i, group in enumerate(kv_cache_groups) if group.is_eagle_group}
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_groups)))

        self._verify_and_split_kv_cache_groups()

    def _verify_and_split_kv_cache_groups(self) -> None:
        attention_groups: list[tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]] = []
        self.group_effective_specs: list[KVCacheSpec] = []

        for group_id, group in enumerate(self.kv_cache_groups):
            spec = _unwrap_spec(group.kv_cache_spec)
            effective_spec = _copy_spec_with_block_size(spec, self.group_effective_block_sizes[group_id])
            if (
                not _uses_reachable_mask(self.group_cache_families[group_id])
                and getattr(effective_spec, "compress_ratio", 1) > 1
            ):
                # The cache family already folds the compression ratio into
                # the external key granularity. Avoid applying it again inside
                # CompressAttentionManager.find_longest_cache_hit().
                effective_spec = replace(effective_spec, compress_ratio=1)
            self.group_effective_specs.append(effective_spec)
            manager_cls = _get_manager_class(spec)

            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == effective_spec:
                    assert manager_cls is existing_cls, "Expected same manager class for identical KV cache specs."
                    group_ids.append(group_id)
                    break
            else:
                attention_groups.append((effective_spec, [group_id], manager_cls))

        self.attention_groups = sorted(
            attention_groups,
            key=lambda item: not isinstance(item[0], FullAttentionSpec),
        )
        self.eagle_attn_group_indices: set[int] = {
            index
            for index, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(group_id in self.eagle_group_ids for group_id in group_ids)
        }
        if self.use_eagle and not self.eagle_attn_group_indices:
            self.eagle_attn_group_indices = set(range(len(self.attention_groups)))
        self.eagle_reachable_group_ids: set[int] = {
            group_id for index in self.eagle_attn_group_indices for group_id in self.attention_groups[index][1]
        }

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
        *,
        apply_eagle: bool = True,
    ) -> tuple[tuple[list[bool], ...], int]:
        blocks_per_group, hit_length = self._find_hit_blocks(
            block_hashes,
            max_length,
            cached_block_pool,
            apply_eagle=apply_eagle,
        )
        masks = tuple([block is not cached_block_pool.null_block for block in blocks] for blocks in blocks_per_group)
        return masks, hit_length

    def load_mask(
        self,
        block_hashes: list[BlockHash],
        token_len: int,
    ) -> tuple[list[bool], ...]:
        masks, _ = self.find_longest_cache_hit(
            block_hashes,
            token_len,
            ExternalCachedBlockPool(self.hash_block_size),
            apply_eagle=False,
        )
        return tuple(
            [True] * _num_chunks(token_len, self.group_effective_block_sizes[group_id])
            if not _uses_reachable_mask(self.group_cache_families[group_id])
            else mask
            for group_id, mask in enumerate(masks)
        )

    def store_mask(
        self,
        aligned_token_len: int,
        num_prompt_tokens: int | None = None,
    ) -> tuple[list[bool], ...]:
        assert aligned_token_len % self.lcm_block_size == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of lcm_block_size ({self.lcm_block_size})"
        )
        masks: list[list[bool]] = []
        for group_id, spec in enumerate(self.group_effective_specs):
            num_chunks = aligned_token_len // self.group_effective_block_sizes[group_id]
            if not _uses_reachable_mask(self.group_cache_families[group_id]):
                masks.append([True] * num_chunks)
                continue
            manager_cls = _get_manager_class(_unwrap_spec(self.kv_cache_groups[group_id].kv_cache_spec))
            mask = _reachable_block_mask(
                manager_cls,
                start_block=0,
                end_block=num_chunks,
                alignment_tokens=self.lcm_block_size,
                kv_cache_spec=spec,
                use_eagle=group_id in self.eagle_reachable_group_ids,
                retention_interval=self.retention_interval,
                num_prompt_tokens=num_prompt_tokens,
            )
            masks.append([True] * num_chunks if mask is None else mask)
        return tuple(masks)

    def block_hashes_for_spec(self, block_hashes: list[BlockHash], spec: KVCacheSpec) -> BlockHashList:
        if not vllm_version_is("0.25.1"):
            # vLLM #46384 moved hash-size resolution into each manager.
            return block_hashes
        if spec.block_size == self.hash_block_size:
            return block_hashes
        return cast(BlockHashList, get_block_hashes(block_hashes, spec.block_size, self.hash_block_size))

    def _find_hit_blocks(
        self,
        block_hashes: list[BlockHash],
        max_length: int,
        cached_block_pool: ExternalCachedBlockPool,
        *,
        apply_eagle: bool = True,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        eagle_indices = self.eagle_attn_group_indices if apply_eagle else set()
        if len(self.attention_groups) == 1:
            spec, group_ids, manager_cls = self.attention_groups[0]
            hashes = self.block_hashes_for_spec(block_hashes, spec)
            hit_blocks, hit_length = _find_longest_cache_hit(
                manager_cls,
                block_hashes=hashes,
                max_length=max_length,
                kv_cache_group_ids=group_ids,
                block_pool=cast(BlockPool, cached_block_pool),
                kv_cache_spec=spec,
                drop_eagle_block=0 in eagle_indices,
                alignment_tokens=spec.block_size,
            )
            blocks_by_group: list[list[KVCacheBlock]] = [[] for _ in range(len(self.kv_cache_groups))]
            for group_id, blocks in zip(group_ids, hit_blocks, strict=True):
                blocks_by_group[group_id] = blocks
            return tuple(blocks_by_group), hit_length

        hit_length = max_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * len(self.kv_cache_groups)
        hit_length_by_group: list[int] = [0] * len(self.kv_cache_groups)
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for index, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                first_group_id = group_ids[0]
                cached = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached is not None:
                    curr_hit_length = min(curr_hit_length, hit_length_by_group[first_group_id])
                    continue

                drop_eagle_block = index in eagle_indices and index not in eagle_verified
                max_group_length = curr_hit_length
                if drop_eagle_block:
                    max_group_length = min(curr_hit_length + spec.block_size, max_length)
                hashes = self.block_hashes_for_spec(block_hashes, spec)
                hit_blocks, new_hit_length = _find_longest_cache_hit(
                    manager_cls,
                    block_hashes=hashes,
                    max_length=max_group_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=cast(BlockPool, cached_block_pool),
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self.lcm_block_size,
                )
                if drop_eagle_block:
                    eagle_verified.add(index)
                elif new_hit_length < curr_hit_length:
                    eagle_verified.clear()
                curr_hit_length = new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks, strict=True):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = new_hit_length

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        spec0, group_ids0, _ = self.attention_groups[0]
        if isinstance(spec0, FullAttentionSpec):
            num_blocks = cdiv(hit_length, spec0.block_size)
            for group_id in group_ids0:
                full_blocks = hit_blocks_by_group[group_id]
                assert full_blocks is not None
                del full_blocks[num_blocks:]
                hit_length_by_group[group_id] = hit_length

        return (
            tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group),
            hit_length,
        )


def _unwrap_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


def _copy_spec_with_block_size(spec: KVCacheSpec, block_size: int) -> KVCacheSpec:
    if spec.block_size == block_size:
        return spec
    copy_with_new_block_size = getattr(spec, "copy_with_new_block_size", None)
    if copy_with_new_block_size is not None:
        return copy_with_new_block_size(block_size)
    return replace(spec, block_size=block_size)


def _get_manager_class_cache() -> dict[str, Any]:
    cache = getattr(_get_manager_class, _MANAGER_CLASS_CACHE_ATTR, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(_get_manager_class, _MANAGER_CLASS_CACHE_ATTR, cache)
    return cast(dict[str, Any], cache)


def _get_manager_class(spec: KVCacheSpec) -> type[SingleTypeKVCacheManager]:
    cache = _get_manager_class_cache()
    compress_ratio = getattr(spec, "compress_ratio", None)
    if compress_ratio is not None and compress_ratio > 1:
        compress_manager = cache.get("compress_manager", _CACHE_MISSING)
        if compress_manager is _CACHE_MISSING:
            try:
                from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager
            except ImportError:
                compress_manager = None
            else:
                compress_manager = CompressAttentionManager
            cache["compress_manager"] = compress_manager
        if compress_manager is not None:
            return cast(type[SingleTypeKVCacheManager], compress_manager)

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


def _find_longest_cache_hit(
    manager_cls: type[SingleTypeKVCacheManager],
    **kwargs: Any,
) -> tuple[tuple[list[KVCacheBlock], ...], int]:
    hit_result = manager_cls.find_longest_cache_hit(**kwargs)
    if vllm_version_is("0.25.1"):
        hit_blocks = cast(tuple[list[KVCacheBlock], ...], hit_result)
        hit_length = len(hit_blocks[0]) * kwargs["kv_cache_spec"].block_size
        return hit_blocks, hit_length
    return cast(tuple[tuple[list[KVCacheBlock], ...], int], hit_result)


def _reachable_block_mask(
    manager_cls: type[SingleTypeKVCacheManager],
    **kwargs: Any,
) -> list[bool] | None:
    reachable_block_mask = getattr(manager_cls, "reachable_block_mask", None)
    if reachable_block_mask is None:
        return None
    try:
        return reachable_block_mask(**kwargs)
    except TypeError as exc:
        if "retention_interval" not in str(exc) and "num_prompt_tokens" not in str(exc):
            logger.debug("KV cache manager does not support reachable_block_mask kwargs: %s", exc)
            return reachable_block_mask(
                start_block=kwargs["start_block"],
                end_block=kwargs["end_block"],
                alignment_tokens=kwargs["alignment_tokens"],
                kv_cache_spec=kwargs["kv_cache_spec"],
                use_eagle=kwargs["use_eagle"],
            )
        kwargs.pop("retention_interval", None)
        kwargs.pop("num_prompt_tokens", None)
        return reachable_block_mask(**kwargs)


def _cache_family_granularity(block_size: int, cache_family: str | None) -> int:
    if not cache_family or not cache_family.startswith("c"):
        return block_size
    ratio = cache_family[1:]
    return block_size * int(ratio) if ratio.isdigit() else block_size


def _uses_reachable_mask(cache_family: str | None) -> bool:
    return cache_family in (None, "default", "c1")


def _num_chunks(token_len: int, block_size: int) -> int:
    return (token_len + block_size - 1) // block_size
