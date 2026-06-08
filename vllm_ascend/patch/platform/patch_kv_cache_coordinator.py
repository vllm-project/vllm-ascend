# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM projectx
import sys
from collections.abc import Mapping
from math import lcm

import vllm
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
)

from vllm_ascend.core.single_type_kv_cache_manager import get_manager_for_kv_cache_spec
from vllm_ascend.utils import vllm_version_is

USE_MULTI_GROUPS_KV_CACHE = True

_orig_get_kv_cache_coordinator = vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator


def _is_deepseek_v4_kv_cache_spec(kv_cache_spec: KVCacheSpec) -> bool:
    if getattr(kv_cache_spec, "model_version", None) == "deepseek_v4":
        return True

    nested_specs = getattr(kv_cache_spec, "kv_cache_specs", None)
    if nested_specs is None:
        return False

    if isinstance(nested_specs, Mapping):
        nested_specs = nested_specs.values()
    elif not isinstance(nested_specs, (list, tuple, set)):
        return False

    return any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in nested_specs)


def _is_deepseek_v4_kv_cache_config(kv_cache_config: KVCacheConfig) -> bool:
    return any(_is_deepseek_v4_kv_cache_spec(group.kv_cache_spec) for group in kv_cache_config.kv_cache_groups)


class AscendHybridKVCacheCoordinator(HybridKVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        eagle_attn_layer_names: list[str] | None = None,
        metrics_collector: KVCacheMetricsCollector | None = None,
        max_num_batched_tokens: int | None = None,
        scheduler_block_size: int | None = None,
    ):
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        self.scheduler_block_size = scheduler_block_size
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # Fall back to `max_model_len` when unset so the recycling-aware
        # admission cap (vLLM PR #40946) collapses to the prior uncapped
        # behavior. The scheduler always supplies the real value at runtime.
        if max_num_batched_tokens is None:
            max_num_batched_tokens = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens

        self.block_pool = BlockPool(
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )

        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group}
        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        # v0.21.0 managers don't accept `scheduler_block_size`.
        extra_mgr_kwargs: dict = {}
        if not vllm_version_is("0.21.0"):
            extra_mgr_kwargs["scheduler_block_size"] = scheduler_block_size
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                **extra_mgr_kwargs,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        if enable_caching:
            assert all(
                self._get_effective_block_size(g.kv_cache_spec) % hash_block_size == 0
                for g in kv_cache_config.kv_cache_groups
            ), "block_size must be divisible by hash_block_size"
        self.verify_and_split_kv_cache_groups()

        self.use_eagle = use_eagle

    def _get_effective_block_size(self, kv_cache_spec: KVCacheSpec) -> int:
        block_size = kv_cache_spec.block_size
        if isinstance(kv_cache_spec, MambaSpec) and self.enable_caching:
            return block_size
        if self.dcp_world_size * self.pcp_world_size > 1:
            block_size *= self.dcp_world_size * self.pcp_world_size
        if hasattr(kv_cache_spec, "compress_ratio"):
            compress_ratio = kv_cache_spec.compress_ratio or 1
            compress_ratio = compress_ratio if compress_ratio >= 1 else 1
            block_size *= compress_ratio
        return block_size

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        attention_groups: list[tuple[KVCacheSpec, list[int], type[SingleTypeKVCacheManager]]] = []

        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec

            # Try to find an existing group with the same spec
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    assert manager_cls is existing_cls, "Expected same manager class for identical KV cache specs."
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        assert len(attention_groups) > 1, "HybridKVCacheCoordinator requires at least two attention groups."

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups = sorted(
            attention_groups,
            key=lambda x: not isinstance(x[0], FullAttentionSpec),
        )

        # Attention-group indices (into ``self.attention_groups``) that
        # contain at least one EAGLE/MTP KV cache group.
        self.eagle_attn_group_indices: set[int] = {
            i
            for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(gid in self.eagle_group_ids for gid in group_ids)
        }

        # The LCM of the block sizes of all attention types.
        # The cache hit length must be a multiple of the LCM of the block sizes
        # to make sure the cache hit length is a multiple of the block size of
        # each attention type. Requiring this because we don't support partial
        # block cache hit yet.
        # NOTE: use 16k as the alignment tokens for model with compress ratio
        block_sizes = [self._get_effective_block_size(spec) for spec, _, _ in self.attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A tuple of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """

        def _get_block_hashes(kv_cache_spec: KVCacheSpec) -> BlockHashList:
            target_block_size = kv_cache_spec.block_size
            if not isinstance(kv_cache_spec, MambaSpec) and self.dcp_world_size * self.pcp_world_size > 1:
                target_block_size *= self.dcp_world_size * self.pcp_world_size
            if target_block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(block_hashes, self.hash_block_size, target_block_size)

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0][0], FullAttentionSpec
        )

        # Attention-group indices whose EAGLE drop is verified at the current
        # ``curr_hit_length``. Each eagle group applies the drop at most once
        # per candidate length (see issue #32802).
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length
            for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
                effective_block_size = self._get_effective_block_size(spec)
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    num_blocks = curr_hit_length // effective_block_size
                    curr_hit_length = num_blocks * effective_block_size
                    continue

                use_eagle = idx in self.eagle_attn_group_indices and idx not in eagle_verified

                _max_length = curr_hit_length
                if use_eagle:
                    # Eagle needs to match one more block and then pop the last.
                    _max_length = min(curr_hit_length + spec.block_size, max_cache_hit_length)
                # vLLM B renamed the ``use_eagle`` kwarg to ``drop_eagle_block``.
                if vllm_version_is("0.21.0"):
                    eagle_kwarg = {"use_eagle": use_eagle}
                else:
                    eagle_kwarg = {"drop_eagle_block": use_eagle}
                hit_blocks = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    **eagle_kwarg,
                    alignment_tokens=self.lcm_block_size,
                    dcp_world_size=self.dcp_world_size,
                    pcp_world_size=self.pcp_world_size,
                )
                _new_hit_length = len(hit_blocks[0]) * effective_block_size
                if use_eagle:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                curr_hit_length = len(hit_blocks[0]) * effective_block_size
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length (if present)
        # NOTE(zxr): for deepseek-v4, there is two fullattn groups, but
        # in this function, only the first fullattn group is truncate by
        # the belowing codes(c4), c128 layer does not truncate, which may
        # have prefix cache block hit.
        # Due to slidingwindow attn, deepseek-v4 decode node can't have
        # any prefix cache hit, because `hit_length` of SWA is 0.
        spec, group_ids, _ = self.attention_groups[0]
        if isinstance(spec, FullAttentionSpec):
            num_blocks = hit_length // self._get_effective_block_size(spec)
            for group_id in group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]

        return tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group), hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_num_batched_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    scheduler_block_size: int | None = None,
    eagle_attn_layer_names: list[str] | None = None,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    if _is_deepseek_v4_kv_cache_config(kv_cache_config):
        return AscendHybridKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            eagle_attn_layer_names=eagle_attn_layer_names,
            metrics_collector=metrics_collector,
            max_num_batched_tokens=max_num_batched_tokens,
            scheduler_block_size=scheduler_block_size,
        )

    cp_enabled = dcp_world_size > 1 or pcp_world_size > 1

    # Only CP hybrid prefix caching needs AscendHybridKVCacheCoordinator.
    # Otherwise keep upstream coordinators (non-CP / unitary / no-prefix-cache).
    if not cp_enabled or len(kv_cache_config.kv_cache_groups) == 1 or not enable_caching:
        orig_kwargs = dict(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        if not vllm_version_is("0.21.0"):
            orig_kwargs["scheduler_block_size"] = scheduler_block_size
        return _orig_get_kv_cache_coordinator(**orig_kwargs)
    return AscendHybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        eagle_attn_layer_names=eagle_attn_layer_names,
        metrics_collector=metrics_collector,
        max_num_batched_tokens=max_num_batched_tokens,
        scheduler_block_size=scheduler_block_size,
    )


vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]

# `kv_cache_manager` imports `get_kv_cache_coordinator` with
# `from ... import ...`, so if it was loaded before this patch runs
# (for example through the recompute scheduler path), it keeps the
# old function object. Update that cached binding as well.
_kv_cache_manager = sys.modules.get("vllm.v1.core.kv_cache_manager")
if _kv_cache_manager is not None:
    _kv_cache_manager.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]
