# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM projectx
from collections.abc import Mapping
from math import lcm

import vllm
import vllm.envs as envs_vllm
import vllm.v1.core.kv_cache_coordinator as vllm_kv_cache_coordinator
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
    SpecGroup,
)
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    SlidingWindowManager,
)
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


def _select_kv_token_budget(
    max_model_len: int,
    max_in_flight_tokens: int | None,
    max_num_batched_tokens: int | None,
) -> int:
    token_budget = max_num_batched_tokens if vllm_version_is("0.25.1") else max_in_flight_tokens
    return token_budget if token_budget is not None else max_model_len


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
    thus multiple KV cache groups.
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
        max_in_flight_tokens: int | None = None,
        max_num_batched_tokens: int | None = None,
        scheduler_block_size: int | None = None,
    ):
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        self.scheduler_block_size = scheduler_block_size
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        token_budget = _select_kv_token_budget(max_model_len, max_in_flight_tokens, max_num_batched_tokens)
        self.max_in_flight_tokens = token_budget
        self.max_num_batched_tokens = token_budget
        self.retention_interval = getattr(envs_vllm, "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", None)
        validate_retention_interval = getattr(
            vllm_kv_cache_coordinator,
            "_validate_prefix_cache_retention_interval",
            None,
        )
        if self.retention_interval is not None and validate_retention_interval is not None:
            validate_retention_interval(
                self.retention_interval,
                self.scheduler_block_size,
                kv_cache_config,
            )
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
        )

        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group}
        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        extra_mgr_kwargs: dict = {"scheduler_block_size": scheduler_block_size}
        if not vllm_version_is("0.25.1"):
            extra_mgr_kwargs["needs_kv_cache_zeroing"] = kv_cache_config.needs_kv_cache_zeroing
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                max_in_flight_tokens=token_budget,
                max_num_batched_tokens=token_budget,
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
        self.enable_partial_hash_hits = not vllm_version_is("0.25.1") and (
            dcp_world_size == 1
            and any(
                isinstance(g.kv_cache_spec, MambaSpec)
                and g.kv_cache_spec.mamba_cache_mode == "align"
                and g.kv_cache_spec.block_size > hash_block_size
                for g in kv_cache_config.kv_cache_groups
            )
        )
        self.verify_and_split_kv_cache_groups()
        if vllm_version_is("0.25.1") and self.scheduler_block_size is None:
            self.scheduler_block_size = self.lcm_block_size

        # Align the WRITE-path mask granularity (reachable_block_mask) with the
        # READ-path hit granularity (find_longest_cache_hit) so SlidingWindowManager
        # only caches blocks that land on a boundary where future cache hits can
        # actually be matched.
        # TODO (Csrayz): Consider unified all single_type_managers to simplify logic.
        for mgr in self.single_type_managers:
            if isinstance(mgr, SlidingWindowManager):
                mgr.scheduler_block_size = self.lcm_block_size

        self.use_eagle = use_eagle

    @property
    def _cache_hit_alignment_tokens(self) -> int:
        if vllm_version_is("0.25.1"):
            return self.lcm_block_size
        if self.enable_partial_hash_hits:
            return self.hash_block_size
        return self.scheduler_block_size or self.lcm_block_size

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

    def _get_physical_hit_block_size(self, kv_cache_spec: KVCacheSpec) -> int:
        """Return the token span represented by one v0.25.1 hit block."""
        block_size = kv_cache_spec.block_size
        if not isinstance(kv_cache_spec, MambaSpec) and self.dcp_world_size * self.pcp_world_size > 1:
            block_size *= self.dcp_world_size * self.pcp_world_size
        return block_size

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        self.attention_groups: list[SpecGroup] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec
            use_eagle = i in self.eagle_group_ids

            # Try to find an existing group with the same spec
            for idx, group in enumerate(self.attention_groups):
                if group.spec == spec:
                    assert manager_cls is group.manager_cls, "Expected same manager class for identical KV cache specs."
                    group.group_ids.append(i)
                    if use_eagle and not group.use_eagle:
                        self.attention_groups[idx] = group._replace(use_eagle=True)
                    break
            else:
                self.attention_groups.append(SpecGroup(spec, [i], manager_cls, use_eagle))

        assert len(self.attention_groups) > 1, "HybridKVCacheCoordinator requires at least two attention groups."

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups.sort(key=lambda g: not isinstance(g.spec, FullAttentionSpec))

        # Propagate the eagle bit to each manager (default to ``use_eagle=False``).
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True

        block_sizes = [self._get_effective_block_size(group.spec) for group in self.attention_groups]
        self.lcm_block_size = lcm(*block_sizes)

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int] | tuple[tuple[list[KVCacheBlock], ...], int, int]:
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
            if not vllm_version_is("0.25.1"):
                return block_hashes
            block_size = self._get_physical_hit_block_size(kv_cache_spec)
            if block_size == self.hash_block_size:
                return block_hashes
            return BlockHashListWithBlockSize(block_hashes, self.hash_block_size, block_size)

        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        longest_hit_length = 0
        hit_blocks_by_group: list[list[KVCacheBlock] | None] = [None] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        # Simple hybrid (1 full attn + 1 other): one iteration suffices.
        # Full attn is always first if it exists.
        is_simple_hybrid = len(self.attention_groups) == 2 and isinstance(
            self.attention_groups[0].spec, FullAttentionSpec
        )

        # Attention-group indices whose EAGLE drop is verified at the current
        # ``curr_hit_length``. Each eagle group applies the drop at most once
        # per candidate length (see issue #32802).
        eagle_verified: set[int] = set()

        while True:
            curr_hit_length = hit_length

            for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(self.attention_groups):
                group_block_size = self._get_effective_block_size(spec)
                cached_blocks = hit_blocks_by_group[group_ids[0]]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    if vllm_version_is("0.25.1"):
                        curr_hit_length = min(curr_hit_length, hit_length_by_group[group_ids[0]])
                    else:
                        curr_hit_length = curr_hit_length // group_block_size * group_block_size
                    continue

                drop_eagle_block = use_eagle and idx not in eagle_verified

                _max_length = curr_hit_length
                if drop_eagle_block:
                    # Eagle needs to match one more block and then pop the last.
                    _max_length = min(curr_hit_length + group_block_size, max_cache_hit_length)
                hit_result = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self.lcm_block_size,
                    dcp_world_size=self.dcp_world_size,
                    pcp_world_size=self.pcp_world_size,
                )
                if vllm_version_is("0.25.1"):
                    hit_blocks = hit_result
                    _new_hit_length = len(hit_blocks[0]) * self._get_physical_hit_block_size(spec)
                else:
                    hit_blocks, _new_hit_length = hit_result
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)

            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full attention blocks to final hit_length (if present)
        first_group = self.attention_groups[0]
        if isinstance(first_group.spec, FullAttentionSpec):
            group_block_size = self._get_effective_block_size(first_group.spec)
            num_blocks = cdiv(hit_length, group_block_size)
            for group_id in first_group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        self.num_uncached_common_prefix_tokens = longest_hit_length - hit_length
        cache_hit_blocks = tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group)
        if vllm_version_is("0.25.1"):
            return cache_hit_blocks, hit_length
        return cache_hit_blocks, hit_length, longest_hit_length - hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_in_flight_tokens: int | None = None,
    use_eagle: bool = False,
    enable_caching: bool = True,
    enable_kv_cache_events: bool = False,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
    hash_block_size: int = 0,
    scheduler_block_size: int | None = None,
    eagle_attn_layer_names: list[str] | None = None,
    metrics_collector: KVCacheMetricsCollector | None = None,
    max_num_batched_tokens: int | None = None,
) -> KVCacheCoordinator:
    token_budget = _select_kv_token_budget(max_model_len, max_in_flight_tokens, max_num_batched_tokens)
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
            max_in_flight_tokens=token_budget,
            max_num_batched_tokens=token_budget,
            scheduler_block_size=scheduler_block_size,
        )

    if len(kv_cache_config.kv_cache_groups) == 1 or not enable_caching:
        orig_kwargs = dict(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        if vllm_version_is("0.25.1"):
            orig_kwargs["max_num_batched_tokens"] = token_budget
        else:
            orig_kwargs["max_in_flight_tokens"] = token_budget
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
        max_in_flight_tokens=token_budget,
        max_num_batched_tokens=token_budget,
        scheduler_block_size=scheduler_block_size,
    )


vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]
