# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from math import lcm
from typing import cast

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
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    MambaManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable
from vllm_ascend.utils import vllm_version_is

USE_MULTI_GROUPS_KV_CACHE = True

_orig_get_kv_cache_coordinator = vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator


def _select_kv_token_budget(
    max_model_len: int,
    max_in_flight_tokens: int | None,
    max_num_batched_tokens: int | None,
) -> int:
    token_budget = max_in_flight_tokens
    return token_budget if token_budget is not None else max_model_len


def _is_deepseek_v4_kv_cache_spec(kv_cache_spec: KVCacheSpec) -> bool:
    if getattr(kv_cache_spec, "model_version", None) in {"deepseek_v4", "deepseek_v41"}:
        return True

    nested_specs = getattr(kv_cache_spec, "kv_cache_specs", None)
    if nested_specs is None:
        return False

    if isinstance(nested_specs, Mapping):
        nested_specs = nested_specs.values()
    elif not isinstance(nested_specs, (list, tuple, set)):
        return False

    return any(getattr(spec, "model_version", None) in {"deepseek_v4", "deepseek_v41"} for spec in nested_specs)


def _is_deepseek_v4_kv_cache_config(kv_cache_config: KVCacheConfig) -> bool:
    return any(_is_deepseek_v4_kv_cache_spec(group.kv_cache_spec) for group in kv_cache_config.kv_cache_groups)


def _is_a5_packed_cache_config(kv_cache_config: KVCacheConfig) -> bool:
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        nested = getattr(spec, "kv_cache_specs", None)
        specs = nested.values() if isinstance(nested, Mapping) else (spec,)
        if any(
            getattr(item, "cache_dtype_str", None) == "a5_mxfp8_bf16_scale"
            for item in specs
        ):
            return True
    return False


def _manager_spec(spec: KVCacheSpec) -> KVCacheSpec:
    # The scheduler normally unwraps uniform groups. Also accept the original
    # planner representation when constructing the coordinator directly.
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


class _GroupStableBlockPool(BlockPool):
    """Keep packed physical pages in the cache group that first owns them.

    A5 V4.1 cache groups are different typed views over the same physical page
    slots. Reassigning a recycled page to another group is not graph-safe, so
    free pages are reused only by their owning group; untouched pages remain
    available to every group.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._owners = [-1] * self.num_gpu_blocks
        self._unowned = deque(self.blocks[1:])
        self._free_by_group: dict[int, deque[KVCacheBlock]] = defaultdict(deque)
        self._queued_by_group: dict[int, set[int]] = defaultdict(set)

    @staticmethod
    def _is_free(block: KVCacheBlock) -> bool:
        return (
            block.ref_cnt == 0
            and not block.is_null
            and block.prev_free_block is not None
            and block.next_free_block is not None
        )

    def get_new_blocks_for_group(
        self, num_blocks: int, group_id: int
    ) -> list[KVCacheBlock]:
        if num_blocks == 0:
            return []
        preferred: list[KVCacheBlock] = []
        group_queue = self._free_by_group[group_id]
        queued = self._queued_by_group[group_id]
        while group_queue and len(preferred) < num_blocks:
            block = group_queue.popleft()
            if block.block_id not in queued:
                continue
            queued.remove(block.block_id)
            if self._is_free(block):
                preferred.append(block)
        while self._unowned and len(preferred) < num_blocks:
            block = self._unowned.popleft()
            if self._owners[block.block_id] == -1 and self._is_free(block):
                preferred.append(block)
        if len(preferred) != num_blocks:
            raise ValueError(
                f"Cache group {group_id} needs {num_blocks} stable pages, "
                f"but only {len(preferred)} owned or unowned pages are free"
            )
        for block in preferred:
            self.free_block_queue.remove(block)
        self.free_block_queue.prepend_n(preferred)
        blocks = super().get_new_blocks(num_blocks)
        for block in blocks:
            owner = self._owners[block.block_id]
            assert owner in (-1, group_id)
            self._owners[block.block_id] = group_id
        return blocks

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        blocks = list(ordered_blocks)
        super().free_blocks(blocks)
        for block in blocks:
            group_id = self._owners[block.block_id]
            if group_id < 0 or not self._is_free(block):
                continue
            queued = self._queued_by_group[group_id]
            if block.block_id not in queued:
                self._free_by_group[group_id].appendleft(block)
                queued.add(block.block_id)

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        for block in blocks:
            group_id = self._owners[block.block_id]
            if group_id >= 0:
                self._queued_by_group[group_id].discard(block.block_id)
        super().touch(blocks)

    def free_blocks_for_group(
        self, ordered_blocks: Iterable[KVCacheBlock], group_id: int
    ) -> None:
        blocks = list(ordered_blocks)
        assert all(
            block.is_null or self._owners[block.block_id] == group_id
            for block in blocks
        )
        self.free_blocks(blocks)


class _CacheGroupBlockPoolView:
    """Bind an ordinary single-type manager to one stable-page domain."""

    def __init__(self, pool: _GroupStableBlockPool, group_id: int):
        self._pool = pool
        self._group_id = group_id

    def __getattr__(self, name):
        return getattr(self._pool, name)

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        return self._pool.get_new_blocks_for_group(num_blocks, self._group_id)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        self._pool.free_blocks_for_group(ordered_blocks, self._group_id)

    def touch(self, blocks: list[KVCacheBlock]) -> None:
        assert all(
            block.is_null
            or self._pool._owners[block.block_id] == self._group_id
            for block in blocks
        )
        self._pool.touch(blocks)


class AscendHybridKVCacheCoordinator(HybridKVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
    """

    def __init__(  # type: ignore[misc]
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
        num_prefill_lookahead: int = 0,
    ):
        # Keep pcp_world_size in this patched constructor for compatibility
        # with the upstream coordinator interface. PCP is rejected by the platform.
        del pcp_world_size
        # main (cdc4824a21): upstream cache_blocks reads num_reprefillable_tokens
        self.num_reprefillable_tokens = max(0, (num_prefill_lookahead or 0) - 1)
        self.dcp_world_size = dcp_world_size
        self.scheduler_block_size = scheduler_block_size
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # Fall back to `max_model_len` when unset so the recycling-aware
        # admission cap (vLLM PR #40946) collapses to the prior uncapped
        # behavior. The scheduler always supplies the real value at runtime.
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
        pool_cls = (
            _GroupStableBlockPool
            if _is_a5_packed_cache_config(kv_cache_config)
            else BlockPool
        )
        self.block_pool = pool_cls(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=enable_kv_cache_events,
            metrics_collector=metrics_collector,
        )

        # KV cache group indices that get the EAGLE last-block drop.
        self.eagle_group_ids: set[int] = {  # type: ignore[no-redef]
            i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
        }
        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

        extra_mgr_kwargs: dict = {"scheduler_block_size": scheduler_block_size}
        extra_mgr_kwargs["needs_kv_cache_zeroing"] = kv_cache_config.needs_kv_cache_zeroing
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=_manager_spec(kv_cache_group.kv_cache_spec),
                block_pool=(
                    cast(
                        BlockPool,
                        _CacheGroupBlockPoolView(self.block_pool, i),
                    )
                    if isinstance(self.block_pool, _GroupStableBlockPool)
                    else self.block_pool
                ),
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=1,
                max_in_flight_tokens=token_budget,
                max_model_len=max_model_len,
                **extra_mgr_kwargs,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )
        # A CircularBufferSpec can share the global block-ID pool with
        # attention caches even though its physical state is not an attention
        # tensor.  Ascend's physical-page zeroer clears every backing for each
        # reported ID, so circular-state allocations must participate as well;
        # otherwise an ID recycled directly into the state group can retain a
        # previous request's state indefinitely.
        if kv_cache_config.needs_kv_cache_zeroing:
            for manager in self.single_type_managers:
                if isinstance(manager.kv_cache_spec, CircularBufferSpec):
                    manager._record_new_block_ids = True
        # vLLM #53614 aligns exported Mamba checkpoints with EAGLE replay.
        if use_eagle and not vllm_version_is("0.29.0"):
            for manager in self.single_type_managers:
                if isinstance(manager, MambaManager):
                    manager.drop_eagle_checkpoint_block = True

        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        if enable_caching:
            # The GLM kpool tail spec uses block_size=index_kpool and opts
            # out of prefix caching, so it is not bound by the MLA hash
            # block size.
            assert all(
                self._get_effective_block_size(g.kv_cache_spec) % hash_block_size == 0
                for g in kv_cache_config.kv_cache_groups
                if is_prefix_cacheable(g.kv_cache_spec)
            ), "block_size must be divisible by hash_block_size"
        self.enable_partial_hash_hits = dcp_world_size == 1 and any(
            isinstance(g.kv_cache_spec, MambaSpec)
            and g.kv_cache_spec.mamba_cache_mode == "align"
            and g.kv_cache_spec.block_size > hash_block_size
            for g in kv_cache_config.kv_cache_groups
        )
        self.verify_and_split_kv_cache_groups()

        # Align the WRITE-path mask granularity (reachable_block_mask) with the
        # READ-path hit granularity (find_longest_cache_hit) so SlidingWindowManager
        # only caches blocks that land on a boundary where future cache hits can
        # actually be matched.
        # TODO (Csrayz): Consider unified all single_type_managers to simplify logic.
        for mgr in self.single_type_managers:
            # Both supported versions use a separate write-mask alignment.
            # Match the lookup boundary, including the raw-tail pool alignment.
            mgr.cache_hit_alignment_tokens = self._cache_hit_alignment_tokens

        self.use_eagle = use_eagle

    @property
    def _cache_hit_alignment_tokens(self) -> int:
        tail_alignment = getattr(self, "tail_pool_alignment", 1)
        if self.enable_partial_hash_hits:
            return lcm(self.hash_block_size, tail_alignment)
        return lcm(self.scheduler_block_size or self.lcm_block_size, tail_alignment)

    def _get_effective_block_size(self, kv_cache_spec: KVCacheSpec) -> int:
        block_size = kv_cache_spec.block_size
        if isinstance(kv_cache_spec, MambaSpec) and self.enable_caching:
            return block_size
        if self.dcp_world_size > 1:
            block_size *= self.dcp_world_size
        return block_size

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Groups KV cache groups by their spec type for efficient batch processing
        during cache hit lookup.
        """
        self.attention_groups: list[SpecGroup] = []
        self.tail_pool_alignment = 1
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            if not is_prefix_cacheable(g.kv_cache_spec):
                specs = (
                    g.kv_cache_spec.kv_cache_specs.values()
                    if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
                    else (g.kv_cache_spec,)
                )
                # A new tail has no raw history at a prefix hit. Resume only
                # at whole-pool boundaries, including fine-grained Mamba hits.
                for tail_spec in specs:
                    self.tail_pool_alignment = lcm(self.tail_pool_alignment, getattr(tail_spec, "compress_ratio", 1))
                continue
            manager_cls = self.single_type_managers[i].__class__
            spec = _manager_spec(g.kv_cache_spec)
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

        assert self.attention_groups, "Prefix caching requires at least one cacheable KV cache group."

        # Put full attention first: its efficient left-to-right scan provides
        # a tighter initial bound, reducing work for subsequent groups.
        self.attention_groups.sort(key=lambda group: not isinstance(group.spec, FullAttentionSpec))

        # Dense reference group for per-group lookups (None when the model
        # has no full-attention layers): full attention is downward-closed,
        # so any group reporting a longer per-group hit implies the union of
        # per-group hits is not consistent at a single boundary (#46453).
        first = self.attention_groups[0]
        self.full_attention_group_id: int | None = (
            first.group_ids[0] if isinstance(first.spec, FullAttentionSpec) else None
        )

        # Propagate the eagle bit to every manager in an eagle-containing
        # attention group, mirroring upstream
        # HybridKVCacheCoordinator.verify_and_split_kv_cache_groups. Managers
        # default to ``use_eagle=False`` ("initialized lazily by the
        # coordinator", see SingleTypeKVCacheManager.__init__).
        #
        # Required for prefix-cache correctness on DeepSeek-V4 + MTP/EAGLE: the
        # SWA write path (``cache_blocks`` -> ``reachable_block_mask``) keys the
        # retained checkpoint tail on ``manager.use_eagle``, while the read path
        # (``find_longest_cache_hit``) applies ``drop_eagle_block`` to every gid
        # merged into the eagle attention group (and ``get_cached_block``
        # requires the block cached for *all* of them). If any such manager
        # keeps the default False, its retained tail ends one block short of the
        # eagle "peek" boundary the read looks at, the SWA group never hits, and
        # the min-over-groups hybrid hit collapses to 0%. Note the upstream
        # ``_annotate_eagle_groups_deepseek_v4`` flags only the single group
        # holding the MTP layer, so iterating ``eagle_group_ids`` alone would
        # miss its same-spec siblings.
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True

        # The LCM of the block sizes of all attention types.
        # The cache hit length must be a multiple of the LCM of the block sizes
        # to make sure the cache hit length is a multiple of the block size of
        # each attention type. Requiring this because we don't support partial
        # block cache hit yet.
        # NOTE: use 16k as the alignment tokens for model with compress ratio
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
            return block_hashes

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
            for idx, (
                spec,
                group_ids,
                manager_cls,
                use_eagle,
            ) in enumerate(self.attention_groups):
                group_block_size = self._get_effective_block_size(spec)
                first_group_id = group_ids[0]
                cached_blocks = hit_blocks_by_group[first_group_id]
                if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                    # Full attention is downward-closed: we only need to look
                    # up cached blocks once; on subsequent iterations just trim
                    # to the (reduced) current hit length.
                    curr_hit_length = min(curr_hit_length, hit_length_by_group[first_group_id])
                    continue

                drop_eagle_block = use_eagle and idx not in eagle_verified

                _max_length = curr_hit_length
                if drop_eagle_block and not isinstance(spec, MambaSpec):
                    # Eagle needs to match one more block and then pop the last.
                    eagle_margin = (
                        self.hash_block_size
                        if self.enable_partial_hash_hits
                        and manager_cls.supports_fine_grained_hash_lookup
                        and group_block_size > self.hash_block_size
                        else group_block_size
                    )
                    _max_length = min(curr_hit_length + eagle_margin, max_cache_hit_length)
                hit_result = manager_cls.find_longest_cache_hit(
                    block_hashes=_get_block_hashes(spec),
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self._cache_hit_alignment_tokens,
                    dcp_world_size=self.dcp_world_size,
                    pcp_world_size=1,
                )
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

        # Truncate all full attention groups to the final hit_length.
        # NOTE(zxr): DeepSeek-V4 has two full-attention groups, C4 and
        # C128. Truncate both groups with their own effective block sizes
        # so neither group keeps prefix-cache blocks beyond final hit_length.
        for group in self.attention_groups:
            if not isinstance(group.spec, FullAttentionSpec):
                continue
            num_blocks = cdiv(
                hit_length,
                self._get_effective_block_size(group.spec),
            )
            for group_id in group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        cache_hit_blocks = tuple(blocks if blocks is not None else [] for blocks in hit_blocks_by_group)
        return cache_hit_blocks, hit_length, longest_hit_length - hit_length


def get_kv_cache_coordinator(  # type: ignore[misc]
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
    num_prefill_lookahead: int = 0,
) -> KVCacheCoordinator:
    # Keep pcp_world_size in this patched function for upstream call
    # compatibility; platform validation guarantees that it is one.
    del pcp_world_size
    token_budget = _select_kv_token_budget(max_model_len, max_in_flight_tokens, max_num_batched_tokens)
    if _is_deepseek_v4_kv_cache_config(kv_cache_config):
        return AscendHybridKVCacheCoordinator(  # type: ignore[call-arg]
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=1,
            hash_block_size=hash_block_size,
            eagle_attn_layer_names=eagle_attn_layer_names,
            metrics_collector=metrics_collector,
            max_in_flight_tokens=token_budget,
            max_num_batched_tokens=token_budget,
            scheduler_block_size=scheduler_block_size,
            num_prefill_lookahead=num_prefill_lookahead,
        )

    if len(kv_cache_config.kv_cache_groups) == 1 or not enable_caching:
        orig_kwargs = dict(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=1,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        orig_kwargs["max_in_flight_tokens"] = token_budget
        orig_kwargs["scheduler_block_size"] = scheduler_block_size
        orig_kwargs["num_prefill_lookahead"] = num_prefill_lookahead
        return _orig_get_kv_cache_coordinator(**orig_kwargs)

    return AscendHybridKVCacheCoordinator(  # type: ignore[call-arg]
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=1,
        hash_block_size=hash_block_size,
        eagle_attn_layer_names=eagle_attn_layer_names,
        metrics_collector=metrics_collector,
        max_in_flight_tokens=token_budget,
        max_num_batched_tokens=token_budget,
        scheduler_block_size=scheduler_block_size,
        num_prefill_lookahead=num_prefill_lookahead,
    )


vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]

# `kv_cache_manager` imports `get_kv_cache_coordinator` with
# `from ... import ...`, so if it was loaded before this patch runs
# (for example through the recompute scheduler path), it keeps the
# old function object. Update that cached binding as well.
_kv_cache_manager = sys.modules.get("vllm.v1.core.kv_cache_manager")
if _kv_cache_manager is not None:
    _kv_cache_manager.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]
