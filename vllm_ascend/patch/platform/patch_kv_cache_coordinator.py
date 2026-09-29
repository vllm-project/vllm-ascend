# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM projectx
import sys
from collections.abc import Mapping
from math import lcm

import vllm
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
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import is_prefix_cacheable

_orig_get_kv_cache_coordinator = vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator


def _skips_eagle_block_drop(kv_transfer_config) -> bool:
    """Whether the EAGLE last-block drop must be suppressed on this process.

    Suppressed for a pure PD prefill producer (``is_kv_producer`` and not
    ``is_kv_consumer``; ``getattr`` fallbacks keep the check working with
    partial config doubles in unit tests and across vLLM revisions) and for
    a standalone instance (``kv_transfer_config is None``). A standalone
    instance has no connector: ``num_external_computed_tokens`` is always
    zero and every content-hash match comes from verified local prompt
    blocks, so the drop only erases hit length - on hybrid mamba-align
    models with a fine ``prefix_match_unit`` it trims the full-attention
    hit below the mamba partial-tail entry and collapses the reconciled
    hybrid hit to 0 (the single-instance counterpart of the P-side kill
    band). Consumers and ``kv_both`` instances keep upstream behavior: they
    receive external loads whose verifier window the drop protects.
    """
    return kv_transfer_config is None or (
        getattr(kv_transfer_config, "is_kv_producer", False)
        and not getattr(kv_transfer_config, "is_kv_consumer", False)
    )


def _select_kv_token_budget(max_model_len: int, max_in_flight_tokens: int | None) -> int:
    return max_in_flight_tokens if max_in_flight_tokens is not None else max_model_len


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


def _manager_spec(spec: KVCacheSpec) -> KVCacheSpec:
    # The scheduler normally unwraps uniform groups. Also accept the original
    # planner representation when constructing the coordinator directly.
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


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
        scheduler_block_size: int | None = None,
        num_prefill_lookahead: int = 0,
        allow_partial_hash_hits: bool = True,
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
        # vLLM main (#54736) added allow_partial_hash_hits to the upstream
        # coordinator interface (fine-grained hybrid prefix hits).
        self.allow_partial_hash_hits = allow_partial_hash_hits
        # Fall back to `max_model_len` when unset so the recycling-aware
        # admission cap (vLLM PR #40946) collapses to the prior uncapped
        # behavior. The scheduler always supplies the real value at runtime.
        token_budget = _select_kv_token_budget(max_model_len, max_in_flight_tokens)
        self.max_in_flight_tokens = token_budget
        self.retention_interval = kv_cache_config.prefix_cache_retention_interval
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
        self.eagle_group_ids: set[int] = {  # type: ignore[no-redef]
            i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
        }
        # Fall back to flagging only full-attention groups when no group is
        # flagged. Mamba/GDN state hits do not use the eagle drop (a draft
        # model has no mamba layers), and flagging mamba groups truncates
        # cached state writes, collapsing hybrid prefix-cache hits to 0.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = {
                i
                for i, g in enumerate(kv_cache_config.kv_cache_groups)
                if isinstance(g.kv_cache_spec, FullAttentionSpec)
            }

        extra_mgr_kwargs: dict = {"scheduler_block_size": scheduler_block_size}
        extra_mgr_kwargs["needs_kv_cache_zeroing"] = kv_cache_config.needs_kv_cache_zeroing
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=_manager_spec(kv_cache_group.kv_cache_spec),
                block_pool=self.block_pool,
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
        # vLLM #53614 aligns exported Mamba checkpoints with EAGLE replay.
        if use_eagle:
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
        # Roles are derived here, where they are used, from the kv-transfer
        # config attached by the ``get_kv_cache_config_from_groups`` builder
        # in ``patch_kv_cache_utils`` (``KVCacheConfig`` has no native field;
        # the attach survives the scheduler-side deepcopy and is dropped by
        # worker pickle IPC, which never reads it). Configs built without it
        # (e.g. unit tests) read as standalone (``kv_transfer_config is
        # None``).
        #
        # A PD prefill producer only schedules fresh-request prefills;
        # every content-hash block it can match is a verified prompt block
        # (draft/lookahead tokens live in the request-private tail, whose
        # hash can never match another request). The EAGLE last-block drop
        # is therefore never needed on the producer, and with hybrid
        # mamba-align pages (1536 tokens) it erases the whole shared prefix
        # of typical ~2K prompts, pinning P-side prefix hits to 0.
        kv_transfer_config = getattr(kv_cache_config, "kv_transfer_config", None)
        self.skips_eagle_block_drop = _skips_eagle_block_drop(kv_transfer_config)

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

                drop_eagle_block = use_eagle and idx not in eagle_verified and not self.skips_eagle_block_drop

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

    def find_longest_cache_hit_per_group(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], tuple[int, ...]]:
        # PD + hybrid connector path. Skip the EAGLE drop on the prefill
        # producer and on standalone instances (see
        # ``self.skips_eagle_block_drop``): matched content blocks are
        # always verified prompt blocks there.
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_blocks: list[list[KVCacheBlock]] = [[] for _ in range(num_groups)]
        hit_lengths: list[int] = [0] * num_groups
        for spec, group_ids, manager_cls, use_eagle in self.attention_groups:
            blocks, group_hit = manager_cls.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=use_eagle and not self.skips_eagle_block_drop,
                alignment_tokens=self._cache_hit_alignment_tokens,
                dcp_world_size=self.dcp_world_size,
                pcp_world_size=1,
            )
            for gid, blks in zip(group_ids, blocks):
                hit_blocks[gid] = blks
                hit_lengths[gid] = group_hit
        return tuple(hit_blocks), tuple(hit_lengths)


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
    num_prefill_lookahead: int = 0,
    allow_partial_hash_hits: bool = True,
) -> KVCacheCoordinator:
    # Keep pcp_world_size in this patched function for upstream call
    # compatibility; platform validation guarantees that it is one.
    del pcp_world_size
    token_budget = _select_kv_token_budget(max_model_len, max_in_flight_tokens)
    hybrid_kwargs = dict(
        kv_cache_config=kv_cache_config,
        max_model_len=max_model_len,
        use_eagle=use_eagle,
        enable_caching=enable_caching,
        enable_kv_cache_events=enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=1,
        hash_block_size=hash_block_size,
        eagle_attn_layer_names=eagle_attn_layer_names,
        metrics_collector=metrics_collector,
        max_in_flight_tokens=token_budget,
        scheduler_block_size=scheduler_block_size,
        num_prefill_lookahead=num_prefill_lookahead,
    )
    # vLLM main (#54736) added allow_partial_hash_hits.
    hybrid_kwargs["allow_partial_hash_hits"] = allow_partial_hash_hits
    if _is_deepseek_v4_kv_cache_config(kv_cache_config):
        return AscendHybridKVCacheCoordinator(**hybrid_kwargs)  # type: ignore[call-arg]

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
        orig_kwargs["allow_partial_hash_hits"] = allow_partial_hash_hits
        return _orig_get_kv_cache_coordinator(**orig_kwargs)

    return AscendHybridKVCacheCoordinator(**hybrid_kwargs)  # type: ignore[call-arg]


vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]

# `kv_cache_manager` imports `get_kv_cache_coordinator` with
# `from ... import ...`, so if it was loaded before this patch runs
# (for example through the recompute scheduler path), it keeps the
# old function object. Update that cached binding as well.
_kv_cache_manager = sys.modules.get("vllm.v1.core.kv_cache_manager")
if _kv_cache_manager is not None:
    _kv_cache_manager.get_kv_cache_coordinator = get_kv_cache_coordinator  # type: ignore[attr-defined]
