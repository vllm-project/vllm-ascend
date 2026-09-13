# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM projectx
import functools
import sys
from collections.abc import Callable, Mapping
from math import lcm
from typing import Any

import vllm
import vllm.envs as envs_vllm
import vllm.v1.core.kv_cache_coordinator as vllm_kv_cache_coordinator
import vllm.v1.core.kv_cache_utils as vllm_kv_cache_utils
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
    SlidingWindowManager,
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
from vllm_ascend.utils import vllm_version_is

USE_MULTI_GROUPS_KV_CACHE = True

_orig_get_kv_cache_coordinator = vllm.v1.core.kv_cache_coordinator.get_kv_cache_coordinator


def _is_pure_kv_producer(vllm_config) -> bool:
    """Whether this process is a PD prefill producer that never consumes.

    ``kv_both`` instances (both ``is_kv_producer`` and ``is_kv_consumer``)
    also accept consumer traffic and intentionally keep upstream behavior.
    ``getattr`` fallbacks keep this callable with partial config doubles in
    unit tests and across vLLM revisions.
    """
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    return (
        kv_transfer_config is not None
        and getattr(kv_transfer_config, "is_kv_producer", False)
        and not getattr(kv_transfer_config, "is_kv_consumer", False)
    )


# Module anchors populated by _install_producer_role_kv_cache_config_tag:
# the captured upstream/Ascend builder and its role-tagging wrapper. Keeping
# them module-global lets the unit tests substitute a fake upstream builder.
_orig_get_kv_cache_config_from_groups: Callable[..., Any]
_get_kv_cache_config_from_groups: Callable[..., Any]


def _install_producer_role_kv_cache_config_tag() -> None:
    """Make every built ``KVCacheConfig`` carry the pure-PD-producer role.

    The coordinator factory below receives ``KVCacheConfig`` but not
    ``VllmConfig``, so the role has to ride on the config object. This one
    routine performs the whole hand-off:

    1. Wrap whichever builder is currently registered (the Ascend
       stride-aware planner in ``patch_kv_cache_utils`` may already have
       replaced the upstream one) so the resulting config gets an
       ``is_kv_producer`` tag.
    2. Publish the wrapper on the ``kv_cache_utils`` module global -
       ``get_kv_cache_configs`` invokes the builder as a module global, so
       this covers the engine's multi-config path.
    3. Compensate importers that may already hold a direct symbol binding
       (e.g. the worker profile-run path in ``gpu_model_runner``) when this
       patch loads late.

    In the supported vLLM revisions ``KVCacheConfig`` is a plain (non-frozen,
    no-``__slots__``) dataclass; the scheduler-side config is derived via
    ``copy.deepcopy`` in ``generate_scheduler_kv_cache_config``, which keeps
    the attribute, while worker IPC is pickle-based and simply carries (or,
    on an msgspec boundary, silently drops) the extra bool - nothing on the
    worker side reads it. A marker attribute makes re-installation
    idempotent under module reload.
    """
    global _orig_get_kv_cache_config_from_groups, _get_kv_cache_config_from_groups

    current_builder = vllm_kv_cache_utils.get_kv_cache_config_from_groups
    if getattr(current_builder, "_ascend_producer_role_tag", False):
        # Module reload: keep the existing tagged wrapper (its __wrapped__
        # points at the builder captured on first install).
        _get_kv_cache_config_from_groups = current_builder
        _orig_get_kv_cache_config_from_groups = getattr(current_builder, "__wrapped__", current_builder)
        return
    _orig_get_kv_cache_config_from_groups = current_builder

    @functools.wraps(current_builder)
    def _get_kv_cache_config_from_groups(vllm_config, *args, **kwargs):
        kv_cache_config = _orig_get_kv_cache_config_from_groups(vllm_config, *args, **kwargs)
        kv_cache_config.is_kv_producer = _is_pure_kv_producer(vllm_config)
        return kv_cache_config

    _get_kv_cache_config_from_groups._ascend_producer_role_tag = True  # type: ignore[attr-defined]
    vllm_kv_cache_utils.get_kv_cache_config_from_groups = (  # type: ignore[attr-defined]
        _get_kv_cache_config_from_groups
    )
    for _importer_module_name in ("vllm.v1.worker.gpu_model_runner",):
        _importer = sys.modules.get(_importer_module_name)
        if _importer is not None and hasattr(_importer, "get_kv_cache_config_from_groups"):
            _importer.get_kv_cache_config_from_groups = (  # type: ignore[attr-defined]
                _get_kv_cache_config_from_groups
            )


_install_producer_role_kv_cache_config_tag()


def _select_kv_token_budget(
    max_model_len: int,
    max_in_flight_tokens: int | None,
    max_num_batched_tokens: int | None,
) -> int:
    token_budget = max_in_flight_tokens
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
        if use_eagle and not vllm_version_is("0.28.0"):
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
            if not vllm_version_is("0.28.0"):
                # Main uses a separate write-mask alignment. Match the lookup
                # boundary, including the raw-tail pool alignment.
                mgr.cache_hit_alignment_tokens = self._cache_hit_alignment_tokens
            elif isinstance(mgr, SlidingWindowManager):
                mgr.scheduler_block_size = self.lcm_block_size

        self.use_eagle = use_eagle
        # A PD prefill producer only schedules fresh-request prefills;
        # every content-hash block it can match is a verified prompt block
        # (draft/lookahead tokens live in the request-private tail, whose
        # hash can never match another request). The EAGLE last-block drop
        # is therefore never needed on the producer, and with hybrid
        # mamba-align pages (1536 tokens) it erases the whole shared prefix
        # of typical ~2K prompts, pinning P-side prefix hits to 0.
        #
        # The role is tagged onto KVCacheConfig by the
        # ``get_kv_cache_config_from_groups`` wrapper above; configs built
        # without the tag (e.g. unit tests) default to non-producer.
        self.is_kv_producer = getattr(kv_cache_config, "is_kv_producer", False)

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

                drop_eagle_block = use_eagle and idx not in eagle_verified and not self.is_kv_producer

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
        # producer (see ``self.is_kv_producer``): matched content blocks
        # are always verified prompt blocks there.
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
                drop_eagle_block=use_eagle and not self.is_kv_producer,
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


def _install_producer_mamba_block_aligned_split_patch(scheduler_cls: Any = None) -> None:
    """Suppress the EAGLE-block-drop backoff in ``_mamba_block_aligned_split``
    on a PD prefill producer.

    This is the scheduler-side companion of the producer drop exemption in
    ``AscendHybridKVCacheCoordinator`` above.

    ``Scheduler._mamba_block_aligned_split`` backs the last cacheable
    mamba-align page off by one block (and, on newer vLLM revisions, shifts
    the partial-tail checkpoint boundary) whenever the EAGLE block drop is
    active. Consequently a producer prefill can never end a chunk at its
    final full-page boundary. In mamba "align" mode the recurrent state of a
    page is only materialized across a chunk boundary (copy-on-write in
    ``MambaManager.allocate_new_blocks``), so the suppressed split leaves the
    final full state page unhashed. Hybrid coordinator hits reconcile to the
    per-group minimum: even with the full-attention group fixed by the
    coordinator exemption above, the mamba groups report one full page less
    (1600-token prompts -> 0 hit, 3200-token prompts -> 1536 with 1536-token
    align pages) - the observed MTP prefix-cache kill band.

    Rather than copy the scheduler method (its body moves between vLLM
    revisions), invoke the original with the drop bit temporarily cleared:
    every read of the bit inside the method exists solely to compensate for
    the block drop, and on the producer matched blocks are always verified
    prompt blocks and the coordinator never drops. Scheduling is
    single-threaded per scheduler instance, so the temporary toggle is safe.
    vLLM 0.28.x names the bit ``use_eagle``; newer revisions expose the
    dedicated ``use_eagle_block_drop`` knob.

    The wrapper is installed unconditionally and self-gates at call time on
    ``self.vllm_config.kv_transfer_config`` (the same PD role source used by
    the coordinator tag and the neighboring mamba split patch), so consumers
    and standalone instances pass straight through with upstream behavior.

    The producer wrapper is always installed around the method currently
    registered on ``Scheduler``.  This ordering is important: the neighboring
    consumer/sparse-index patch has early-return paths which never delegate to
    its saved original.  An inner producer wrapper would therefore leave the
    EAGLE drop enabled on those paths.
    """
    if scheduler_cls is None:
        from vllm.v1.core.sched.scheduler import Scheduler

        scheduler_cls = Scheduler

    if not hasattr(scheduler_cls, "_mamba_block_aligned_split"):
        return
    registered_split = scheduler_cls._mamba_block_aligned_split
    if getattr(registered_split, "_ascend_producer_no_eagle_drop", False):
        # Idempotent under module reload.
        return

    original_split = registered_split

    @functools.wraps(original_split)
    def _producer_mamba_block_aligned_split(
        self,
        request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        if not _is_pure_kv_producer(getattr(self, "vllm_config", None)):
            return original_split(
                self,
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        # vLLM 0.28.x and newer revisions use different names.  Some
        # transitional scheduler implementations expose both and different
        # wrapper layers consult different attributes, so clear every
        # attribute that exists and restore all of them after the call.
        drop_attrs = tuple(
            name for name in ("use_eagle", "use_eagle_block_drop") if hasattr(self, name)
        )
        original_drop_values = {name: getattr(self, name) for name in drop_attrs}
        for name in drop_attrs:
            setattr(self, name, False)
        try:
            return original_split(
                self,
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        finally:
            for name, value in original_drop_values.items():
                setattr(self, name, value)

    _producer_mamba_block_aligned_split._ascend_producer_no_eagle_drop = True  # type: ignore[attr-defined]
    scheduler_cls._mamba_block_aligned_split = _producer_mamba_block_aligned_split


# The wrapper self-gates on the PD role at call time, so installation is
# unconditional (mirrors the other scheduler-side patch imported earlier in
# this package).
_install_producer_mamba_block_aligned_split_patch()
