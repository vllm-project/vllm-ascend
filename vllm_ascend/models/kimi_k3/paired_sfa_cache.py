# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired MLA/indexer ownership with native, independently sized A5 pages.

Each full-history block owns both components. Recurrent groups use disjoint
IDs from the same pool and may alias main-page physical slots. Token ownership
never depends on the relative byte sizes of the two attention components.
MLA DSpark shares full-history block IDs. GQA DSpark has its own groups.
Matching BF16 sliding-window draft K/V planes reuse adjacent indexer slots.
"""

from dataclasses import replace

import torch
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    KVQuantMode,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import (
    AscendGQADraftFullAttentionSpec,
    AscendGQADraftSlidingWindowSpec,
    AscendMLAAttentionSpec,
    AscendSFAIndexerCacheSpec,
    supports_component_major_mla_pd,
    uses_kimi_k31_bf16_cache,
)
from vllm_ascend.models.kimi_k3.cache_config import make_kimi_cache_groups
from vllm_ascend.utils import vllm_version_is

# vLLM main moved this helper to kv_cache_planning after v0.32.0.
if vllm_version_is("0.30.0") or vllm_version_is("0.32.0"):
    from vllm.v1.core.kv_cache_utils import may_override_num_blocks
else:
    from vllm.v1.core.kv_cache_planning import may_override_num_blocks


def _draft_layout(groups, num_indexer_slots, indexer_page):
    """Offsets in the indexer/draft region, in bytes per manager block ID."""
    region_size = num_indexer_slots * indexer_page
    layers = []
    for group in groups:
        group_spec = group.kv_cache_spec
        # All layers within this group hold the same IDs and need distinct
        # planes. Other groups acquire disjoint IDs from the same BlockPool.
        next_slot = 0
        for name in group.layer_names:
            spec = group_spec.kv_cache_specs[name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
            if not getattr(spec, "is_draft", False):
                continue
            if (
                isinstance(group_spec, AscendGQADraftSlidingWindowSpec)
                and group_spec.can_overlay_indexer
                and next_slot + 2 <= num_indexer_slots
            ):
                offset = next_slot * indexer_page
                next_slot += 2
            else:
                offset = region_size
                region_size += spec.page_size_bytes
            layers.append((name, spec.page_size_bytes, offset))
    return layers, region_size


def _can_overlay_indexer(spec, target):
    """Both interpretations of each slot must address exactly the same bytes per ID."""
    plane_bytes = spec.block_size * spec.num_kv_heads * spec.head_size * torch.bfloat16.itemsize
    if (
        spec.dtype != torch.bfloat16
        or spec.kv_quant_mode != KVQuantMode.NONE
        or spec.head_size != spec.head_size_v
        or spec.tokens_per_state != 1
        or spec.block_size % 128
        or spec.page_size_bytes != 2 * plane_bytes
        or spec.real_page_size_bytes != spec.page_size_bytes
    ):
        return False
    for component in target.values():
        if not isinstance(component, (AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec)):
            continue
        if (
            component.dtype != torch.bfloat16
            or component.block_size != spec.block_size
            or component.kv_quant_mode != KVQuantMode.NONE
            or component.tokens_per_state != 1
            or component.scale_dim
        ):
            return False
        if isinstance(component, AscendMLAAttentionSpec):
            if component.uses_packed_sfa_main_cache or component.static_fp8_pa_nz:
                return False
        elif (
            component.cache_sparse_li_c4
            or component.cache_sparse_li_c8
            or component.sfa_dcp_replicated_indexer_size != 1
            or component.page_size_padded not in (None, plane_bytes)
            or component.page_size_bytes != plane_bytes
        ):
            return False
    return True


def _validate_paired_sfa_pages(specs) -> None:
    """Check the page geometry settled by NPUPlatform before spec creation."""
    main = [s for s in specs.values() if isinstance(s, AscendMLAAttentionSpec) and not getattr(s, "is_draft", False)]
    recurrent = [s for s in specs.values() if isinstance(s, MambaSpec)]
    if not main:
        raise ValueError("A paired SFA cache requires main MLA specs.")
    page = max(s.real_page_size_bytes for s in main)
    if any(s.real_page_size_bytes != page for s in main):
        raise ValueError("Paired SFA main pages must have identical geometry.")
    if any(replace(s, page_size_padded=None).page_size_bytes > page for s in recurrent):
        raise ValueError("The MLA token block must cover one recurrent checkpoint.")
    if any(s.page_size_bytes != page for s in recurrent):
        raise ValueError("Paired SFA recurrent pages must match the main MLA page size.")


def supports_k31_draft_separate_pool() -> bool:
    """Check the installed core before selecting a layout with overlapping numeric IDs."""
    fields = KVCacheConfig.__dataclass_fields__
    if not all(name in fields and fields[name].init for name in ("block_pool_ids", "num_blocks_per_group")):
        return False

    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler

    return (
        getattr(kv_cache_utils, "SUPPORTS_INDEPENDENT_KV_BLOCK_POOLS", False)
        and {"new_block_ids_to_zero_by_pool", "kv_cache_block_copies_by_pool"}
        <= SchedulerOutput.__dataclass_fields__.keys()
        and hasattr(KVCacheConfig, "pool_id_for_group")
        and hasattr(KVCacheConfig, "num_blocks_for_group")
        and hasattr(KVCacheCoordinator, "get_num_blocks_to_allocate_by_pool")
        and hasattr(KVCacheManager, "get_pool_block_counts")
        and hasattr(Scheduler, "_take_kv_cache_operations")
        and hasattr(Scheduler, "_skip_zeroing_for_remote_load")
    )


def paired_sfa_cache_groups(config, specs):
    specs = {
        name: replace(spec, is_paired_sfa=True) if isinstance(spec, AscendSFAIndexerCacheSpec) else spec
        for name, spec in specs.items()
    }
    draft = {name: spec for name, spec in specs.items() if getattr(spec, "is_draft", False)}
    if not draft:
        return _paired_sfa_target_cache_groups(config, specs)
    target = {name: spec for name, spec in specs.items() if not getattr(spec, "is_draft", False)}
    groups = _paired_sfa_target_cache_groups(config, target)
    gqa = {
        name: spec
        for name, spec in draft.items()
        if isinstance(spec, (AscendGQADraftFullAttentionSpec, AscendGQADraftSlidingWindowSpec))
    }
    if gqa:
        if len(gqa) != len(draft):
            raise ValueError("Paired SFA does not support a mixture of MLA and GQA draft layers.")
        # Specs with identical geometry share a block table. Keep the target
        # group IDs stable and preserve the draft's native window semantics.
        by_spec = {}
        for name, spec in sorted(gqa.items(), key=lambda item: (extract_layer_index(item[0]), item[0])):
            by_spec.setdefault(spec, []).append(name)
        use_indexer_slots = uses_kimi_k31_bf16_cache(config) and supports_component_major_mla_pd(config)
        transfer = config.kv_transfer_config
        options = getattr(transfer, "kv_connector_extra_config", {}) or {}
        separate_pool = options.get("k31_draft_separate_pool", False)
        if type(separate_pool) is not bool:
            raise ValueError("k31_draft_separate_pool must be a boolean")
        if separate_pool and not (use_indexer_slots and config.use_v2_model_runner):
            raise ValueError("Separate K3.1 draft pages require BF16, the V2 runner and the Mooncake pull connector")
        if separate_pool and not supports_k31_draft_separate_pool():
            logger.warning_once(
                "K3.1 k31_draft_separate_pool requested=true, effective=false. "
                "The installed vLLM core lacks independent KV pool support. Using the shared target/draft pool."
            )
            separate_pool = False
        for spec, names in by_spec.items():
            spec = replace(spec, separate_block_pool=separate_pool)
            if isinstance(spec, AscendGQADraftSlidingWindowSpec):
                spec = replace(
                    spec,
                    can_overlay_indexer=not separate_pool and use_indexer_slots and _can_overlay_indexer(spec, target),
                )
            logger.info_once(
                "K3.1 DSpark GQA KV group %d: %d %s layers, block=%d, window=%s, dtype=%s, bytes/page/layer=%d",
                len(groups),
                len(names),
                type(spec).__name__,
                spec.block_size,
                getattr(spec, "sliding_window", None),
                spec.dtype,
                spec.page_size_bytes,
            )
            groups.append(KVCacheGroupSpec(names, spec))
        return groups
    if any(not isinstance(spec, AscendMLAAttentionSpec) or spec.uses_packed_sfa_main_cache for spec in draft.values()):
        raise ValueError("Kimi DSpark draft groups require ordinary MLA cache specs.")
    ordered = dict(sorted(draft.items(), key=lambda item: extract_layer_index(item[0])))

    # Share logical ownership while retaining separate physical regions.
    # UniformType retains per-layer dtype/layout/non-causal flags; the strict
    # AscendMLAAttentionSpec.merge must not merge these different MLA specs.
    full = groups[0]
    assert isinstance(full.kv_cache_spec, UniformTypeKVCacheSpecs)
    shared = {**full.kv_cache_spec.kv_cache_specs, **ordered}
    if any(spec.tokens_per_state != 1 for spec in shared.values()):
        raise ValueError("Shared DSpark ownership requires uncompressed full-history token slots.")
    uniform = UniformTypeKVCacheSpecs.from_specs(shared)
    if uniform is None:
        raise ValueError("Target and DSpark must share full-attention semantics and logical block size.")
    max_len = config.model_config.max_model_len
    widths = {spec.max_num_blocks_per_req(config, max_len) for spec in shared.values()}
    if len(widths) != 1:
        raise ValueError("Target and DSpark must have identical block-table coverage, including DCP sharding.")
    # Keep target first for scheduler conversion and keep recurrent group IDs.
    # One request now owns one full-history block sequence instead of two.
    # Retain drafter prefix-tail protection after scheduler spec unwrapping;
    # this flag does not change the target's causal attention metadata.
    groups[0] = replace(
        full,
        layer_names=list(shared),
        kv_cache_spec=uniform,
        is_eagle_group=full.is_eagle_group or any(spec.non_causal_multi_token_decode for spec in ordered.values()),
    )
    return groups


def _paired_sfa_target_cache_groups(config, specs):
    if config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("Paired SFA cache requires the hybrid KV cache manager.")
    _validate_paired_sfa_pages(specs)
    main = sorted((n for n, s in specs.items() if isinstance(s, AscendMLAAttentionSpec)), key=extract_layer_index)
    indexer = sorted((n for n, s in specs.items() if isinstance(s, AscendSFAIndexerCacheSpec)), key=extract_layer_index)
    if not main or [extract_layer_index(n) for n in main] != [extract_layer_index(n) for n in indexer]:
        raise ValueError("Every sparse MLA layer must own one matching indexer cache.")
    paired = {n: specs[n] for pair in zip(main, indexer) for n in pair}
    if len({s.block_size for s in paired.values()}) != 1:
        raise ValueError("Paired MLA and indexer must use the same local token block size.")
    recurrent = {n: s for n, s in specs.items() if isinstance(s, MambaSpec)}
    recurrent = {n: recurrent[n] for n in sorted(recurrent, key=extract_layer_index)}
    # Each recurrent group overlays at most one main-page slot per target MLA
    # layer. Indexer and draft pages must not enlarge that physical region.
    groups = make_kimi_cache_groups(
        paired,
        recurrent,
        recurrent_group_size=len(main),
        uniform_recurrent=False,
    )
    if groups is None:
        raise ValueError("Paired SFA components must share full-attention block semantics.")
    if len(paired) + len(recurrent) != len(specs):
        raise ValueError("Unexpected cache role in paired SFA layout.")
    return groups


def _layout(groups):
    full = next(
        g
        for g in groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        and any(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in g.kv_cache_spec.kv_cache_specs.values())
    )
    specs = full.kv_cache_spec.kv_cache_specs
    # Draft layers share logical IDs but must not enlarge the target/KDA
    # overlay or inherit the target's packed page geometry. Retain global
    # specs here: a PP-local group can have no local target MLA layer.
    target_main = {
        n: spec
        for n, spec in specs.items()
        if isinstance(spec, AscendMLAAttentionSpec) and not getattr(spec, "is_draft", False)
    }
    main = [n for n in full.layer_names if n in target_main]
    indexer = [n for n in full.layer_names if isinstance(specs[n], AscendSFAIndexerCacheSpec)]
    recurrent = [g for g in groups if isinstance(g.kv_cache_spec, MambaSpec)]
    main_page = next(iter(target_main.values())).page_size_bytes
    indexer_page = next(s.page_size_bytes for s in specs.values() if isinstance(s, AscendSFAIndexerCacheSpec))
    slots = max([len(main), *(len(g.layer_names) for g in recurrent)])
    return full, main, indexer, recurrent, main_page, indexer_page, slots


def paired_sfa_pool_bytes(groups):
    _, _, indexer, _, main_page, indexer_page, slots = _layout(groups)
    _, region_size = _draft_layout(groups, len(indexer), indexer_page)
    return slots * main_page + region_size


def _pool_demand(config, groups):
    return sum(
        (g.kv_cache_spec.max_memory_usage_bytes(config) + g.kv_cache_spec.page_size_bytes - 1)
        // g.kv_cache_spec.page_size_bytes
        for g in groups
    )


def _split_pool_counts(config, groups, pool_ids, available_memory, target_page, draft_page):
    override = config.cache_config.num_gpu_blocks_override
    if override is not None:
        return override, override
    target_required, draft_required = (
        _pool_demand(config, [g for g, pid in zip(groups, pool_ids) if pid == pool]) for pool in (0, 1)
    )
    usable = available_memory - target_page - draft_page  # One reserved null ID per pool.
    request_bytes = target_page * target_required + draft_page * draft_required
    if usable < 0:
        raise ValueError("K3.1 KV budget cannot allocate both pools' null pages")
    if not request_bytes:
        # A PP rank without any cache layers adopts the other ranks' counts.
        return 1, 1
    target_blocks = 1 + usable * target_required // request_bytes
    draft_blocks = 1 + usable * draft_required // request_bytes
    if draft_page:
        # Fill the rounding remainder with whole draft pages.
        draft_blocks += (available_memory - target_blocks * target_page - draft_blocks * draft_page) // draft_page
    return target_blocks, draft_blocks


def paired_sfa_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
    *,
    num_blocks_per_group: tuple[int, ...] | None = None,
) -> KVCacheConfig:
    _, main_mla_names, indexer_names, kda_groups, main_page_size, indexer_page_size, num_main_slots = _layout(
        kv_cache_groups
    )
    draft_layers, region_size = _draft_layout(kv_cache_groups, len(indexer_names), indexer_page_size)
    pool_bytes = num_main_slots * main_page_size + region_size
    pool_ids = tuple(int(getattr(g.kv_cache_spec, "separate_block_pool", False)) for g in kv_cache_groups)
    split = 1 in pool_ids
    if split:
        # Separate all draft layers from the target/indexer span. Within the
        # draft pool, the layers have private planes even when their IDs match.
        draft_layers, draft_page = _draft_layout(kv_cache_groups, 0, indexer_page_size)
        target_page = num_main_slots * main_page_size + len(indexer_names) * indexer_page_size
        if num_blocks_per_group is not None:
            num_blocks = num_blocks_per_group[pool_ids.index(0)]
            draft_blocks = num_blocks_per_group[pool_ids.index(1)]
        else:
            num_blocks, draft_blocks = _split_pool_counts(
                vllm_config, kv_cache_groups, pool_ids, available_memory, target_page, draft_page
            )
        backing_size = num_blocks * target_page + draft_blocks * draft_page
    else:
        num_blocks = may_override_num_blocks(vllm_config, available_memory // pool_bytes)
        draft_blocks = num_blocks
        backing_size = pool_bytes * num_blocks
    main_layer_size = main_page_size * num_blocks
    # Recurrent groups overlay the target MLA region. Their block IDs are
    # disjoint. The indexer/draft region follows these main-page slots.
    tensors = [
        KVCacheTensor(
            size=backing_size,
            layers=main_mla_names,
            offset=0,
            layer_stride=main_layer_size,
            block_stride=main_page_size,
        )
    ]
    for group in kda_groups:
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=group.layer_names,
                offset=0,
                layer_stride=main_layer_size,
                block_stride=main_page_size,
            )
        )
    offset = num_main_slots * main_layer_size
    indexer_layer_size = indexer_page_size * num_blocks
    tensors.append(
        KVCacheTensor(
            size=backing_size,
            layers=indexer_names,
            offset=offset,
            layer_stride=indexer_layer_size,
            block_stride=indexer_page_size,
        )
    )
    if split:
        offset += len(indexer_names) * indexer_layer_size
    for name, page_bytes, draft_offset in draft_layers:
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[name],
                offset=offset + draft_offset * draft_blocks,
                layer_stride=page_bytes * draft_blocks,
                block_stride=page_bytes,
            )
        )
    pool_kwargs = (
        {
            "block_pool_ids": pool_ids,
            "num_blocks_per_group": tuple(draft_blocks if pid else num_blocks for pid in pool_ids),
        }
        if split
        else {}
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
        **pool_kwargs,
    )


def paired_sfa_max_memory(config, groups):
    draft = [g for g in groups if getattr(g.kv_cache_spec, "separate_block_pool", False)]
    if draft:
        target = [g for g in groups if not getattr(g.kv_cache_spec, "separate_block_pool", False)]
        draft_page = sum(len(g.layer_names) * g.kv_cache_spec.page_size_bytes for g in draft)
        target_page = paired_sfa_pool_bytes(target)
        target_required, draft_required = _pool_demand(config, target), _pool_demand(config, draft)
        if config.cache_config.num_gpu_blocks_override is not None:
            # The override sets an identical ID count in both pools.
            return max(target_required, draft_required) * (target_page + draft_page)
        return target_required * target_page + draft_required * draft_page
    return _pool_demand(config, groups) * paired_sfa_pool_bytes(groups)
