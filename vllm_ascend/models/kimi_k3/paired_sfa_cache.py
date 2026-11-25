# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired MLA/indexer ownership with native, independently sized A5 pages.

Each full-history block owns both components. Recurrent groups use disjoint
IDs from the same pool and may alias main-page physical slots. Token ownership
never depends on the relative byte sizes of the two attention components.
DSpark shares full-history block IDs when its sharding matches the target,
with a separate region in the backing for every draft layer.
"""

from collections.abc import Collection
from dataclasses import replace

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.kv_cache_planning import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheTensor,
    MambaSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec, KVCacheGroup
from vllm_ascend.models.kimi_k3.cache_config import make_kimi_cache_groups
from vllm_ascend.utils import vllm_version_is


def _entry_spec(entry):
    return entry if vllm_version_is("0.34.0") else entry.spec


def _is_target_main_layer(name: str, spec, draft_layer_names: Collection[str] | None = None) -> bool:
    if not isinstance(spec, AscendMLAAttentionSpec):
        return False
    if vllm_version_is("0.34.0"):
        return not spec.is_draft
    if draft_layer_names is not None:
        return name not in draft_layer_names
    return bool(spec.uses_packed_sfa_main_cache)


def _is_draft_layer(name: str, spec, draft_layer_names: Collection[str] | None = None) -> bool:
    if vllm_version_is("0.34.0"):
        return bool(spec.is_draft)
    if draft_layer_names is not None:
        return name in draft_layer_names
    return not isinstance(spec, (AscendSFAIndexerCacheSpec, MambaSpec)) and not _is_target_main_layer(name, spec)


def _needs_draft_tail_protection(config, draft_entries) -> bool:
    if vllm_version_is("0.34.0"):
        return any(getattr(spec, "non_causal_multi_token_decode", False) for spec in draft_entries.values())
    # vLLM #61452 removed non_causal_multi_token_decode from MLAAttentionSpec.
    # Single-process / UT runs expose it on static_forward_context layers, while
    # EngineCore (which does not instantiate model layers) resolves it from
    # speculative_config.draft_model_config.hf_config.
    compilation_config = getattr(config, "compilation_config", None)
    static_ctx = getattr(compilation_config, "static_forward_context", None) or {}
    if any(getattr(static_ctx.get(name), "non_causal_multi_token_decode", False) for name in draft_entries):
        return True
    spec_config = getattr(config, "speculative_config", None)
    if spec_config is not None and getattr(spec_config, "use_dspark", lambda: False)():
        draft_model_config = getattr(spec_config, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        if draft_hf_config is not None:
            dflash_config = getattr(draft_hf_config, "dflash_config", None)
            if isinstance(dflash_config, dict) and "causal" in dflash_config:
                return not bool(dflash_config["causal"])
            return not bool(getattr(draft_hf_config, "full_attention_causal", False))
    return False


def _draft_layer_specs(groups, draft_layer_names: Collection[str] | None = None):
    """Return local draft specs; empty PP-local groups allocate no tensors."""
    result = {}
    for group in groups:
        for name, spec in group.kv_cache_specs.items():
            if _is_draft_layer(name, spec, draft_layer_names):
                result[name] = spec
    return result


def _validate_paired_sfa_pages(entries) -> None:
    """Check the page geometry settled by NPUPlatform before spec creation."""
    main = [
        _entry_spec(entry)
        for entry in entries.values()
        if isinstance(_entry_spec(entry), AscendMLAAttentionSpec) and not entry.is_draft
    ]
    recurrent = [_entry_spec(entry) for entry in entries.values() if isinstance(_entry_spec(entry), MambaSpec)]
    if not main:
        raise ValueError("A paired SFA cache requires main MLA specs.")
    page = max(s.real_page_size_bytes for s in main)
    if any(s.real_page_size_bytes != page for s in main):
        raise ValueError("Paired SFA main pages must have identical geometry.")
    if any(replace(s, page_size_padded=None).page_size_bytes > page for s in recurrent):
        raise ValueError("The MLA token block must cover one recurrent checkpoint.")
    if any(s.page_size_bytes != page for s in recurrent):
        raise ValueError("Paired SFA recurrent pages must match the main MLA page size.")


def _attach_draft_groups_v034(config, groups, separate, ordered):
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

    full = groups[0]
    assert isinstance(full.kv_cache_spec, UniformTypeKVCacheSpecs)
    if separate:
        uniform_draft = UniformTypeKVCacheSpecs.from_specs(separate)
        if uniform_draft is None:
            raise ValueError("Draft layers must share full-attention semantics and logical block size.")
        groups.append(
            KVCacheGroupSpec(
                list(separate),
                uniform_draft,
                is_eagle_group=_needs_draft_tail_protection(config, separate),
            )
        )
    # UniformType retains per-layer dtype/layout/non-causal flags; the strict
    # AscendMLAAttentionSpec.merge must not merge these different MLA specs.
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
        is_eagle_group=full.is_eagle_group or _needs_draft_tail_protection(config, ordered),
    )
    return groups


def _attach_draft_groups_from_layer_configs(config, groups, separate, ordered):
    full = groups[0]
    if separate:
        first_sep = next(iter(separate.values()))
        if any(
            lc.manager_config != first_sep.manager_config
            or lc.spec.block_size != first_sep.spec.block_size
            or lc.spec.dcp_sharded != first_sep.spec.dcp_sharded
            for lc in separate.values()
        ):
            raise ValueError("Draft layers must share full-attention semantics and logical block size.")
        groups.append(
            KVCacheGroup.from_layers(
                separate,
                is_eagle_group=_needs_draft_tail_protection(config, separate),
            )
        )
    if any(
        lc.manager_config != full.manager_config or lc.spec.block_size != full.block_size for lc in ordered.values()
    ):
        raise ValueError("Target and DSpark must share full-attention semantics and logical block size.")
    shared_specs = {**full.kv_cache_specs, **{name: lc.spec for name, lc in ordered.items()}}
    if any(spec.tokens_per_state != 1 for spec in shared_specs.values()):
        raise ValueError("Shared DSpark ownership requires uncompressed full-history token slots.")
    max_len = config.model_config.max_model_len
    widths = {
        full.manager_cls.max_num_blocks_per_req(full.manager_config, spec.block_size, spec.dcp_sharded, config, max_len)
        for spec in shared_specs.values()
    }
    if len(widths) != 1:
        raise ValueError("Target and DSpark must have identical block-table coverage, including DCP sharding.")
    # Keep target first for scheduler conversion and keep recurrent group IDs.
    # One request now owns one full-history block sequence instead of two.
    # Retain drafter prefix-tail protection after scheduler spec unwrapping;
    # this flag does not change the target's causal attention metadata.
    groups[0] = replace(
        full,
        kv_cache_specs=shared_specs,
        is_eagle_group=full.is_eagle_group or _needs_draft_tail_protection(config, ordered),
    )
    return groups


def paired_sfa_cache_groups(config, specs):
    draft = {name: entry for name, entry in specs.items() if entry.is_draft}
    if not vllm_version_is("0.34.0"):
        config._ascend_draft_layer_names = frozenset(draft)
    if not draft:
        return _paired_sfa_target_cache_groups(config, specs)
    target = {name: entry for name, entry in specs.items() if not entry.is_draft}
    groups = _paired_sfa_target_cache_groups(config, target)
    specs.update(target)
    if any(
        isinstance(_entry_spec(entry), AscendMLAAttentionSpec) and _entry_spec(entry).uses_packed_sfa_main_cache
        for entry in draft.values()
    ):
        raise ValueError("Kimi DSpark draft groups require ordinary attention cache specs.")
    ordered = dict(sorted(draft.items(), key=lambda item: extract_layer_index(item[0])))

    # Share logical ownership while retaining separate physical regions.
    full = groups[0]
    # Replicated and DCP-sharded attention have different logical coverage.
    # Preserve their own block tables while retaining independent physical
    # regions in the paired backing allocation.
    separate = {name: entry for name, entry in ordered.items() if _entry_spec(entry).dcp_sharded != full.dcp_sharded}
    ordered = {name: entry for name, entry in ordered.items() if name not in separate}
    if vllm_version_is("0.34.0"):
        return _attach_draft_groups_v034(config, groups, separate, ordered)
    return _attach_draft_groups_from_layer_configs(config, groups, separate, ordered)


def _paired_sfa_target_cache_groups(config, specs):
    if config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("Paired SFA cache requires the hybrid KV cache manager.")
    _validate_paired_sfa_pages(specs)
    main = sorted(
        (n for n, s in specs.items() if isinstance(_entry_spec(s), AscendMLAAttentionSpec)),
        key=extract_layer_index,
    )
    indexer = sorted(
        (n for n, s in specs.items() if isinstance(_entry_spec(s), AscendSFAIndexerCacheSpec)),
        key=extract_layer_index,
    )
    if not main or [extract_layer_index(n) for n in main] != [extract_layer_index(n) for n in indexer]:
        raise ValueError("Every sparse MLA layer must own one matching indexer cache.")
    paired = {n: specs[n] for pair in zip(main, indexer) for n in pair}
    if len({_entry_spec(s).block_size for s in paired.values()}) != 1:
        raise ValueError("Paired MLA and indexer must use the same local token block size.")
    recurrent = {n: s for n, s in specs.items() if isinstance(_entry_spec(s), MambaSpec)}
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


def _layout(groups, draft_layer_names: Collection[str] | None = None):
    full = next(g for g in groups if any(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in g.distinct_specs))
    specs = full.kv_cache_specs
    # Draft layers share logical IDs but must not enlarge the target/KDA
    # overlay or inherit the target's packed page geometry. Retain global
    # specs via distinct_specs here: a PP-local group can have no local target MLA layer.
    main = [n for n in full.layer_names if _is_target_main_layer(n, specs[n], draft_layer_names)]
    indexer = [n for n in full.layer_names if isinstance(specs[n], AscendSFAIndexerCacheSpec)]
    recurrent = [g for g in groups if g.distinct_specs and isinstance(g.distinct_specs[0], MambaSpec)]
    if main:
        main_page = specs[main[0]].page_size_bytes
    elif vllm_version_is("0.34.0"):
        main_page = next(
            s.page_size_bytes for s in full.distinct_specs if isinstance(s, AscendMLAAttentionSpec) and not s.is_draft
        )
    else:
        # Paired SFA always inserts target main MLA first in groups[0], and
        # distinct_specs preserves that insertion order across PP projection.
        assert isinstance(full.distinct_specs[0], AscendMLAAttentionSpec)
        main_page = full.distinct_specs[0].page_size_bytes
    indexer_page = next(s.page_size_bytes for s in full.distinct_specs if isinstance(s, AscendSFAIndexerCacheSpec))
    slots = max([len(main), *(len(g.layer_names) for g in recurrent)])
    return full, main, indexer, recurrent, main_page, indexer_page, slots


def paired_sfa_pool_bytes(groups, *, draft_layer_names: Collection[str] | None = None):
    _, _, indexer, _, main_page, indexer_page, slots = _layout(groups, draft_layer_names=draft_layer_names)
    # Every pool ID still reserves independent target/indexer/draft storage.
    draft_bytes = sum(spec.page_size_bytes for spec in _draft_layer_specs(groups, draft_layer_names).values())
    return slots * main_page + len(indexer) * indexer_page + draft_bytes


def paired_sfa_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroup],
    available_memory: int,
) -> KVCacheConfig:
    draft_layer_names = getattr(vllm_config, "_ascend_draft_layer_names", None)
    _, main_mla_names, indexer_names, kda_groups, main_page_size, indexer_page_size, num_main_slots = _layout(
        kv_cache_groups,
        draft_layer_names=draft_layer_names,
    )
    pool_bytes = paired_sfa_pool_bytes(kv_cache_groups, draft_layer_names=draft_layer_names)
    num_blocks = may_override_num_blocks(vllm_config, available_memory // pool_bytes)
    backing_size = pool_bytes * num_blocks
    main_layer_size = main_page_size * num_blocks
    # Recurrent groups overlay the target MLA region. Their block IDs are
    # disjoint, while indexer and draft layers always own separate regions.
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
    offset += len(indexer_names) * indexer_layer_size
    for name, spec in _draft_layer_specs(kv_cache_groups, draft_layer_names).items():
        draft_layer_size = spec.page_size_bytes * num_blocks
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[name],
                offset=offset,
                layer_stride=draft_layer_size,
                block_stride=spec.page_size_bytes,
            )
        )
        offset += draft_layer_size
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )
