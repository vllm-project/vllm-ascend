# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired MLA/indexer ownership with native, independently sized A5 pages.

Each full-history block owns both components. Recurrent groups use disjoint
IDs from the same pool and may alias main-page physical slots. Token ownership
never depends on the relative byte sizes of the two attention components.
DSpark shares full-history block IDs, with separate physical draft tensors.
"""

from dataclasses import replace

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.kv_cache_utils import create_kv_cache_group_specs, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec


def is_paired_sfa_cache(specs) -> bool:
    """Recognize uncompressed SFA/recurrent hybrids, including PP projection.

    Pure DSA models and GLM's compressed indexer/tail retain their planners.
    Retained group specs allow recognition on a PP worker with no local MLA.
    """
    values = (
        list(specs.values())
        if isinstance(specs, dict)
        else [
            inner
            for group in specs
            for inner in (
                group.kv_cache_spec.kv_cache_specs.values()
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else (group.kv_cache_spec,)
            )
        ]
    )
    return (
        any(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in values)
        and any(isinstance(spec, AscendMLAAttentionSpec) for spec in values)
        and any(isinstance(spec, MambaSpec) for spec in values)
    )


def _is_dspark_draft_spec(spec) -> bool:
    return isinstance(spec, AscendMLAAttentionSpec) and spec.is_dspark_draft


def _draft_layer_specs(groups):
    """Return local draft specs; empty PP-local groups allocate no tensors."""
    result = {}
    for group in groups:
        group_spec = group.kv_cache_spec
        for name in group.layer_names:
            spec = group_spec.kv_cache_specs[name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
            if _is_dspark_draft_spec(spec):
                result[name] = spec
    return result


def align_paired_sfa_cache(specs) -> None:
    """Pad recurrent states to main pages without changing token block sizes."""
    main = [s for s in specs.values() if isinstance(s, AscendMLAAttentionSpec) and not _is_dspark_draft_spec(s)]
    recurrent = [s for s in specs.values() if isinstance(s, MambaSpec)]
    if not main:
        raise ValueError("A paired SFA cache requires main MLA specs.")
    page = max(s.real_page_size_bytes for s in main)
    if any(s.real_page_size_bytes != page for s in main):
        raise ValueError("Paired SFA main pages must have identical geometry.")
    if any(replace(s, page_size_padded=None).page_size_bytes > page for s in recurrent):
        raise ValueError("The MLA token block must cover one recurrent checkpoint.")
    for name, spec in list(specs.items()):
        if isinstance(spec, MambaSpec):
            specs[name] = replace(spec, page_size_padded=page)


def paired_sfa_cache_groups(config, specs):
    draft = {name: spec for name, spec in specs.items() if _is_dspark_draft_spec(spec)}
    if not draft:
        return _paired_sfa_target_cache_groups(config, specs)
    target = {name: spec for name, spec in specs.items() if not _is_dspark_draft_spec(spec)}
    groups = _paired_sfa_target_cache_groups(config, target)
    specs.update(target)
    if any(spec.uses_packed_sfa_main_cache for spec in draft.values()):
        raise ValueError("Kimi DSpark draft groups require ordinary MLA cache specs.")
    ordered = dict(sorted(draft.items(), key=lambda item: extract_layer_index(item[0])))

    # Share logical ownership, not the MLA spec or its backing allocation.
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
    align_paired_sfa_cache(specs)
    main = sorted((n for n, s in specs.items() if isinstance(s, AscendMLAAttentionSpec)), key=extract_layer_index)
    indexer = sorted((n for n, s in specs.items() if isinstance(s, AscendSFAIndexerCacheSpec)), key=extract_layer_index)
    if not main or [extract_layer_index(n) for n in main] != [extract_layer_index(n) for n in indexer]:
        raise ValueError("Every sparse MLA layer must own one matching indexer cache.")
    paired = {n: specs[n] for pair in zip(main, indexer) for n in pair}
    if len({s.block_size for s in paired.values()}) != 1:
        raise ValueError("Paired MLA and indexer must use the same local token block size.")
    uniform = UniformTypeKVCacheSpecs.from_specs(paired)
    if uniform is None:
        raise ValueError("Paired SFA components must share full-attention block semantics.")
    groups = [KVCacheGroupSpec(list(paired), uniform)]
    recurrent = {n: s for n, s in specs.items() if isinstance(s, MambaSpec)}
    if len(paired) + len(recurrent) != len(specs):
        raise ValueError("Unexpected cache role in paired SFA layout.")
    if recurrent:
        # Limit each recurrent group to one physical main-page slot per MLA
        # layer. Consecutive KDA states remain independent scheduler groups.
        names = sorted(recurrent, key=extract_layer_index)
        count = (len(names) + len(main) - 1) // len(main)
        groups.extend(create_kv_cache_group_specs(recurrent, [names[i::count] for i in range(count)]))
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
        if isinstance(spec, AscendMLAAttentionSpec) and not _is_dspark_draft_spec(spec)
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
    # Every pool ID still reserves independent target/indexer/draft storage.
    draft_bytes = sum(spec.page_size_bytes for spec in _draft_layer_specs(groups).values())
    return slots * main_page + len(indexer) * indexer_page + draft_bytes


def paired_sfa_cache_config(config, groups, available_memory):
    _, main, indexer, recurrent, main_page, indexer_page, slots = _layout(groups)
    blocks = may_override_num_blocks(config, max(available_memory // paired_sfa_pool_bytes(groups), 0))
    tensors = []
    for slot in range(slots):
        names = main[slot : slot + 1]
        names += [g.layer_names[slot] for g in recurrent if slot < len(g.layer_names)]
        tensors.append(
            KVCacheTensor(size=main_page * blocks, layers=names, offset=0, layer_stride=0, block_stride=main_page)
        )
    for name in indexer:
        tensors.append(
            KVCacheTensor(
                size=indexer_page * blocks, layers=[name], offset=0, layer_stride=0, block_stride=indexer_page
            )
        )
    # Shared block IDs never alias draft tensors with target or KDA tensors.
    for name, spec in _draft_layer_specs(groups).items():
        tensors.append(
            KVCacheTensor(
                size=spec.page_size_bytes * blocks,
                layers=[name],
                offset=0,
                layer_stride=0,
                block_stride=spec.page_size_bytes,
            )
        )
    return KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
        prefix_cache_retention_interval=config.cache_config.prefix_cache_retention_interval,
    )


def paired_sfa_max_memory(config, groups):
    blocks = sum(
        (g.kv_cache_spec.max_memory_usage_bytes(config) + g.kv_cache_spec.page_size_bytes - 1)
        // g.kv_cache_spec.page_size_bytes
        for g in groups
    )
    return blocks * paired_sfa_pool_bytes(groups)
