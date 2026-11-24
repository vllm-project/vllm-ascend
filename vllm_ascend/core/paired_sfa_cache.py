# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired MLA/indexer ownership with native, independently sized A5 pages.

Each full-history block owns both components. Recurrent groups use disjoint
IDs from the same pool and may alias main-page physical slots. Token ownership
never depends on the relative byte sizes of the two attention components.
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


def align_paired_sfa_cache(specs) -> None:
    """Pad recurrent states to main pages without changing token block sizes."""
    main = [s for s in specs.values() if isinstance(s, AscendMLAAttentionSpec)]
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
    full = next(g for g in groups if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs))
    specs = full.kv_cache_spec.kv_cache_specs
    main = [n for n in full.layer_names if isinstance(specs[n], AscendMLAAttentionSpec)]
    indexer = [n for n in full.layer_names if isinstance(specs[n], AscendSFAIndexerCacheSpec)]
    recurrent = [g for g in groups if isinstance(g.kv_cache_spec, MambaSpec)]
    main_page = next(s.page_size_bytes for s in specs.values() if isinstance(s, AscendMLAAttentionSpec))
    indexer_page = next(s.page_size_bytes for s in specs.values() if isinstance(s, AscendSFAIndexerCacheSpec))
    slots = max([len(main), *(len(g.layer_names) for g in recurrent)])
    return full, main, indexer, recurrent, main_page, indexer_page, slots


def paired_sfa_pool_bytes(groups):
    _, _, indexer, _, main_page, indexer_page, slots = _layout(groups)
    return slots * main_page + len(indexer) * indexer_page


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
    return KVCacheConfig(num_blocks=blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)


def paired_sfa_max_memory(config, groups):
    blocks = sum(
        (g.kv_cache_spec.max_memory_usage_bytes(config) + g.kv_cache_spec.page_size_bytes - 1)
        // g.kv_cache_spec.page_size_bytes
        for g in groups
    )
    return blocks * paired_sfa_pool_bytes(groups)
