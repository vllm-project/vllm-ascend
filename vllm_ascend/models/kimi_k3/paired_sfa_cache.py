# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired MLA/indexer ownership with native, independently sized A5 pages.

Each full-history block owns both components. Recurrent groups use disjoint
IDs from the same pool and may alias main-page physical slots. Token ownership
never depends on the relative byte sizes of the two attention components.
DSpark shares full-history block IDs, with a separate region in the backing.
"""

from dataclasses import replace

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.models.kimi_k3.cache_config import make_kimi_cache_groups
from vllm_ascend.utils import vllm_version_is

# vLLM main moved this helper to kv_cache_planning after v0.32.0.
if vllm_version_is("0.32.0"):
    from vllm.v1.core.kv_cache_utils import may_override_num_blocks
else:
    from vllm.v1.core.kv_cache_planning import may_override_num_blocks


def _draft_layer_specs(groups):
    """Return local draft specs; empty PP-local groups allocate no tensors."""
    result = {}
    for group in groups:
        group_spec = group.kv_cache_spec
        for name in group.layer_names:
            spec = group_spec.kv_cache_specs[name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
            if spec.is_draft:
                result[name] = spec
    return result


def _validate_paired_sfa_pages(specs) -> None:
    """Check the page geometry settled by NPUPlatform before spec creation."""
    main = [s for s in specs.values() if isinstance(s, AscendMLAAttentionSpec) and not s.is_draft]
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


def paired_sfa_cache_groups(config, specs):
    draft = {name: spec for name, spec in specs.items() if spec.is_draft}
    if not draft:
        return _paired_sfa_target_cache_groups(config, specs)
    target = {name: spec for name, spec in specs.items() if not spec.is_draft}
    groups = _paired_sfa_target_cache_groups(config, target)
    specs.update(target)
    if any(spec.uses_packed_sfa_main_cache for spec in draft.values()):
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
        n: spec for n, spec in specs.items() if isinstance(spec, AscendMLAAttentionSpec) and not spec.is_draft
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


def paired_sfa_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    _, main_mla_names, indexer_names, kda_groups, main_page_size, indexer_page_size, num_main_slots = _layout(
        kv_cache_groups
    )
    pool_bytes = paired_sfa_pool_bytes(kv_cache_groups)
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
    for name, spec in _draft_layer_specs(kv_cache_groups).items():
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


def paired_sfa_max_memory(config, groups):
    blocks = sum(
        (g.kv_cache_spec.max_memory_usage_bytes(config) + g.kv_cache_spec.page_size_bytes - 1)
        // g.kv_cache_spec.page_size_bytes
        for g in groups
    )
    return blocks * paired_sfa_pool_bytes(groups)
