# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Layer-outer KV placement for GLM-MoE-DSA with a DFlash draft."""

from collections import defaultdict

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.ascend_config import KVPPConfig
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec


def applies_to(config: VllmConfig, layout: str | None = None) -> bool:
    text_config = getattr(getattr(config, "model_config", None), "hf_text_config", None)
    if layout is None:
        layout = getattr(getattr(config, "cache_config", None), "kv_cache_layout", None)
    return (
        layout in ("HND", "LBHNC")
        and getattr(text_config, "model_type", None) == "glm_moe_dsa"
        and getattr(getattr(config, "speculative_config", None), "method", None) == "dflash"
        and not getattr(getattr(config, "scheduler_config", None), "disable_hybrid_kv_cache_manager", False)
    )


def is_target_spec(spec: KVCacheSpec) -> bool:
    return isinstance(spec, (AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec)) and not spec.is_draft


def build_groups(config: VllmConfig, specs: dict[str, KVCacheSpec]) -> list[KVCacheGroupSpec]:
    target = {}
    draft = {}
    uses_kvpp = KVPPConfig.from_vllm_config(config).size > 1
    for name, spec in specs.items():
        if is_target_spec(spec):
            target[name] = spec
        # KVPP can promote a windowed draft to full allocation on the producer.
        elif spec.is_draft and (isinstance(spec, SlidingWindowSpec) or (uses_kvpp and type(spec) is FullAttentionSpec)):
            draft[name] = spec
        else:
            raise ValueError(f"Unexpected GLM DFlash HND cache spec for {name}: {type(spec).__name__}")
        if getattr(spec, "store_on_host", False) or spec.page_size_bytes % spec.get_block_stride_alignment():
            raise ValueError(f"GLM DFlash HND requires device-resident pages with an aligned stride for {name}")

    # The scheduler unwraps a uniform group to its first spec. Prefer the
    # main MLA layer over an indexer even when the indexer registered first.
    target = dict(sorted(target.items(), key=lambda item: isinstance(item[1], AscendSFAIndexerCacheSpec)))
    if uses_kvpp and target and draft and all(type(spec) is FullAttentionSpec for spec in draft.values()):
        # KVPP promotes draft pages to full allocation. Compatible full layers
        # share scheduler block IDs while retaining independent physical pages.
        combined = {**target, **draft}
        uniform = UniformTypeKVCacheSpecs.from_specs(combined)
        if uniform is not None:
            uniform.max_num_blocks_per_req(config, config.model_config.max_model_len)
            return [KVCacheGroupSpec(list(combined), uniform)]
    groups = []
    for layers in (target, draft):
        if not layers:
            continue
        uniform = UniformTypeKVCacheSpecs.from_specs(layers)
        if uniform is None:
            raise ValueError("GLM DFlash HND group members must have compatible attention and block-table semantics")
        uniform.max_num_blocks_per_req(config, config.model_config.max_model_len)
        groups.append(KVCacheGroupSpec(list(layers), uniform))
    return groups


def layers_by_spec(groups: list[KVCacheGroupSpec]) -> dict[KVCacheSpec, list[str]]:
    """Use only PP-local names; empty projected groups still retain their global spec."""
    regions = defaultdict(list)
    for group in groups:
        uniform = group.kv_cache_spec
        assert isinstance(uniform, UniformTypeKVCacheSpecs)
        for name in group.layer_names:
            regions[uniform.kv_cache_specs[name]].append(name)
    return dict(regions)


def bytes_per_block(regions: dict[KVCacheSpec, list[str]]) -> int:
    return sum(spec.page_size_bytes * len(names) for spec, names in regions.items())


def build_cache_config(config: VllmConfig, groups: list[KVCacheGroupSpec], num_blocks: int) -> KVCacheConfig:
    regions = layers_by_spec(groups)
    size = num_blocks * bytes_per_block(regions)
    tensors = []
    offset = 0
    for spec, names in regions.items():
        # Packed MLA stores one latent per token regardless of its query head count.
        page = spec.page_size_bytes
        layer_stride = num_blocks * page
        tensors.append(
            KVCacheTensor(size=size, layers=names, offset=offset, block_stride=page, layer_stride=layer_stride)
        )
        offset += len(names) * layer_stride
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=tensors,
        prefix_cache_retention_interval=config.cache_config.prefix_cache_retention_interval,
    )
