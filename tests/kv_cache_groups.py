# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct cache planner inputs with each supported vLLM interface."""

from dataclasses import replace

from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec
from vllm_ascend.core.kv_cache_layers import CacheGroup
from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.33.0"):
    from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

    ring_spec_type = ascend_ring_spec_type = AttentionSpec
    full_attention_spec_type = AttentionSpec
    window_mla_spec_type = MLAAttentionSpec
else:
    from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec

    from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec, AscendSlidingWindowMLASpec

    ring_spec_type = CircularBufferSpec
    ascend_ring_spec_type = AscendCircularBufferSpec
    full_attention_spec_type = FullAttentionSpec
    window_mla_spec_type = AscendSlidingWindowMLASpec


def window_requirement(*, sliding_window, extra_retained_tokens=0, **fields):
    if not vllm_version_is("0.33.0"):
        return SlidingWindowManager.layer_config(
            AttentionSpec(**fields), sliding_window=sliding_window, extra_retained_tokens=extra_retained_tokens
        )
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    return SlidingWindowSpec(sliding_window=sliding_window, extra_retained_tokens=extra_retained_tokens, **fields)


def mamba_requirement(*, mamba_cache_mode="none", num_speculative_blocks=0, **layout):
    policy = dict(mamba_cache_mode=mamba_cache_mode, num_speculative_blocks=num_speculative_blocks)
    if not vllm_version_is("0.33.0"):
        from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager

        return AscendMambaManager.layer_config(MambaSpec(**layout), **policy)
    return MambaSpec(**layout, **policy)


def ring_requirement(spec, *, zero_new_blocks=False):
    if vllm_version_is("0.33.0"):
        return spec
    if zero_new_blocks:
        from vllm_ascend.core.single_type_kv_cache_manager import AscendCircularBufferManager

        return AscendCircularBufferManager.layer_config(spec)
    return CircularBufferManager.layer_config(spec)


def layer_config(layout):
    from vllm.v1.kv_cache_interface import KVCacheLayerConfig

    if isinstance(layout, KVCacheLayerConfig):
        return layout
    if isinstance(layout, MambaSpec):
        return MambaManager.layer_config(layout)
    if isinstance(layout, AscendIndexerKPoolTailSpec):
        from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager

        return KpoolTailManager.layer_config(layout)
    return FullAttentionManager.layer_config(layout)


def cache_group(layer_names, kv_cache_spec, is_eagle_group=False, **group_fields):
    if vllm_version_is("0.33.0"):
        if isinstance(kv_cache_spec, dict):
            from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

            kv_cache_spec = UniformTypeKVCacheSpecs(
                block_size=next(iter(kv_cache_spec.values())).block_size, kv_cache_specs=kv_cache_spec
            )
        return CacheGroup(layer_names, kv_cache_spec, is_eagle_group=is_eagle_group, **group_fields)

    members = (
        kv_cache_spec if isinstance(kv_cache_spec, dict) else dict.fromkeys(layer_names or ["layer"], kv_cache_spec)
    )
    configs = {name: layer_config(layout) for name, layout in members.items()}
    group = CacheGroup.from_layers(configs, is_eagle_group=is_eagle_group, **group_fields)
    if list(configs) == layer_names:
        return group
    return replace(
        group, kv_cache_specs={n: configs[n].spec for n in layer_names}, unprojected_specs=group.distinct_specs
    )
