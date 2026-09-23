# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec, UniformTypeKVCacheSpecs


def get_310p_shared_cache_slots(kv_cache_groups, layout) -> dict[str, int]:
    """Map layers to slots when uniform Mamba groups can share storage."""
    if not (layout.is_layer_compact and layout.is_block_compact) or not kv_cache_groups:
        return {}
    group_size = len(kv_cache_groups[0].layer_names)
    if not group_size or any(len(group.layer_names) != group_size for group in kv_cache_groups):
        return {}

    page_sizes = set()
    attention_group_count = 0
    mamba_spec = None
    shared_slots: dict[str, int] = {}
    for group in kv_cache_groups:
        has_attention = False
        for slot, layer_name in enumerate(group.layer_names):
            group_spec = group.kv_cache_spec
            spec = (
                group_spec.kv_cache_specs[layer_name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
            )
            page_sizes.add(spec.page_size_bytes)
            if isinstance(spec, AttentionSpec):
                has_attention = True
            elif isinstance(spec, MambaSpec):
                if mamba_spec is not None and spec != mamba_spec:
                    return {}
                mamba_spec = spec
                shared_slots[layer_name] = slot
            else:
                return {}
        attention_group_count += has_attention

    if len(page_sizes) != 1 or attention_group_count != 1 or mamba_spec is None:
        return {}
    return shared_slots
