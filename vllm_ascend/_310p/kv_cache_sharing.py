# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec, UniformTypeKVCacheSpecs


def get_310p_shared_cache_slots(kv_cache_groups, layout) -> dict[str, int]:
    """Identify uniform hybrid groups whose Mamba pages can share storage."""
    if not (layout.is_layer_compact and layout.is_block_compact) or len(kv_cache_groups) < 2:
        return {}
    group_sizes = {len(group.layer_names) for group in kv_cache_groups}
    if len(group_sizes) != 1 or 0 in group_sizes:
        return {}

    page_sizes = set()
    attention_groups = set()
    attention_specs = []
    mamba_specs = []
    for group_id, group in enumerate(kv_cache_groups):
        for layer_name in group.layer_names:
            group_spec = group.kv_cache_spec
            spec = (
                group_spec.kv_cache_specs[layer_name] if isinstance(group_spec, UniformTypeKVCacheSpecs) else group_spec
            )
            page_sizes.add(spec.page_size_bytes)
            if isinstance(spec, AttentionSpec):
                attention_groups.add(group_id)
                attention_specs.append(spec)
            elif isinstance(spec, MambaSpec):
                mamba_specs.append(spec)
            else:
                return {}
    if (
        len(page_sizes) != 1
        or len(attention_groups) != 1
        or not mamba_specs
        or any(spec != attention_specs[0] for spec in attention_specs[1:])
        or any(spec != mamba_specs[0] for spec in mamba_specs[1:])
    ):
        return {}
    return {layer_name: slot for group in kv_cache_groups for slot, layer_name in enumerate(group.layer_names)}
