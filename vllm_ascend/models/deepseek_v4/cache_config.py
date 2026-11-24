# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4 component ownership and ordinary/shared physical cache geometry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from types import MappingProxyType

from vllm.v1.kv_cache_interface import KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec


DEEPSEEK_V4_SPEC_REUSE_RULES = MappingProxyType(
    {
        AscendMLAAttentionSpec: lambda spec: spec.model_version == "deepseek_v4",
        AscendSlidingWindowMLASpec: True,
    }
)


def is_deepseek_v4_cache(groups_or_specs) -> bool:
    specs = (
        groups_or_specs.values()
        if isinstance(groups_or_specs, dict)
        else (
            spec
            for group in groups_or_specs
            for spec in (
                group.kv_cache_spec.kv_cache_specs.values()
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else (group.kv_cache_spec,)
            )
        )
    )
    return any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs)


def component_layout_identity(groups) -> str:
    components = []
    for group_id, group in enumerate(groups):
        if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            raise ValueError("DSV4 layout identity requires unprojected component specs")
        for name in sorted(group.layer_names):
            spec = group.kv_cache_spec.kv_cache_specs[name]
            spec_fields = asdict(spec)
            # The view-selection policy does not change existing object bytes.
            spec_fields.pop("transfer_separate_page_views", None)
            components.append((group_id, name, type(spec).__name__, spec_fields, spec.page_size_bytes))
    payload = json.dumps(("dsv4-components-v2", components), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_dsv4_transfer_speculation(config) -> None:
    speculative = getattr(config, "speculative_config", None)
    if speculative is not None and speculative.method not in ("mtp", "dspark"):
        raise ValueError("DSV4 Mooncake transfer supports native MTP and DSpark speculation")


def prepare_deepseek_v4_reuse_config(config, cache_config):
    """Expand native KV tensors into descriptors for individual components."""
    parallel = config.parallel_config
    if any(
        getattr(parallel, field, 1) != 1
        for field in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        raise ValueError("DSV4 Mooncake temporal offload requires PP=PCP=DCP=1")
    validate_dsv4_transfer_speculation(config)
    if not config.model_config.enforce_eager:
        raise ValueError("DSV4 Mooncake temporal offload requires enforce_eager")
    if getattr(config.cache_config, "cache_dtype", None) == "turboquant":
        raise ValueError("DSV4 Mooncake temporal offload does not support TurboQuant")

    tensors = []
    for group in cache_config.kv_cache_groups:
        for name in group.layer_names:
            page_size = group.kv_cache_spec.kv_cache_specs[name].page_size_bytes
            tensors.append(
                KVCacheTensor(
                    size=cache_config.num_blocks * page_size,
                    layers=[name],
                    offset=0,
                    layer_stride=0,
                    block_stride=page_size,
                )
            )
    return replace(cache_config, kv_cache_tensors=tensors)


def finalize_deepseek_v4_reuse_layout(config, cache_config, layout):
    """Use one backing per slot and attach the window and transfer metadata."""
    groups = cache_config.kv_cache_groups
    num_blocks = cache_config.num_blocks
    if layout.has_layer_reuse:
        # The DSV4 allocator uses an independent, device-aligned backing per
        # slot. Preserve the planner's alias relationships and page strides.
        tensors = tuple(
            KVCacheTensor(
                size=num_blocks * slot.block_stride,
                layers=slot.layers,
                offset=0,
                layer_stride=0,
                block_stride=slot.block_stride,
            )
            for slot in layout.kv_cache_tensors
        )
        layout = replace(layout, kv_cache_tensors=tensors, separate_slot_backings=True)
    cache_config.dsv4_component_identity = component_layout_identity(groups)
    cache_config.kv_cache_allocation_alignment = 2 * 1024 * 1024
    return layout
