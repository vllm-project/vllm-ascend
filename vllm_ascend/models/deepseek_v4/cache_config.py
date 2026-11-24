# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4 component ownership and ordinary/shared physical cache geometry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec



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
            components.append((group_id, name, type(spec).__name__, asdict(spec), spec.page_size_bytes))
    payload = json.dumps(("dsv4-components-v2", components), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_dsv4_transfer_speculation(config) -> None:
    speculative = getattr(config, "speculative_config", None)
    if speculative is not None and speculative.method not in ("mtp", "dspark"):
        raise ValueError("DSV4 Mooncake transfer supports native MTP and DSpark speculation")


def build_deepseek_v4_reuse_config(config, groups, num_blocks):
    """Plan the same named component slots for the capacity probe and allocation."""
    if not is_deepseek_v4_cache(groups):
        return None
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
        LayerwiseCachePlanner,
        get_layerwise_offload_config,
    )

    extra = get_layerwise_offload_config(getattr(config, "kv_transfer_config", None))
    if extra is None or str(extra.get("backend", "mooncake")).lower() != "mooncake":
        return None
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

    specs = {name: group.kv_cache_spec.kv_cache_specs[name] for group in groups for name in group.layer_names}
    native = [
        KVCacheTensor(
            size=num_blocks * spec.page_size_bytes,
            layers=[name],
            offset=0,
            layer_stride=0,
            block_stride=spec.page_size_bytes,
        )
        for name, spec in specs.items()
    ]
    plan = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=native, kv_cache_groups=groups)
    plan.prefix_cache_retention_interval = getattr(config.cache_config, "prefix_cache_retention_interval", None)
    # DSV4 groups contain only attention history, window pages and compressor
    # state. The common planner excludes draft and independently resident layers.
    rules = {
        AscendMLAAttentionSpec: lambda spec: spec.model_version == "deepseek_v4",
        AscendSlidingWindowMLASpec: True,
    }
    planner = LayerwiseCachePlanner(plan, config, extra, spec_reuse_rules=rules)
    layout = planner.build()
    if layout.has_layer_reuse:
        # The DSV4 allocator uses an independent, device-aligned backing per
        # slot. Preserve the planner's alias relationships and page strides.
        plan.kv_cache_tensors = [
            KVCacheTensor(
                size=num_blocks * slot.block_stride,
                layers=slot.layers,
                offset=0,
                layer_stride=0,
                block_stride=slot.block_stride,
            )
            for slot in layout.kv_cache_tensors
        ]
        layout = replace(layout, kv_cache_tensors=tuple(plan.kv_cache_tensors))
    plan.layerwise_reuse_layout = layout
    plan.layerwise_kv_cache_reuse = layout.has_layer_reuse
    plan.layerwise_kv_cache_scratch = layout.has_layer_reuse
    plan.layerwise_store_verified_prefix = layout.has_layer_reuse
    plan.layerwise_draft_layer_names = tuple(sorted(planner.draft_names))
    plan.dsv4_resident_cache_names = tuple(sorted(specs.keys() - layout.reused_cache_names))
    plan.dsv4_shared_cache_names = tuple(sorted(layout.reused_cache_names))
    plan.dsv4_window_group_ids = tuple(
        group_id
        for group_id, group in enumerate(groups)
        if all(getattr(specs[name], "sliding_window", None) is not None for name in group.layer_names)
    )
    plan.dsv4_nominal_windows = {
        group_id: max(
            specs[name].sliding_window + (getattr(specs[name], "extra_retained_tokens", 0) or 0)
            for name in groups[group_id].layer_names
        )
        for group_id in plan.dsv4_window_group_ids
    }
    if getattr(config, "speculative_config", None) is None:
        plan.dsv4_continuation_windows = plan.dsv4_nominal_windows
    plan.dsv4_component_identity = component_layout_identity(groups)
    plan.kv_cache_allocation_alignment = 2 * 1024 * 1024
    plan.kv_transfer_config = config.kv_transfer_config
    return plan
