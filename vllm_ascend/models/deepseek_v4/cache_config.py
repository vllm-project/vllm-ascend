# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4 component ownership and ordinary/shared physical cache geometry."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, replace

from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import LayerwiseCachePlanner
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    get_layerwise_reuse_config,
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
            components.append((group_id, name, type(spec).__name__, asdict(spec), spec.page_size_bytes))
    return hashlib.sha256(json.dumps(components, sort_keys=True, default=str).encode()).hexdigest()


def validate_dsv4_transfer_speculation(config) -> None:
    speculative = getattr(config, "speculative_config", None)
    if speculative is not None and speculative.method not in ("mtp", "dspark"):
        raise ValueError("DSV4 Mooncake transfer supports native MTP and DSpark speculation")


def build_deepseek_v4_reuse_config(config, groups, num_blocks):
    """Plan the same named component slots for the capacity probe and allocation."""
    if not is_deepseek_v4_cache(groups):
        return None
    extra = get_layerwise_reuse_config(getattr(config, "kv_transfer_config", None))
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
    rules = dict.fromkeys(map(type, specs.values()), True)
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
    plan.dsv4_component_identity = component_layout_identity(groups)
    plan.kv_cache_allocation_alignment = 2 * 1024 * 1024
    plan.kv_transfer_config = config.kv_transfer_config
    logger.info(
        "DSV4_OFFLOAD plan blocks=%d bytes=%d descriptors=%d predecessors=%s resident_layers=%s identity=%s",
        num_blocks,
        sum(t.size for t in plan.kv_cache_tensors),
        len(plan.kv_cache_tensors),
        layout.prefetch_layer_map,
        layout.independent_layers,
        plan.dsv4_component_identity,
    )
    return plan


def _get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[list[int], list[dict[int, list[str]]], list[str], int, int]:
    """Return the geometry shared by DSV4 planning and memory accounting.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream so every layer's page_size matches one of the
    full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each layer
    at tuple_idx = position-within-bucket.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted({s.page_size_bytes for s in full_mla_spec.kv_cache_specs.values()})

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            padded = getattr(specs[name], "page_size_padded", None)
            page_size = padded if padded is not None else specs[name].page_size_bytes
            if "mtp" not in name:
                b[page_size].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = page_size
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    return page_sizes, bucketed, mtp_layer_names, mtp_page_size, num_layer_tuples
