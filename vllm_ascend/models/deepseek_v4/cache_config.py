# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4 component ownership and ordinary/shared physical cache geometry."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict

from vllm.logger import logger
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LayerwiseLayerCacheSpecs,
    LayerwiseReuseLayout,
    NamedKVCacheSpec,
    build_layerwise_cache_layout,
    get_layerwise_reuse_config,
    get_physical_layer_index,
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


def is_shared_compressor_cache(name: str) -> bool:
    # The attention history is registered as .self_attn; indexer histories and
    # both compressors' .state_cache entries must remain resident.
    return name.endswith((".self_attn", ".attn"))


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
    """Separate resident allocations from the two compressed-history rings.

    Splitting the ordinary cross-group overlay is deliberate: a resident view
    must never pin or overlap a scratch slot. Every descriptor below owns its
    allocation; capacity probes and device initialization use these same bytes.
    """
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
    base = config.model_config.get_num_layers(parallel)
    options = build_layerwise_cache_layout(base, extra)
    specs = {name: group.kv_cache_spec.kv_cache_specs[name] for group in groups for name in group.layer_names}
    per_layer = defaultdict(list)
    for name, spec in specs.items():
        per_layer[get_physical_layer_index(name, base)].append(NamedKVCacheSpec(name, spec))
    if not set(range(base)).issubset(per_layer):
        raise ValueError("DSV4 component plan must cover every target layer")
    layer_specs = {}
    resident_layers = set(options.independent_layers)
    buckets = []
    resident_names = []
    for layer, components in sorted(per_layer.items()):
        if layer >= base:
            # Draft KV remains resident, including across prefill chunks.
            resident_names.extend(item.layer_name for item in components)
            continue
        mains = [item for item in components if is_shared_compressor_cache(item.layer_name)]
        if len(mains) > 1:
            raise ValueError("DSV4 layer has multiple compressor histories")
        main = mains[0] if mains else components[0]
        layer_specs[layer] = LayerwiseLayerCacheSpecs(
            main, extra_main_specs=tuple(item for item in components if item is not main)
        )
        if not mains:
            resident_layers.add(layer)
        resident_names.extend(
            item.layer_name for item in components if item is not main or not mains or layer in resident_layers
        )
        if mains and layer not in resident_layers:
            for signature, layers in buckets:
                if main.spec == signature:
                    layers.append(layer)
                    break
            else:
                buckets.append((main.spec, [layer]))
    slots = [(layer,) for layer in sorted(resident_layers)]
    predecessors = {}
    for _, layers in buckets:
        count = min(options.num_shared_buffers, len(layers))
        for index in range(count):
            slot = tuple(layers[index::count])
            slots.append(slot)
            predecessors.update(zip(slot[1:], slot[:-1]))
    layout = LayerwiseReuseLayout(
        layer_specs,
        tuple(slots),
        predecessors,
        sorted(resident_layers),
        options.num_prefetch_layers,
        bool(predecessors),
    )
    tensors = []
    covered = set()
    for slot in slots:
        names = [layer_specs[layer].main.layer_name for layer in slot]
        page_sizes = {specs[name].page_size_bytes for name in names}
        if len(page_sizes) != 1:
            raise ValueError("Incompatible DSV4 compressed history slot")
        size = page_sizes.pop()
        tensors.append(KVCacheTensor(size=num_blocks * size, layers=names, offset=0, layer_stride=0, block_stride=size))
        covered.update(names)
    for name in specs:
        if name not in covered:
            size = specs[name].page_size_bytes
            tensors.append(
                KVCacheTensor(size=num_blocks * size, layers=[name], offset=0, layer_stride=0, block_stride=size)
            )
    plan = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)
    plan.prefix_cache_retention_interval = getattr(config.cache_config, "prefix_cache_retention_interval", None)
    plan.layerwise_reuse_layout = layout
    # Already final, including the all-independent control configuration.
    plan.layerwise_kv_cache_reuse = layout.has_layer_reuse
    plan.layerwise_kv_cache_scratch = layout.has_layer_reuse
    plan.layerwise_store_verified_prefix = layout.has_layer_reuse
    plan.dsv4_resident_cache_names = tuple(sorted(set(resident_names)))
    plan.dsv4_shared_cache_names = tuple(
        sorted(layer_specs[layer].main.layer_name for slot in slots if len(slot) > 1 for layer in slot)
    )
    plan.dsv4_component_identity = component_layout_identity(groups)
    plan.kv_cache_allocation_alignment = 2 * 1024 * 1024
    plan.kv_transfer_config = config.kv_transfer_config
    logger.info(
        "DSV4_OFFLOAD plan blocks=%d bytes=%d descriptors=%d predecessors=%s resident_layers=%s identity=%s",
        num_blocks,
        sum(t.size for t in tensors),
        len(tensors),
        predecessors,
        sorted(resident_layers),
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
