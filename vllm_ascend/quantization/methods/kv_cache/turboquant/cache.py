# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack DeepSeek C4 TQ pages using upstream byte-offset cache descriptors.

Each physical slot has the same page stride for every group that shares it.
Block IDs can therefore never alias a different block ID in another group.
"""

import math

from vllm.logger import logger
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from . import SLOT_BYTES, TURBOQUANT_CACHE_DTYPE

# PA attention's fast copy path requires aligned page starts. Preserve the
# integral compact-row stride used by the scatter update path as well.
PHYSICAL_STRIDE_ALIGNMENT = 256


def uses_turboquant_groups(groups):
    # A pipeline rank may own only C128/SWA layers. Its grouping and physical
    # allocation must still use the same TQ planner as ranks containing C4.
    return any(
        getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE
        for group in groups
        for spec in (
            group.kv_cache_spec.kv_cache_specs.values()
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else (group.kv_cache_spec,)
        )
    )


def group_specs(grouped_specs):
    # Keep C4 and its indexer in one metadata group. Preserve all per-layer
    # specs, including state padding; only C4 values are quantized.
    largest_page = max(spec.page_size_bytes for group in grouped_specs for spec in group.kv_cache_specs.values())
    c4_group_bytes = [
        group.page_size_bytes
        for group in grouped_specs
        if any(
            getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE and spec.head_size == SLOT_BYTES
            for spec in group.kv_cache_specs.values()
        )
    ]
    budget = round_up(max(c4_group_bytes, default=largest_page), largest_page)
    groups = []
    for group in grouped_specs:
        specs = group.kv_cache_specs
        chunk = {}
        used = 0
        for name, spec in specs.items():
            if chunk and used + spec.page_size_bytes > budget:
                merged = UniformTypeKVCacheSpecs.from_specs(chunk)
                assert merged is not None
                groups.append(KVCacheGroupSpec(layer_names=list(chunk), kv_cache_spec=merged))
                chunk, used = {}, 0
            chunk[name] = spec
            used += spec.page_size_bytes
        if chunk:
            merged = UniformTypeKVCacheSpecs.from_specs(chunk)
            assert merged is not None
            groups.append(KVCacheGroupSpec(layer_names=list(chunk), kv_cache_spec=merged))
    return groups


def _alignment(spec):
    alignment = get_dtype_size(spec.dtype)
    if getattr(spec, "scale_dim", 0):
        alignment = math.lcm(alignment, get_dtype_size(spec.scale_dtype))
    if getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE and spec.head_size == SLOT_BYTES:
        # The fused kernel computes the physical row stride from the page
        # stride. Preserve an integral number of compact 258-byte rows.
        alignment = math.lcm(alignment, SLOT_BYTES)
    return alignment


def packed_layout(groups):
    specs = {name: spec for group in groups for name, spec in group.kv_cache_spec.kv_cache_specs.items()}
    page_alignment = math.lcm(PHYSICAL_STRIDE_ALIGNMENT, *(_alignment(spec) for spec in specs.values()))
    # A block ID belongs to one group at a time. Overlay complete groups
    # with a common stride to avoid fragmentation between small-page bins.
    offsets = {}
    page_size = 0
    for group in groups:
        used = 0
        for name in sorted(group.layer_names, key=lambda name: -specs[name].page_size_bytes):
            spec = specs[name]
            offset = round_up(used, _alignment(spec))
            offsets[name] = offset
            used = offset + spec.page_size_bytes
        page_size = max(page_size, used)
    return [(round_up(page_size, page_alignment), offsets)]


def pool_bytes_per_block(groups):
    return sum(size for size, _ in packed_layout(groups))


def max_memory_usage(vllm_config, groups):
    request_blocks = sum(group.kv_cache_spec.max_memory_usage_pages(vllm_config) for group in groups)
    return pool_bytes_per_block(groups) * request_blocks


def cache_config(vllm_config, groups, available_memory):
    layout = packed_layout(groups)
    page_bytes = sum(size for size, _ in layout)
    num_blocks = may_override_num_blocks(vllm_config, available_memory // page_bytes)
    logger.info(
        "DeepSeek V4 TurboQuant KV: C4 slot=%d bytes, pool bytes/block=%d, groups=%d, blocks=%d",
        SLOT_BYTES,
        page_bytes,
        len(groups),
        num_blocks,
    )
    backing_size = num_blocks * page_bytes
    descriptors = []
    slot_base = 0
    for stride, offsets in layout:
        for name, offset in offsets.items():
            descriptors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=[name],
                    offset=slot_base + offset,
                    layer_stride=0,
                    block_stride=stride,
                )
            )
        slot_base += stride * num_blocks
    return num_blocks, descriptors
