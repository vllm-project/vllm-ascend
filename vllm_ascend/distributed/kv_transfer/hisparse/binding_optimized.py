# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bind Ascend indexer views while retaining upstream HiSparse allocation."""

import math
from dataclasses import replace
from typing import Any

import torch
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec


def align_hisparse_indexer_specs(
    specs: dict[str, KVCacheSpec],
    *,
    packed_row_bytes: int | None = None,
) -> dict[str, KVCacheSpec]:
    """Declare dense indexer-page addressing to Core's shared-slab planner.

    Core includes this alignment when sizing each physical device block, before
    choosing the block count. Source MLA specs and their host layout stay intact.
    """
    if packed_row_bytes is not None and packed_row_bytes <= 0:
        raise ValueError("HiSparse packed row width must be positive")
    aligned = {}
    for name, spec in specs.items():
        alignment = spec.get_block_stride_alignment()
        updates = {}
        if isinstance(spec, AscendSFAIndexerCacheSpec):
            _validate_indexer_spec(spec)
            if spec.cache_sparse_li_c8:
                # K and scale occupy separate contiguous regions within each
                # page. Reserve both before Core computes the layer stride.
                updates["page_size_padded"] = max(spec.page_size_bytes, spec.page_size_padded or 0)
                alignment = math.lcm(alignment, spec.head_size, torch.float32.itemsize)
            else:
                alignment = math.lcm(alignment, spec.page_size_bytes)
        elif packed_row_bytes is not None and isinstance(spec, (HiSparseResidentSpec, HiSparseHotSpec)):
            # Core derives these synthetic specs without the source row alignment.
            # Declare it before shared-slab sizing, never round runtime addresses.
            alignment = math.lcm(alignment, packed_row_bytes)
        if alignment != spec.get_block_stride_alignment():
            updates["block_stride_alignment"] = alignment
        aligned[name] = replace(spec, **updates) if updates else spec
    return aligned


def get_hisparse_indexer_specs(
    kv_cache_config: KVCacheConfig,
) -> dict[str, AscendSFAIndexerCacheSpec]:
    """Validate the static indexer format before common storage allocation."""
    indexer_specs = {}
    for group in kv_cache_config.kv_cache_groups:
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if isinstance(spec, AscendSFAIndexerCacheSpec):
                _validate_indexer_spec(spec)
                indexer_specs[name] = spec

    return indexer_specs


def _validate_indexer_spec(spec: AscendSFAIndexerCacheSpec) -> None:
    if spec.cache_sparse_li_c4:
        raise ValueError("HiSparse indexer C4 is not supported by this adapter.")
    if spec.cache_sparse_li_c8:
        if not (
            spec.dtype == torch.float8_e4m3fn
            and spec.scale_dtype == torch.float32
            and spec.scale_dim == 1
            and spec.num_kv_heads == 1
            and spec.head_size == 128
            and spec.block_size == 128
            and spec.sfa_dcp_replicated_indexer_size == 1
        ):
            raise ValueError("HiSparse indexer C8 requires E4M3FN K128 with one FP32 scale per token, block128.")
    elif spec.scale_dim:
        raise ValueError("HiSparse indexer scales require an explicit supported quantization format.")


def _quantized_indexer_views(cache: torch.Tensor, spec: AscendSFAIndexerCacheSpec):
    """Alias the current Core allocation; retain no additional storage owner."""
    _validate_indexer_spec(spec)
    pages, heads, block, width = cache.shape
    stride = cache.stride(0)
    key_bytes = block * heads * width
    scale_bytes = block * heads * torch.float32.itemsize
    offset = cache.storage_offset()
    if (
        cache.dtype != torch.float8_e4m3fn
        or (heads, block, width) != (1, spec.block_size, spec.head_size)
        or cache.stride()[1:] != (block * width, width, 1)
        or pages <= 0
        or stride < key_bytes + scale_bytes
        or stride % width
        or (offset + key_bytes) % torch.float32.itemsize
        or stride % torch.float32.itemsize
        or offset + (pages - 1) * stride + key_bytes + scale_bytes > cache.untyped_storage().nbytes()
    ):
        raise ValueError("Invalid HiSparse FP8 indexer K/scale page geometry.")
    key = torch.as_strided(cache, (pages, block, heads, width), (stride, width, width, 1))
    # Reinterpret only a zero-length contiguous alias before constructing the
    # scale view; dtype conversion of a strided tensor would impose unrelated
    # constraints on its last axis. Offsets below are absolute storage offsets.
    raw = torch.as_strided(cache, (0,), (1,), storage_offset=0).view(torch.float32)
    scale = torch.as_strided(
        raw, (pages, block, heads, 1), (stride // 4, 1, 1, 1), storage_offset=(offset + key_bytes) // 4
    )
    return key, scale


def write_hisparse_indexer_c8(key, scale, values, scales, slots, scatter) -> None:
    """Write byte-preserving K and FP32 scales using this layer's page stride.

    Derive aliases from the current cache tuple, so profiling-cache replacement
    and teardown cannot leave a second set of persistent cache references.
    """
    from vllm_ascend.ops.triton.hisparse_indexer_c8_write import try_write

    if try_write(key, scale, values, scales, slots):
        return
    pages, block, heads, width = key.shape
    stride = key.stride(0)
    count = slots.numel()
    if values.numel() != count * width or scales.numel() != count:
        raise ValueError("HiSparse FP8 indexer writes require matching token/slot counts.")
    if values.dtype != key.dtype or scales.dtype != torch.float32:
        raise ValueError("HiSparse FP8 indexer writes require E4M3FN data and FP32 scales.")
    key_rows = ((pages - 1) * stride + block * width) // width
    scale_rows = (pages - 1) * scale.stride(0) + block
    raw_key = torch.as_strided(key, (0,), (1,), storage_offset=0).view(torch.int8)
    key_write = torch.as_strided(raw_key, (key_rows, width), (width, 1), storage_offset=key.storage_offset())
    scale_write = torch.as_strided(scale, (scale_rows, 1), (1, 1))
    # Ascend V2 supplies int32 slots. Page-strided scale addresses can exceed
    # INT32_MAX even when logical slots fit; widen before multiplying strides.
    logical = slots.reshape(-1).to(torch.int64)
    page = logical // block
    row = logical % block
    # Preserve the native -1 padding sentinel; never turn it into a valid row.
    key_slots = torch.where(logical >= 0, page * (stride // width) + row, -1)
    scale_slots = torch.where(logical >= 0, page * scale.stride(0) + row, -1)
    scatter(key_write, key_slots.view(-1, 1), values.contiguous().view(torch.int8).reshape(-1, width))
    scatter(scale_write, scale_slots.view(-1, 1), scales.reshape(-1, 1))


def _bind_indexer_views(
    kv_caches: dict[str, Any], indexer_specs: dict[str, AscendSFAIndexerCacheSpec], forward_context: dict[str, Any]
) -> None:
    for name, spec in indexer_specs.items():
        cache = kv_caches[name]
        # The indexer kernels address dense pages. Expand the view over the
        # shared slab and let the layer scale its block IDs and write slots.
        if cache.ndim != 4 or cache.shape[1] != spec.num_kv_heads:
            raise ValueError(f"Unexpected HiSparse indexer cache shape for {name}: {tuple(cache.shape)}")
        if spec.cache_sparse_li_c8:
            kv_caches[name] = _quantized_indexer_views(cache, spec)
            forward_context[name].hisparse_block_stride = 1
            # Declare only after validating this allocation's actual K/scale views.
            forward_context[name].hisparse_indexer_payload_format = "ascend_indexer_e4m3fn_key_fp32_scale_v1"
            continue
        block_size = cache.shape[2]
        page_elements = block_size * spec.num_kv_heads * spec.head_size
        if cache.stride(0) % page_elements:
            raise ValueError("HiSparse indexer slab stride must be a whole number of indexer pages.")
        block_stride = cache.stride(0) // page_elements
        physical_blocks = (cache.shape[0] - 1) * block_stride + 1
        indexer_cache = torch.as_strided(
            cache,
            size=(physical_blocks, block_size, spec.num_kv_heads, spec.head_size),
            stride=(page_elements, spec.num_kv_heads * spec.head_size, spec.head_size, 1),
        )
        forward_context[name].hisparse_block_stride = block_stride
        forward_context[name].hisparse_indexer_payload_format = "bf16"
        kv_caches[name] = (indexer_cache,)
