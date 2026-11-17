# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bind shared-slab views while retaining Core HiSparse allocation."""

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

import torch
from vllm.v1.kv_cache_interface import (
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec

_SLAB_ALIGNMENT = 2 * 1024 * 1024
# The runner reserves padding before Core copies the config. The backend hook
# consumes the logical size within that same synchronous initialization call.
_slab_size: ContextVar[int | None] = ContextVar("hisparse_device_slab_size", default=None)


@contextmanager
def reserve_hisparse_slab_alignment(kv_cache_config: KVCacheConfig) -> Iterator[None]:
    """Leave room to align the single device allocation made by Core."""
    original_tensors = kv_cache_config.kv_cache_tensors
    device_sizes = {tensor.size for tensor in original_tensors if not tensor.host_resident}
    if len(device_sizes) != 1:
        raise ValueError(f"HiSparse alignment requires one shared device slab size, got {sorted(device_sizes)}.")
    (size,) = device_sizes
    kv_cache_config.kv_cache_tensors = [
        tensor if tensor.host_resident else replace(tensor, size=size + _SLAB_ALIGNMENT - 1)
        for tensor in original_tensors
    ]
    token = _slab_size.set(size)
    try:
        yield
    finally:
        _slab_size.reset(token)
        kv_cache_config.kv_cache_tensors = original_tensors


def _align_hisparse_slab_views(
    kv_cache_config: KVCacheConfig, kv_caches: dict[str, Any]
) -> tuple[list[KVCacheTensor], int] | None:
    """Rebase Core's device views onto one aligned slice, without allocating or copying KV."""
    size = _slab_size.get()
    if size is None:
        from .admission import AscendStoreHiSparseResidentSpec

        if any(
            isinstance(group.kv_cache_spec, AscendStoreHiSparseResidentSpec)
            for group in kv_cache_config.kv_cache_groups
        ):
            raise RuntimeError("HiSparse AscendStore device views require an active slab alignment reservation.")
        return None
    device_tensors = [tensor for tensor in kv_cache_config.kv_cache_tensors if not tensor.host_resident]
    first_view = kv_caches[device_tensors[0].layers[0]]
    owner = first_view.untyped_storage()
    start = -owner.data_ptr() % _SLAB_ALIGNMENT
    if owner.nbytes() != size + _SLAB_ALIGNMENT - 1:
        raise ValueError("HiSparse device slab is missing its reserved alignment padding.")

    # Core binds Resident and Hot using offsets from the storage base. Use
    # one storage slice for all device views so those offsets still start at 0.
    aligned_storage = owner[start : start + size]
    for tensor in device_tensors:
        for name in tensor.layers:
            view = kv_caches[name]
            if view.untyped_storage().data_ptr() != owner.data_ptr():
                raise ValueError("HiSparse device cache views must share one backing allocation.")
            if view.ndim == 1 and view.numel() == owner.nbytes():  # Raw Resident and Hot slab.
                kv_caches[name] = view.new_empty(0).set_(aligned_storage, 0, (size,), (1,))
            else:
                kv_caches[name] = view.new_empty(0).set_(
                    aligned_storage, view.storage_offset(), view.size(), view.stride()
                )
    return device_tensors, size


def prepare_hisparse_cache_views(
    kv_cache_config: KVCacheConfig, kv_caches: dict[str, Any], forward_context: dict[str, Any]
) -> None:
    """Publish the aligned slab and indexer views only after every view is valid."""
    prepared_caches = kv_caches.copy()
    alignment = _align_hisparse_slab_views(kv_cache_config, prepared_caches)
    _bind_indexer_views(prepared_caches, get_hisparse_indexer_specs(kv_cache_config), forward_context)
    kv_caches.update(prepared_caches)
    if alignment is not None:
        device_tensors, size = alignment
        for tensor in device_tensors:
            tensor.size = size


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
    """Collect the Ascend indexer specs keyed by layer."""
    indexer_specs = {}
    for group in kv_cache_config.kv_cache_groups:
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            if isinstance(spec, AscendSFAIndexerCacheSpec):
                indexer_specs[name] = spec

    return indexer_specs


def hisparse_indexer_transfer_views(caches: Sequence[torch.Tensor], layer: Any) -> tuple[torch.Tensor, ...]:
    """Recover logical pages from the dense Indexer alias used by attention."""
    stride = getattr(layer, "hisparse_block_stride", 1)
    if stride > 1:
        (dense_cache,) = caches
        return (dense_cache[::stride],)
    return tuple(caches)


def _validate_indexer_spec(spec: AscendSFAIndexerCacheSpec) -> None:
    if spec.cache_sparse_li_c4:
        raise ValueError("HiSparse indexer C4 is not supported by this adapter.")
    if spec.cache_sparse_li_c8:
        if not (
            (spec.dtype, spec.scale_dtype) in ((torch.int8, torch.float16), (torch.float8_e4m3fn, torch.float32))
            and spec.scale_dim == 1
            and spec.num_kv_heads == 1
            and spec.head_size == 128
            and spec.block_size == 128
            and spec.sfa_dcp_replicated_indexer_size == 1
        ):
            raise ValueError("HiSparse indexer C8 requires INT8/FP16 or E4M3FN/FP32 K128, block128.")
    elif spec.scale_dim:
        raise ValueError("HiSparse indexer scales require an explicit supported quantization format.")


def _quantized_indexer_views(cache: torch.Tensor, spec: AscendSFAIndexerCacheSpec):
    """Alias the current Core allocation; retain no additional storage owner."""
    pages, heads, block, row_bytes = cache.shape
    stride = cache.stride(0)
    width = spec.head_size
    key_bytes = block * heads * width
    scale_size = spec.scale_dtype.itemsize
    scale_bytes = block * heads * scale_size
    offset = cache.storage_offset()
    # Core's view covers all bytes reserved for each token. The Indexer
    # kernels use a page with contiguous K followed by contiguous scales.
    if (
        cache.dtype != spec.dtype
        or (heads, block, row_bytes) != (1, spec.block_size, spec.state_content_size_bytes)
        or cache.stride()[2:] != (row_bytes, 1)
        or pages <= 0
        or stride < key_bytes + scale_bytes
        or stride % width
        or (offset + key_bytes) % scale_size
        or stride % scale_size
        or offset + (pages - 1) * stride + key_bytes + scale_bytes > cache.untyped_storage().nbytes()
    ):
        raise ValueError("Invalid HiSparse C8 indexer K/scale page geometry.")
    key = torch.as_strided(cache, (pages, block, heads, width), (stride, width, width, 1))
    # Reinterpret only a zero-length contiguous alias before constructing the
    # scale view; dtype conversion of a strided tensor would impose unrelated
    # constraints on its last axis. Offsets below are absolute storage offsets.
    raw = torch.as_strided(cache, (0,), (1,), storage_offset=0).view(spec.scale_dtype)
    scale = torch.as_strided(
        raw,
        (pages, block, heads, 1),
        (stride // scale_size, 1, 1, 1),
        storage_offset=(offset + key_bytes) // scale_size,
    )
    return key, scale


def write_hisparse_indexer_c8(key, scale, values, scales, slots) -> None:
    """Write byte-preserving K and scales using this layer's page stride.

    Derive aliases from the current cache tuple, so profiling-cache replacement
    and teardown cannot leave a second set of persistent cache references.
    """
    from vllm_ascend.ops.triton.sfa_indexer_store import store_indexer_key_scale

    block, width = key.shape[1], key.shape[-1]
    count = slots.numel()
    if values.numel() != count * width or scales.numel() != count:
        raise ValueError("HiSparse C8 indexer writes require matching token/slot counts.")
    if (
        values.dtype != key.dtype
        or (key.dtype, scale.dtype) not in ((torch.int8, torch.float16), (torch.float8_e4m3fn, torch.float32))
        or scales.dtype not in (scale.dtype, torch.float32)
    ):
        raise ValueError("HiSparse C8 indexer writes require matching K and scale dtypes.")
    scale_bits = torch.int16 if scale.dtype == torch.float16 else torch.int32
    store_indexer_key_scale(
        values.contiguous().view(torch.int8).reshape(-1, width),
        scales.to(scale.dtype).contiguous().view(scale_bits).reshape(-1),
        slots.reshape(-1),
        key.view(torch.int8),
        scale.view(scale_bits),
        block_size=block,
        key_block_stride=key.stride(0),
        scale_block_stride=scale.stride(0),
    )


def _bind_indexer_views(
    kv_caches: dict[str, Any], indexer_specs: dict[str, AscendSFAIndexerCacheSpec], forward_context: dict[str, Any]
) -> None:
    block_strides = {}
    for name, spec in indexer_specs.items():
        cache = kv_caches[name]
        # The indexer kernels address dense pages. Expand the view over the
        # shared slab and let the layer scale its block IDs and write slots.
        if cache.ndim != 4 or cache.shape[1] != spec.num_kv_heads:
            raise ValueError(f"Unexpected HiSparse indexer cache shape for {name}: {tuple(cache.shape)}")
        if spec.cache_sparse_li_c8:
            kv_caches[name] = _quantized_indexer_views(cache, spec)
            block_strides[name] = 1
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
        block_strides[name] = block_stride
        kv_caches[name] = (indexer_cache,)
    for name, block_stride in block_strides.items():
        forward_context[name].hisparse_block_stride = block_stride
