# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def dense_paged_kv_views(
    key: torch.Tensor, value: torch.Tensor, block_table: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Expose paged KV caches as dense aliases without copying their contents.

    A logical page ``p`` starts at ``p * stride(0)``. Represent gaps as unused
    physical blocks, then remap FIA's block IDs by the stride ratio. Unsupported
    layouts retain the existing contiguous-conversion fallback.
    """
    if key.is_contiguous() and value.is_contiguous():
        return key, value, 1
    if block_table.device != key.device or block_table.dtype != torch.int32:
        return key, value, 1
    return _dense_paged_kv_aliases(key, value)


def _dense_paged_kv_aliases(key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Validate page geometry independently of scheduler read/write metadata."""
    if (
        key.ndim != 3
        or value.shape != key.shape
        or key.dtype not in (torch.float16, torch.bfloat16)
        or value.dtype != key.dtype
        or value.device != key.device
        or key.shape[0] <= 1
        or key.shape[1] == 0
        or key.shape[2] == 0
    ):
        return key, value, 1
    pages, block_size, hidden_size = key.shape
    page_elements = block_size * hidden_size
    if (
        key.stride() != value.stride()
        or key.stride()[1:] != (hidden_size, 1)
        or key.stride(0) <= page_elements
        or key.stride(0) % page_elements
    ):
        return key, value, 1
    block_stride = key.stride(0) // page_elements
    physical_blocks = (pages - 1) * block_stride + 1
    if physical_blocks > torch.iinfo(torch.int32).max:
        return key, value, 1
    for cache in (key, value):
        end = cache.storage_offset() + physical_blocks * page_elements
        if end * cache.element_size() > cache.untyped_storage().nbytes():
            return key, value, 1
    dense_stride = (page_elements, hidden_size, 1)
    dense_key = torch.as_strided(key, (physical_blocks, block_size, hidden_size), dense_stride)
    dense_value = torch.as_strided(value, (physical_blocks, block_size, hidden_size), dense_stride)
    return dense_key, dense_value, block_stride


def dense_paged_kv_write_views(
    key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return contiguous 4D aliases accepted by ScatterPaKvCache's ref inputs.

    The caller must map slots with ``remap_paged_slots``. Original cache handles
    and scheduler slots retain their logical page numbering.
    """
    if key.is_contiguous() and value.is_contiguous():
        return key, value, 1
    if (
        key.ndim != 4
        or value.shape != key.shape
        or slot_mapping.ndim != 1
        or slot_mapping.dtype not in (torch.int32, torch.int64)
        or slot_mapping.device != key.device
    ):
        return key, value, 1
    pages, block_size, heads, head_dim = key.shape
    inner_stride = (heads * head_dim, head_dim, 1)
    if key.stride()[1:] != inner_stride or value.stride()[1:] != inner_stride:
        return key, value, 1
    dense_key, dense_value, block_stride = _dense_paged_kv_aliases(
        key.view(pages, block_size, heads * head_dim),
        value.view(pages, block_size, heads * head_dim),
    )
    if block_stride == 1 or dense_key.shape[0] * block_size - 1 > torch.iinfo(slot_mapping.dtype).max:
        return key, value, 1
    return (
        dense_key.view(-1, block_size, heads, head_dim),
        dense_value.view(-1, block_size, heads, head_dim),
        block_stride,
    )


def remap_paged_slots(slot_mapping: torch.Tensor, block_size: int, block_stride: int) -> torch.Tensor:
    """Map page origins, preserving in-page offsets and negative padding slots."""
    if block_stride == 1:
        return slot_mapping
    if block_size <= 0 or block_stride <= 0:
        raise ValueError("KV block size and stride must be positive")
    pages = torch.div(slot_mapping.clamp_min(0), block_size, rounding_mode="floor")
    return torch.add(slot_mapping, pages, alpha=(block_stride - 1) * block_size)


def validate_paged_block_table_buffer(block_table: torch.Tensor, mapped_block_table: torch.Tensor) -> None:
    """Reject resizing or scheduler aliasing before starting a graph update."""
    if (
        mapped_block_table.shape != block_table.shape
        or mapped_block_table.dtype != block_table.dtype
        or mapped_block_table.device != block_table.device
    ):
        raise ValueError("Mapped KV block table must match source shape, dtype and device; recapture the graph")
    if mapped_block_table.untyped_storage().data_ptr() == block_table.untyped_storage().data_ptr():
        raise ValueError("Mapped KV block table must not share scheduler storage")


def remap_paged_block_table(
    block_table: torch.Tensor,
    block_stride: int,
    mapped_block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map logical page IDs to their physical rows in the dense alias.

    With an output buffer, remap in place into persistent storage for graph
    replay. Without one, return the eager-path result without mutating the input.
    """
    if block_stride == 1:
        return block_table
    if mapped_block_table is None:
        return block_table * block_stride
    validate_paged_block_table_buffer(block_table, mapped_block_table)
    torch.mul(block_table, block_stride, out=mapped_block_table)
    return mapped_block_table
