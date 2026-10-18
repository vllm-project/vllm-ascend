"""Precompute head-slice addresses from tensor layout; never rearrange KV memory.

Example: one Entry, NHD, one kernel block, two tokens, four 4-byte heads,
two head slices, no padding. Numbers below are byte offsets from the Entry base.

    token                  t0                            t1
    heads          [h0  h1 | h2  h3]              [h0  h1 | h2  h3]
    byte offsets   [ 0   4 |  8  12]              [16  20 | 24  28]
    head slice     [   0   |   1   ]              [   0   |   1   ]
                        |                              |
                        +---------------+--------------+
                                        v
    slice 0 segments: [0:8], [16:24]    slice 1 segments: [8:16], [24:32]
    segment lengths:       [8, 8]                         [8, 8]
    next group block:       +32                            +32

TP splits heads inside EVERY Entry. Slices describe bytes in place, not copied
buffers; padding can increase strides but never payload lengths. Equal layouts
share compiled offsets before each Entry's base is added.

    Wire traversal (no format conversion):
    NHD         kernel -> token    -> head run
    HND/LBHNC   kernel -> head run -> token

LBHNC uses registered per-layer views. Whole-Entry binding requires one head slice,
compatible internal strides and full payload size; HND/LBHNC also requires one
head. Padding between group blocks is allowed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .types import LocalMemorySegments, UInt64Array

if TYPE_CHECKING:
    import torch

    from ..configuration.topology import KVPoolGroupTopology


@dataclass(frozen=True, slots=True)
class HeadSliceLayout:
    """One Entry's block/token/head geometry for a selected head slice.

    Kernel blocks are the first-axis units of the registered tensor; a group
    block may contain several of them. Counts describe physically stored data,
    not request token coverage. A segment holds heads_per_segment adjacent heads
    for one stored token in one kernel block. Strides and vector size are bytes.
    """

    kernel_blocks_per_group_block: int
    stored_tokens_per_kernel_block: int
    heads_per_slice: int
    heads_per_segment: int
    head_vector_bytes: int
    kernel_stride_bytes: int
    token_stride_bytes: int
    head_stride_bytes: int


@dataclass(frozen=True, slots=True)
class HeadSliceGeometry:
    """Precomputed [head slice, segment] byte offsets relative to Entry bases.

    segment_count totals all Entries for ONE head slice. Equal Entry layouts
    reuse one offset array; uniform_segment_length enables singleton lengths.
    """

    segment_offsets_bytes_by_layout: dict[HeadSliceLayout, UInt64Array]
    segment_count: int
    first_segment_length_bytes: int
    uniform_segment_length: bool


# ============================================
# Binding stages
# ============================================


def inspect_entries(
    group: KVPoolGroupTopology,
    entry_block_payload_bytes: UInt64Array,
    group_block_strides_bytes: UInt64Array,
    head_slices_per_rank: int,
    head_major: bool,
    kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]] | None,
) -> tuple[list[HeadSliceLayout], bool]:
    """Read registered layouts and prove whether whole Entry blocks preserve the wire order."""
    if kv_caches is None:
        raise ValueError("TP head binding requires the registered KV tensors")
    token_axis, head_axis = (2, 1) if head_major else (1, 2)
    entry_layouts: list[HeadSliceLayout] = []
    whole_entries_contiguous = head_slices_per_rank == 1
    # Match registration's empty-tensor filtering and traversal order exactly.
    for layer in group.layers:
        for name in layer.layer_names:
            cache = kv_caches[name]
            tensors = (cache,) if hasattr(cache, "shape") else cache
            for tensor in tensors:
                if not tensor.numel():
                    continue
                entry_index = len(entry_layouts)
                block_stride_bytes = int(group_block_strides_bytes[entry_index])
                layout = _read_head_layout(
                    tensor, block_stride_bytes, group.block_size, head_slices_per_rank, token_axis, head_axis
                )
                entry_layouts.append(layout)
                # Stop proving contiguity after the first failure, but validate every layout.
                if whole_entries_contiguous:
                    token_bytes = layout.heads_per_slice * layout.head_vector_bytes
                    kernel_bytes = layout.stored_tokens_per_kernel_block * token_bytes
                    whole_entries_contiguous = (
                        (not head_major or layout.heads_per_slice == 1)
                        and (layout.heads_per_slice == 1 or layout.head_stride_bytes == layout.head_vector_bytes)
                        and (layout.stored_tokens_per_kernel_block == 1 or layout.token_stride_bytes == token_bytes)
                        and (layout.kernel_blocks_per_group_block == 1 or layout.kernel_stride_bytes == kernel_bytes)
                        and layout.kernel_blocks_per_group_block * kernel_bytes
                        == int(entry_block_payload_bytes[entry_index])
                    )
    return entry_layouts, whole_entries_contiguous


def compile_geometries(
    entry_layouts: list[HeadSliceLayout],
    head_slices_per_rank: int,
    head_major: bool,
    entry_segment_boundaries: list[int],
) -> HeadSliceGeometry:
    """Share compiled offsets across equal layouts and append cumulative Entry boundaries.

    entry_segment_boundaries is caller-owned and must start with zero; appended
    values are segment-array indices used by layer/PP selection, not byte offsets.
    """
    geometry_by_layout = {}
    segment_count = 0
    first_segment_length_bytes = entry_layouts[0].heads_per_segment * entry_layouts[0].head_vector_bytes
    uniform_segment_length = True
    slice_indices = np.arange(head_slices_per_rank, dtype=np.uint64)[:, None]
    for layout in entry_layouts:
        geometry = geometry_by_layout.get(layout)
        if geometry is None:
            geometry = _compile_head_slice_geometry(layout, slice_indices, head_major)
            geometry_by_layout[layout] = geometry
        segment_count += geometry.shape[1]
        entry_segment_boundaries.append(segment_count)
        uniform_segment_length = (
            uniform_segment_length and layout.heads_per_segment * layout.head_vector_bytes == first_segment_length_bytes
        )
    return HeadSliceGeometry(geometry_by_layout, segment_count, first_segment_length_bytes, uniform_segment_length)


def materialize_segments(
    entry_addresses: UInt64Array,
    group_block_strides_bytes: UInt64Array,
    entry_layouts: list[HeadSliceLayout],
    head_slices_per_rank: int,
    geometry: HeadSliceGeometry,
) -> tuple[LocalMemorySegments, ...]:
    """Return read-only segments whose head slices share stride and length arrays."""
    segment_count = geometry.segment_count
    slice_addresses = np.empty((head_slices_per_rank, segment_count), dtype=np.uint64)
    uniform_block_stride = np.all(group_block_strides_bytes == group_block_strides_bytes[0])
    segment_strides = (
        group_block_strides_bytes[:1] if uniform_block_stride else np.empty(segment_count, dtype=np.uint64)
    )
    segment_lengths = (
        np.full(1, geometry.first_segment_length_bytes, dtype=np.uint64)
        if geometry.uniform_segment_length
        else np.empty(segment_count, dtype=np.uint64)
    )
    cursor = 0
    for address, block_stride, layout in zip(entry_addresses, group_block_strides_bytes, entry_layouts, strict=True):
        slice_offsets = geometry.segment_offsets_bytes_by_layout[layout]
        end = cursor + slice_offsets.shape[1]
        np.add(address, slice_offsets, out=slice_addresses[:, cursor:end])
        if not uniform_block_stride:
            segment_strides[cursor:end] = block_stride
        if not geometry.uniform_segment_length:
            segment_lengths[cursor:end] = layout.heads_per_segment * layout.head_vector_bytes
        cursor = end
    # Address rows share strides and lengths; read-only arrays protect all shared NumPy slices.
    for array in (slice_addresses, segment_strides, segment_lengths):
        array.flags.writeable = False
    return tuple(LocalMemorySegments(addresses, segment_strides, segment_lengths) for addresses in slice_addresses)


# ============================================
# Tensor layout and byte-offset calculations
# ============================================


def _read_head_layout(
    cache: torch.Tensor,
    group_block_stride_bytes: int,
    group_block_size: int,
    head_slices_per_rank: int,
    token_axis: int,
    head_axis: int,
) -> HeadSliceLayout:
    if cache.ndim != 4:
        raise ValueError("TP mismatch requires a block/token/head/vector tensor")
    element_bytes = cache.element_size()
    byte_strides = tuple(stride * element_bytes for stride in cache.stride())
    kernel_stride_bytes = byte_strides[0]
    if kernel_stride_bytes == 0:
        raise ValueError("TP kernel block stride must be nonzero")
    # Registration multiplied this very Tensor stride by the integer block scale.
    kernel_blocks_per_group_block = group_block_stride_bytes // kernel_stride_bytes
    stored_tokens_per_kernel_block = cache.shape[token_axis]
    stored_tokens_per_group_block = kernel_blocks_per_group_block * stored_tokens_per_kernel_block
    if group_block_size % stored_tokens_per_group_block:
        raise ValueError("Physical token extent must divide the logical group block")
    local_head_count = cache.shape[head_axis]
    heads_per_slice, remainder = divmod(local_head_count, head_slices_per_rank)
    if remainder or not heads_per_slice:
        raise ValueError("TP key slices must divide the local heads")
    if cache.shape[-1] > 1 and byte_strides[-1] != element_bytes:
        raise ValueError("TP head vectors must be contiguous")
    head_vector_bytes = cache.shape[-1] * element_bytes
    heads_per_segment = heads_per_slice if head_axis == 2 and byte_strides[head_axis] == head_vector_bytes else 1
    return HeadSliceLayout(
        kernel_blocks_per_group_block,
        stored_tokens_per_kernel_block,
        heads_per_slice,
        heads_per_segment,
        head_vector_bytes,
        kernel_stride_bytes,
        byte_strides[token_axis],
        byte_strides[head_axis],
    )


def _compile_head_slice_geometry(layout: HeadSliceLayout, slice_indices: UInt64Array, head_major: bool) -> UInt64Array:
    """Return [head slice, segment] byte offsets, independent of entry base addresses.

    Flattening must preserve the layout's wire order, not sort by local address.
    """
    kernel_indices = np.arange(layout.kernel_blocks_per_group_block, dtype=np.uint64)[:, None, None]
    token_indices = np.arange(layout.stored_tokens_per_kernel_block, dtype=np.uint64)
    head_starts = np.arange(0, layout.heads_per_slice, layout.heads_per_segment, dtype=np.uint64)
    token_indices = token_indices[None, None, :] if head_major else token_indices[None, :, None]
    head_starts = head_starts[None, :, None] if head_major else head_starts[None, None, :]
    offsets = (
        kernel_indices * layout.kernel_stride_bytes
        + token_indices * layout.token_stride_bytes
        + head_starts * layout.head_stride_bytes
    )
    return offsets.ravel() + slice_indices * (layout.heads_per_slice * layout.head_stride_bytes)
