"""Static effective-rank keys and head-slice ranges for TP-mismatched Bulk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from ...topology import KVPoolTopology
from ..reachability import UnitaryReachability
from .common import (
    BulkRangeBatch,
    ByteArray,
    KeyAxes,
    bind_contiguous_layout,
    concatenate,
    contiguous_ranges,
    key_prefixes,
    lookup_prefixes,
    readonly_uint64,
    scatter_range_counts,
    selected_rows,
    splits,
)


@dataclass(frozen=True, slots=True)
class TPMismatchBulkProjection:
    """Registration-bound formula inputs for one TP-mismatched Bulk group."""

    group_id: int
    block_size: int
    hash_block_size: int
    physical_layer_ids: tuple[int, ...]
    local_key_prefixes: tuple[str, ...]
    lookup_key_prefixes: tuple[str, ...]
    key_slices_per_rank: int
    base_addresses: ByteArray
    block_lengths: ByteArray
    block_strides: ByteArray
    bases_by_slice: tuple[ByteArray, ...]
    strides_by_slice: tuple[ByteArray, ...]
    sizes_by_slice: tuple[ByteArray, ...]
    token_indices: ByteArray
    object_size: int
    reachability: UnitaryReachability


def bind_tp_mismatch_bulk_projection(
    topology: KVPoolTopology,
    max_model_len: int,
    base_addresses: Mapping[int, Sequence[int]],
    block_lengths: Mapping[int, Sequence[int]],
    block_strides: Mapping[int, Sequence[int]],
    layer_entry_offsets: Mapping[int, Sequence[int]],
    *,
    kv_cache_layout: str = "NHD",
    kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]] | None = None,
) -> TPMismatchBulkProjection:
    group = topology.transfer_groups[0]
    try:
        bases, lengths, strides = bind_contiguous_layout(
            group,
            base_addresses[group.group_id],
            block_lengths[group.group_id],
            block_strides[group.group_id],
            layer_entry_offsets[group.group_id],
        )
    except KeyError as error:
        raise RuntimeError(f"KV cache group {group.group_id} has not registered local memory") from error

    slice_count = topology.tp_partition.key_slices_per_rank
    local_key_prefixes = tuple(
        _effective_rank_prefix(group.key_metadata, topology.tp_rank * slice_count + slice_index)
        for slice_index in range(slice_count)
    )
    bases_by_slice: list[ByteArray] = []
    strides_by_slice: list[ByteArray] = []
    sizes_by_slice: list[ByteArray] = []
    head_major = kv_cache_layout in ("HND", "LBHNC")
    tensors: list[torch.Tensor] | None = None
    if kv_caches is not None:
        tensors = []
        for layer in group.layers:
            for layer_name in layer.layer_names:
                cache_or_caches = kv_caches[layer_name]
                entries = (cache_or_caches,) if isinstance(cache_or_caches, torch.Tensor) else cache_or_caches
                tensors.extend(cache for cache in entries if cache.numel())
        if len(tensors) != len(bases):
            raise ValueError("TP mismatch tensors do not align the registered cache entries")
    if tensors is None and head_major:
        raise ValueError("Head-major TP mismatch requires the registered cache tensors")
    token_indices: list[int] = []
    packed_full_rows = slice_count == 1 and (
        not head_major or (tensors is not None and all(cache.ndim == 4 and cache.shape[1] == 1 for cache in tensors))
    )
    if slice_count > 1 or tensors is not None:
        for slice_index in range(slice_count):
            slice_bases: list[int] = []
            slice_strides: list[int] = []
            slice_sizes: list[int] = []
            for entry_index, (base, length, stride) in enumerate(zip(bases, lengths, strides, strict=True)):
                if tensors is None:
                    slice_size, remainder = divmod(int(length), group.block_size * slice_count)
                    if remainder or slice_size == 0:
                        raise ValueError(
                            f"KV cache group {group.group_id} block length {length} cannot form "
                            f"{slice_count} Strided slices"
                        )
                    token_bytes = slice_size * slice_count
                    offsets = [
                        slice_index * slice_size + token_index * token_bytes for token_index in range(group.block_size)
                    ]
                    entry_sizes = [slice_size] * group.block_size
                    entry_tokens = list(range(group.block_size))
                else:
                    offsets, entry_sizes, entry_tokens = _tensor_head_slice_ranges(
                        tensors[entry_index],
                        int(stride),
                        group.block_size,
                        slice_index,
                        slice_count,
                        head_major=head_major,
                    )
                slice_bases.extend(int(base) + offset for offset in offsets)
                slice_strides.extend([int(stride)] * len(offsets))
                slice_sizes.extend(entry_sizes)
                if slice_index == 0:
                    token_indices.extend(entry_tokens)
                if packed_full_rows:
                    next_offset = 0
                    for offset, size in zip(offsets, entry_sizes, strict=True):
                        if offset != next_offset:
                            packed_full_rows = False
                            break
                        next_offset += size
                    packed_full_rows = packed_full_rows and next_offset == int(length)
            bases_by_slice.append(readonly_uint64(slice_bases))
            strides_by_slice.append(readonly_uint64(slice_strides))
            sizes_by_slice.append(readonly_uint64(slice_sizes))
    # A packed whole row needs no scatter plan; single-head HND also qualifies.
    if packed_full_rows:
        bases_by_slice.clear()
        strides_by_slice.clear()
        sizes_by_slice.clear()
    return TPMismatchBulkProjection(
        group_id=group.group_id,
        block_size=group.block_size,
        hash_block_size=topology.hash_block_size,
        physical_layer_ids=tuple(layer.physical_layer_id for layer in group.layers),
        local_key_prefixes=local_key_prefixes,
        lookup_key_prefixes=lookup_prefixes(topology, group),
        key_slices_per_rank=slice_count,
        base_addresses=bases,
        block_lengths=lengths,
        block_strides=strides,
        bases_by_slice=tuple(bases_by_slice),
        strides_by_slice=tuple(strides_by_slice),
        sizes_by_slice=tuple(sizes_by_slice),
        token_indices=readonly_uint64(token_indices),
        object_size=int(lengths.sum()),
        reachability=UnitaryReachability(group.group_id, max_model_len, topology.cache_transfer_granularity),
    )


def _tensor_head_slice_ranges(
    cache: torch.Tensor,
    block_stride: int,
    block_size: int,
    slice_index: int,
    slice_count: int,
    *,
    head_major: bool,
) -> tuple[list[int], list[int], list[int]]:
    """Compile local ranges while preserving this layout's existing Bulk wire order."""

    token_axis, head_axis = (2, 1) if head_major else (1, 2)
    if cache.ndim != 4:
        raise ValueError("Dense TP mismatch requires a block/token/head/vector tensor")
    element_size = cache.element_size()
    byte_strides = tuple(stride * element_size for stride in cache.stride())
    if byte_strides[0] <= 0:
        raise ValueError("TP mismatch tensor must have a positive kernel block stride")
    kernel_blocks_per_cache_block, stride_remainder = divmod(block_stride, byte_strides[0])
    if stride_remainder or kernel_blocks_per_cache_block <= 0:
        raise ValueError("TP mismatch registered block stride does not align the tensor's kernel blocks")
    kernel_tokens = cache.shape[token_axis]
    head_count = cache.shape[head_axis]
    physical_tokens = kernel_tokens * kernel_blocks_per_cache_block
    if physical_tokens <= 0 or block_size % physical_tokens:
        raise ValueError("TP mismatch tensor token extent does not divide the logical cache block")
    heads_per_slice, remainder = divmod(head_count, slice_count)
    if remainder or heads_per_slice <= 0:
        raise ValueError("TP mismatch key slices do not divide the tensor's local heads")
    if cache.shape[-1] > 1 and byte_strides[-1] != element_size:
        raise ValueError("Dense TP mismatch requires contiguous per-head vectors")
    vector_bytes = cache.shape[-1] * element_size
    if vector_bytes <= 0:
        raise ValueError("TP mismatch tensor has an empty head vector")
    head_stride = byte_strides[head_axis]
    heads_per_range = heads_per_slice if not head_major and head_stride == vector_bytes else 1
    raw_tokens_per_physical_token = block_size // physical_tokens
    offsets: list[int] = []
    sizes: list[int] = []
    token_indices: list[int] = []
    for kernel_block in range(kernel_blocks_per_cache_block):
        head_indices = range(0, heads_per_slice, heads_per_range)
        positions = (
            ((token, head) for head in head_indices for token in range(kernel_tokens))
            if head_major
            else ((token, head) for token in range(kernel_tokens) for head in head_indices)
        )
        for kernel_token, head_index in positions:
            token_offset = kernel_block * byte_strides[0] + kernel_token * byte_strides[token_axis]
            offsets.append(token_offset + (slice_index * heads_per_slice + head_index) * head_stride)
            sizes.append(heads_per_range * vector_bytes)
            physical_token = kernel_block * kernel_tokens + kernel_token
            token_indices.append(physical_token * raw_tokens_per_physical_token)
    return offsets, sizes, token_indices


def tp_mismatch_load_keys(projection: TPMismatchBulkProjection, hashes) -> KeyAxes:
    return key_prefixes(projection.local_key_prefixes, hashes)


def tp_mismatch_store_keys(projection: TPMismatchBulkProjection, hashes) -> KeyAxes:
    return key_prefixes(projection.local_key_prefixes, hashes)


def tp_mismatch_lookup_keys(projection: TPMismatchBulkProjection, hashes) -> KeyAxes:
    return key_prefixes(projection.lookup_key_prefixes, hashes)


def tp_mismatch_bulk_ranges(
    projection: TPMismatchBulkProjection,
    block_ids,
    token_counts,
    selected_objects=None,
) -> BulkRangeBatch:
    if not projection.bases_by_slice:
        return contiguous_ranges(
            block_ids,
            token_counts,
            selected_objects=selected_objects,
            bases=projection.base_addresses,
            lengths=projection.block_lengths,
            strides=projection.block_strides,
            block_size=projection.block_size,
            entry_slices=((0, len(projection.base_addresses)),),
        )

    ids = np.asarray(block_ids, dtype=np.uint64)
    counts = np.asarray(token_counts, dtype=np.uint64)
    if len(ids) != len(counts):
        raise ValueError("Block IDs and token counts must describe the same rows")
    selections = selected_rows(selected_objects, projection.key_slices_per_rank, len(ids))
    address_parts: list[ByteArray] = []
    size_parts: list[ByteArray] = []
    object_range_counts: list[int] = []
    for bases, strides, sizes, selected in zip(
        projection.bases_by_slice,
        projection.strides_by_slice,
        projection.sizes_by_slice,
        selections,
        strict=True,
    ):
        selected_ids = ids if selected is None else ids[selected]
        selected_counts = counts if selected is None else counts[selected]
        addresses = bases[None, :] + selected_ids[:, None] * strides[None, :]
        active = projection.token_indices[None, :] < selected_counts[:, None]
        address_parts.append(addresses[active])
        size_parts.append(np.broadcast_to(sizes, addresses.shape)[active])
        selected_counts_by_row = active.sum(axis=1, dtype=np.intp)
        object_range_counts.extend(scatter_range_counts(selected, len(ids), selected_counts_by_row))
    return concatenate(address_parts), concatenate(size_parts), splits(object_range_counts)


def _effective_rank_prefix(metadata, effective_rank: int) -> str:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import PoolKey

    return PoolKey(replace(metadata, head_or_tp_rank=effective_rank), "").to_string()
