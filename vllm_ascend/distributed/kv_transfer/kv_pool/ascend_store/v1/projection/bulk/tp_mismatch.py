"""Static effective-rank keys and head-slice ranges for TP-mismatched Bulk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

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
    if slice_count > 1:
        for slice_index in range(slice_count):
            slice_bases: list[int] = []
            slice_strides: list[int] = []
            slice_sizes: list[int] = []
            for base, length, stride in zip(bases, lengths, strides, strict=True):
                slice_size, remainder = divmod(int(length), group.block_size * slice_count)
                if remainder or slice_size == 0:
                    raise ValueError(
                        f"KV cache group {group.group_id} block length {length} cannot form "
                        f"{slice_count} Strided slices"
                    )
                token_bytes = slice_size * slice_count
                for token_index in range(group.block_size):
                    slice_bases.append(int(base) + slice_index * slice_size + token_index * token_bytes)
                    slice_strides.append(int(stride))
                    slice_sizes.append(slice_size)
            bases_by_slice.append(readonly_uint64(slice_bases))
            strides_by_slice.append(readonly_uint64(slice_strides))
            sizes_by_slice.append(readonly_uint64(slice_sizes))
    token_indices = np.tile(np.arange(group.block_size, dtype=np.uint64), len(bases))
    token_indices.flags.writeable = False
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
        token_indices=token_indices,
        object_size=int(lengths.sum()),
        reachability=UnitaryReachability(group.group_id, max_model_len, topology.cache_transfer_granularity),
    )


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
    if projection.key_slices_per_rank == 1:
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
