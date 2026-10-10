"""Small formulas shared by the concrete Bulk projection variants.

Only registration-time facts live here. Request row selection and writer
ownership remain Worker decisions; these helpers merely join fixed key
prefixes and evaluate bound memory layouts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    PoolKey,
    block_hash_to_str,
)

from ...topology import KVPoolGroupTopology, KVPoolTopology

ByteArray: TypeAlias = NDArray[np.uint64]
IndexArray: TypeAlias = NDArray[np.intp]
BulkRangeBatch: TypeAlias = tuple[ByteArray, ByteArray, IndexArray]
KeyAxes: TypeAlias = tuple[tuple[str, ...], ...]


def key_prefixes(prefixes: tuple[str, ...], hashes: Sequence[BlockHash | str]) -> KeyAxes:
    """Append dynamic content hashes to registration-time key prefixes."""

    hash_strings = tuple(block_hash_to_str(value) for value in hashes)
    return tuple(tuple(prefix + value for value in hash_strings) for prefix in prefixes)


def local_prefix(group: KVPoolGroupTopology) -> str:
    return _prefix(group.key_metadata)


def lookup_prefixes(topology: KVPoolTopology, group: KVPoolGroupTopology) -> tuple[str, ...]:
    head_rank_count = topology.tp_size if group.uses_align_state else topology.tp_partition.key_rank_count
    return tuple(
        _prefix(replace(group.key_metadata, pp_rank=pp_rank, dcp_rank=dcp_rank, head_or_tp_rank=head_rank))
        for pp_rank in range(topology.pp_size)
        for dcp_rank in range(topology.dcp_size)
        for head_rank in range(head_rank_count)
    )


def bind_contiguous_layout(
    group: KVPoolGroupTopology,
    base_addresses: Sequence[int],
    block_lengths: Sequence[int],
    block_strides: Sequence[int],
    layer_entry_offsets: Sequence[int],
) -> tuple[ByteArray, ByteArray, ByteArray]:
    """Freeze one group's dense registered entries after validating its layer bounds."""

    physical_layers = tuple(layer.physical_layer_id for layer in group.layers)
    if len(base_addresses) == 0 or not (len(base_addresses) == len(block_lengths) == len(block_strides)):
        raise ValueError(f"KV cache group {group.group_id} registered misaligned memory arrays")
    if (
        len(layer_entry_offsets) != len(physical_layers) + 1
        or layer_entry_offsets[0] != 0
        or layer_entry_offsets[-1] != len(base_addresses)
    ):
        raise ValueError(f"KV cache group {group.group_id} registered invalid layer entry bounds")
    return readonly_uint64(base_addresses), readonly_uint64(block_lengths), readonly_uint64(block_strides)


def contiguous_ranges(
    block_ids: Sequence[int] | ByteArray,
    token_counts: Sequence[int] | ByteArray,
    *,
    selected_objects: Sequence[bool] | None,
    bases: ByteArray,
    lengths: ByteArray,
    strides: ByteArray,
    block_size: int,
    entry_slices: tuple[tuple[int, int], ...],
    full_extent: bool = False,
) -> BulkRangeBatch:
    """Evaluate axis-aligned contiguous ranges for ordinary or Consumer-PP Bulk."""

    ids = np.asarray(block_ids, dtype=np.uint64)
    counts = np.asarray(token_counts, dtype=np.uint64)
    if len(ids) != len(counts):
        raise ValueError("Block IDs and token counts must describe the same rows")

    selections = selected_rows(selected_objects, len(entry_slices), len(ids))
    address_parts: list[ByteArray] = []
    size_parts: list[ByteArray] = []
    object_range_counts: list[int] = []
    for (start, end), selected in zip(entry_slices, selections, strict=True):
        axis_bases = bases[start:end]
        axis_lengths = lengths[start:end]
        axis_strides = strides[start:end]
        selected_ids = ids if selected is None else ids[selected]
        addresses = axis_bases[None, :] + selected_ids[:, None] * axis_strides[None, :]
        if full_extent:
            sizes = np.broadcast_to(axis_lengths, addresses.shape)
        else:
            selected_counts = counts if selected is None else counts[selected]
            sizes = axis_lengths[None, :] * selected_counts[:, None] // np.uint64(block_size)
        address_parts.append(addresses.ravel())
        size_parts.append(sizes.ravel())
        object_range_counts.extend(range_counts(selected, len(ids), end - start))
    return concatenate(address_parts), concatenate(size_parts), splits(object_range_counts)


def bulk_arguments(
    key_axes: KeyAxes,
    ranges: BulkRangeBatch,
    selected_objects: Sequence[bool] | None,
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    """Group flat local ranges into the Bulk Backend's per-key argument shape."""

    local, sizes, range_splits = ranges
    keys = tuple(key for axis in key_axes for key in axis)
    selected = selected_object_indices(len(keys), range_splits, selected_objects)
    return (
        [keys[index] for index in selected],
        [local[range_splits[index] : range_splits[index + 1]].tolist() for index in selected],
        [sizes[range_splits[index] : range_splits[index + 1]].tolist() for index in selected],
    )


def selected_rows(
    selected_objects: Sequence[bool] | None,
    axis_count: int,
    row_count: int,
) -> tuple[IndexArray | None, ...]:
    if selected_objects is None:
        return (None,) * axis_count
    selected = np.asarray(selected_objects, dtype=np.bool_)
    if len(selected) != axis_count * row_count:
        raise ValueError("Object selection does not align the mapped rows")
    return tuple(
        np.flatnonzero(selected[axis_index * row_count : (axis_index + 1) * row_count])
        for axis_index in range(axis_count)
    )


def range_counts(selected: IndexArray | None, row_count: int, range_count: int) -> list[int]:
    if selected is None:
        return [range_count] * row_count
    selected_counts: IndexArray = np.full(len(selected), range_count, dtype=np.intp)
    return scatter_range_counts(selected, row_count, selected_counts)


def scatter_range_counts(
    selected: IndexArray | None,
    row_count: int,
    selected_counts: IndexArray,
) -> list[int]:
    if selected is None:
        return selected_counts.tolist()
    counts: IndexArray = np.zeros(row_count, dtype=np.intp)
    counts[selected] = selected_counts
    return counts.tolist()


def selected_object_indices(
    count: int,
    range_splits: IndexArray,
    selected: Sequence[bool] | None,
) -> tuple[int, ...]:
    if len(range_splits) != count + 1:
        raise ValueError("Key axes do not align the mapped range objects")
    if selected is None:
        return tuple(range(count))
    mask = np.asarray(selected, dtype=np.bool_)
    if len(mask) != count:
        raise ValueError("Object selection does not align the mapped range objects")
    return tuple(np.flatnonzero(mask).tolist())


def readonly_uint64(values: Sequence[int]) -> ByteArray:
    result = np.array(values, dtype=np.uint64, copy=True)
    result.flags.writeable = False
    return result


def splits(object_range_counts: Sequence[int]) -> IndexArray:
    result: IndexArray = np.empty(len(object_range_counts) + 1, dtype=np.intp)
    result[0] = 0
    if object_range_counts:
        np.cumsum(np.asarray(object_range_counts, dtype=np.intp), out=result[1:])
    return result


def concatenate(parts: list[ByteArray]) -> ByteArray:
    if not parts:
        return np.empty(0, dtype=np.uint64)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _prefix(metadata) -> str:
    return PoolKey(metadata, "").to_string()
