"""Bind registered KV memory layouts to reusable transfer-range rules.

Memory rules determine:

- how local Block IDs map to physical byte addresses for each cache Group;
- which contiguous or Strided formula evaluates that mapping;
- how full and partial token extents become byte lengths;
- how Layerwise access selects one layer inside a remote object;
- which registered entries a consumer-PP Store operation includes;
- how local ranges become Bulk, KeyRange, or GVA Backend arguments.

The bound rules retain only registered layout facts and selected formulas.
Request Block IDs, token counts, layer choices, object selection, and leased GVA
bases remain caller-owned. Logical object identity, Backend keys, and request
lifetime remain outside this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

ByteArray: TypeAlias = NDArray[np.uint64]
IndexArray: TypeAlias = NDArray[np.intp]
MemoryValues: TypeAlias = Mapping[int, Sequence[int]]
MemoryDataPlane: TypeAlias = Literal["bulk", "key_range", "gva"]

# Bulk needs only local ranges grouped by Backend key. Layerwise access also
# needs offsets inside each remote object and its registered global size.
BulkRangeBatch: TypeAlias = tuple[ByteArray, ByteArray, IndexArray]
RangeBatch: TypeAlias = tuple[ByteArray, ByteArray, ByteArray, IndexArray, ByteArray]


# =============================================================================
# Registered Memory Binding
# =============================================================================
#
# Bind each cache Group's registered layout to the selected contiguous or
# Strided formula, including Load and Store views of consumer-PP entries.


class KVMemoryRule:
    """Dispatch each cache Group to its bound full and partial range rules."""

    __slots__ = ("full", "partial", "store_full", "store_partial")

    def __init__(
        self,
        *,
        group_ids: tuple[int, ...],
        block_sizes: Mapping[int, int],
        align_state_group_ids: frozenset[int],
        physical_layers: Mapping[int, tuple[int, ...]],
        base_addresses: MemoryValues,
        block_lengths: MemoryValues,
        block_strides: MemoryValues,
        layer_entry_offsets: MemoryValues,
        strided_slice_count: int,
        consumer_pipeline_partitions: tuple[int, ...] | None,
        store_pipeline_ranks: Mapping[int, tuple[int, ...]] | None,
        data_plane: MemoryDataPlane,
        requires_global_offsets: bool,
        object_sizes: Mapping[int, int] | None,
        object_offsets: Mapping[int, int] | None,
    ) -> None:
        if data_plane == "gva" and object_sizes is None:
            raise ValueError("GVA memory rules require registered global object sizes")
        if data_plane == "gva" and requires_global_offsets and object_offsets is None:
            raise ValueError("Parallel GVA memory rules require registered global object offsets")

        load_full: dict[int, Callable] = {}
        load_partial: dict[int, Callable] = {}
        store_full: dict[int, Callable] = {}
        store_partial: dict[int, Callable] = {}
        for group_id in group_ids:
            if data_plane == "gva" and object_sizes is not None and group_id not in object_sizes:
                raise ValueError(f"GVA group {group_id} has no registered global object size")
            if (
                data_plane == "gva"
                and requires_global_offsets
                and object_offsets is not None
                and group_id not in object_offsets
            ):
                raise ValueError(f"GVA group {group_id} has no registered global object offset")
            try:
                bases = base_addresses[group_id]
                lengths = block_lengths[group_id]
                strides = block_strides[group_id]
                layer_offsets = layer_entry_offsets[group_id]
            except KeyError as error:
                raise RuntimeError(f"KV cache group {group_id} has not registered local memory") from error

            if strided_slice_count > 1:
                full, partial_extent = _bind_strided_group(
                    group_id,
                    bases,
                    lengths,
                    strides,
                    block_sizes[group_id],
                    strided_slice_count,
                )
                load_full[group_id] = store_full[group_id] = full
                load_partial[group_id] = store_partial[group_id] = partial_extent
                continue

            full, partial_extent, store_full_rule, store_partial_rule = _bind_contiguous_group(
                group_id=group_id,
                base_addresses=bases,
                block_lengths=lengths,
                block_strides=strides,
                layer_entry_offsets=layer_offsets,
                block_size=block_sizes[group_id],
                physical_layers=physical_layers[group_id],
                align_state=group_id in align_state_group_ids,
                consumer_pipeline_partitions=consumer_pipeline_partitions,
                store_pipeline_ranks=(None if store_pipeline_ranks is None else store_pipeline_ranks[group_id]),
                data_plane=data_plane,
                object_size=None if object_sizes is None else object_sizes.get(group_id),
                object_offset=None if object_offsets is None else object_offsets.get(group_id),
            )
            load_full[group_id] = full
            load_partial[group_id] = partial_extent
            store_full[group_id] = store_full_rule
            store_partial[group_id] = store_partial_rule

        self.full = partial(_dispatch, load_full)
        self.partial = partial(_dispatch, load_partial)
        self.store_full = partial(_dispatch, store_full)
        self.store_partial = partial(_dispatch, store_partial)


def _bind_contiguous_group(
    *,
    group_id: int,
    base_addresses: Sequence[int],
    block_lengths: Sequence[int],
    block_strides: Sequence[int],
    layer_entry_offsets: Sequence[int],
    block_size: int,
    physical_layers: tuple[int, ...],
    align_state: bool,
    consumer_pipeline_partitions: tuple[int, ...] | None,
    store_pipeline_ranks: tuple[int, ...] | None,
    data_plane: MemoryDataPlane,
    object_size: int | None,
    object_offset: int | None,
) -> tuple[Callable, Callable, Callable, Callable]:
    _validate_registration(group_id, base_addresses, block_lengths, block_strides, layer_entry_offsets, physical_layers)
    bases = _readonly(base_addresses)
    lengths = _readonly(block_lengths)
    strides = _readonly(block_strides)
    layer_bounds = {
        layer_id: (int(layer_entry_offsets[index]), int(layer_entry_offsets[index + 1]))
        for index, layer_id in enumerate(physical_layers)
    }
    all_entries = ((0, len(bases)),)
    store_entries = _pipeline_entry_slices(
        physical_layers,
        layer_bounds,
        consumer_pipeline_partitions,
        store_pipeline_ranks,
        len(bases),
    )

    load: Callable
    store: Callable
    if data_plane == "bulk":
        load = partial(
            _bulk_contiguous_ranges,
            bases=bases,
            lengths=lengths,
            strides=strides,
            block_size=block_size,
            align_state=align_state,
            entry_slices=all_entries,
        )
        store = (
            load
            if store_entries == all_entries
            else partial(
                _bulk_contiguous_ranges,
                bases=bases,
                lengths=lengths,
                strides=strides,
                block_size=block_size,
                align_state=align_state,
                entry_slices=store_entries,
            )
        )
    else:
        local_size = int(lengths.sum(dtype=np.uint64))
        if data_plane == "gva":
            if object_size is None:
                raise ValueError(f"GVA group {group_id} has no registered global object layout")
            bound_object_offset = 0 if object_offset is None else object_offset
            if object_size < bound_object_offset + local_size:
                raise ValueError(f"KV cache group {group_id} does not fit inside its registered remote object")
            bound_object_size = object_size
        else:
            bound_object_offset = 0 if object_offset is None else object_offset
            bound_object_size = bound_object_offset + local_size if object_size is None else object_size
            if bound_object_size < bound_object_offset + local_size:
                raise ValueError(f"KV cache group {group_id} does not fit inside its registered remote object")

        offsets: ByteArray = np.empty(len(lengths), dtype=np.uint64)
        offsets[0] = np.uint64(bound_object_offset)
        if len(lengths) > 1:
            np.cumsum(lengths[:-1], out=offsets[1:])
            offsets[1:] += np.uint64(bound_object_offset)
        offsets.flags.writeable = False
        load = partial(
            _object_ranges,
            bases=bases,
            lengths=lengths,
            strides=strides,
            offsets=offsets,
            block_size=block_size,
            align_state=align_state,
            layer_bounds=layer_bounds,
            object_size=bound_object_size,
        )
        store = load

    return (
        partial(load, partial_extent=False),
        partial(load, partial_extent=True),
        partial(store, partial_extent=False),
        partial(store, partial_extent=True),
    )


def _bind_strided_group(
    group_id: int,
    base_addresses: Sequence[int],
    block_lengths: Sequence[int],
    block_strides: Sequence[int],
    block_size: int,
    slice_count: int,
) -> tuple[Callable, Callable]:
    if not (len(base_addresses) == len(block_lengths) == len(block_strides)) or len(base_addresses) == 0:
        raise ValueError(f"KV cache group {group_id} registered misaligned memory arrays")

    bases_by_slice: list[ByteArray] = []
    strides_by_slice: list[ByteArray] = []
    sizes_by_slice: list[ByteArray] = []
    token_indices = np.tile(np.arange(block_size, dtype=np.uint64), len(base_addresses))
    token_indices.flags.writeable = False
    for slice_index in range(slice_count):
        slice_bases: list[int] = []
        slice_strides: list[int] = []
        slice_sizes: list[int] = []
        for base, length, stride in zip(base_addresses, block_lengths, block_strides, strict=True):
            slice_size, remainder = divmod(int(length), block_size * slice_count)
            if remainder or slice_size == 0:
                raise ValueError(
                    f"KV cache group {group_id} block length {length} cannot form {slice_count} Strided slices"
                )
            token_bytes = slice_size * slice_count
            for token_index in range(block_size):
                slice_bases.append(int(base) + slice_index * slice_size + token_index * token_bytes)
                slice_strides.append(int(stride))
                slice_sizes.append(slice_size)
        bases_by_slice.append(_readonly(slice_bases))
        strides_by_slice.append(_readonly(slice_strides))
        sizes_by_slice.append(_readonly(slice_sizes))

    evaluator = partial(
        _bulk_strided_ranges,
        bases_by_slice=tuple(bases_by_slice),
        strides_by_slice=tuple(strides_by_slice),
        sizes_by_slice=tuple(sizes_by_slice),
        token_indices=token_indices,
    )
    return partial(evaluator, partial_extent=False), partial(evaluator, partial_extent=True)


def _validate_registration(
    group_id: int,
    bases: Sequence[int],
    lengths: Sequence[int],
    strides: Sequence[int],
    layer_offsets: Sequence[int],
    physical_layers: tuple[int, ...],
) -> None:
    if len(bases) == 0 or not (len(bases) == len(lengths) == len(strides)):
        raise ValueError(f"KV cache group {group_id} registered misaligned memory arrays")
    if len(layer_offsets) != len(physical_layers) + 1 or layer_offsets[0] != 0 or layer_offsets[-1] != len(bases):
        raise ValueError(f"KV cache group {group_id} registered invalid layer entry bounds")


def _pipeline_entry_slices(
    physical_layers: tuple[int, ...],
    layer_bounds: Mapping[int, tuple[int, int]],
    partitions: tuple[int, ...] | None,
    selected_ranks: tuple[int, ...] | None,
    entry_count: int,
) -> tuple[tuple[int, int], ...]:
    if partitions is None or selected_ranks is None:
        return ((0, entry_count),)
    partition_bounds = []
    first_layer = 0
    for layer_count in partitions:
        last_layer = first_layer + layer_count
        partition_bounds.append((first_layer, last_layer))
        first_layer = last_layer
    slices = []
    for partition_index in selected_ranks:
        first_layer, last_layer = partition_bounds[partition_index]
        selected = tuple(layer_id for layer_id in physical_layers if first_layer <= layer_id < last_layer)
        if not selected:
            raise ValueError("Selected consumer pipeline partition has no registered KV entries")
        slices.append((layer_bounds[selected[0]][0], layer_bounds[selected[-1]][1]))
    if sum(end - start for start, end in slices) != entry_count:
        raise ValueError("Consumer pipeline partitions do not cover the registered KV entries")
    return tuple(slices)


# =============================================================================
# Local Range Evaluation
# =============================================================================
#
# Apply the bound formula to caller-owned Block IDs, token counts, and optional
# layer selection, producing flat local byte ranges grouped by remote object.


def _bulk_contiguous_ranges(
    block_ids: Sequence[int] | ByteArray,
    token_counts: Sequence[int] | ByteArray | None = None,
    *,
    layer_id: int | None = None,
    partial_extent: bool,
    bases: ByteArray,
    lengths: ByteArray,
    strides: ByteArray,
    block_size: int,
    align_state: bool,
    entry_slices: tuple[tuple[int, int], ...],
) -> BulkRangeBatch:
    if layer_id is not None:
        raise ValueError("Bulk memory rules do not select individual layers")
    block_ids_array, counts = _dynamic_rows(block_ids, token_counts, partial_extent)
    address_parts: list[ByteArray] = []
    size_parts: list[ByteArray] = []
    object_range_counts: list[int] = []
    for start, end in entry_slices:
        selected_bases = bases[start:end]
        selected_lengths = lengths[start:end]
        selected_strides = strides[start:end]
        addresses = selected_bases[None, :] + block_ids_array[:, None] * selected_strides[None, :]
        if not partial_extent or align_state:
            sizes = np.broadcast_to(selected_lengths, addresses.shape)
        else:
            assert counts is not None
            sizes = selected_lengths[None, :] * counts[:, None] // np.uint64(block_size)
        address_parts.append(addresses.ravel())
        size_parts.append(sizes.ravel())
        object_range_counts.extend([end - start] * len(block_ids_array))
    return _concat(address_parts), _concat(size_parts), _splits(object_range_counts)


def _bulk_strided_ranges(
    block_ids: Sequence[int] | ByteArray,
    token_counts: Sequence[int] | ByteArray | None = None,
    *,
    layer_id: int | None = None,
    partial_extent: bool,
    bases_by_slice: tuple[ByteArray, ...],
    strides_by_slice: tuple[ByteArray, ...],
    sizes_by_slice: tuple[ByteArray, ...],
    token_indices: ByteArray,
) -> BulkRangeBatch:
    if layer_id is not None:
        raise ValueError("Strided memory rules do not select individual layers")
    block_ids_array, counts = _dynamic_rows(block_ids, token_counts, partial_extent)
    address_parts: list[ByteArray] = []
    size_parts: list[ByteArray] = []
    object_range_counts: list[int] = []
    for bases, strides, sizes in zip(bases_by_slice, strides_by_slice, sizes_by_slice, strict=True):
        addresses = bases[None, :] + block_ids_array[:, None] * strides[None, :]
        if partial_extent:
            assert counts is not None
            active = token_indices[None, :] < counts[:, None]
            address_parts.append(addresses[active])
            size_parts.append(np.broadcast_to(sizes, addresses.shape)[active])
            object_range_counts.extend(active.sum(axis=1, dtype=np.intp).tolist())
        else:
            address_parts.append(addresses.ravel())
            size_parts.append(np.tile(sizes, len(block_ids_array)))
            object_range_counts.extend([len(bases)] * len(block_ids_array))
    return _concat(address_parts), _concat(size_parts), _splits(object_range_counts)


def _object_ranges(
    block_ids: Sequence[int] | ByteArray,
    token_counts: Sequence[int] | ByteArray | None = None,
    *,
    layer_id: int | None = None,
    partial_extent: bool,
    bases: ByteArray,
    lengths: ByteArray,
    strides: ByteArray,
    offsets: ByteArray,
    block_size: int,
    align_state: bool,
    layer_bounds: Mapping[int, tuple[int, int]],
    object_size: int,
) -> RangeBatch:
    block_ids_array, counts = _dynamic_rows(block_ids, token_counts, partial_extent)
    start, end = (0, len(bases)) if layer_id is None else layer_bounds[layer_id]
    selected_bases = bases[start:end]
    selected_lengths = lengths[start:end]
    selected_strides = strides[start:end]
    selected_offsets = offsets[start:end]
    addresses = selected_bases[None, :] + block_ids_array[:, None] * selected_strides[None, :]
    if not partial_extent or align_state:
        sizes = np.broadcast_to(selected_lengths, addresses.shape)
    else:
        assert counts is not None
        sizes = selected_lengths[None, :] * counts[:, None] // np.uint64(block_size)
    range_count = end - start
    return (
        addresses.ravel(),
        sizes.ravel(),
        np.tile(selected_offsets, len(block_ids_array)),
        _splits([range_count] * len(block_ids_array)),
        np.full(len(block_ids_array), object_size, dtype=np.uint64),
    )


def _dynamic_rows(
    block_ids: Sequence[int] | ByteArray,
    token_counts: Sequence[int] | ByteArray | None,
    partial_extent: bool,
) -> tuple[ByteArray, ByteArray | None]:
    ids = np.asarray(block_ids, dtype=np.uint64)
    counts = None if token_counts is None else np.asarray(token_counts, dtype=np.uint64)
    if partial_extent and counts is None:
        raise ValueError("Partial memory ranges require token counts")
    if counts is not None and len(counts) != len(ids):
        raise ValueError("Block IDs and token counts must describe the same rows")
    return ids, counts


# =============================================================================
# Backend Argument Shapes
# =============================================================================
#
# Combine logical keys with evaluated ranges, retain the caller's object
# selection, and emit the exact argument shape required by each Backend API.


def bulk_arguments(
    key_axes: tuple[tuple[str, ...], ...],
    ranges: BulkRangeBatch,
    selected_objects: Sequence[bool] | None = None,
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    """Convert Bulk ranges without constructing Layerwise-only metadata."""

    local, sizes, splits = ranges
    keys = tuple(key for axis in key_axes for key in axis)
    selected = _selected_object_indices(len(keys), splits, selected_objects)
    return (
        [keys[index] for index in selected],
        [local[splits[index] : splits[index + 1]].tolist() for index in selected],
        [sizes[splits[index] : splits[index + 1]].tolist() for index in selected],
    )


def key_range_arguments(
    key_axes: tuple[tuple[str, ...], ...],
    ranges: RangeBatch,
    selected_objects: Sequence[bool] | None = None,
) -> tuple[list[str], list[list[int]], list[list[int]], list[list[int]]]:
    """Convert selected range objects to the existing KeyRange Backend shape."""

    local, sizes, offsets, splits, _ = ranges
    keys = tuple(key for axis in key_axes for key in axis)
    selected = _selected_object_indices(len(keys), splits, selected_objects)
    return (
        [keys[index] for index in selected],
        [local[splits[index] : splits[index + 1]].tolist() for index in selected],
        [sizes[splits[index] : splits[index + 1]].tolist() for index in selected],
        [offsets[splits[index] : splits[index + 1]].tolist() for index in selected],
    )


def required_object_sizes(
    ranges: RangeBatch,
    selected_objects: Sequence[bool] | None = None,
) -> ByteArray:
    """Return the minimum leased extents required by the selected copies."""

    _, sizes, offsets, splits, _ = ranges
    object_count = len(splits) - 1
    if object_count == 0:
        return np.empty(0, dtype=np.uint64)
    required = np.maximum.reduceat(offsets + sizes, splits[:-1])
    selected = _selected_object_indices(object_count, splits, selected_objects)
    return required[np.asarray(selected, dtype=np.intp)]


def gva_arguments(
    ranges: RangeBatch,
    object_bases: Sequence[int] | ByteArray,
    selected_objects: Sequence[bool] | None = None,
) -> tuple[ByteArray, ByteArray, ByteArray]:
    """Evaluate GVA addresses from caller-owned, already admitted bases."""

    local, sizes, offsets, splits, _ = ranges
    object_count = len(splits) - 1
    selected = _selected_object_indices(object_count, splits, selected_objects)
    bases = np.asarray(object_bases, dtype=np.uint64)
    if len(bases) != len(selected):
        raise ValueError("GVA bases must contain exactly the selected remote objects")
    range_counts = np.diff(splits)[np.asarray(selected, dtype=np.intp)]
    if selected:
        selected_ranges = np.concatenate(
            [np.arange(splits[index], splits[index + 1], dtype=np.intp) for index in selected]
        )
    else:
        selected_ranges = np.empty(0, dtype=np.intp)
    return (
        np.repeat(bases, range_counts) + offsets[selected_ranges],
        local[selected_ranges],
        sizes[selected_ranges],
    )


def _selected_object_indices(
    count: int,
    splits: IndexArray,
    selected: Sequence[bool] | None,
) -> tuple[int, ...]:
    if len(splits) != count + 1:
        raise ValueError("Key axes do not align the mapped range objects")
    if selected is None:
        return tuple(range(count))
    mask = np.asarray(selected, dtype=np.bool_)
    if len(mask) != count:
        raise ValueError("Object selection does not align the mapped range objects")
    return tuple(np.flatnonzero(mask).tolist())


# =============================================================================
# Range Grouping and Dispatch
# =============================================================================
#
# Preserve the flat-ranges-plus-splits representation shared by all data
# planes, keep bound layout arrays immutable, and route calls by cache Group.


def _dispatch(functions: Mapping[int, Callable], group_id: int, *args, **kwargs):
    return functions[group_id](*args, **kwargs)


def _splits(object_range_counts: Sequence[int]) -> IndexArray:
    splits: IndexArray = np.empty(len(object_range_counts) + 1, dtype=np.intp)
    splits[0] = 0
    if object_range_counts:
        np.cumsum(np.asarray(object_range_counts, dtype=np.intp), out=splits[1:])
    return splits


def _concat(parts: list[ByteArray]) -> ByteArray:
    if not parts:
        return np.empty(0, dtype=np.uint64)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _readonly(values: Sequence[int]) -> ByteArray:
    result = np.array(values, dtype=np.uint64, copy=True)
    result.flags.writeable = False
    return result
