"""Materialize bound transfer spans into native Backend arguments.

All topology, identity and bounds checks have already happened in the bound plan
and request rows.  This module owns only the final arithmetic and API shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ...program.values.selection import RowIndices, RowSelection, TransferWork
from ...program.values.transfer import ContiguousLayoutPlan, StridedLayoutPlan, TransferRows


@dataclass(frozen=True, slots=True)
class MaterializedRanges:
    keys: list[str]
    addresses: list[list[int]]
    sizes: list[list[int]]
    offsets: list[list[int]]


@dataclass(frozen=True, slots=True)
class MaterializedKeyRanges:
    keys: list[str]
    addresses: list[list[int]]
    sizes: list[list[int]]
    offsets: list[list[int]]


@dataclass(frozen=True, slots=True)
class MaterializedGVA:
    remote_addresses: np.ndarray
    local_addresses: np.ndarray
    sizes: np.ndarray


@dataclass(frozen=True, slots=True)
class ResolvedGVARows:
    """GVA session facts aligned once with a stable request row axis."""

    object_bases: np.ndarray
    object_sizes: np.ndarray
    available: np.ndarray
    validated_selections: frozenset[int]


@dataclass(frozen=True, slots=True)
class ResolvedGVABatch:
    """Dynamic row axes shared by compatible layer submissions."""

    block_ids: np.ndarray
    token_counts: np.ndarray
    object_bases: np.ndarray
    object_sizes: np.ndarray


ResolvedGVASessions = dict[tuple[TransferRows, int], ResolvedGVARows]
GVABatchSignature = tuple[int, tuple[tuple[TransferRows, RowSelection | None, RowIndices], ...]]
ResolvedGVABatches = dict[GVABatchSignature, ResolvedGVABatch]


def materialize_ranges(work: TransferWork) -> MaterializedRanges:
    """Evaluate static affine geometry for the selected request rows."""

    keys: list[str] = []
    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    for span in work.spans:
        rows = span.rows
        if len(span.layout_indices) == 1 and isinstance(
            layout := rows.plan.layouts[layout_index := span.layout_indices[0]], ContiguousLayoutPlan
        ):
            key_axis = rows.keys_by_coordinate[layout.coordinate_index]
            for selected_rows in span.selected_ranges(layout_index):
                row_indexer = _numpy_row_indexer(selected_rows)
                block_ids = rows.block_ids[row_indexer]
                token_counts = rows.memory_token_counts[row_indexer]
                span_addresses = layout.bases[None, :] + block_ids[:, None] * layout.block_strides[None, :]
                span_sizes = token_counts[:, None] * layout.bytes_per_token[None, :]
                row_offsets = layout.offsets.tolist()
                keys.extend(key_axis[row_index] for row_index in selected_rows)
                addresses.extend(span_addresses.tolist())
                sizes.extend(span_sizes.tolist())
                offsets.extend(row_offsets.copy() for _ in selected_rows)
            continue
        for row_index in span.row_indices:
            for layout_index in span.layout_indices:
                if not span.includes(layout_index, row_index):
                    continue
                layout = rows.plan.layouts[layout_index]
                key_axis = rows.keys_by_coordinate[layout.coordinate_index]
                row_addresses, row_sizes, row_offsets = _materialize_row(
                    layout,
                    int(rows.block_ids[row_index]),
                    int(rows.memory_token_counts[row_index]),
                )
                key = key_axis[row_index]
                keys.append(key)
                addresses.append(row_addresses)
                sizes.append(row_sizes)
                offsets.append(row_offsets)
    return MaterializedRanges(keys, addresses, sizes, offsets)


def materialize_key_ranges(work: TransferWork) -> MaterializedKeyRanges:
    """Coalesce already-enumerated ranges by Backend object key."""

    ranges = materialize_ranges(work)
    if len(set(ranges.keys)) == len(ranges.keys):
        return MaterializedKeyRanges(ranges.keys, ranges.addresses, ranges.sizes, ranges.offsets)
    key_indices: dict[str, int] = {}
    keys: list[str] = []
    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    for key, row_addresses, row_sizes, row_offsets in zip(
        ranges.keys,
        ranges.addresses,
        ranges.sizes,
        ranges.offsets,
        strict=True,
    ):
        index = key_indices.get(key)
        if index is None:
            index = len(keys)
            key_indices[key] = index
            keys.append(key)
            addresses.append([])
            sizes.append([])
            offsets.append([])
        addresses[index].extend(row_addresses)
        sizes[index].extend(row_sizes)
        offsets[index].extend(row_offsets)
    return MaterializedKeyRanges(
        keys,
        addresses,
        sizes,
        offsets,
    )


def materialize_gva(
    work: TransferWork,
    sessions: Mapping[str, tuple[int, int] | None],
    resolved_sessions: ResolvedGVASessions | None = None,
    resolved_batches: ResolvedGVABatches | None = None,
) -> MaterializedGVA:
    """Resolve leased object bases only at the GVA Backend boundary."""

    resolved_sessions = resolve_gva_sessions(work, sessions, resolved_sessions)
    contiguous = _materialize_contiguous_gva(
        work,
        resolved_sessions,
        {} if resolved_batches is None else resolved_batches,
    )
    if contiguous is not None:
        return contiguous
    remote_addresses = []
    local_addresses = []
    sizes = []
    for span in work.spans:
        rows = span.rows
        if len(span.layout_indices) == 1 and isinstance(
            layout := rows.plan.layouts[layout_index := span.layout_indices[0]], ContiguousLayoutPlan
        ):
            key_axis = rows.keys_by_coordinate[layout.coordinate_index]
            for selected_rows in span.selected_ranges(layout_index):
                row_indexer = _numpy_row_indexer(selected_rows)
                block_ids = rows.block_ids[row_indexer]
                token_counts = rows.memory_token_counts[row_indexer]
                span_addresses = layout.bases[None, :] + block_ids[:, None] * layout.block_strides[None, :]
                span_sizes = token_counts[:, None] * layout.bytes_per_token[None, :]
                session_rows = resolved_sessions[(rows, layout.coordinate_index)]
                object_bases = session_rows.object_bases[row_indexer]
                object_sizes = session_rows.object_sizes[row_indexer]
                if np.any(layout.offsets[None, :] + span_sizes > object_sizes[:, None]):
                    raise RuntimeError("GVA ranges exceed a leased object")
                span_remote = object_bases[:, None] + layout.offsets[None, :]
                remote_addresses.append(span_remote.ravel())
                local_addresses.append(span_addresses.ravel())
                sizes.append(span_sizes.ravel())
            continue
        for row_index in span.row_indices:
            for layout_index in span.layout_indices:
                if not span.includes(layout_index, row_index):
                    continue
                layout = rows.plan.layouts[layout_index]
                key_axis = rows.keys_by_coordinate[layout.coordinate_index]
                row_addresses, row_sizes, row_offsets = _materialize_row(
                    layout,
                    int(rows.block_ids[row_index]),
                    int(rows.memory_token_counts[row_index]),
                )
                key = key_axis[row_index]
                session_rows = resolved_sessions[(rows, layout.coordinate_index)]
                object_base = int(session_rows.object_bases[row_index])
                object_size = int(session_rows.object_sizes[row_index])
                if any(offset + size > object_size for offset, size in zip(row_offsets, row_sizes, strict=True)):
                    raise RuntimeError(f"GVA ranges for {key!r} exceed the leased object")
                remote_addresses.append(np.asarray(row_offsets, dtype=np.uint64) + np.uint64(object_base))
                local_addresses.append(np.asarray(row_addresses, dtype=np.uint64))
                sizes.append(np.asarray(row_sizes, dtype=np.uint64))
    return MaterializedGVA(
        _concatenate_gva_parts(remote_addresses),
        _concatenate_gva_parts(local_addresses),
        _concatenate_gva_parts(sizes),
    )


def resolve_gva_sessions(
    work: TransferWork,
    sessions: Mapping[str, tuple[int, int] | None],
    resolved_sessions: ResolvedGVASessions | None = None,
) -> ResolvedGVASessions:
    """Align GVA sessions with each row axis once, then reuse them by layer."""

    resolved = {} if resolved_sessions is None else resolved_sessions
    for span in work.spans:
        rows = span.rows
        for layout_index in span.layout_indices:
            coordinate_index = rows.plan.layouts[layout_index].coordinate_index
            cache_key = (rows, coordinate_index)
            keys = rows.keys_by_coordinate[coordinate_index]
            session_rows = resolved.get(cache_key)
            selection_id = 0 if span.selection is None else id(span.selection)
            if session_rows is None:
                object_bases: np.ndarray = np.zeros(rows.row_count, dtype=np.uint64)
                object_sizes: np.ndarray = np.zeros(rows.row_count, dtype=np.uint64)
                available: np.ndarray = np.zeros(rows.row_count, dtype=np.bool_)
                for row_index, key in enumerate(keys):
                    session = sessions.get(key)
                    if session is None:
                        continue
                    object_bases[row_index], object_sizes[row_index] = session
                    available[row_index] = True
                for values in (object_bases, object_sizes, available):
                    values.flags.writeable = False
                session_rows = ResolvedGVARows(object_bases, object_sizes, available, frozenset())
                resolved[cache_key] = session_rows
            if selection_id in session_rows.validated_selections:
                continue
            selected_ranges = (
                (range(rows.row_count),)
                if span.selection is None
                else span.selection.runs_by_coordinate[coordinate_index]
            )
            for row_range in selected_ranges:
                available = session_rows.available[row_range]
                if not np.all(available):
                    unavailable = row_range.start + int(np.argmax(~available))
                    key = keys[unavailable]
                    raise RuntimeError(f"GVA session for {key!r} is unavailable")
            resolved[cache_key] = ResolvedGVARows(
                session_rows.object_bases,
                session_rows.object_sizes,
                session_rows.available,
                session_rows.validated_selections | {selection_id},
            )
    return resolved


def _materialize_contiguous_gva(
    work: TransferWork,
    resolved_sessions: ResolvedGVASessions,
    resolved_batches: ResolvedGVABatches,
) -> MaterializedGVA | None:
    """Batch compatible request spans before one vectorized GVA calculation."""

    if not work.spans or any(len(span.layout_indices) != 1 for span in work.spans):
        return None
    layout_indices = tuple(span.layout_indices[0] for span in work.spans)
    layouts = tuple(
        span.rows.plan.layouts[layout_index] for span, layout_index in zip(work.spans, layout_indices, strict=True)
    )
    layout = layouts[0]
    if not isinstance(layout, ContiguousLayoutPlan) or any(item is not layout for item in layouts[1:]):
        return None

    signature: GVABatchSignature = (
        layout.coordinate_index,
        tuple((span.rows, span.selection, span.row_indices) for span in work.spans),
    )
    batch = resolved_batches.get(signature)
    if batch is None:
        block_ids = []
        token_counts = []
        object_bases = []
        object_sizes = []
        for span, layout_index in zip(work.spans, layout_indices, strict=True):
            rows = span.rows
            session_rows = resolved_sessions[(rows, layout.coordinate_index)]
            for selected_rows in span.selected_ranges(layout_index):
                row_indexer = _numpy_row_indexer(selected_rows)
                block_ids.append(rows.block_ids[row_indexer])
                token_counts.append(rows.memory_token_counts[row_indexer])
                object_bases.append(session_rows.object_bases[row_indexer])
                object_sizes.append(session_rows.object_sizes[row_indexer])
        if block_ids:
            batch = ResolvedGVABatch(
                _concatenate_gva_parts(block_ids),
                _concatenate_gva_parts(token_counts),
                _concatenate_gva_parts(object_bases),
                _concatenate_gva_parts(object_sizes),
            )
            resolved_batches[signature] = batch
    if batch is None:
        empty: np.ndarray = np.empty(0, dtype=np.uint64)
        return MaterializedGVA(empty, empty, empty)

    local = layout.bases[None, :] + batch.block_ids[:, None] * layout.block_strides[None, :]
    sizes = batch.token_counts[:, None] * layout.bytes_per_token[None, :]
    if np.any(layout.offsets[None, :] + sizes > batch.object_sizes[:, None]):
        raise RuntimeError("GVA ranges exceed a leased object")
    remote = batch.object_bases[:, None] + layout.offsets[None, :]
    return MaterializedGVA(remote.ravel(), local.ravel(), sizes.ravel())


def _concatenate_gva_parts(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=np.uint64)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _numpy_row_indexer(row_indices: range | tuple[int, ...]) -> slice | np.ndarray:
    if isinstance(row_indices, range):
        return slice(row_indices.start, row_indices.stop)
    return np.asarray(row_indices, dtype=np.intp)


def _materialize_row(
    layout: ContiguousLayoutPlan | StridedLayoutPlan,
    block_id: int,
    token_count: int,
) -> tuple[list[int], list[int], list[int]]:
    if isinstance(layout, ContiguousLayoutPlan):
        addresses = layout.bases + np.uint64(block_id) * layout.block_strides
        sizes = np.uint64(token_count) * layout.bytes_per_token
        return addresses.tolist(), sizes.tolist(), layout.offsets.tolist()

    active = layout.token_indices < token_count
    addresses = layout.bases[active] + np.uint64(block_id) * layout.block_strides[active]
    return addresses.tolist(), layout.sizes[active].tolist(), layout.offsets[active].tolist()
