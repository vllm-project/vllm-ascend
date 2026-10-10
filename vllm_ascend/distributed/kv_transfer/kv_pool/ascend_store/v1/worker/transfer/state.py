"""Step and batch state shared by concrete Worker routes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ...protocol.transfer import KVTransferStep
from .batch import KeyAxes, KVGroupBatch, LayerStoreGroup, TransferSource
from .rows import BlockRows, StoreCandidateRows


@dataclass(slots=True)
class KVPoolStepContext:
    """Adapt one upstream hook batch without owning cross-step executions."""

    step: KVTransferStep
    failed_request_ids: set[str] = field(default_factory=set)
    failed_block_ids: set[int] = field(default_factory=set)
    failed_sources: list[TransferSource] = field(default_factory=list)
    store_submitted: bool = False


@dataclass(frozen=True, slots=True)
class StoreGroupCandidates:
    """One cache Group's list-backed rows and object-key coordinates."""

    group_id: int
    rows_by_request: tuple[StoreCandidateRows, ...]
    key_axes: KeyAxes

    @property
    def row_count(self) -> int:
        return sum(len(rows[0]) for rows in self.rows_by_request)

    @property
    def object_count(self) -> int:
        return len(self.key_axes) * self.row_count


@dataclass(frozen=True, slots=True)
class StoreCandidates:
    """All pre-admission Store objects in canonical group/axis/row order."""

    groups: tuple[StoreGroupCandidates, ...]


@dataclass(frozen=True, slots=True)
class MaterializedStoreGroup:
    """One domain Group plus its optional Layerwise execution operands."""

    batch: KVGroupBatch
    layer_store: LayerStoreGroup | None


def store_candidate_keys(candidates: StoreCandidates) -> tuple[str, ...]:
    """Flatten candidate keys in their canonical group/axis/row order."""

    return tuple(key for group in candidates.groups for axis in group.key_axes for key in axis)


def select_store_candidate_objects(
    candidate_keys: tuple[str, ...], accepted: set[str], *, claim_once: bool
) -> tuple[bool, ...]:
    """Map key admission back to ordered candidate objects."""

    if not claim_once:
        return tuple(key in accepted for key in candidate_keys)
    claimed = set()
    selected = []
    for key in candidate_keys:
        include = key in accepted and key not in claimed
        selected.append(include)
        if include:
            claimed.add(key)
    return tuple(selected)


def merge_key_axes(group_id: int, key_parts: list[KeyAxes]) -> KeyAxes:
    """Concatenate per-request key coordinates without changing axes."""

    key_axis_count = len(key_parts[0]) if key_parts else 0
    if any(len(keys) != key_axis_count for keys in key_parts):
        raise RuntimeError(f"Cache group {group_id} changed its key coordinate count at runtime")
    return tuple(tuple(key for keys in key_parts for key in keys[axis_index]) for axis_index in range(key_axis_count))


def split_selection(selection: tuple[bool, ...], axis_count: int) -> tuple[tuple[bool, ...], ...]:
    """Restore flattened object admission to key-axis coordinates."""

    if axis_count == 0:
        return ()
    if len(selection) % axis_count:
        raise RuntimeError("Store selection does not form aligned key axes")
    row_count = len(selection) // axis_count
    return tuple(selection[index * row_count : (index + 1) * row_count] for index in range(axis_count))


def compact_single_axis_rows(
    rows_by_request: tuple[StoreCandidateRows, ...],
    key_axis: tuple[str, ...],
    selected_rows: tuple[bool, ...],
) -> tuple[list[int], list[int], list[int], tuple[str, ...]]:
    """Compact a one-axis Store batch while preserving request splits."""

    token_counts: list[int] = []
    block_ids: list[int] = []
    selected_keys: list[str] = []
    request_splits = [0]
    row_offset = 0
    for counts, _hashes, ids in rows_by_request:
        request_row_count = len(counts)
        selection = selected_rows[row_offset : row_offset + request_row_count]
        keys = key_axis[row_offset : row_offset + request_row_count]
        for count, block_id, key, keep in zip(counts, ids, keys, selection, strict=True):
            if keep:
                token_counts.append(count)
                block_ids.append(block_id)
                selected_keys.append(key)
        request_splits.append(len(block_ids))
        row_offset += request_row_count
    return token_counts, block_ids, request_splits, tuple(selected_keys)


def empty_candidate_rows() -> StoreCandidateRows:
    return [], [], []


def empty_rows() -> BlockRows:
    empty: np.ndarray = np.empty(0, dtype=np.uint64)
    empty.flags.writeable = False
    return empty, (), empty


def concat_uint64(parts: Sequence[np.ndarray]) -> np.ndarray:
    if not parts:
        result: np.ndarray = np.empty(0, dtype=np.uint64)
    elif len(parts) == 1:
        result = parts[0]
    else:
        result = np.concatenate(parts)
        result.flags.writeable = False
    return result


def readonly_indices(values: Sequence[int]) -> np.ndarray:
    result = np.asarray(values, dtype=np.intp)
    result.flags.writeable = False
    return result


def readonly_uint64(values: Sequence[int]) -> np.ndarray:
    result = np.asarray(values, dtype=np.uint64)
    result.flags.writeable = False
    return result


def readonly_bool(values: Sequence[bool]) -> np.ndarray:
    result = np.asarray(values, dtype=np.bool_)
    result.flags.writeable = False
    return result


__all__ = (
    "KVPoolStepContext",
    "MaterializedStoreGroup",
    "StoreCandidates",
    "StoreGroupCandidates",
)
