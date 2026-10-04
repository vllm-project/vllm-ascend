"""Carry one Runtime invocation's rows from rule selection to Backend I/O.

The batch values retain dynamic Block IDs, token counts, keys, request
ownership, and current object selection together with the minimal bound facts
needed to interpret them: physical layers and remote object size. Reusable
layout formulas remain in ``KVPoolRules``; session progress and visibility
fences remain in Timeline state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..rules.identity import KeyAxes

ByteArray: TypeAlias = NDArray[np.uint64]
IndexArray: TypeAlias = NDArray[np.intp]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True, slots=True, eq=False)
class KVGroupBatch:
    """One cache Group's dynamic row axis, shared across every layer."""

    group_id: int
    block_ids: ByteArray
    token_counts: ByteArray
    key_axes: KeyAxes
    request_splits: IndexArray
    physical_layer_ids: tuple[int, ...]
    object_size: int
    selected_objects: BoolArray | None = None
    selected_key_values: tuple[str, ...] | None = None

    @property
    def row_count(self) -> int:
        return len(self.block_ids)

    @property
    def object_count(self) -> int:
        return len(self.key_axes) * self.row_count

    @property
    def selection(self) -> BoolArray | None:
        return self.selected_objects

    def selected_keys(self) -> tuple[str, ...]:
        if self.selected_key_values is not None:
            return self.selected_key_values
        if self.selected_objects is None:
            return tuple(key for axis in self.key_axes for key in axis)
        return tuple(
            key
            for axis_index, axis in enumerate(self.key_axes)
            for row_index, key in enumerate(axis)
            if self.selected_objects[axis_index * self.row_count + row_index]
        )

    def select_keys(self, accepted: set[str], claimed: set[str] | None = None) -> KVGroupBatch:
        current: BoolArray = (
            np.ones(self.object_count, dtype=np.bool_) if self.selected_objects is None else self.selected_objects
        )
        selected: BoolArray = np.zeros(self.object_count, dtype=np.bool_)
        selected_keys: list[str] = []
        for axis_index, axis in enumerate(self.key_axes):
            axis_offset = axis_index * self.row_count
            for row_index, key in enumerate(axis):
                index = axis_offset + row_index
                if not current[index] or key not in accepted or (claimed is not None and key in claimed):
                    continue
                selected[index] = True
                selected_keys.append(key)
                if claimed is not None:
                    claimed.add(key)
        selected.flags.writeable = False
        return replace(self, selected_objects=selected, selected_key_values=tuple(selected_keys))

    def request_index(self, object_index: int) -> int:
        row_index = object_index % self.row_count
        return int(np.searchsorted(self.request_splits, row_index, side="right") - 1)

    def source(self, object_index: int, layer_id: int | None) -> TransferSource:
        axis_index, row_index = divmod(object_index, self.row_count)
        layer_ids = self.physical_layer_ids if layer_id is None else (layer_id,)
        request_index = self.request_index(object_index)
        key = self.key_axes[axis_index][row_index]
        return TransferSource(self, object_index, row_index, request_index, layer_ids, key)


@dataclass(frozen=True, slots=True)
class KVTransferBatch:
    """One Runtime invocation with request ownership retained on Group rows."""

    request_ids: tuple[str, ...]
    groups: tuple[KVGroupBatch, ...]
    store_job_ids: tuple[int | None, ...] | None = None
    selected_key_values: tuple[str, ...] | None = None

    @property
    def empty(self) -> bool:
        return not self.selected_keys()

    def selected_keys(self) -> tuple[str, ...]:
        if self.selected_key_values is not None:
            return self.selected_key_values
        return tuple(key for group in self.groups for key in group.selected_keys())

    def select_keys(self, accepted: set[str], *, claim_once: bool = False) -> KVTransferBatch:
        claimed: set[str] | None = set() if claim_once else None
        groups = tuple(group.select_keys(accepted, claimed) for group in self.groups)
        selected_keys = tuple(key for group in groups for key in group.selected_keys())
        return replace(self, groups=groups, selected_key_values=selected_keys)

    def for_layer(self, layer_id: int) -> KVTransferBatch:
        groups = tuple(group for group in self.groups if layer_id in group.physical_layer_ids)
        selected_keys = tuple(key for group in groups for key in group.selected_keys())
        return replace(self, groups=groups, selected_key_values=selected_keys)


@dataclass(frozen=True, slots=True)
class TransferSource:
    """Immutable request and cache provenance for one Backend object result."""

    group: KVGroupBatch
    object_index: int
    row_index: int
    request_index: int
    physical_layer_ids: tuple[int, ...]
    key: str

    @property
    def group_id(self) -> int:
        return self.group.group_id

    @property
    def block_id(self) -> int:
        return int(self.group.block_ids[self.row_index])
