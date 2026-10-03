"""Request rows shared by Runtime selection, Timeline fences and Backend I/O.

The two batch values contain only facts produced by the current request batch:
Block IDs, token counts, keys, request ownership and the current object
selection.  Static layout formulas remain in ``KVPoolRules``; layer order and
session progress remain in Timeline state.
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
        for axis_index, axis in enumerate(self.key_axes):
            axis_offset = axis_index * self.row_count
            for row_index, key in enumerate(axis):
                index = axis_offset + row_index
                if not current[index] or key not in accepted or (claimed is not None and key in claimed):
                    continue
                selected[index] = True
                if claimed is not None:
                    claimed.add(key)
        selected.flags.writeable = False
        return replace(self, selected_objects=selected)

    def request_index(self, object_index: int) -> int:
        row_index = object_index % self.row_count
        return int(np.searchsorted(self.request_splits, row_index, side="right") - 1)

    def source(self, object_index: int, layer_id: int | None) -> TransferSource:
        axis_index, row_index = divmod(object_index, self.row_count)
        layer_ids = self.physical_layer_ids if layer_id is None else (layer_id,)
        return TransferSource(
            self,
            object_index,
            row_index,
            self.request_index(object_index),
            layer_ids,
            self.key_axes[axis_index][row_index],
        )


@dataclass(frozen=True, slots=True)
class KVTransferBatch:
    """One Runtime invocation with request ownership retained on Group rows."""

    request_ids: tuple[str, ...]
    groups: tuple[KVGroupBatch, ...]
    store_job_ids: tuple[int | None, ...] | None = None

    @property
    def empty(self) -> bool:
        return not any(group.selected_keys() for group in self.groups)

    def selected_keys(self) -> tuple[str, ...]:
        return tuple(key for group in self.groups for key in group.selected_keys())

    def select_keys(self, accepted: set[str], *, claim_once: bool = False) -> KVTransferBatch:
        claimed: set[str] | None = set() if claim_once else None
        return replace(
            self,
            groups=tuple(group.select_keys(accepted, claimed) for group in self.groups),
        )

    def for_layer(self, layer_id: int) -> KVTransferBatch:
        return replace(
            self,
            groups=tuple(group for group in self.groups if layer_id in group.physical_layer_ids),
        )


@dataclass(frozen=True, slots=True)
class TransferSource:
    """Provenance for one Backend object result, created only at submission."""

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
