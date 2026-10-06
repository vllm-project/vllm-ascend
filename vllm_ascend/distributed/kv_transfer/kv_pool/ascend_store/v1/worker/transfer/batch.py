"""Carry one Worker transfer's rows from projection selection to Backend I/O.

The batch values retain dynamic Block IDs, token counts, keys, request
ownership, and current object selection together with the minimal bound facts
needed to interpret them: physical layers and remote object size. Reusable
layout formulas remain in the bound variant projection; session progress and
visibility fences remain in Timeline state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ByteArray: TypeAlias = NDArray[np.uint64]
IndexArray: TypeAlias = NDArray[np.intp]
BoolArray: TypeAlias = NDArray[np.bool_]
KeyAxes: TypeAlias = tuple[tuple[str, ...], ...]


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
    physical_layer_ids_by_axis: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.physical_layer_ids_by_axis is not None and len(self.physical_layer_ids_by_axis) != len(self.key_axes):
            raise ValueError("Per-axis physical layers must align the key axes")

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
        layer_ids: tuple[int, ...]
        if layer_id is not None:
            layer_ids = (layer_id,)
        elif self.physical_layer_ids_by_axis is None:
            layer_ids = self.physical_layer_ids
        else:
            layer_ids = self.physical_layer_ids_by_axis[axis_index]
        request_index = self.request_index(object_index)
        key = self.key_axes[axis_index][row_index]
        return TransferSource(self, object_index, row_index, request_index, layer_ids, key)


@dataclass(frozen=True, slots=True)
class LayerTransferGroup:
    """One Group's selected object rows shared by every Layerwise copy."""

    batch: KVGroupBatch
    keys: tuple[str, ...]
    block_ids: ByteArray
    token_counts: ByteArray
    object_indices: range | IndexArray

    @property
    def group_id(self) -> int:
        return self.batch.group_id

    @property
    def physical_layer_ids(self) -> tuple[int, ...]:
        return self.batch.physical_layer_ids


@dataclass(frozen=True, slots=True)
class LayerTransferPlan:
    """Selected object rows and layer membership lowered once per session."""

    batch: KVTransferBatch
    groups: tuple[LayerTransferGroup, ...]
    groups_by_layer: dict[int, tuple[LayerTransferGroup, ...]]
    keys_by_layer: dict[int, list[str]]
    object_bases_by_group: dict[int, ByteArray] | None = None


def make_layer_transfer_group(
    group: KVGroupBatch,
    selected_object_indices: IndexArray | None = None,
) -> LayerTransferGroup:
    """Align one materialized Group's dynamic rows with its selected objects."""

    keys = group.selected_keys()
    if not keys:
        return LayerTransferGroup(
            group,
            (),
            group.block_ids[:0],
            group.token_counts[:0],
            range(0),
        )
    if selected_object_indices is None and group.selection is not None:
        selected_object_indices = np.flatnonzero(group.selection)
    if selected_object_indices is None:
        axis_count = len(group.key_axes)
        object_indices: range | IndexArray = range(group.object_count)
        block_ids = group.block_ids if axis_count == 1 else np.tile(group.block_ids, axis_count)
        token_counts = group.token_counts if axis_count == 1 else np.tile(group.token_counts, axis_count)
    else:
        row_indices = selected_object_indices % group.row_count
        block_ids = group.block_ids[row_indices]
        token_counts = group.token_counts[row_indices]
        selected_object_indices.flags.writeable = False
        object_indices = selected_object_indices
    if not (len(keys) == len(block_ids) == len(token_counts)):
        raise RuntimeError(f"Layerwise group {group.group_id} has misaligned keys and object rows")
    block_ids.flags.writeable = False
    token_counts.flags.writeable = False
    return LayerTransferGroup(group, keys, block_ids, token_counts, object_indices)


def make_layer_transfer_plan(
    batch: KVTransferBatch,
    groups: tuple[LayerTransferGroup, ...],
) -> LayerTransferPlan:
    """Compile reusable group membership and key axes without touching object rows."""

    active_groups = tuple(group for group in groups if group.keys)
    mutable_groups_by_layer: dict[int, list[LayerTransferGroup]] = {}
    for group in active_groups:
        for layer_id in group.physical_layer_ids:
            mutable_groups_by_layer.setdefault(layer_id, []).append(group)

    groups_by_layer = {layer_id: tuple(layer_groups) for layer_id, layer_groups in mutable_groups_by_layer.items()}
    keys_by_membership: dict[tuple[int, ...], list[str]] = {}
    keys_by_layer = {}
    for layer_id, layer_groups in groups_by_layer.items():
        membership = tuple(group.group_id for group in layer_groups)
        if membership not in keys_by_membership:
            keys_by_membership[membership] = [key for group in layer_groups for key in group.keys]
        keys_by_layer[layer_id] = keys_by_membership[membership]
    return LayerTransferPlan(batch, active_groups, groups_by_layer, keys_by_layer)


@dataclass(frozen=True, slots=True)
class KVTransferBatch:
    """One Worker invocation with request ownership retained on Group rows."""

    request_ids: tuple[str, ...]
    groups: tuple[KVGroupBatch, ...]
    store_job_ids: tuple[int | None, ...] | None = None
    selected_key_values: tuple[str, ...] | None = None
    layer_store_plan: LayerTransferPlan | None = None

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
        layer_store_plan = (
            None
            if self.layer_store_plan is None
            else make_layer_transfer_plan(
                replace(self, groups=groups, selected_key_values=selected_keys, layer_store_plan=None),
                tuple(make_layer_transfer_group(group) for group in groups),
            )
        )
        return replace(
            self,
            groups=groups,
            selected_key_values=selected_keys,
            layer_store_plan=layer_store_plan,
        )

    def for_layer(self, layer_id: int) -> KVTransferBatch:
        groups = tuple(group for group in self.groups if layer_id in group.physical_layer_ids)
        selected_keys = tuple(key for group in groups for key in group.selected_keys())
        return replace(self, groups=groups, selected_key_values=selected_keys, layer_store_plan=None)


# Store builds this plan during Worker materialization. Load builds the same
# shape after session admission, so both directions retain one cross-layer
# lowering without sharing mutable lifecycle state.
LayerStoreGroup = LayerTransferGroup
LayerStorePlan = LayerTransferPlan
make_layer_store_group = make_layer_transfer_group


def make_layer_store_plan(groups: tuple[LayerTransferGroup, ...], batch: KVTransferBatch) -> LayerTransferPlan:
    return make_layer_transfer_plan(batch, groups)


def make_layer_load_plan(batch: KVTransferBatch) -> LayerTransferPlan:
    return make_layer_transfer_plan(batch, tuple(make_layer_transfer_group(group) for group in batch.groups))


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
