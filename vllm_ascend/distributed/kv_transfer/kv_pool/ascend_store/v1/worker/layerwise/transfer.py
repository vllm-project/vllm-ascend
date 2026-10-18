"""Prepare reusable Layerwise plans and interpret layer copy results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...infrastructure.mapping import LayerwiseMapping, LayerwiseSelection
from ..transfer.batch import ByteArray, IndexArray, KVGroupBatch, KVTransferBatch
from ..transfer.completion import LoadCompletion, TransferEvidence, _load_completions


@dataclass(frozen=True, slots=True)
class LayerTransferGroup:
    """One Group's selected object rows shared by every Layerwise copy."""

    batch: KVGroupBatch
    keys: tuple[str, ...]
    block_ids: ByteArray
    token_counts: ByteArray
    object_indices: range | IndexArray
    # Load resolves ownership after session admission; Store does not need this lowering.
    request_indices: tuple[int, ...] | None = None
    selection: LayerwiseSelection | None = None

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


@dataclass(frozen=True, slots=True)
class LayerStoreResult:
    """One layer copy result; scalar values apply to every prepared key."""

    result_codes: tuple[int | None, ...] | int | None
    source_release_confirmed: tuple[bool, ...] | bool
    error: Exception | None = None


ObjectBaseResolver = Callable[[KVGroupBatch, tuple[str, ...]], np.ndarray]


def prepare_layer_store(
    batch: KVTransferBatch,
    mappings: dict[int, LayerwiseMapping],
    resolve_object_bases: ObjectBaseResolver | None = None,
) -> LayerTransferPlan:
    """Lower admitted Store objects once; every physical layer reuses this plan."""

    return _prepare_layer_plan(batch, mappings, resolve_object_bases, resolve_requests=False)


def prepare_layer_load(
    batch: KVTransferBatch,
    mappings: dict[int, LayerwiseMapping],
    resolve_object_bases: ObjectBaseResolver | None = None,
) -> LayerTransferPlan:
    """Lower selected Load rows once after session admission."""

    return _prepare_layer_plan(batch, mappings, resolve_object_bases, resolve_requests=True)


def _prepare_layer_plan(
    batch: KVTransferBatch,
    mappings: dict[int, LayerwiseMapping],
    resolve_object_bases: ObjectBaseResolver | None,
    *,
    resolve_requests: bool,
) -> LayerTransferPlan:
    groups = []
    object_bases_by_group: dict[int, ByteArray] | None = {} if resolve_object_bases is not None else None
    for batch_group in batch.groups:
        selection = mappings[batch_group.group_id].prepare(
            batch_group.key_axes, batch_group.block_ids, selected_keys=batch_group.selected_objects
        )
        group = make_layer_transfer_group(batch_group, selection=selection, resolve_requests=resolve_requests)
        groups.append(group)
        if resolve_object_bases is not None and group.keys:
            assert object_bases_by_group is not None
            object_bases_by_group[group.group_id] = resolve_object_bases(batch_group, group.keys)
    return make_layer_transfer_plan(batch, tuple(groups), object_bases_by_group)


def make_layer_transfer_group(
    group: KVGroupBatch,
    selected_object_indices: IndexArray | None = None,
    *,
    selection: LayerwiseSelection | None = None,
    resolve_requests: bool = False,
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
            () if resolve_requests else None,
            selection,
        )
    if selected_object_indices is None and group.selected_objects is not None:
        selected_object_indices = np.flatnonzero(group.selected_objects)
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
    block_ids.flags.writeable = False
    token_counts.flags.writeable = False
    request_indices = tuple(group.request_index(int(index)) for index in object_indices) if resolve_requests else None
    return LayerTransferGroup(group, keys, block_ids, token_counts, object_indices, request_indices, selection)


def make_layer_transfer_plan(
    batch: KVTransferBatch,
    groups: tuple[LayerTransferGroup, ...],
    object_bases_by_group: dict[int, ByteArray] | None = None,
) -> LayerTransferPlan:
    """Compile reusable group membership and key axes without touching object rows."""

    active_groups = []
    mutable_groups_by_layer: dict[int, list[LayerTransferGroup]] = {}
    for group in groups:
        if not group.keys:
            continue
        active_groups.append(group)
        for layer_id in group.physical_layer_ids:
            mutable_groups_by_layer.setdefault(layer_id, []).append(group)

    groups_by_layer = {}
    keys_by_membership: dict[tuple[int, ...], list[str]] = {}
    keys_by_layer = {}
    for layer_id, layer_groups in mutable_groups_by_layer.items():
        groups_by_layer[layer_id] = tuple(layer_groups)
        membership = tuple(group.group_id for group in layer_groups)
        if membership not in keys_by_membership:
            keys_by_membership[membership] = [key for group in layer_groups for key in group.keys]
        keys_by_layer[layer_id] = keys_by_membership[membership]
    return LayerTransferPlan(batch, tuple(active_groups), groups_by_layer, keys_by_layer, object_bases_by_group)


def make_layer_load_plan(batch: KVTransferBatch) -> LayerTransferPlan:
    groups = tuple(make_layer_transfer_group(group, resolve_requests=True) for group in batch.groups)
    return make_layer_transfer_plan(batch, groups)


def _layer_load_completions(
    plan: LayerTransferPlan,
    layer_id: int,
    failed_codes: dict[str, int] | int | None,
) -> tuple[LoadCompletion, ...]:
    """Interpret failed keys or a batch-wide GVA code using the plan's original row ownership.

    KeyRange supplies only nonzero results, keyed after Backend argument merging.
    A failed key must still identify every original destination, including duplicates
    across requests. None means completion is unknown for every participating object.
    """

    if failed_codes == 0 or failed_codes == {}:
        return _load_completions(plan.batch, ())
    failures = []
    for group in plan.groups_by_layer[layer_id]:
        assert group.request_indices is not None
        for index, request_index, key in zip(group.object_indices, group.request_indices, group.keys, strict=True):
            code = failed_codes.get(key, 0) if isinstance(failed_codes, dict) else failed_codes
            if code != 0:
                source = group.batch.source(int(index), layer_id, request_index=request_index)
                failures.append(TransferEvidence(source, code))
    return _load_completions(plan.batch, tuple(failures))
