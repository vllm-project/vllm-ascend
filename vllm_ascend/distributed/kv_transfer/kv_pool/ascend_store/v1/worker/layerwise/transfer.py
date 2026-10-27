"""Prepare reusable Layerwise plans and interpret layer copy results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...infrastructure.mapping import LayerwiseMapping, LayerwiseSelection
from ..transfer.batch import IndexArray, KVGroupBatch, KVTransferBatch, UInt64Array
from ..transfer.completion import LoadCompletion, TransferEvidence, _load_completions


@dataclass(frozen=True, slots=True)
class LayerTransferGroup:
    """One Group's selected object rows shared by every Layerwise copy."""

    batch: KVGroupBatch
    keys: tuple[str, ...]
    selection: LayerwiseSelection
    # Load resolves ownership after session admission; Store does not need this lowering.
    selected_transfer_candidate_indices: range | IndexArray | None = None
    request_indices: tuple[int, ...] | None = None

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
    selected_keys_by_layer: dict[int, list[str]]
    remote_object_bases_by_group: dict[int, UInt64Array] | None = None


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
    remote_object_bases_by_group: dict[int, UInt64Array] | None = {} if resolve_object_bases is not None else None
    for batch_group in batch.groups:
        keys = batch_group.selected_keys()
        if not keys:
            continue
        selection = mappings[batch_group.group_id].prepare(
            batch_group.keys_by_partition,
            batch_group.local_block_ids,
            selected_keys=batch_group.transfer_candidate_mask,
        )
        group = make_layer_transfer_group(batch_group, keys, selection, resolve_requests=resolve_requests)
        groups.append(group)
        if resolve_object_bases is not None:
            assert remote_object_bases_by_group is not None
            remote_object_bases_by_group[group.group_id] = resolve_object_bases(batch_group, group.keys)
    return make_layer_transfer_plan(batch, tuple(groups), remote_object_bases_by_group)


def make_layer_transfer_group(
    group: KVGroupBatch,
    keys: tuple[str, ...],
    selection: LayerwiseSelection,
    *,
    resolve_requests: bool = False,
) -> LayerTransferGroup:
    """Retain Mapping's selected rows; lower request ownership only for Load."""
    if not resolve_requests:
        return LayerTransferGroup(group, keys, selection)
    selected_transfer_candidate_indices: range | IndexArray
    if selection.selected_transfer_count == group.transfer_candidate_count:
        selected_transfer_candidate_indices = range(group.transfer_candidate_count)
    elif len(selection.by_partition) == 1:
        selected = selection.by_partition[0]
        assert selected is not None
        selected_transfer_candidate_indices = selected.selected_transfer_record_indices
    else:
        selected_transfer_candidate_indices = np.fromiter(
            (
                axis * group.transfer_record_count + row
                for axis, selected in enumerate(selection.by_partition)
                if selected is not None
                for row in selected.selected_transfer_record_indices
            ),
            dtype=np.intp,
            count=selection.selected_transfer_count,
        )
        selected_transfer_candidate_indices.flags.writeable = False
    request_indices = tuple(group.request_index(int(index)) for index in selected_transfer_candidate_indices)
    return LayerTransferGroup(group, keys, selection, selected_transfer_candidate_indices, request_indices)


def make_layer_transfer_plan(
    batch: KVTransferBatch,
    groups: tuple[LayerTransferGroup, ...],
    remote_object_bases_by_group: dict[int, UInt64Array] | None = None,
) -> LayerTransferPlan:
    """Compile reusable group membership and key axes without touching object rows."""

    mutable_groups_by_layer: dict[int, list[LayerTransferGroup]] = {}
    for group in groups:
        for layer_id in group.physical_layer_ids:
            mutable_groups_by_layer.setdefault(layer_id, []).append(group)

    groups_by_layer = {}
    keys_by_membership: dict[tuple[int, ...], list[str]] = {}
    selected_keys_by_layer = {}
    for layer_id, layer_groups in mutable_groups_by_layer.items():
        groups_by_layer[layer_id] = tuple(layer_groups)
        membership = tuple(group.group_id for group in layer_groups)
        if membership not in keys_by_membership:
            keys_by_membership[membership] = [key for group in layer_groups for key in group.keys]
        selected_keys_by_layer[layer_id] = keys_by_membership[membership]
    return LayerTransferPlan(batch, groups, groups_by_layer, selected_keys_by_layer, remote_object_bases_by_group)


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
        assert group.selected_transfer_candidate_indices is not None
        for index, request_index, key in zip(
            group.selected_transfer_candidate_indices, group.request_indices, group.keys, strict=True
        ):
            code = failed_codes.get(key, 0) if isinstance(failed_codes, dict) else failed_codes
            if code != 0:
                source = group.batch.source(int(index), layer_id, request_index=request_index)
                failures.append(TransferEvidence(source, code))
    return _load_completions(plan.batch, tuple(failures))
