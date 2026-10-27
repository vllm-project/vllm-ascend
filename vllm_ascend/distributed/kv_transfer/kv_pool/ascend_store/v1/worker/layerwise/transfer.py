"""Prepare reusable Layerwise plans and interpret layer copy results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ...infrastructure.mapping import LayerwiseMapping
from ..transfer.batch import IndexArray, KVGroupBatch, KVTransferBatch, UInt64Array
from ..transfer.completion import LoadCompletion, TransferEvidence, _load_completions


@dataclass(frozen=True, slots=True)
class LayerTransferGroup:
    """One group's selected transfers and original positions, shared across layers."""

    batch: KVGroupBatch
    selected_keys: tuple[str, ...]
    selected_transfer_record_indices_by_partition: tuple[range | IndexArray, ...]
    selected_local_block_ids_by_partition: tuple[UInt64Array, ...]
    selected_transfer_slices_by_partition: tuple[slice, ...]

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


# Fixed (group, partitions) membership is compiled when Worker binds its mappings.
LayerMemberships = dict[int, dict[int, tuple[int, tuple[int, ...]]]]


def bind_layer_memberships(mappings: dict[int, LayerwiseMapping]) -> LayerMemberships:
    return {
        group_id: {
            layer_id: (group_id, tuple(layout.partition_index for layout in layouts))
            for layer_id, layouts in mapping.layer_layouts.items()
        }
        for group_id, mapping in mappings.items()
    }


def prepare_layer_transfer(
    batch: KVTransferBatch,
    memberships: LayerMemberships,
    resolve_object_bases: ObjectBaseResolver | None = None,
) -> LayerTransferPlan:
    """Snapshot admitted transfers once; resolve request provenance only on failure."""

    groups = []
    remote_object_bases_by_group: dict[int, UInt64Array] | None = {} if resolve_object_bases is not None else None
    for batch_group in batch.groups:
        keys = batch_group.selected_keys()
        if not keys:
            continue
        group = make_layer_transfer_group(batch_group, selected_keys=keys)
        groups.append(group)
        if resolve_object_bases is not None:
            assert remote_object_bases_by_group is not None
            remote_object_bases_by_group[group.group_id] = resolve_object_bases(batch_group, group.selected_keys)
    return make_layer_transfer_plan(batch, tuple(groups), memberships, remote_object_bases_by_group)


def make_layer_transfer_group(
    group: KVGroupBatch,
    *,
    selected_keys: tuple[str, ...],
) -> LayerTransferGroup:
    """Snapshot one final candidate selection; derive all correspondence from it."""
    record_count = group.transfer_record_count
    masks = (
        None
        if group.transfer_candidate_mask is None
        else group.transfer_candidate_mask.reshape(len(group.keys_by_partition), record_count)
    )
    shared_ids = None
    shared_indices = range(record_count)
    indices_by_partition = []
    ids_by_partition = []
    slices_by_partition = []
    selected_count_total = 0
    for partition_index in range(len(group.keys_by_partition)):
        indices: range | IndexArray
        selected_count = record_count if masks is None else int(np.count_nonzero(masks[partition_index]))
        if selected_count == record_count:
            indices = shared_indices
            if shared_ids is None:
                shared_ids = group.local_block_ids.copy()
                shared_ids.flags.writeable = False
            ids = shared_ids
        elif selected_count:
            assert masks is not None
            indices = np.flatnonzero(masks[partition_index])
            indices.flags.writeable = False
            ids = group.local_block_ids[indices]
            ids.flags.writeable = False
        else:
            indices = range(0)
            ids = group.local_block_ids[:0].view()
            ids.flags.writeable = False
        start = selected_count_total
        selected_count_total += selected_count
        slices_by_partition.append(slice(start, selected_count_total))
        indices_by_partition.append(indices)
        ids_by_partition.append(ids)

    return LayerTransferGroup(
        group,
        selected_keys,
        tuple(indices_by_partition),
        tuple(ids_by_partition),
        tuple(slices_by_partition),
    )


def make_layer_transfer_plan(
    batch: KVTransferBatch,
    groups: tuple[LayerTransferGroup, ...],
    memberships: LayerMemberships,
    remote_object_bases_by_group: dict[int, UInt64Array] | None = None,
) -> LayerTransferPlan:
    """Reuse key lists only when groups and participating partitions match."""

    mutable_groups_by_layer: dict[int, list[LayerTransferGroup]] = {}
    members_by_layer: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for group in groups:
        for layer_id, member in memberships[group.group_id].items():
            mutable_groups_by_layer.setdefault(layer_id, []).append(group)
            members_by_layer.setdefault(layer_id, []).append(member)

    groups_by_layer = {}
    keys_by_membership: dict[tuple[tuple[int, tuple[int, ...]], ...], list[str]] = {}
    selected_keys_by_layer = {}
    for layer_id, layer_groups in mutable_groups_by_layer.items():
        groups_by_layer[layer_id] = tuple(layer_groups)
        membership = tuple(members_by_layer[layer_id])
        if membership not in keys_by_membership:
            keys: list[str] = []
            for group, (_, partitions) in zip(layer_groups, membership, strict=True):
                for partition_index in partitions:
                    keys.extend(group.selected_keys[group.selected_transfer_slices_by_partition[partition_index]])
            keys_by_membership[membership] = keys
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
        for partition_index, selected_slice in enumerate(group.selected_transfer_slices_by_partition):
            partition_layers = group.batch.physical_layer_ids_by_partition
            if partition_layers is not None and layer_id not in partition_layers[partition_index]:
                continue
            for record_index, key in zip(
                group.selected_transfer_record_indices_by_partition[partition_index],
                group.selected_keys[selected_slice],
                strict=True,
            ):
                code = failed_codes.get(key, 0) if isinstance(failed_codes, dict) else failed_codes
                if code != 0:
                    index = partition_index * group.batch.transfer_record_count + int(record_index)
                    source = group.batch.source(index, layer_id)
                    failures.append(TransferEvidence(source, code))
    return _load_completions(plan.batch, tuple(failures))
