"""Evaluate bound projection into the argument shape required by each Backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from ...projection import GVALayerwiseProjection, KeyRangeLayerwiseProjection
from ...projection.layerwise.gva import gva_layer_ranges
from ...projection.layerwise.key_range import key_range_layer_ranges
from ..batch import (
    KVGroupBatch,
    KVTransferBatch,
    LayerTransferGroup,
    LayerTransferPlan,
    TransferSource,
    make_layer_load_plan,
    make_layer_transfer_group,
    make_layer_transfer_plan,
)


@dataclass(frozen=True, slots=True)
class BulkBackendArguments:
    """Key-aligned local ranges prepared by the concrete Bulk route."""

    keys: list[str]
    addresses: list[list[int]]
    sizes: list[list[int]]
    sources: tuple[TransferSource, ...]


@dataclass(frozen=True, slots=True)
class KeyRangeArguments:
    """Per-key local ranges and remote object offsets for one physical layer."""

    keys: list[str]
    addresses: list[list[int]]
    sizes: list[list[int]]
    offsets: list[list[int]]
    sources: tuple[TransferSource, ...]


@dataclass(frozen=True, slots=True)
class GVAArguments:
    """Flattened global and local ranges evaluated for one GVA copy."""

    keys: tuple[str, ...]
    remote_addresses: np.ndarray
    local_addresses: np.ndarray
    sizes: np.ndarray
    sources: tuple[TransferSource, ...]


ObjectBaseResolver = Callable[[KVGroupBatch, tuple[str, ...]], np.ndarray]


def prepare_layer_store(
    batch: KVTransferBatch,
    resolve_object_bases: ObjectBaseResolver | None = None,
) -> LayerTransferPlan:
    """Bind session-only GVA bases to the plan produced by materialization."""

    plan = batch.layer_store_plan
    if plan is None:
        # Partial session-start failure creates a new selected batch. Rebuild only
        # on that cold failure path; the successful path reuses the original plan.
        plan = make_layer_transfer_plan(
            batch,
            tuple(make_layer_transfer_group(group) for group in batch.groups),
        )
    return _bind_object_bases(plan, resolve_object_bases)


def prepare_layer_load(
    batch: KVTransferBatch,
    resolve_object_bases: ObjectBaseResolver | None = None,
) -> LayerTransferPlan:
    """Lower selected Load rows once after session admission."""

    return _bind_object_bases(make_layer_load_plan(batch), resolve_object_bases)


def _bind_object_bases(
    plan: LayerTransferPlan,
    resolve_object_bases: ObjectBaseResolver | None,
) -> LayerTransferPlan:
    if resolve_object_bases is None:
        return plan

    object_bases_by_group = {}
    for group in plan.groups:
        object_bases = resolve_object_bases(group.batch, group.keys)
        if len(object_bases) != len(group.keys):
            raise RuntimeError(f"Layerwise group {group.group_id} has misaligned GVA bases")
        object_bases_by_group[group.group_id] = object_bases
    return replace(plan, object_bases_by_group=object_bases_by_group)


def materialize_store_layer_ranges(
    projection: KeyRangeLayerwiseProjection,
    groups: tuple[LayerTransferGroup, ...],
    layer_id: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Evaluate only layer-dependent KeyRange arrays from prepared objects."""

    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    for group in groups:
        local, group_sizes, group_offsets = key_range_layer_ranges(
            projection.groups[group.group_id],
            group.block_ids,
            group.token_counts,
            layer_id=layer_id,
        )
        addresses.extend(local.tolist())
        sizes.extend(group_sizes.tolist())
        offsets.extend(group_offsets.tolist())
    return addresses, sizes, offsets


def materialize_store_layer_gva(
    projection: GVALayerwiseProjection,
    plan: LayerTransferPlan,
    layer_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate only layer-dependent GVA arrays from prepared objects."""

    object_bases_by_group = plan.object_bases_by_group
    if object_bases_by_group is None:
        raise RuntimeError("Layerwise Store GVA bases were not prepared")
    remote_parts = []
    local_parts = []
    size_parts = []
    for group in plan.groups_by_layer[layer_id]:
        remote, local, sizes = gva_layer_ranges(
            projection.groups[group.group_id],
            group.block_ids,
            group.token_counts,
            object_bases_by_group[group.group_id],
            layer_id=layer_id,
        )
        remote_parts.append(remote)
        local_parts.append(local)
        size_parts.append(sizes)
    return (
        _concatenate_gva_parts(remote_parts),
        _concatenate_gva_parts(local_parts),
        _concatenate_gva_parts(size_parts),
    )


def materialize_load_layer_ranges(
    projection: KeyRangeLayerwiseProjection,
    plan: LayerTransferPlan,
    layer_id: int,
) -> KeyRangeArguments:
    """Evaluate one KeyRange Load layer from cross-layer reusable rows."""

    keys: list[str] = []
    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    sources: list[TransferSource] = []
    for group in plan.groups_by_layer[layer_id]:
        group_addresses, group_sizes, group_offsets = key_range_layer_ranges(
            projection.groups[group.group_id],
            group.block_ids,
            group.token_counts,
            layer_id=layer_id,
        )
        keys.extend(group.keys)
        addresses.extend(group_addresses.tolist())
        sizes.extend(group_sizes.tolist())
        offsets.extend(group_offsets.tolist())
        sources.extend(_layer_sources(group, layer_id))
    return KeyRangeArguments(keys, addresses, sizes, offsets, tuple(sources))


def materialize_load_layer_gva(
    projection: GVALayerwiseProjection,
    plan: LayerTransferPlan,
    layer_id: int,
) -> GVAArguments:
    """Evaluate one GVA Load layer from leased bases resolved once per session."""

    object_bases_by_group = plan.object_bases_by_group
    if object_bases_by_group is None:
        raise RuntimeError("Layerwise Load GVA bases were not prepared")
    remote_parts = []
    local_parts = []
    size_parts = []
    keys: list[str] = []
    sources: list[TransferSource] = []
    for group in plan.groups_by_layer[layer_id]:
        remote, local, sizes = gva_layer_ranges(
            projection.groups[group.group_id],
            group.block_ids,
            group.token_counts,
            object_bases_by_group[group.group_id],
            layer_id=layer_id,
        )
        remote_parts.append(remote)
        local_parts.append(local)
        size_parts.append(sizes)
        keys.extend(group.keys)
        sources.extend(_layer_sources(group, layer_id))
    return GVAArguments(
        tuple(keys),
        _concatenate_gva_parts(remote_parts),
        _concatenate_gva_parts(local_parts),
        _concatenate_gva_parts(size_parts),
        tuple(sources),
    )


def merge_key_range_arguments(ranges: KeyRangeArguments) -> KeyRangeArguments:
    """Coalesce repeated object keys before one KeyRange Backend call."""

    if len(set(ranges.keys)) == len(ranges.keys):
        return ranges
    key_indices: dict[str, int] = {}
    keys: list[str] = []
    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    for key, row_addresses, row_sizes, row_offsets in zip(
        ranges.keys, ranges.addresses, ranges.sizes, ranges.offsets, strict=True
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
    return KeyRangeArguments(keys, addresses, sizes, offsets, ranges.sources)


def _layer_sources(group: LayerTransferGroup, layer_id: int) -> tuple[TransferSource, ...]:
    return tuple(group.batch.source(int(index), layer_id) for index in group.object_indices)


def _concatenate_gva_parts(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=np.uint64)
    return parts[0] if len(parts) == 1 else np.concatenate(parts)
