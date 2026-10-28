"""Layerwise Worker route fixed to the Backend key-range data plane."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat
from typing import Any, cast

import numpy as np
from vllm.logger import logger

from ....attention_fence import reset_attention_compute_start_gate
from ...backend import KeyRangeBackend
from ...backend.naming import WireKeyBuilder
from ...infrastructure.coordinator import Coordinator
from ...infrastructure.mapping import (
    LayerTable,
    LocalSegmentData,
    LocalTable,
    RemoteTable,
    build_key_range_remote,
    local_addresses,
    local_lengths,
)
from ...infrastructure.resources import KVPoolResources
from ...infrastructure.topology import KVPoolTopology
from ..transfer.batch import KVTransferBatch
from ..transfer.completion import LoadCompletion
from .transfer import (
    LayerStoreResult,
    LayerTransferGroup,
    LayerTransferPlan,
    _layer_load_completions,
    prepare_layer_transfer,
)
from .worker import LayerwiseWorker, _attention_store_layer_ids


class KeyRangeLayerwiseWorker(LayerwiseWorker):
    """Select layer ranges while Backend owns the underlying object sessions."""

    def __init__(
        self,
        topology: KVPoolTopology,
        resources: KVPoolResources,
        *,
        coordinator: Coordinator,
        kv_cache_layout: str = "NHD",
        store_enabled: bool = True,
        layerwise_prefetch_layers: int = 2,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
        key_builder: WireKeyBuilder | None = None,
    ) -> None:
        backend = cast(KeyRangeBackend, resources.backend)
        sessions = backend.create_sessions()
        self._key_range_backend = backend
        self._load_key_merges: dict[int, KeyRangeKeyMerge | None] = {}
        super().__init__(
            topology,
            resources,
            sessions,
            store_enabled=store_enabled,
            coordinator=coordinator,
            kv_cache_layout=kv_cache_layout,
            layerwise_prefetch_layers=layerwise_prefetch_layers,
            start_gate_factory=start_gate_factory,
            source_ready_event_factory=source_ready_event_factory,
            key_builder=key_builder,
        )

        self._attention_store_layers = _attention_store_layer_ids(topology) if store_enabled else frozenset()

    def _bind_mappings(self) -> None:
        super()._bind_mappings()
        self._remote = {group_id: build_key_range_remote(local) for group_id, local in self._local.items()}
        self._key_range_layouts = bind_key_range_layouts(self._local, self._layer_slices, self._remote)

    def _prepare_load_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._load_plan = prepare_layer_transfer(batch, self._layer_memberships)
        self._load_key_merges = prepare_load_key_merges(self._load_plan)
        return self._load_plan.selected_keys_by_layer

    def _prepare_store_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._store_plan = prepare_layer_transfer(batch, self._layer_memberships)
        return self._store_plan.selected_keys_by_layer

    def _finish_load(self) -> None:
        self._load_key_merges.clear()
        super()._finish_load()

    def _copy_load_layer(self, layer_id: int) -> tuple[LoadCompletion, ...]:
        plan = self._prepared_load_plan()
        try:
            arguments = materialize_load_layer_ranges(
                self._key_range_layouts, plan, layer_id, self._load_key_merges[layer_id]
            )
            if not arguments.keys:
                return _layer_load_completions(plan, layer_id, {})
            result_codes = self._key_range_backend.copy_key_range_load(
                arguments.keys, arguments.addresses, arguments.sizes, arguments.offsets
            )
        except Exception as error:
            logger.error(
                "KeyRange Load failed for requests %s. type=%s, error=%s",
                plan.batch.request_ids,
                type(error).__name__,
                error,
            )
            return _layer_load_completions(plan, layer_id, None)
        failed_codes = {key: code for key, code in zip(arguments.keys, result_codes, strict=True) if code != 0}
        return _layer_load_completions(plan, layer_id, failed_codes)

    def _copy_store_layer(self, source_ready_event: Any | None, layer_id: int) -> LayerStoreResult:
        """Prepare addresses ahead of source readiness, then copy this layer."""

        source_handed_off = False
        groups, keys = self._prepared_store_layer(layer_id)
        try:
            addresses, sizes, offsets = materialize_store_layer_ranges(self._key_range_layouts, groups, layer_id)
            if source_ready_event is not None:
                source_ready_event.synchronize()
            source_handed_off = True
            result_codes = self._key_range_backend.copy_key_range_store(keys, addresses, sizes, offsets)
        except Exception as error:
            return LayerStoreResult(None, not source_handed_off, error)
        return LayerStoreResult(result_codes, (True,) * len(result_codes))

    def _prepared_store_layer(self, layer_id: int) -> tuple[tuple[LayerTransferGroup, ...], list[str]]:
        plan = self._prepared_store_plan()
        try:
            return plan.groups_by_layer[layer_id], plan.selected_keys_by_layer[layer_id]
        except KeyError as error:
            raise RuntimeError(f"Layerwise Store layer {layer_id} was not prepared") from error


__all__ = ("KeyRangeLayerwiseWorker",)


@dataclass(frozen=True, slots=True)
class KeyRangeArguments:
    """Per-key local ranges and remote object offsets for one physical layer."""

    keys: list[str]
    addresses: list[list[int]]
    sizes: list[list[int]]
    offsets: list[list[int]]


@dataclass(frozen=True, slots=True)
class KeyRangeLayout:
    """Worker-owned native scalar templates for a bound layer partition."""

    partition_index: int
    segments: LocalSegmentData
    segment_slice: slice
    sizes: tuple[int, ...]
    offsets: tuple[int, ...]


KeyRangeLayouts = dict[int, dict[int, tuple[KeyRangeLayout, ...]]]


def bind_key_range_layouts(
    local: dict[int, LocalTable], layers: dict[int, LayerTable], remote: dict[int, RemoteTable]
) -> KeyRangeLayouts:
    result = {}
    for group_id, partitions in local.items():
        by_layer: dict[int, list[KeyRangeLayout]] = {}
        for index, ((pp, _, _), segments) in enumerate(partitions.items()):
            for layer_id, selected in layers[group_id][pp].items():
                if selected.start == selected.stop:
                    continue
                sizes = np.broadcast_to(local_lengths(segments, selected), (selected.stop - selected.start,))
                offsets = remote[group_id][pp]["offsets"][selected]
                by_layer.setdefault(layer_id, []).append(
                    KeyRangeLayout(index, segments, selected, tuple(sizes.tolist()), tuple(offsets.tolist()))
                )
        result[group_id] = {layer: tuple(layouts) for layer, layouts in by_layer.items()}
    return result


@dataclass(frozen=True, slots=True)
class KeyRangeKeyMerge:
    """Duplicate-key destinations, shared by layers with the same participating partitions."""

    keys: list[str]
    merge_indices: tuple[int, ...]


def prepare_load_key_merges(plan: LayerTransferPlan) -> dict[int, KeyRangeKeyMerge | None]:
    by_membership: dict[int, KeyRangeKeyMerge | None] = {}
    by_layer = {}
    for layer_id, keys in plan.selected_keys_by_layer.items():
        membership = id(keys)  # The plan shares this list exactly when participating partitions match.
        if membership not in by_membership:
            if len(set(keys)) == len(keys):
                selection = None
            else:
                indices: dict[str, int] = {}
                merge_indices = tuple(indices.setdefault(key, len(indices)) for key in keys)
                selection = KeyRangeKeyMerge(list(indices), merge_indices)
            by_membership[membership] = selection
        by_layer[layer_id] = by_membership[membership]
    return by_layer


def materialize_group_layer_ranges(
    layouts: tuple[KeyRangeLayout, ...], group: LayerTransferGroup
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Combine the Worker's partition selection with independent byte layouts."""
    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    for native in layouts:
        ids = group.selected_local_block_ids_by_partition[native.partition_index]
        if not len(ids):
            continue
        addresses.extend(local_addresses(native.segments, ids, native.segment_slice).tolist())
        sizes.extend(list(native.sizes) for _ in repeat(None, len(ids)))
        offsets.extend(list(native.offsets) for _ in repeat(None, len(ids)))
    return addresses, sizes, offsets


def materialize_store_layer_ranges(
    layouts: KeyRangeLayouts,
    groups: tuple[LayerTransferGroup, ...],
    layer_id: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Evaluate only layer-dependent KeyRange arrays from prepared objects."""

    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    for group in groups:
        local, group_sizes, group_offsets = materialize_group_layer_ranges(
            layouts[group.group_id].get(layer_id, ()), group
        )
        addresses.extend(local)
        sizes.extend(group_sizes)
        offsets.extend(group_offsets)
    return addresses, sizes, offsets


def materialize_load_layer_ranges(
    layouts: KeyRangeLayouts,
    plan: LayerTransferPlan,
    layer_id: int,
    merge: KeyRangeKeyMerge | None,
) -> KeyRangeArguments:
    """Evaluate one KeyRange Load layer from cross-layer reusable rows."""

    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    for group in plan.groups_by_layer[layer_id]:
        group_addresses, group_sizes, group_offsets = materialize_group_layer_ranges(
            layouts[group.group_id].get(layer_id, ()), group
        )
        addresses.extend(group_addresses)
        sizes.extend(group_sizes)
        offsets.extend(group_offsets)
    if merge is None:
        return KeyRangeArguments(plan.selected_keys_by_layer[layer_id], addresses, sizes, offsets)

    merged_addresses: list[list[int]] = []
    merged_sizes: list[list[int]] = []
    merged_offsets: list[list[int]] = []
    for index, row_addresses, row_sizes, row_offsets in zip(
        merge.merge_indices, addresses, sizes, offsets, strict=True
    ):
        if index == len(merged_addresses):
            merged_addresses.append(row_addresses)
            merged_sizes.append(row_sizes)
            merged_offsets.append(row_offsets)
        else:
            merged_addresses[index].extend(row_addresses)
            merged_sizes[index].extend(row_sizes)
            merged_offsets[index].extend(row_offsets)
    return KeyRangeArguments(merge.keys, merged_addresses, merged_sizes, merged_offsets)
