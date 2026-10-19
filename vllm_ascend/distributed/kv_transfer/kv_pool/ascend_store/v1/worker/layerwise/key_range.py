"""Layerwise Worker route fixed to the Backend key-range data plane."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from vllm.logger import logger

from ....attention_fence import reset_attention_compute_start_gate
from ...backend import KeyRangeBackend
from ...infrastructure.configuration.topology import KVPoolTopology
from ...infrastructure.mapping import LayerwiseMapping, LayerwiseSelection
from ...infrastructure.mapping.object_layout import ObjectLayoutFactory
from ...infrastructure.mapping.object_naming import KeyBuilder
from ...infrastructure.reachability import HybridReachability, UnitaryReachability
from ...infrastructure.resources import KVPoolResources
from ..transfer.batch import KVTransferBatch
from ..transfer.completion import LoadCompletion
from .transfer import (
    LayerStoreResult,
    LayerTransferGroup,
    LayerTransferPlan,
    _layer_load_completions,
    prepare_layer_load,
    prepare_layer_store,
)
from .worker import LayerwiseWorker, _attention_store_layer_ids


class KeyRangeLayerwiseWorker(LayerwiseWorker):
    """Select layer ranges while Backend owns the underlying object sessions."""

    def __init__(
        self,
        topology: KVPoolTopology,
        resources: KVPoolResources,
        *,
        reachability: UnitaryReachability | HybridReachability,
        kv_cache_layout: str = "NHD",
        store_enabled: bool = True,
        layerwise_prefetch_layers: int = 2,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
        key_builder: KeyBuilder | None = None,
        object_layout_factory: ObjectLayoutFactory | None = None,
    ) -> None:
        backend = cast(KeyRangeBackend, resources.backend)
        sessions = backend.create_sessions()
        self._key_range_backend = backend
        super().__init__(
            topology,
            resources,
            sessions,
            store_enabled=store_enabled,
            reachability=reachability,
            kv_cache_layout=kv_cache_layout,
            layerwise_prefetch_layers=layerwise_prefetch_layers,
            start_gate_factory=start_gate_factory,
            source_ready_event_factory=source_ready_event_factory,
            key_builder=key_builder,
            object_layout_factory=object_layout_factory,
        )

        self._attention_store_layers = _attention_store_layer_ids(topology) if store_enabled else frozenset()

    def _prepare_load_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._load_plan = prepare_layer_load(batch, self._mappings)
        return self._load_plan.keys_by_layer

    def _prepare_store_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._store_plan = prepare_layer_store(batch, self._mappings)
        return self._store_plan.keys_by_layer

    def _copy_load_layer(self, layer_id: int) -> tuple[LoadCompletion, ...]:
        plan = self._prepared_load_plan()
        try:
            arguments = materialize_load_layer_ranges(self._mappings, plan, layer_id)
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
            addresses, sizes, offsets = materialize_store_layer_ranges(self._mappings, groups, layer_id)
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
            return plan.groups_by_layer[layer_id], plan.keys_by_layer[layer_id]
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


def materialize_store_layer_ranges(
    mappings: dict[int, LayerwiseMapping],
    groups: tuple[LayerTransferGroup, ...],
    layer_id: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Evaluate only layer-dependent KeyRange arrays from prepared objects."""

    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    for group in groups:
        _, local, group_sizes, group_offsets = mappings[group.group_id].arguments(
            cast(LayerwiseSelection, group.selection), layer_id=layer_id
        )
        addresses.extend(local)
        sizes.extend(group_sizes)
        offsets.extend(group_offsets)
    return addresses, sizes, offsets


def materialize_load_layer_ranges(
    mappings: dict[int, LayerwiseMapping],
    plan: LayerTransferPlan,
    layer_id: int,
) -> KeyRangeArguments:
    """Evaluate one KeyRange Load layer from cross-layer reusable rows."""

    addresses: list[list[int]] = []
    sizes: list[list[int]] = []
    offsets: list[list[int]] = []
    for group in plan.groups_by_layer[layer_id]:
        _, group_addresses, group_sizes, group_offsets = mappings[group.group_id].arguments(
            cast(LayerwiseSelection, group.selection), layer_id=layer_id
        )
        addresses.extend(group_addresses)
        sizes.extend(group_sizes)
        offsets.extend(group_offsets)
    keys = plan.keys_by_layer[layer_id]
    if len(set(keys)) == len(keys):
        return KeyRangeArguments(keys, addresses, sizes, offsets)

    # Keep the common path in bulk list operations; only duplicate keys need per-object merging.
    key_indices: dict[str, int] = {}
    merged_keys: list[str] = []
    merged_addresses: list[list[int]] = []
    merged_sizes: list[list[int]] = []
    merged_offsets: list[list[int]] = []
    for key, row_addresses, row_sizes, row_offsets in zip(keys, addresses, sizes, offsets, strict=True):
        index = key_indices.get(key)
        if index is None:
            key_indices[key] = len(merged_keys)
            merged_keys.append(key)
            merged_addresses.append(row_addresses)
            merged_sizes.append(row_sizes)
            merged_offsets.append(row_offsets)
        else:
            merged_addresses[index].extend(row_addresses)
            merged_sizes[index].extend(row_sizes)
            merged_offsets[index].extend(row_offsets)
    return KeyRangeArguments(merged_keys, merged_addresses, merged_sizes, merged_offsets)
