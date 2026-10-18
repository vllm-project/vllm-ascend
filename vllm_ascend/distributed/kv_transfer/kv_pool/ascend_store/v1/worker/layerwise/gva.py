"""Layerwise Worker route fixed to the GVA data plane."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from vllm.logger import logger

from ....attention_fence import reset_attention_compute_start_gate
from ...backend import GVABackend, GVAObjectSessions, GVARegion
from ...infrastructure.configuration.topology import KVPoolTopology
from ...infrastructure.mapping import LayerwiseMapping, LayerwiseSelection
from ...infrastructure.reachability import HybridReachability, UnitaryReachability
from ...infrastructure.resources import KVPoolResources
from ..transfer.batch import KVGroupBatch, KVTransferBatch
from ..transfer.completion import LoadCompletion
from .transfer import (
    LayerStoreResult,
    LayerTransferPlan,
    _layer_load_completions,
    prepare_layer_load,
    prepare_layer_store,
)
from .worker import LayerwiseWorker


class GVALayerwiseWorker(LayerwiseWorker):
    """Select GVA transfers while Backend owns leases, addresses, and publication mechanics."""

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
    ) -> None:
        backend = cast(GVABackend, resources.backend)
        sessions = backend.create_sessions()
        self._gva_backend = backend
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
        )

    def _exists(self, keys: list[str]) -> tuple[bool, ...]:
        return tuple(region is not None for region in self._gva_backend.query_gva_regions(keys))

    def _prepare_load_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._load_plan = prepare_layer_load(batch, self._layer_mappings, self._resolve_load_bases)
        return self._load_plan.keys_by_layer

    def _prepare_store_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._store_plan = prepare_layer_store(batch, self._layer_mappings, self._resolve_store_bases)
        return self._store_plan.keys_by_layer

    def _copy_load_layer(self, layer_id: int) -> tuple[LoadCompletion, ...]:
        plan = self._prepared_load_plan()
        if not plan.groups:
            return _layer_load_completions(plan, layer_id, 0)
        try:
            remote, local, sizes = materialize_layer_gva(self._layer_mappings, plan, layer_id)
            result_code = self._gva_backend.copy_from_gva(remote, local, sizes)
        except Exception as error:
            logger.error(
                "GVA Load failed for requests %s. type=%s, error=%s",
                plan.batch.request_ids,
                type(error).__name__,
                error,
            )
            return _layer_load_completions(plan, layer_id, None)
        return _layer_load_completions(plan, layer_id, result_code)

    def _copy_store_layer(self, source_ready_event: Any, layer_id: int) -> LayerStoreResult:
        """Prepare native arguments ahead of source readiness, then copy this layer."""

        source_handed_off = False
        plan = self._prepared_store_plan()
        try:
            remote_addresses, local_addresses, sizes = materialize_layer_gva(self._layer_mappings, plan, layer_id)
            source_ready_event.synchronize()
            source_handed_off = True
            result_code = self._gva_backend.copy_to_gva(remote_addresses, local_addresses, sizes)
        except Exception as error:
            return LayerStoreResult(None, not source_handed_off, error)
        return LayerStoreResult(result_code, True)

    def _resolve_store_bases(self, group: KVGroupBatch, keys: tuple[str, ...]) -> np.ndarray:
        return self._region_bases(
            cast(GVAObjectSessions, self._backend_sessions).store_regions(keys, group.object_size)
        )

    def _resolve_load_bases(self, group: KVGroupBatch, keys: tuple[str, ...]) -> np.ndarray:
        return self._region_bases(cast(GVAObjectSessions, self._backend_sessions).load_regions(keys, group.object_size))

    @staticmethod
    def _region_bases(regions: tuple[GVARegion, ...]) -> np.ndarray:
        object_bases = np.asarray([region.address for region in regions], dtype=np.uint64)
        object_bases.flags.writeable = False
        return object_bases


__all__ = ("GVALayerwiseWorker",)


def materialize_layer_gva(
    mappings: dict[int, LayerwiseMapping],
    plan: LayerTransferPlan,
    layer_id: int,
) -> tuple[list[int], list[int], list[int]]:
    """Evaluate only layer-dependent GVA arrays from prepared objects."""

    object_bases_by_group = plan.object_bases_by_group
    assert object_bases_by_group is not None
    remote_addresses: list[int] = []
    local_addresses: list[int] = []
    sizes: list[int] = []
    for group in plan.groups_by_layer[layer_id]:
        _, local_rows, size_rows, offset_rows = mappings[group.group_id].arguments(
            cast(LayerwiseSelection, group.selection), layer_id=layer_id
        )
        remote_addresses.extend(
            int(base) + offset
            for base, offsets in zip(object_bases_by_group[group.group_id], offset_rows, strict=True)
            for offset in offsets
        )
        local_addresses.extend(address for row in local_rows for address in row)
        sizes.extend(size for row in size_rows for size in row)
    return remote_addresses, local_addresses, sizes
