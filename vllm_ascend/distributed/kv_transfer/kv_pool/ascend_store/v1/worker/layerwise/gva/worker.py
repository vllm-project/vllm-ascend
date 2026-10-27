"""Layerwise Worker route fixed to the GVA data plane."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from vllm.logger import logger

from .....attention_fence import reset_attention_compute_start_gate
from ....backend import GVABackend, GVAObjectSessions, GVARegion
from ....infrastructure.configuration.topology import KVPoolTopology
from ....infrastructure.mapping import LayerwiseMapping
from ....infrastructure.mapping.object_layout import RemoteObjectLayoutFactory
from ....infrastructure.mapping.object_naming import KeyBuilder
from ....infrastructure.reachability import HybridReachability, UnitaryReachability
from ....infrastructure.resources import KVPoolResources
from ...transfer.batch import KVGroupBatch, KVTransferBatch
from ...transfer.completion import LoadCompletion
from ..transfer import (
    LayerStoreResult,
    LayerTransferPlan,
    _layer_load_completions,
    prepare_layer_load,
    prepare_layer_store,
)
from ..worker import LayerwiseWorker, _StoreState
from .collective import StoreCollective
from .shared_store import GVAWriteOutcome, SharedGVAWrites


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
        key_builder: KeyBuilder | None = None,
        remote_object_layout_factory: RemoteObjectLayoutFactory | None = None,
        store_collective: StoreCollective | None = None,
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
            key_builder=key_builder,
            remote_object_layout_factory=remote_object_layout_factory,
        )

        self._shared_writes = (
            SharedGVAWrites(self._store_writers, store_collective) if store_collective is not None else None
        )

    def _exists(self, keys: list[str]) -> tuple[bool, ...]:
        return tuple(region is not None for region in self._gva_backend.query_gva_regions(keys))

    def _prepare_load_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._load_plan = prepare_layer_load(batch, self._mappings, self._resolve_load_bases)
        return self._load_plan.selected_keys_by_layer

    def _prepare_store_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._store_plan = prepare_layer_store(batch, self._mappings, self._resolve_store_bases)
        return self._store_plan.selected_keys_by_layer

    def _copy_load_layer(self, layer_id: int) -> tuple[LoadCompletion, ...]:
        plan = self._prepared_load_plan()
        if not plan.groups:
            return _layer_load_completions(plan, layer_id, 0)
        try:
            remote, local, sizes = materialize_layer_gva(self._mappings, plan, layer_id)
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
            remote_addresses, local_addresses, sizes = materialize_layer_gva(self._mappings, plan, layer_id)
            source_ready_event.synchronize()
            source_handed_off = True
            result_code = self._gva_backend.copy_to_gva(remote_addresses, local_addresses, sizes)
        except Exception as error:
            return LayerStoreResult(None, not source_handed_off, error)
        return LayerStoreResult(result_code, True)

    def _resolve_store_bases(self, group: KVGroupBatch, keys: tuple[str, ...]) -> np.ndarray:
        return self._region_bases(
            cast(GVAObjectSessions, self._backend_sessions).store_regions(keys, group.remote_object_size_bytes)
        )

    def _resolve_load_bases(self, group: KVGroupBatch, keys: tuple[str, ...]) -> np.ndarray:
        return self._region_bases(
            cast(GVAObjectSessions, self._backend_sessions).load_regions(keys, group.remote_object_size_bytes)
        )

    def _complete_shared_store(self, state: _StoreState, commit_keys: list[str]) -> list[str]:
        if self._shared_writes is None:
            return commit_keys
        regions: dict[str, tuple[int, int]] = {}
        sessions = cast(GVAObjectSessions, self._backend_sessions)
        selected = set(commit_keys)
        try:
            if state.batch is not None:
                for group in state.batch.groups:
                    keys = tuple(key for key in group.selected_keys() if key in selected)
                    regions.update(
                        (key, (region.address, region.size))
                        for key, region in zip(
                            keys, sessions.store_regions(keys, group.remote_object_size_bytes), strict=True
                        )
                    )
        except Exception as error:
            # Still participate: peers must learn that this writer cannot prove completion.
            selected.clear()
            state.errors_by_key.update((key, error) for key in commit_keys)
        try:
            completed = self._shared_writes.complete_keys(GVAWriteOutcome(frozenset(selected), regions))
        except Exception as error:
            completed = set()
            state.errors_by_key.update((key, error) for key in commit_keys)
        for key in commit_keys:
            if key not in completed:
                state.failure_codes[key] = None
                state.errors_by_key.setdefault(key, RuntimeError("Shared GVA object has an incomplete context shard"))
        return [key for key in commit_keys if key in completed]

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

    remote_object_bases_by_group = plan.remote_object_bases_by_group
    assert remote_object_bases_by_group is not None
    remote_addresses: list[int] = []
    local_addresses: list[int] = []
    sizes: list[int] = []
    for group in plan.groups_by_layer[layer_id]:
        _, local_rows, size_rows, offset_rows = mappings[group.group_id].arguments(
            group.selected_keys,
            group.selected_local_block_ids_by_partition,
            group.selected_transfer_slices_by_partition,
            layer_id=layer_id,
        )
        layer_bases = [
            int(base)
            for layout in mappings[group.group_id].layer_layouts.get(layer_id, ())
            for base in remote_object_bases_by_group[group.group_id][
                group.selected_transfer_slices_by_partition[layout.partition_index]
            ]
        ]
        remote_addresses.extend(
            base + offset for base, offsets in zip(layer_bases, offset_rows, strict=True) for offset in offsets
        )
        local_addresses.extend(address for row in local_rows for address in row)
        sizes.extend(size for row in size_rows for size in row)
    return remote_addresses, local_addresses, sizes
