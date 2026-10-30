"""Layerwise Worker route fixed to the GVA data plane."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import numpy as np
from vllm.logger import logger

from .....attention_fence import reset_attention_compute_start_gate
from ....backend import GVABackend, GVAObjectSessions, GVARegion
from ....backend.naming import WireKeyBuilder
from ....infrastructure.coordinator import Coordinator
from ....infrastructure.mapping import (
    LayerTable,
    LocalTable,
    RemoteTable,
    build_gva_remote,
    local_addresses,
    local_lengths,
)
from ....infrastructure.resources import Resources
from ....infrastructure.topology import KVPoolTopology
from ...transfer.batch import KVGroupBatch, KVTransferBatch
from ...transfer.completion import LoadCompletion
from ..transfer import (
    LayerStoreResult,
    LayerTransferPlan,
    PreparedLayerLoad,
    _layer_load_completions,
    prepare_layer_transfer,
)
from ..worker import LayerwiseWorker, _StoreState
from .collective import StoreCollective
from .shared_store import GVAWriteOutcome, SharedGVAWrites


class GVALayerwiseWorker(LayerwiseWorker):
    """Select GVA transfers while Backend owns leases, addresses, and publication mechanics."""

    def __init__(
        self,
        topology: KVPoolTopology,
        resources: Resources,
        *,
        coordinator: Coordinator,
        kv_cache_layout: str = "NHD",
        store_enabled: bool = True,
        layerwise_prefetch_layers: int = 2,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
        key_builder: WireKeyBuilder | None = None,
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
            coordinator=coordinator,
            kv_cache_layout=kv_cache_layout,
            layerwise_prefetch_layers=layerwise_prefetch_layers,
            start_gate_factory=start_gate_factory,
            source_ready_event_factory=source_ready_event_factory,
            key_builder=key_builder,
        )

        self._shared_writes = (
            SharedGVAWrites(self._store_writers, store_collective) if store_collective is not None else None
        )

    def _bind_mappings(self) -> None:
        super()._bind_mappings()
        self._remote = {
            group_id: build_gva_remote(self._topology, local, self._layer_slices[group_id])
            for group_id, local in self._local.items()
        }

    def _exists(self, keys: list[str]) -> tuple[bool, ...]:
        return tuple(region is not None for region in self._gva_backend.query_gva_regions(keys))

    def _prepare_load_layers(self, batch: KVTransferBatch) -> PreparedLayerLoad:
        plan = prepare_layer_transfer(batch, self._layer_memberships, self._resolve_load_bases)
        return PreparedLayerLoad(plan, partial(self._copy_load_layer, plan))

    def _prepare_store_layers(self, batch: KVTransferBatch) -> LayerTransferPlan:
        return prepare_layer_transfer(batch, self._layer_memberships, self._resolve_store_bases)

    def _copy_load_layer(self, plan: LayerTransferPlan, layer_id: int) -> tuple[LoadCompletion, ...]:
        if not plan.groups:
            return _layer_load_completions(plan, layer_id, 0)
        try:
            remote, local, sizes = materialize_layer_gva(self._local, self._layer_slices, self._remote, plan, layer_id)
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

    def _copy_store_layer(self, source_ready_event: Any, layer_id: int, plan: LayerTransferPlan) -> LayerStoreResult:
        """Prepare native arguments ahead of source readiness, then copy this layer."""

        source_handed_off = False
        try:
            remote_addresses, local_addresses, sizes = materialize_layer_gva(
                self._local, self._layer_slices, self._remote, plan, layer_id
            )
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
    local: dict[int, LocalTable],
    layers: dict[int, LayerTable],
    remote: dict[int, RemoteTable],
    plan: LayerTransferPlan,
    layer_id: int,
) -> tuple[list[int], list[int], list[int]]:
    """Evaluate only layer-dependent GVA arrays from prepared objects."""

    remote_object_bases_by_group = plan.remote_object_bases_by_group
    assert remote_object_bases_by_group is not None
    remote_addresses: list[int] = []
    selected_addresses: list[int] = []
    sizes: list[int] = []
    for group in plan.groups_by_layer[layer_id]:
        for partition_index, ((pp, _, _), segments) in enumerate(local[group.group_id].items()):
            selected = layers[group.group_id][pp].get(layer_id)
            if selected is None or selected.start == selected.stop:
                continue
            ids = group.selected_local_block_ids_by_partition[partition_index]
            if not len(ids):
                continue
            bases = remote_object_bases_by_group[group.group_id][
                group.selected_transfer_slices_by_partition[partition_index]
            ]
            addresses = local_addresses(segments, ids, selected)
            # Remote addition previously used Python integers. Preserve that
            # range instead of silently wrapping large GVA sums to uint64.
            offsets = remote[group.group_id][pp]["offsets"][selected]
            remote_rows = np.add(bases[:, None], offsets[None, :], dtype=object)
            remote_addresses.extend(remote_rows.ravel().tolist())
            selected_addresses.extend(addresses.ravel().tolist())
            sizes.extend(np.broadcast_to(local_lengths(segments, selected), addresses.shape).ravel().tolist())
    return remote_addresses, selected_addresses, sizes
