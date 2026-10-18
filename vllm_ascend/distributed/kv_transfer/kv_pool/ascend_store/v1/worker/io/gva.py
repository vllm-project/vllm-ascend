"""Prepare GVA copy arguments and request results, without owning remote leases."""

from __future__ import annotations

from typing import Any

import numpy as np
from vllm.logger import logger

from ...backend import BackendSpec, GVABackend, GVAObjectSessions, GVARegion
from ...projection import GVALayerwiseProjection
from ..transfer.batch import KVGroupBatch, KVTransferBatch, LayerTransferPlan
from ..transfer.evidence import LayerStoreResult, TransferEvidence
from .arguments import (
    materialize_load_layer_gva,
    materialize_store_layer_gva,
    prepare_layer_load,
    prepare_layer_store,
)
from .io import (
    BackendIO,
    _batch_sources,
    _failed_load_completions,
    _is_integer_result,
    _load_completions,
    _store_completions,
)


class GVABackendIO(BackendIO):
    """Use Backend-owned regions to prepare Worker layer plans and request evidence."""

    def __init__(self, backend: GVABackend, backend_spec: BackendSpec, sessions: GVAObjectSessions) -> None:
        super().__init__(backend, backend_spec)
        self._gva_backend = backend
        self._sessions = sessions
        self._projection: GVALayerwiseProjection | None = None
        self._load_plan: LayerTransferPlan | None = None
        self._store_plan: LayerTransferPlan | None = None

    def bind_projection(self, projection: GVALayerwiseProjection) -> None:
        if self._projection is not None:
            raise RuntimeError("GVA projection is already bound")
        self._projection = projection

    def exists(self, keys: list[str]) -> tuple[bool, ...]:
        return tuple(region is not None for region in self._gva_backend.query_gva_regions(keys))

    def prepare_store_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._store_plan = prepare_layer_store(batch, self._resolve_store_bases)
        return self._store_plan.keys_by_layer

    def prepare_load_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._load_plan = prepare_layer_load(batch, self._resolve_load_bases)
        return self._load_plan.keys_by_layer

    def load_layer(self, layer_id: int):
        plan = self._prepared_load_plan()
        if not plan.groups:
            return _load_completions(plan.batch, ())
        try:
            arguments = materialize_load_layer_gva(self._bound_projection, plan, layer_id)
            result_code = self._gva_backend.copy_from_gva(
                arguments.remote_addresses.tolist(),
                arguments.local_addresses.tolist(),
                arguments.sizes.tolist(),
            )
            if not _is_integer_result(result_code):
                raise RuntimeError("GVA batch_copy returned a non-integer result")
        except Exception as error:
            logger.error(
                "GVA Load failed for requests %s. type=%s, error=%s",
                plan.batch.request_ids,
                type(error).__name__,
                error,
            )
            return _failed_load_completions(plan.batch, layer_id)
        evidence = tuple(TransferEvidence(source, int(result_code)) for source in arguments.sources)
        return _load_completions(plan.batch, evidence)

    def load_batch(self, batch: KVTransferBatch, layer_id: int | None = None):
        """Compatibility entry for direct adapter tests; Timeline uses the plan."""

        if layer_id is None:
            raise ValueError("GVA Load requires a physical layer")
        self.prepare_load_layers(batch)
        return self.load_layer(layer_id)

    def store_batch(self, batch: KVTransferBatch, layer_id: int | None = None):
        if layer_id is None:
            raise ValueError("GVA Store requires a physical layer")
        if batch.empty:
            return _store_completions(batch, ())
        self.prepare_store_layers(batch)
        source_handed_off = False
        try:
            remote_addresses, local_addresses, sizes = materialize_store_layer_gva(
                self._bound_projection,
                self._prepared_store_plan(),
                layer_id,
            )
            source_handed_off = True
            result_code = self._gva_backend.copy_to_gva(
                remote_addresses.tolist(),
                local_addresses.tolist(),
                sizes.tolist(),
            )
            if not _is_integer_result(result_code):
                raise RuntimeError("GVA batch_copy returned a non-integer result")
        except Exception as error:
            sources = _batch_sources(batch, layer_id)
            evidence = tuple(TransferEvidence(source, None, not source_handed_off) for source in sources)
            return _store_completions(batch, evidence, error, force_failed=True)
        evidence = tuple(TransferEvidence(source, int(result_code), True) for source in _batch_sources(batch, layer_id))
        return _store_completions(batch, evidence)

    def store_layer(self, source_ready_event: Any, layer_id: int) -> LayerStoreResult:
        """Prepare native arguments ahead of source readiness, then copy this layer."""

        source_handed_off = False
        plan = self._prepared_store_plan()
        try:
            remote_addresses, local_addresses, sizes = materialize_store_layer_gva(
                self._bound_projection, plan, layer_id
            )
            remote_buffers = remote_addresses.tolist()
            local_buffers = local_addresses.tolist()
            buffer_sizes = sizes.tolist()
            source_ready_event.synchronize()
            source_handed_off = True
            result_code = self._gva_backend.copy_to_gva(remote_buffers, local_buffers, buffer_sizes)
            if not _is_integer_result(result_code):
                raise RuntimeError("GVA batch_copy returned a non-integer result")
        except Exception as error:
            return LayerStoreResult(None, not source_handed_off, error)
        return LayerStoreResult(int(result_code), True)

    def clear_load_plan(self) -> None:
        self._load_plan = None

    def clear_store_plan(self) -> None:
        self._store_plan = None

    def _resolve_store_bases(self, group: KVGroupBatch, keys: tuple[str, ...]) -> np.ndarray:
        return self._region_bases(self._sessions.store_regions(keys, group.object_size))

    def _resolve_load_bases(self, group: KVGroupBatch, keys: tuple[str, ...]) -> np.ndarray:
        return self._region_bases(self._sessions.load_regions(keys, group.object_size))

    @staticmethod
    def _region_bases(regions: tuple[GVARegion, ...]) -> np.ndarray:
        object_bases = np.asarray([region.address for region in regions], dtype=np.uint64)
        object_bases.flags.writeable = False
        return object_bases

    @property
    def _bound_projection(self) -> GVALayerwiseProjection:
        if self._projection is None:
            raise RuntimeError("GVA projection is unavailable before cache registration")
        return self._projection

    def _prepared_load_plan(self) -> LayerTransferPlan:
        if self._load_plan is None:
            raise RuntimeError("Layerwise Load plan was not prepared")
        return self._load_plan

    def _prepared_store_plan(self) -> LayerTransferPlan:
        if self._store_plan is None:
            raise RuntimeError("Layerwise Store plan was not prepared")
        return self._store_plan
