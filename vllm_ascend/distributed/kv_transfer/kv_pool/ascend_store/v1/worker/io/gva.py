"""Execute Layerwise transfer work through GVA Backend sessions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from vllm.logger import logger

from ...backend import BackendSpec, GVABackend, GVARegion
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

GVA_SESSION_FAILURE = -1


class GVABackendIO(BackendIO):
    """Keep leased GVA addresses inside the Backend boundary."""

    def __init__(self, backend: GVABackend, backend_spec: BackendSpec) -> None:
        super().__init__(backend, backend_spec)
        self._gva_backend = backend
        self._load_sessions: dict[str, tuple[int, int] | None] = {}
        self._store_sessions: dict[str, tuple[int, int]] = {}
        self._projection: GVALayerwiseProjection | None = None
        self._load_plan: LayerTransferPlan | None = None
        self._store_plan: LayerTransferPlan | None = None
        self.validate_support()

    def bind_projection(self, projection: GVALayerwiseProjection) -> None:
        if self._projection is not None:
            raise RuntimeError("GVA projection is already bound")
        self._projection = projection

    def validate_support(self) -> None:
        self._gva_backend.validate_gva_support()

    def exists(self, keys: list[str]) -> tuple[bool, ...]:
        return tuple(region is not None for region in self._query_regions(keys))

    def start_load_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        self._load_plan = None
        if not keys:
            return ()
        if any(key in self._load_sessions for key in keys):
            raise RuntimeError("Previous GVA Load sessions have not ended")
        result_codes = list(self._gva_backend.add_gva_leases(keys))
        leased_keys = [key for key, code in zip(keys, result_codes, strict=True) if code == 0]
        self._load_sessions.update((key, None) for key in leased_keys)
        # Resolve after acquiring the lease: an earlier Lookup address may already have been evicted.
        regions = dict(zip(leased_keys, self._query_regions(leased_keys), strict=True))
        rejected_keys = []
        for index, (key, object_size) in enumerate(zip(keys, object_sizes, strict=True)):
            if result_codes[index] != 0:
                continue
            region = regions[key]
            if region is None or region.size != object_size:
                result_codes[index] = GVA_SESSION_FAILURE
                rejected_keys.append(key)
            else:
                self._load_sessions[key] = (region.address, region.size)
        self.finish_load_sessions(rejected_keys)
        return tuple(result_codes)

    def finish_load_sessions(self, keys: list[str]) -> None:
        leased_keys = [key for key in keys if key in self._load_sessions]
        if not leased_keys:
            return
        self._gva_backend.remove_gva_leases(leased_keys)
        for key in leased_keys:
            del self._load_sessions[key]
        self._load_plan = None

    def start_store_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        self._store_plan = None
        if not keys:
            return ()
        if any(key in self._store_sessions for key in keys):
            raise RuntimeError("Previous GVA Store sessions have not ended")
        gvas = self._gva_backend.allocate_gva(keys, object_sizes)
        result_codes = []
        for key, gva, object_size in zip(keys, gvas, object_sizes, strict=True):
            result_codes.append(0 if gva > 0 else GVA_SESSION_FAILURE)
            if gva > 0:
                self._store_sessions[key] = (gva, object_size)
        return tuple(result_codes)

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
        succeeded = result_code == 0
        evidence = tuple(
            TransferEvidence(source, int(result_code), succeeded) for source in _batch_sources(batch, layer_id)
        )
        return _store_completions(batch, evidence)

    def store_layer(self, layer_id: int) -> LayerStoreResult:
        """Copy one layer and defer immutable source evidence to session finalization."""

        source_handed_off = False
        plan = self._prepared_store_plan()
        try:
            remote_addresses, local_addresses, sizes = materialize_store_layer_gva(
                self._bound_projection, plan, layer_id
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
            return LayerStoreResult(None, not source_handed_off, error)
        succeeded = result_code == 0
        return LayerStoreResult(int(result_code), succeeded)

    def commit_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        if not keys:
            return ()
        if any(key not in self._store_sessions for key in keys):
            raise RuntimeError("GVA commit has no owning Store session")
        result_codes = self._gva_backend.publish_gva(keys)
        for key, code in zip(keys, result_codes, strict=True):
            if code == 0:
                del self._store_sessions[key]
        self._store_plan = None
        return result_codes

    def revoke_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        # Native failed-write notification deletes by key, even after a duplicate allocation.
        # Do not delete another writer's object; leave uncertain write leases to Backend expiry.
        result = tuple(GVA_SESSION_FAILURE if self._store_sessions.pop(key, None) is not None else 0 for key in keys)
        self._store_plan = None
        return result

    def _resolve_store_bases(self, group: KVGroupBatch, keys: tuple[str, ...]) -> np.ndarray:
        return self._resolve_bases(group, keys, self._store_sessions)

    def _resolve_load_bases(self, group: KVGroupBatch, keys: tuple[str, ...]) -> np.ndarray:
        return self._resolve_bases(group, keys, self._load_sessions)

    @staticmethod
    def _resolve_bases(
        group: KVGroupBatch,
        keys: tuple[str, ...],
        sessions: Mapping[str, tuple[int, int] | None],
    ) -> np.ndarray:
        bases = []
        for key in keys:
            session = sessions.get(key)
            if session is None:
                raise RuntimeError(f"GVA session for {key!r} is unavailable")
            base, object_size = session
            if object_size != group.object_size:
                raise RuntimeError(f"GVA session for {key!r} has an unexpected object size")
            bases.append(base)
        object_bases = np.asarray(bases, dtype=np.uint64)
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

    def _query_regions(self, keys: list[str]) -> tuple[GVARegion | None, ...]:
        return self._gva_backend.query_gva_regions(keys)
