"""Execute Layerwise transfer work through Backend key-range sessions."""

from __future__ import annotations

from vllm.logger import logger

from ...backend import BackendSpec, KeyRangeBackend
from ...projection import KeyRangeLayerwiseProjection
from ..transfer.batch import KVTransferBatch, LayerTransferGroup, LayerTransferPlan
from ..transfer.evidence import LayerStoreResult, TransferEvidence
from .arguments import (
    materialize_load_layer_ranges,
    materialize_store_layer_ranges,
    merge_key_range_arguments,
    prepare_layer_load,
    prepare_layer_store,
)
from .io import BackendIO, _batch_sources, _failed_load_completions, _load_completions, _store_completions


class KeyRangeBackendIO(BackendIO):
    """Apply Backend range-session APIs to layer-restricted bindings."""

    def __init__(self, backend: KeyRangeBackend, backend_spec: BackendSpec) -> None:
        super().__init__(backend, backend_spec)
        self._key_range_backend = backend
        self._projection: KeyRangeLayerwiseProjection | None = None
        self._load_plan: LayerTransferPlan | None = None
        self._store_plan: LayerTransferPlan | None = None

    def bind_projection(self, projection: KeyRangeLayerwiseProjection) -> None:
        if self._projection is not None:
            raise RuntimeError("KeyRange projection is already bound")
        self._projection = projection

    def validate_support(self) -> None:
        self._key_range_backend.validate_key_range_support()

    def start_load_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        self._load_plan = None
        return self._key_range_backend.start_key_range_load(keys)

    def start_store_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        self._store_plan = None
        return self._key_range_backend.start_key_range_store(keys, object_sizes)

    def prepare_load_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._load_plan = prepare_layer_load(batch)
        return self._load_plan.keys_by_layer

    def prepare_store_layers(self, batch: KVTransferBatch) -> dict[int, list[str]]:
        self._store_plan = prepare_layer_store(batch)
        return self._store_plan.keys_by_layer

    def load_layer(self, layer_id: int):
        plan = self._prepared_load_plan()
        try:
            arguments = merge_key_range_arguments(materialize_load_layer_ranges(self._bound_projection, plan, layer_id))
            if not arguments.keys:
                return _load_completions(plan.batch, ())
            native_result = self._key_range_backend.copy_key_range_load(
                arguments.keys, arguments.addresses, arguments.sizes, arguments.offsets
            )
            result_codes = self._require_result_codes("batch_copy_get", arguments.keys, native_result)
        except Exception as error:
            logger.error(
                "KeyRange Load failed for requests %s. type=%s, error=%s",
                plan.batch.request_ids,
                type(error).__name__,
                error,
            )
            return _failed_load_completions(plan.batch, layer_id)
        codes_by_key = dict(zip(arguments.keys, result_codes, strict=True))
        evidence = tuple(TransferEvidence(source, codes_by_key[source.key]) for source in arguments.sources)
        return _load_completions(plan.batch, evidence)

    def load_batch(self, batch: KVTransferBatch, layer_id: int | None = None):
        """Compatibility entry for direct adapter tests; Timeline uses the plan."""

        if layer_id is None:
            raise ValueError("KeyRange Load requires a physical layer")
        self.prepare_load_layers(batch)
        return self.load_layer(layer_id)

    def store_batch(self, batch: KVTransferBatch, layer_id: int | None = None):
        if layer_id is None:
            raise ValueError("KeyRange Store requires a physical layer")
        if batch.empty:
            return _store_completions(batch, ())
        self.prepare_store_layers(batch)
        result = self.store_layer(layer_id)
        groups, keys = self._prepared_store_layer(layer_id)
        del groups
        if isinstance(result.result_codes, tuple):
            codes_by_key = dict(zip(keys, result.result_codes, strict=True))
            evidence = tuple(
                TransferEvidence(
                    source,
                    codes_by_key[source.key],
                    codes_by_key[source.key] == 0,
                )
                for source in _batch_sources(batch, layer_id)
            )
        else:
            evidence = tuple(
                TransferEvidence(source, result.result_codes, bool(result.source_release_confirmed))
                for source in _batch_sources(batch, layer_id)
            )
        return _store_completions(batch, evidence, result.error, force_failed=result.result_codes is None)

    def store_layer(self, layer_id: int) -> LayerStoreResult:
        """Copy one layer and defer immutable source evidence to session finalization."""

        source_handed_off = False
        groups, keys = self._prepared_store_layer(layer_id)
        try:
            addresses, sizes, offsets = materialize_store_layer_ranges(self._bound_projection, groups, layer_id)
            source_handed_off = True
            native_result = self._key_range_backend.copy_key_range_store(keys, addresses, sizes, offsets)
        except Exception as error:
            return LayerStoreResult(None, not source_handed_off, error)
        result_codes, result_error = self._interpret_store_results(len(keys), native_result)
        if result_codes is None:
            return LayerStoreResult(None, False, result_error)
        return LayerStoreResult(result_codes, tuple(code == 0 for code in result_codes), result_error)

    def finish_load_sessions(self, keys: list[str]) -> None:
        self._key_range_backend.finish_key_range_load(keys)
        self._load_plan = None

    def commit_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        try:
            return self._key_range_backend.commit_key_range_store(keys)
        finally:
            self._store_plan = None

    def revoke_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        try:
            return self._key_range_backend.revoke_key_range_store(keys)
        finally:
            self._store_plan = None

    @property
    def _bound_projection(self) -> KeyRangeLayerwiseProjection:
        if self._projection is None:
            raise RuntimeError("KeyRange projection is unavailable before cache registration")
        return self._projection

    def _prepared_load_plan(self) -> LayerTransferPlan:
        if self._load_plan is None:
            raise RuntimeError("Layerwise Load plan was not prepared")
        return self._load_plan

    def _prepared_store_layer(self, layer_id: int) -> tuple[tuple[LayerTransferGroup, ...], list[str]]:
        plan = self._prepared_store_plan()
        try:
            return plan.groups_by_layer[layer_id], plan.keys_by_layer[layer_id]
        except KeyError as error:
            raise RuntimeError(f"Layerwise Store layer {layer_id} was not prepared") from error

    def _prepared_store_plan(self) -> LayerTransferPlan:
        if self._store_plan is None:
            raise RuntimeError("Layerwise Store plan was not prepared")
        return self._store_plan
