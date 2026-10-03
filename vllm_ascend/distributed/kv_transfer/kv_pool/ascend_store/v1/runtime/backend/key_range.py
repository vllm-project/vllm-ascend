"""Execute layer-restricted bindings through Backend key-range sessions."""

from __future__ import annotations

from numbers import Integral

from ..batch import KVTransferBatch
from ..evidence import TransferEvidence as RuntimeTransferEvidence
from .arguments import materialize_rule_ranges, merge_rule_key_ranges
from .io import BackendIO, _batch_sources, _load_completions, _store_completions


class KeyRangeBackendIO(BackendIO):
    """Apply Backend range-session APIs to layer-restricted bindings."""

    def validate_support(self) -> None:
        self._backend.validate_layerwise_support()

    def start_load_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        return self._require_result_codes("batch_get_start", keys, self._backend.batch_get_start(keys))

    def start_store_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        return self._require_result_codes("batch_put_start", keys, self._backend.batch_put_start(keys, object_sizes))

    def load_batch(self, batch: KVTransferBatch, layer_id: int | None = None):
        if layer_id is None:
            raise ValueError("KeyRange Load requires a physical layer")
        arguments = merge_rule_key_ranges(
            materialize_rule_ranges(self._bound_rules, batch, layer_id=layer_id, store=False)
        )
        if not arguments.keys:
            return _load_completions(batch, ())
        native_result = self._backend.batch_copy_get(
            arguments.keys,
            arguments.addresses,
            arguments.sizes,
            arguments.offsets,
        )
        result_codes = self._require_result_codes("batch_copy_get", arguments.keys, native_result)
        codes_by_key = dict(zip(arguments.keys, result_codes, strict=True))
        evidence = tuple(RuntimeTransferEvidence(source, codes_by_key[source.key]) for source in arguments.sources)
        return _load_completions(batch, evidence)

    def store_batch(self, batch: KVTransferBatch, layer_id: int | None = None):
        if layer_id is None:
            raise ValueError("KeyRange Store requires a physical layer")
        if batch.empty:
            return _store_completions(batch, ())
        source_handed_off = False
        try:
            arguments = merge_rule_key_ranges(
                materialize_rule_ranges(self._bound_rules, batch, layer_id=layer_id, store=True)
            )
            source_handed_off = True
            native_result = self._backend.batch_copy_put(
                arguments.keys,
                arguments.addresses,
                arguments.sizes,
                arguments.offsets,
            )
        except Exception as error:
            evidence = tuple(
                RuntimeTransferEvidence(source, None, not source_handed_off)
                for source in _batch_sources(batch, layer_id)
            )
            return _store_completions(batch, evidence, error, force_failed=True)
        result_codes, result_error = self._interpret_store_results(len(arguments.keys), native_result)
        codes_by_key = None if result_codes is None else dict(zip(arguments.keys, result_codes, strict=True))
        evidence = tuple(
            RuntimeTransferEvidence(
                source,
                None if codes_by_key is None else codes_by_key[source.key],
                False if codes_by_key is None else codes_by_key[source.key] == 0,
            )
            for source in arguments.sources
        )
        return _store_completions(batch, evidence, result_error, force_failed=result_codes is None)

    def finish_load_sessions(self, keys: list[str]) -> None:
        native_result = self._backend.batch_get_end(keys)
        if isinstance(native_result, bool) or not isinstance(native_result, Integral):
            raise RuntimeError("batch_get_end returned a non-integer result")
        if native_result != 0:
            raise RuntimeError(f"batch_get_end failed with result code {native_result}")

    def commit_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self._require_result_codes("batch_commit", keys, self._backend.batch_commit(keys))

    def revoke_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self._require_result_codes("batch_revoke", keys, self._backend.batch_revoke(keys))
