"""Execute layer-restricted bindings through Backend key-range sessions."""

from __future__ import annotations

from numbers import Integral

from ...program.values.evidence import StoreEvidence, TransferEvidence
from ...program.values.selection import TransferWork
from .arguments import materialize_key_ranges
from .io import BackendIO


class KeyRangeBackendIO(BackendIO):
    """Apply Backend range-session APIs to layer-restricted bindings."""

    def validate_support(self) -> None:
        self._backend.validate_layerwise_support()

    def start_load_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        return self._require_result_codes("batch_get_start", keys, self._backend.batch_get_start(keys))

    def start_store_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        return self._require_result_codes("batch_put_start", keys, self._backend.batch_put_start(keys, object_sizes))

    def load(self, work: TransferWork) -> tuple[TransferEvidence, ...]:
        arguments = materialize_key_ranges(work)
        if not arguments.keys:
            return ()
        native_result = self._backend.batch_copy_get(
            arguments.keys,
            arguments.addresses,
            arguments.sizes,
            arguments.offsets,
        )
        result_codes = self._require_result_codes("batch_copy_get", arguments.keys, native_result)
        codes_by_key = dict(zip(arguments.keys, result_codes, strict=True))
        return tuple(TransferEvidence(source, codes_by_key[source.key]) for source in work.sources)

    def finish_load_sessions(self, keys: list[str]) -> None:
        native_result = self._backend.batch_get_end(keys)
        if isinstance(native_result, bool) or not isinstance(native_result, Integral):
            raise RuntimeError("batch_get_end returned a non-integer result")
        if native_result != 0:
            raise RuntimeError(f"batch_get_end failed with result code {native_result}")

    def store(self, work: TransferWork) -> StoreEvidence:
        if work.empty:
            return StoreEvidence((), True, source_release_confirmed=True)
        source_addresses_handed_off = False
        try:
            arguments = materialize_key_ranges(work)
            source_addresses_handed_off = True
            native_result = self._backend.batch_copy_put(
                arguments.keys,
                arguments.addresses,
                arguments.sizes,
                arguments.offsets,
            )
        except Exception as error:
            return StoreEvidence(
                self._unknown_transfer_evidence(work.sources, not source_addresses_handed_off),
                False,
                not source_addresses_handed_off,
                error,
            )

        result_codes, result_error = self._interpret_store_results(len(arguments.keys), native_result)
        succeeded = result_codes is not None and all(code == 0 for code in result_codes)
        codes_by_key = None if result_codes is None else dict(zip(arguments.keys, result_codes, strict=True))
        transfer_evidence = tuple(
            TransferEvidence(
                source,
                None if codes_by_key is None else codes_by_key[source.key],
                False if codes_by_key is None else codes_by_key[source.key] == 0,
            )
            for source in work.sources
        )
        return StoreEvidence(
            transfer_evidence,
            succeeded,
            source_release_confirmed=succeeded,
            error=result_error,
        )

    def commit_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self._require_result_codes("batch_commit", keys, self._backend.batch_commit(keys))

    def revoke_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self._require_result_codes("batch_revoke", keys, self._backend.batch_revoke(keys))
