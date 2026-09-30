"""Execute layer-restricted bindings through Backend key-range sessions."""

from __future__ import annotations

from numbers import Integral

from ...program.values.evidence import BindingEvidence, StoreEvidence
from ...program.values.representation import BindingBatch, KVBinding
from .io import BackendIO


class KeyRangeBackendIO(BackendIO):
    """Apply Backend range-session APIs to layer-restricted bindings."""

    def validate_support(self) -> None:
        self._backend.validate_layerwise_support()

    def start_load_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        return self._require_result_codes("batch_get_start", keys, self._backend.batch_get_start(keys))

    def start_store_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        return self._require_result_codes("batch_put_start", keys, self._backend.batch_put_start(keys, object_sizes))

    def load(self, bindings: tuple[KVBinding, ...]) -> tuple[BindingEvidence, ...]:
        if not bindings:
            return ()
        bindings_by_key = self._group_bindings_by_key(bindings)
        keys = list(bindings_by_key)
        native_result = self._backend.batch_copy_get(
            keys,
            [
                [address for binding in bindings_by_key[key] for address in binding.local_region.memory.addresses]
                for key in keys
            ],
            [[size for binding in bindings_by_key[key] for size in binding.local_region.memory.sizes] for key in keys],
            [[offset for binding in bindings_by_key[key] for offset in binding.remote_layout.offsets] for key in keys],
        )
        result_codes = self._require_result_codes("batch_copy_get", keys, native_result)
        codes_by_key = dict(zip(keys, result_codes, strict=True))
        return tuple(BindingEvidence(binding, codes_by_key[binding.remote_object.key]) for binding in bindings)

    def finish_load_sessions(self, keys: list[str]) -> None:
        native_result = self._backend.batch_get_end(keys)
        if isinstance(native_result, bool) or not isinstance(native_result, Integral):
            raise RuntimeError("batch_get_end returned a non-integer result")
        if native_result != 0:
            raise RuntimeError(f"batch_get_end failed with result code {native_result}")

    def store(self, batches: tuple[BindingBatch, ...]) -> StoreEvidence:
        bindings = tuple(binding for batch in batches for binding in batch.bindings)
        if not bindings:
            return StoreEvidence((), True, source_release_confirmed=True)
        bindings_by_key = self._group_bindings_by_key(bindings)
        keys = list(bindings_by_key)
        source_addresses_handed_off = False
        try:
            source_addresses_handed_off = True
            native_result = self._backend.batch_copy_put(
                keys,
                [
                    [address for binding in bindings_by_key[key] for address in binding.local_region.memory.addresses]
                    for key in keys
                ],
                [
                    [size for binding in bindings_by_key[key] for size in binding.local_region.memory.sizes]
                    for key in keys
                ],
                [
                    [offset for binding in bindings_by_key[key] for offset in binding.remote_layout.offsets]
                    for key in keys
                ],
            )
        except Exception as error:
            return StoreEvidence(
                self._unknown_binding_evidence(bindings),
                False,
                not source_addresses_handed_off,
                error,
            )

        result_codes, result_error = self._interpret_store_results(len(keys), native_result)
        succeeded = result_codes is not None and all(code == 0 for code in result_codes)
        codes_by_key = None if result_codes is None else dict(zip(keys, result_codes, strict=True))
        binding_evidence = tuple(
            BindingEvidence(binding, None if codes_by_key is None else codes_by_key[binding.remote_object.key])
            for binding in bindings
        )
        return StoreEvidence(
            binding_evidence,
            succeeded,
            source_release_confirmed=succeeded,
            error=result_error,
        )

    def commit_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self._require_result_codes("batch_commit", keys, self._backend.batch_commit(keys))

    def revoke_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self._require_result_codes("batch_revoke", keys, self._backend.batch_revoke(keys))

    @staticmethod
    def _group_bindings_by_key(bindings: tuple[KVBinding, ...]) -> dict[str, list[KVBinding]]:
        bindings_by_key: dict[str, list[KVBinding]] = {}
        for binding in bindings:
            bindings_by_key.setdefault(binding.remote_object.key, []).append(binding)
        return bindings_by_key
