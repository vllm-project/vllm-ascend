"""Translate KV Pool values to Backend calls and normalize their evidence."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import Any

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

from ..backend import BackendSpec
from ..program.invocation import BindingEvidence, RemoteObjectObservation, StoreEvidence
from ..program.representation import BindingBatch, KVBinding, RemoteObjectBatch


class BackendIO:
    """Own the v1 boundary between domain batches and one AscendStore Backend."""

    def __init__(self, backend: Backend, backend_spec: BackendSpec) -> None:
        self._backend = backend
        self._backend_spec = backend_spec

    def initialize_thread(self) -> None:
        self._backend.set_device()

    def observe_readability(self, batch: RemoteObjectBatch) -> tuple[RemoteObjectObservation, ...]:
        if not batch.remote_objects:
            return ()
        presence = tuple(self._backend.exists([remote_object.key for remote_object in batch.remote_objects]))
        if len(presence) != len(batch.remote_objects):
            raise ValueError(f"Lookup returned {len(presence)} results for {len(batch.remote_objects)} remote objects")
        if any(value not in (0, 1) for value in presence):
            raise ValueError("Lookup returned states other than 0 or 1")
        return tuple(
            RemoteObjectObservation(remote_object, value == 1)
            for remote_object, value in zip(batch.remote_objects, presence, strict=True)
        )

    def observe_presence(self, keys: list[str]) -> tuple[int, ...]:
        return tuple(self._backend.exists(keys))

    def load(self, bindings: tuple[KVBinding, ...]) -> tuple[BindingEvidence, ...]:
        if not bindings:
            return ()
        native_result = self._backend.get(
            [binding.remote_object.key for binding in bindings],
            [list(binding.memory.addresses) for binding in bindings],
            [list(binding.memory.sizes) for binding in bindings],
        )
        if native_result is None:
            return self._unknown_binding_evidence(bindings)
        result_codes = tuple(native_result)
        if len(result_codes) != len(bindings):
            return self._unknown_binding_evidence(bindings)
        return tuple(BindingEvidence(binding, code) for binding, code in zip(bindings, result_codes, strict=True))

    def store(self, batches: tuple[BindingBatch, ...]) -> StoreEvidence:
        bindings = tuple(binding for batch in batches for binding in batch.bindings)
        if not bindings:
            return StoreEvidence((), True, source_release_confirmed=True)

        keys = [binding.remote_object.key for binding in bindings]
        addresses = [list(binding.memory.addresses) for binding in bindings]
        sizes = [list(binding.memory.sizes) for binding in bindings]
        source_addresses_handed_off = False
        try:
            native_put, native_args = self._prepare_put(keys, addresses, sizes)
            source_addresses_handed_off = True
            native_result = native_put(*native_args)
        except Exception as error:
            return StoreEvidence(
                self._unknown_binding_evidence(bindings),
                False,
                not source_addresses_handed_off,
                error,
            )
        if self._backend_spec.name == "yuanrong":
            return StoreEvidence(self._unknown_binding_evidence(bindings), True, source_release_confirmed=True)
        return self._store_evidence_from_aligned_results(bindings, native_result)

    def _prepare_put(
        self,
        keys: list[str],
        addresses: list[list[int]],
        sizes: list[list[int]],
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        backend = self._backend
        if self._backend_spec.name == "mooncake":
            backend.ensure_initialized()
            if backend.store is None:
                raise RuntimeError("Mooncake store is unavailable for put")
            replicate_config = backend._build_replicate_config()
            return backend.store.batch_put_from_multi_buffers, (keys, addresses, sizes, replicate_config)
        if self._backend_spec.name == "memcache":
            backend.ensure_initialized()
            if backend.store is None:
                raise RuntimeError("Memcache store is unavailable for put")
            direction = self._backend_spec.backend_module.MmcDirect.COPY_L2G.value
            return backend.store.batch_put_from_layers, (keys, addresses, sizes, direction)
        if self._backend_spec.name == "yuanrong":
            if backend.store is None:
                raise RuntimeError("Yuanrong store is unavailable for put")
            return backend.store.mset_d2h_from_multi_buffers, (keys, addresses, sizes, backend._ds_set_param)
        return backend.put, (keys, addresses, sizes)

    def _store_evidence_from_aligned_results(
        self,
        bindings: tuple[KVBinding, ...],
        native_result: Any,
    ) -> StoreEvidence:
        result_codes, error = self._interpret_store_results(len(bindings), native_result)
        if result_codes is None:
            binding_evidence = self._unknown_binding_evidence(bindings)
        else:
            binding_evidence = tuple(
                BindingEvidence(binding, code) for binding, code in zip(bindings, result_codes, strict=True)
            )
        return StoreEvidence(
            binding_evidence,
            result_codes is not None and all(code == 0 for code in result_codes),
            source_release_confirmed=True,
            error=error,
        )

    def _interpret_store_results(
        self, expected_count: int, native_result: Any
    ) -> tuple[tuple[int, ...] | None, Exception | None]:
        try:
            result_codes = None if native_result is None else tuple(native_result)
        except Exception as error:
            return None, error
        if result_codes is None:
            return None, RuntimeError(f"{self._backend_spec.name} Store returned no per-key results")
        if len(result_codes) != expected_count:
            error = RuntimeError(
                f"{self._backend_spec.name} Store returned {len(result_codes)} results for {expected_count} keys"
            )
            return None, error
        return result_codes, None

    @staticmethod
    def _unknown_binding_evidence(bindings: tuple[KVBinding, ...]) -> tuple[BindingEvidence, ...]:
        return tuple(BindingEvidence(binding, None) for binding in bindings)


class LayerwiseBackendIO(BackendIO):
    """Apply Backend range-session APIs to layer-restricted bindings."""

    def validate_support(self) -> None:
        self._backend.validate_layerwise_support()

    def start_load_sessions(self, keys: list[str]) -> tuple[int, ...]:
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
            [[address for binding in bindings_by_key[key] for address in binding.memory.addresses] for key in keys],
            [[size for binding in bindings_by_key[key] for size in binding.memory.sizes] for key in keys],
            [[offset for binding in bindings_by_key[key] for offset in binding.remote_offsets] for key in keys],
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
                [[address for binding in bindings_by_key[key] for address in binding.memory.addresses] for key in keys],
                [[size for binding in bindings_by_key[key] for size in binding.memory.sizes] for key in keys],
                [[offset for binding in bindings_by_key[key] for offset in binding.remote_offsets] for key in keys],
            )
        except Exception as error:
            return StoreEvidence(
                self._unknown_binding_evidence(bindings),
                False,
                not source_addresses_handed_off,
                error,
            )

        result_codes, error = self._interpret_store_results(len(keys), native_result)
        codes_by_key = None if result_codes is None else dict(zip(keys, result_codes, strict=True))
        binding_evidence = tuple(
            BindingEvidence(binding, None if codes_by_key is None else codes_by_key[binding.remote_object.key])
            for binding in bindings
        )
        return StoreEvidence(
            binding_evidence,
            result_codes is not None and all(code == 0 for code in result_codes),
            source_release_confirmed=True,
            error=error,
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

    @staticmethod
    def _require_result_codes(operation: str, keys: list[str], native_result: Any) -> tuple[int, ...]:
        try:
            result_codes = tuple(native_result)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{operation} returned non-integer results") from error
        if len(result_codes) != len(keys):
            raise RuntimeError(f"{operation} returned {len(result_codes)} results for {len(keys)} keys")
        if any(isinstance(code, bool) or not isinstance(code, Integral) for code in result_codes):
            raise RuntimeError(f"{operation} returned non-integer results")
        return tuple(int(code) for code in result_codes)
