"""Backend observations and data movement over request-local KV batches."""

from __future__ import annotations

from dataclasses import dataclass

from ..backend import BackendAdapter, BackendStoreEvidence
from ..graph.elements import BindingBatch, KVBinding, RemoteKVObject, RemoteObjectBatch


@dataclass(frozen=True, slots=True)
class RemoteObjectObservation:
    """Readability reported for one exact remote representation."""

    remote_object: RemoteKVObject
    readable: bool


@dataclass(frozen=True, slots=True)
class BindingEvidence:
    """Native result code aligned with one exact transfer edge."""

    binding: KVBinding
    result_code: int | None


@dataclass(frozen=True, slots=True)
class StoreEvidence:
    """Store execution evidence with source-release kept independent."""

    binding_evidence: tuple[BindingEvidence, ...]
    succeeded: bool
    source_release_confirmed: bool
    error: Exception | None = None


class BackendIO:
    """Apply Backend APIs to typed object and binding batches."""

    def __init__(self, backend: BackendAdapter) -> None:
        self.backend = backend

    def observe_readability(self, batch: RemoteObjectBatch) -> tuple[RemoteObjectObservation, ...]:
        if not batch.remote_objects:
            return ()
        presence = tuple(self.backend.exists([remote_object.key for remote_object in batch.remote_objects]))
        if len(presence) != len(batch.remote_objects):
            raise ValueError(f"Lookup returned {len(presence)} results for {len(batch.remote_objects)} remote objects")
        if any(value not in (0, 1) for value in presence):
            raise ValueError("Lookup returned states other than 0 or 1")
        return tuple(
            RemoteObjectObservation(remote_object, value == 1)
            for remote_object, value in zip(batch.remote_objects, presence, strict=True)
        )

    def observe_presence(self, keys: list[str]) -> tuple[int, ...]:
        return tuple(self.backend.exists(keys))

    def load(self, bindings: tuple[KVBinding, ...]) -> tuple[BindingEvidence, ...]:
        if not bindings:
            return ()
        native_result = self.backend.get(
            [binding.remote_object.key for binding in bindings],
            [list(binding.memory.addresses) for binding in bindings],
            [list(binding.memory.sizes) for binding in bindings],
        )
        if native_result is None:
            return tuple(BindingEvidence(binding, None) for binding in bindings)
        result_codes = tuple(native_result)
        if len(result_codes) != len(bindings):
            return tuple(BindingEvidence(binding, None) for binding in bindings)
        return tuple(BindingEvidence(binding, code) for binding, code in zip(bindings, result_codes, strict=True))

    def store(self, batches: tuple[BindingBatch, ...]) -> StoreEvidence:
        bindings = tuple(binding for batch in batches for binding in batch.bindings)
        if not bindings:
            return StoreEvidence((), True, source_release_confirmed=True)
        backend_evidence = self.backend.put(
            [binding.remote_object.key for binding in bindings],
            [list(binding.memory.addresses) for binding in bindings],
            [list(binding.memory.sizes) for binding in bindings],
        )
        binding_evidence = self._align_store_evidence(bindings, backend_evidence)
        return StoreEvidence(
            binding_evidence,
            backend_evidence.succeeded,
            backend_evidence.source_release_confirmed,
            backend_evidence.error,
        )

    @staticmethod
    def _align_store_evidence(
        bindings: tuple[KVBinding, ...], backend_evidence: BackendStoreEvidence
    ) -> tuple[BindingEvidence, ...]:
        if backend_evidence.result_codes is None or len(backend_evidence.result_codes) != len(bindings):
            return tuple(BindingEvidence(binding, None) for binding in bindings)
        return tuple(
            BindingEvidence(binding, code)
            for binding, code in zip(bindings, backend_evidence.result_codes, strict=True)
        )


class LayerwiseBackendIO(BackendIO):
    """Apply Backend range-session APIs to layer-restricted bindings."""

    def validate_support(self) -> None:
        self.backend.validate_layerwise_support()

    def start_load_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self.backend.start_load_sessions(keys)

    def start_store_sessions(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        return self.backend.start_store_sessions(keys, object_sizes)

    def load(self, bindings: tuple[KVBinding, ...]) -> tuple[BindingEvidence, ...]:
        if not bindings:
            return ()
        bindings_by_key: dict[str, list[KVBinding]] = {}
        for binding in bindings:
            bindings_by_key.setdefault(binding.remote_object.key, []).append(binding)
        keys = list(bindings_by_key)
        result_codes = self.backend.load_session_ranges(
            keys,
            [[address for binding in bindings_by_key[key] for address in binding.memory.addresses] for key in keys],
            [[size for binding in bindings_by_key[key] for size in binding.memory.sizes] for key in keys],
            [[offset for binding in bindings_by_key[key] for offset in binding.remote_offsets] for key in keys],
        )
        codes_by_key = dict(zip(keys, result_codes, strict=True))
        return tuple(BindingEvidence(binding, codes_by_key[binding.remote_object.key]) for binding in bindings)

    def finish_load_sessions(self, keys: list[str]) -> None:
        result = self.backend.finish_load_sessions(keys)
        if result != 0:
            raise RuntimeError(f"batch_get_end failed with result code {result}")

    def store(self, batches: tuple[BindingBatch, ...]) -> StoreEvidence:
        bindings = tuple(binding for batch in batches for binding in batch.bindings)
        if not bindings:
            return StoreEvidence((), True, source_release_confirmed=True)
        bindings_by_key: dict[str, list[KVBinding]] = {}
        for binding in bindings:
            bindings_by_key.setdefault(binding.remote_object.key, []).append(binding)
        keys = list(bindings_by_key)
        backend_evidence = self.backend.store_session_ranges(
            keys,
            [[address for binding in bindings_by_key[key] for address in binding.memory.addresses] for key in keys],
            [[size for binding in bindings_by_key[key] for size in binding.memory.sizes] for key in keys],
            [[offset for binding in bindings_by_key[key] for offset in binding.remote_offsets] for key in keys],
        )
        codes_by_key = None
        if backend_evidence.result_codes is not None and len(backend_evidence.result_codes) == len(keys):
            codes_by_key = dict(zip(keys, backend_evidence.result_codes, strict=True))
        binding_evidence = tuple(
            BindingEvidence(binding, None if codes_by_key is None else codes_by_key[binding.remote_object.key])
            for binding in bindings
        )
        return StoreEvidence(
            binding_evidence,
            backend_evidence.succeeded,
            backend_evidence.source_release_confirmed,
            backend_evidence.error,
        )

    def commit_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self.backend.commit_store_sessions(keys)

    def revoke_store_sessions(self, keys: list[str]) -> tuple[int, ...]:
        return self.backend.revoke_store_sessions(keys)
