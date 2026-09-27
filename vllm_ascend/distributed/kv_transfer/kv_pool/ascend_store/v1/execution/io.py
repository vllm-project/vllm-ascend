"""Backend observations and data movement over request-local KV batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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


class MissingFilter(Protocol):
    """Select Store bindings that still require a Backend write."""

    def select_missing(self, batches: tuple[BindingBatch, ...]) -> tuple[BindingBatch, ...]: ...


class IdentityMissingFilter:
    """Preserve every Store binding when the Backend overwrites existing keys safely."""

    @staticmethod
    def select_missing(batches: tuple[BindingBatch, ...]) -> tuple[BindingBatch, ...]:
        return batches


class BackendExistenceMissingFilter:
    """Remove Store bindings whose remote keys already exist in the Backend."""

    def __init__(self, backend: BackendAdapter) -> None:
        self._backend = backend

    def select_missing(self, batches: tuple[BindingBatch, ...]) -> tuple[BindingBatch, ...]:
        bindings = tuple(binding for batch in batches for binding in batch.bindings)
        if not bindings:
            return batches
        presence = tuple(self._backend.exists([binding.remote_object.key for binding in bindings]))
        if len(presence) != len(bindings):
            raise RuntimeError(f"Store exists returned {len(presence)} results for {len(bindings)} bindings")
        if any(value not in (0, 1) for value in presence):
            raise RuntimeError("Store exists returned states other than 0 or 1")
        selected_batches = []
        result_offset = 0
        for batch in batches:
            batch_presence = presence[result_offset : result_offset + len(batch.bindings)]
            result_offset += len(batch.bindings)
            selected = tuple(
                binding for binding, value in zip(batch.bindings, batch_presence, strict=True) if value != 1
            )
            selected_batches.append(BindingBatch(batch.group_id, selected))
        return tuple(selected_batches)


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

    def load(self, bindings: tuple[KVBinding, ...]) -> tuple[BindingEvidence, ...]:
        if not bindings:
            return ()
        native_result = self.backend.get(
            [binding.remote_object.key for binding in bindings],
            [list(binding.local_slice.addresses) for binding in bindings],
            [list(binding.local_slice.sizes) for binding in bindings],
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
            [list(binding.local_slice.addresses) for binding in bindings],
            [list(binding.local_slice.sizes) for binding in bindings],
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
