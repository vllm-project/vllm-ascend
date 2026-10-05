"""Translate KV Pool values to Backend calls and normalize their evidence."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from vllm.logger import logger

from ...backend import BackendSpec, KVStoreBackend
from ..batch import KVTransferBatch
from ..evidence import (
    LoadCompletion as RuntimeLoadCompletion,
)
from ..evidence import (
    StoreCompletion as RuntimeStoreCompletion,
)
from ..evidence import (
    StoreEvidence as RuntimeStoreEvidence,
)
from ..evidence import (
    TransferEvidence as RuntimeTransferEvidence,
)
from .arguments import BulkBackendArguments


class BackendIO:
    """Own the v1 boundary between domain batches and one AscendStore Backend."""

    def __init__(self, backend: KVStoreBackend, backend_spec: BackendSpec) -> None:
        self._backend = backend
        self._backend_spec = backend_spec

    def initialize_thread(self) -> None:
        self._backend.set_device()

    def exists(self, keys: list[str]) -> tuple[bool, ...]:
        if not keys:
            return ()
        presence = tuple(self._backend.exists(keys))
        if len(presence) != len(keys):
            raise ValueError(f"Backend returned {len(presence)} results for {len(keys)} keys")
        if any(value not in (0, 1) for value in presence):
            raise ValueError("Backend returned object states other than 0 or 1")
        return tuple(bool(value) for value in presence)

    def load_materialized(
        self,
        batch: KVTransferBatch,
        arguments: BulkBackendArguments,
    ) -> tuple[RuntimeLoadCompletion, ...]:
        """Call the Backend with key/range arguments already prepared by Worker."""

        if not arguments.sources:
            return tuple(RuntimeLoadCompletion(request_id, ()) for request_id in batch.request_ids)
        try:
            if len(arguments.keys) != len(arguments.sources):
                raise RuntimeError("Bulk Load arguments do not align keys and source evidence")
            native_result = self._backend.load(arguments.keys, arguments.addresses, arguments.sizes)
        except Exception as error:
            logger.error(
                "Bulk Load failed for requests %s. type=%s, error=%s",
                batch.request_ids,
                type(error).__name__,
                error,
            )
            evidence = tuple(RuntimeTransferEvidence(source, None) for source in arguments.sources)
            return _load_completions(batch, evidence)
        result_codes = self._aligned_result_codes(len(arguments.sources), native_result)
        if any(code is None for code in result_codes):
            logger.error("Bulk Load returned malformed evidence for requests %s", batch.request_ids)
        evidence = tuple(
            RuntimeTransferEvidence(source, code) for source, code in zip(arguments.sources, result_codes, strict=True)
        )
        return _load_completions(batch, evidence)

    def store_materialized(
        self,
        batch: KVTransferBatch,
        arguments: BulkBackendArguments,
    ) -> tuple[RuntimeStoreCompletion, ...]:
        """Call the Backend with admitted key/range arguments prepared by Worker."""

        if not arguments.sources:
            return _empty_store_completions(batch)
        source_addresses_handed_off = False
        try:
            if len(arguments.keys) != len(arguments.sources):
                raise RuntimeError("Bulk Store arguments do not align keys and source evidence")
            source_addresses_handed_off = True
            native_result = self._backend.store(arguments.keys, arguments.addresses, arguments.sizes)
        except Exception as error:
            evidence = tuple(
                RuntimeTransferEvidence(source, None, not source_addresses_handed_off) for source in arguments.sources
            )
            return _store_completions(batch, evidence, error)
        result_codes, result_error = self._interpret_store_results(len(arguments.sources), native_result)
        if result_codes is None:
            evidence = tuple(RuntimeTransferEvidence(source, None, False) for source in arguments.sources)
        else:
            evidence = tuple(
                RuntimeTransferEvidence(source, code, code == 0)
                for source, code in zip(arguments.sources, result_codes, strict=True)
            )
        return _store_completions(batch, evidence, result_error, force_failed=result_codes is None)

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
            result_error = RuntimeError(
                f"{self._backend_spec.name} Store returned {len(result_codes)} results for {expected_count} keys"
            )
            return None, result_error
        if any(not _is_integer_result(code) for code in result_codes):
            return None, RuntimeError(f"{self._backend_spec.name} Store returned non-integer results")
        return tuple(int(code) for code in result_codes), None

    @staticmethod
    def _require_result_codes(operation: str, keys: list[str], native_result: Any) -> tuple[int, ...]:
        try:
            result_codes = tuple(native_result)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{operation} returned non-integer results") from error
        if len(result_codes) != len(keys):
            raise RuntimeError(f"{operation} returned {len(result_codes)} results for {len(keys)} keys")
        if any(not _is_integer_result(code) for code in result_codes):
            raise RuntimeError(f"{operation} returned non-integer results")
        return tuple(int(code) for code in result_codes)

    @staticmethod
    def _aligned_result_codes(expected_count: int, native_result: Any) -> tuple[int | None, ...]:
        if native_result is None:
            return (None,) * expected_count
        try:
            result_codes = tuple(native_result)
        except (TypeError, ValueError):
            return (None,) * expected_count
        if len(result_codes) != expected_count:
            return (None,) * expected_count
        if any(not _is_integer_result(code) for code in result_codes):
            return (None,) * expected_count
        return tuple(int(code) for code in result_codes)


def _batch_sources(batch: KVTransferBatch, layer_id: int | None) -> tuple:
    return tuple(
        source
        for group in batch.groups
        for source in (
            group.source(index, layer_id)
            for index in (
                range(group.object_count) if group.selection is None else np.flatnonzero(group.selection).tolist()
            )
        )
    )


def _is_integer_result(value: Any) -> bool:
    """Accept Backend integer scalars without paying ABC lookup for plain ints."""

    return (
        type(value) is int
        or isinstance(value, np.integer)
        or (not isinstance(value, bool) and isinstance(value, Integral))
    )


def _load_completions(
    batch: KVTransferBatch,
    evidence: tuple[RuntimeTransferEvidence, ...],
) -> tuple[RuntimeLoadCompletion, ...]:
    by_request: list[list[RuntimeTransferEvidence]] = [[] for _ in batch.request_ids]
    for item in evidence:
        by_request[item.source.request_index].append(item)
    return tuple(
        RuntimeLoadCompletion(request_id, tuple(items))
        for request_id, items in zip(batch.request_ids, by_request, strict=True)
    )


def _failed_load_completions(
    batch: KVTransferBatch,
    layer_id: int | None,
) -> tuple[RuntimeLoadCompletion, ...]:
    evidence = tuple(RuntimeTransferEvidence(source, None) for source in _batch_sources(batch, layer_id))
    return _load_completions(batch, evidence)


def _empty_store_completions(batch: KVTransferBatch) -> tuple[RuntimeStoreCompletion, ...]:
    job_ids = batch.store_job_ids or (None,) * len(batch.request_ids)
    return tuple(
        RuntimeStoreCompletion(request_id, RuntimeStoreEvidence((), True, True), store_job_id)
        for request_id, store_job_id in zip(batch.request_ids, job_ids, strict=True)
    )


def _store_completions(
    batch: KVTransferBatch,
    evidence: tuple[RuntimeTransferEvidence, ...],
    error: Exception | None = None,
    *,
    force_failed: bool = False,
) -> tuple[RuntimeStoreCompletion, ...]:
    by_request: list[list[RuntimeTransferEvidence]] = [[] for _ in batch.request_ids]
    for item in evidence:
        by_request[item.source.request_index].append(item)
    job_ids = batch.store_job_ids or (None,) * len(batch.request_ids)
    completions = []
    for request_id, store_job_id, items in zip(batch.request_ids, job_ids, by_request, strict=True):
        item_tuple = tuple(items)
        if not item_tuple:
            completions.append(
                RuntimeStoreCompletion(
                    request_id,
                    RuntimeStoreEvidence((), True, True),
                    store_job_id,
                )
            )
            continue
        succeeded = (
            not force_failed
            and error is None
            and all(
                item.result_code == 0 or (item.result_code is None and item.source_release_confirmed is True)
                for item in item_tuple
            )
        )
        source_released = all(item.source_release_confirmed is True for item in item_tuple)
        completions.append(
            RuntimeStoreCompletion(
                request_id,
                RuntimeStoreEvidence(item_tuple, succeeded, source_released, error),
                store_job_id,
            )
        )
    return tuple(completions)
