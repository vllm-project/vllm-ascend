# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side execution and wire metadata for asynchronous Store GET."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeConnectorMetadata,
    ECMooncakeWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    TensorSpec,
    make_embedding_key,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (
    EmbeddingStoreOperationError,
)
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_backend import (
        AscendDeferredStoreBackend,
    )
    from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_client import (
        AscendMooncakeEmbeddingStoreClient,
    )

_GetPayloadCallback = Callable[[dict[str, torch.Tensor]], list[Future[None]]]


@dataclass
class AscendECMooncakeConnectorMetadata(ECMooncakeConnectorMetadata):
    """Async Store GET work and terminal results consumed by this model step."""

    store_get_requests: dict[str, TensorSpec] = field(default_factory=dict)
    store_get_consume: set[str] = field(default_factory=set)
    store_get_discard: set[str] = field(default_factory=set)


@dataclass
class AscendECMooncakeWorkerMetadata(ECMooncakeWorkerMetadata):
    """Completed asynchronous Store GET outcomes returned to the scheduler."""

    store_get_hits: set[str] = field(default_factory=set)
    store_get_misses: set[str] = field(default_factory=set)
    store_get_failures: set[str] = field(default_factory=set)

    def aggregate(
        self,
        other: ECMooncakeWorkerMetadata,
    ) -> AscendECMooncakeWorkerMetadata:
        """Intersect worker hits and combine misses and failures."""
        assert isinstance(other, AscendECMooncakeWorkerMetadata)
        base = super().aggregate(other)
        return AscendECMooncakeWorkerMetadata(
            **vars(base),
            store_get_hits=self.store_get_hits & other.store_get_hits,
            store_get_misses=self.store_get_misses | other.store_get_misses,
            store_get_failures=self.store_get_failures | other.store_get_failures,
        )


@dataclass(frozen=True)
class _StoreGetResult:
    """Hold terminal tensors and confirmed failures for one GET job."""

    tensors: dict[str, torch.Tensor | None]
    failures: set[str]


@dataclass(frozen=True)
class _PendingStoreGet:
    """Track identifiers owned by a queued or running GET job."""

    identifiers: set[str]
    future: Future[_StoreGetResult]


class AsyncStoreGetExecutor:
    """Execute admitted Store GETs off the model thread and own their results."""

    def __init__(self, *, enabled: bool, device: str) -> None:
        """Initialize the single GET executor and model-thread result bookkeeping."""
        self.enabled = enabled
        self._device = device
        self._executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ec-mooncake-store-get",
            )
            if enabled
            else None
        )
        self._pending: list[_PendingStoreGet] = []
        self._ready: dict[str, torch.Tensor | None] = {}
        self._hits: set[str] = set()
        self._misses: set[str] = set()
        self._failures: set[str] = set()
        self._store_client: AscendMooncakeEmbeddingStoreClient | None = None

    def _load_batch(
        self,
        store: AscendDeferredStoreBackend,
        expected: dict[str, TensorSpec],
        *,
        on_payloads: _GetPayloadCallback | None = None,
    ) -> _StoreGetResult:
        """Check existence and load hits with a callback scoped to this job."""
        identifiers = {make_embedding_key(store.namespace, identifier): identifier for identifier in expected}
        client = store.store_client
        try:
            exists = client.batch_exists(list(identifiers))
            hits = {
                pool_key: expected[identifier]
                for (pool_key, identifier), hit in zip(identifiers.items(), exists, strict=True)
                if hit
            }
            # Each submitted job carries its own immutable key mapping. The
            # existing I/O lock scopes the callback through the complete GET;
            # foreground fallback cannot replace it while native I/O runs.
            callback_scope = (
                client.get_payload_callback(on_payloads) if hits and on_payloads is not None else nullcontext()
            )
            with callback_scope:
                loaded = client.load_tensors_ready(hits, self._device)
        except (EmbeddingStoreOperationError, OSError):
            logger.warning(
                "Asynchronous Store GET failed; falling back to encoder",
                exc_info=True,
            )
            return _StoreGetResult({}, set(expected))

        tensors = {identifier: loaded.get(pool_key) for pool_key, identifier in identifiers.items()}
        return _StoreGetResult(tensors, set())

    def dispatch(
        self,
        store: AscendDeferredStoreBackend,
        expected: dict[str, TensorSpec],
        *,
        on_payloads: _GetPayloadCallback | None = None,
    ) -> None:
        """Submit a snapshot of admitted objects to the background GET worker."""
        if not expected:
            return
        if self._executor is None:
            raise RuntimeError("Asynchronous Store GET work received while the feature is disabled")
        self._store_client = store.store_client
        future = self._executor.submit(self._load_batch, store, dict(expected), on_payloads=on_payloads)
        self._pending.append(_PendingStoreGet(set(expected), future))

    def poll(self) -> None:
        # A direct P2P reader can fail after the GET Future has completed. Its
        # completion callback poisons the client; surface that fatal state on
        # the model thread even when no further Store GET is dispatched.
        """Collect completed GETs and surface fatal Store or reader failures."""
        if self._store_client is not None:
            self._store_client._check_healthy()
        pending = []
        for batch in self._pending:
            if not batch.future.done():
                pending.append(batch)
                continue
            # Safe operation failures are explicit. Unexpected failures,
            # including uncertain native completion, remain fatal.
            result = batch.future.result()
            for identifier in batch.identifiers:
                if identifier in result.failures:
                    self._failures.add(identifier)
                    self._ready[identifier] = None
                    continue
                tensor = result.tensors.get(identifier)
                self._ready[identifier] = tensor
                (self._misses if tensor is None else self._hits).add(identifier)
        self._pending = pending

    def consume(
        self,
        metadata: AscendECMooncakeConnectorMetadata,
        encoder_cache: dict[str, torch.Tensor],
    ) -> set[str]:
        """Move terminal hits into the model cache and return handled keys."""
        self.poll()
        for identifier in metadata.store_get_discard:
            self._ready.pop(identifier, None)
        consumed = set(metadata.store_get_consume)
        for identifier in consumed:
            if identifier not in self._ready:
                raise RuntimeError(f"Asynchronous Store GET result is unavailable for {identifier}")
            tensor = self._ready.pop(identifier)
            if tensor is not None:
                encoder_cache[identifier] = tensor
        return consumed

    def extend_worker_metadata(
        self,
        base: ECMooncakeWorkerMetadata,
    ) -> AscendECMooncakeWorkerMetadata:
        """Report newly completed outcomes and clear their notification sets."""
        self.poll()
        metadata = AscendECMooncakeWorkerMetadata(
            **vars(base),
            store_get_hits=self._hits,
            store_get_misses=self._misses,
            store_get_failures=self._failures,
        )
        self._hits = set()
        self._misses = set()
        self._failures = set()
        return metadata

    def close(self) -> None:
        """Drain queued and running GETs before dropping unconsumed results."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
            self.poll()
        self._ready.clear()
