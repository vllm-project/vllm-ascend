"""Worker-side Lookup, Load and queued Store."""

from __future__ import annotations

import torch

from ..metadata import LoadRequestBatch, StoreRequestBatch
from .load import LoadResult, LoadService
from .lookup import LookupService, WorkerLookupRequest
from .resources import WorkerCacheResources
from .store import StoreService


class WorkerService:
    """Orchestrate Worker Lookup, Load and Store operations."""

    def __init__(
        self,
        cache_resources: WorkerCacheResources,
        lookup_service: LookupService,
        load_service: LoadService,
        store_service: StoreService | None,
    ) -> None:
        self._cache_resources = cache_resources
        self._lookup_service = lookup_service
        self._load_service = load_service
        self._store_service = store_service

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        try:
            self._cache_resources.register_kv_caches(kv_caches)
            if self._store_service is not None:
                self._store_service.start()
            self._load_service.start()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        try:
            try:
                if self._store_service is not None:
                    self._store_service.close()
            finally:
                self._load_service.close()
        finally:
            self._cache_resources.close()

    def lookup(self, request: WorkerLookupRequest) -> int:
        return self._lookup_service.lookup(request)

    def load(self, request_batch: LoadRequestBatch) -> None:
        self._load_service.load(request_batch)

    def submit_store(self, request_batch: StoreRequestBatch) -> None:
        if self._store_service is not None:
            self._store_service.submit(request_batch)

    def wait_for_previous_store(self) -> None:
        if self._store_service is not None:
            self._store_service.wait_for_previous_store()

    def finish_store_step(self, request_batch: StoreRequestBatch) -> None:
        if self._store_service is not None:
            self._store_service.finish_step(request_batch)

    def collect_load_result(self) -> LoadResult:
        return self._load_service.collect_result()
