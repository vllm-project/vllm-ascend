"""Execute non-layerwise KV Pool transfers."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..runtime.batch import KVTransferBatch
from ..runtime.evidence import LoadCompletion, StoreCompletion
from . import StoreBatch
from .executor import TimelineExecutor

_STORE_FENCE_POLL_INTERVAL_S = 1.0


class LoadTimeline:
    """Execute and publish Load completions in the submitting call."""

    collects_completions = False

    def __init__(self) -> None:
        self._operation: Callable[[KVTransferBatch, int | None], tuple[LoadCompletion, ...]] | None = None

    def bind_operation(
        self,
        operation: Callable[[KVTransferBatch, int | None], tuple[LoadCompletion, ...]],
    ) -> None:
        if self._operation is not None:
            raise RuntimeError("Load timeline operation is already bound")
        self._operation = operation

    def start(self) -> None:
        if self._operation is None:
            raise RuntimeError("Load timeline has no bound operation")

    def submit(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        if self._operation is None:
            raise RuntimeError("Load timeline has no bound operation")
        return self._operation(batch, None)

    def collect(self) -> tuple[LoadCompletion, ...]:
        return ()

    def abort(self) -> None:
        return

    def close(self) -> None:
        return


class AsyncLoadTimeline:
    """Execute each submitted request batch on one background timeline."""

    collects_completions = True

    def __init__(self, thread_initializer: Callable[[], None]) -> None:
        self._operation: Callable[[KVTransferBatch, int | None], tuple[LoadCompletion, ...]] | None = None
        self._completed_lock = threading.Lock()
        self._completed: list[LoadCompletion] = []
        self._executor = TimelineExecutor("KVPoolLoadExecutor", thread_initializer, self._execute)

    def bind_operation(
        self,
        operation: Callable[[KVTransferBatch, int | None], tuple[LoadCompletion, ...]],
    ) -> None:
        if self._operation is not None:
            raise RuntimeError("Load timeline operation is already bound")
        self._operation = operation

    def start(self) -> None:
        if self._operation is None:
            raise RuntimeError("Load timeline has no bound operation")
        self._executor.start()
        self._raise_if_failed()

    def submit(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        self._raise_if_not_running()
        self._executor.submit(batch)
        return ()

    def collect(self) -> list[LoadCompletion]:
        self._raise_if_failed()
        with self._completed_lock:
            completed = self._completed
            self._completed = []
        return completed

    def abort(self) -> None:
        return

    def close(self) -> None:
        self._executor.close()
        self._raise_if_failed()

    def _execute(self, batch: KVTransferBatch) -> None:
        if self._operation is None:
            raise RuntimeError("Asynchronous Load timeline has no operation")
        completions = self._operation(batch, None)
        with self._completed_lock:
            self._completed.extend(completions)

    def _raise_if_failed(self) -> None:
        if self._executor.failure is None:
            return
        batches = (self._executor.failed_command, *self._executor.discarded_commands)
        request_ids = sorted(request_id for batch in batches if batch is not None for request_id in batch.request_ids)
        requests = f"; unfinished requests: {request_ids}" if request_ids else ""
        raise RuntimeError(
            f"KVPoolLoadExecutor terminated during asynchronous Load{requests}"
        ) from self._executor.failure

    def _raise_if_not_running(self) -> None:
        self._raise_if_failed()
        self._executor.check_running()


@dataclass(frozen=True, slots=True)
class _StoreSubmission:
    batch: StoreBatch
    source_ready_event: Any


class StoreTimeline:
    """Preserve FIFO whole-step Store submission and next-step fencing."""

    def __init__(self, thread_initializer: Callable[[], None]) -> None:
        self._operation: Callable[[KVTransferBatch, Any, int | None], tuple[StoreCompletion, ...]] | None = None
        self._executor = TimelineExecutor("KVPoolStoreExecutor", thread_initializer, self._execute, self._complete)

    def bind_operation(
        self,
        operation: Callable[[KVTransferBatch, Any, int | None], tuple[StoreCompletion, ...]],
    ) -> None:
        if self._operation is not None:
            raise RuntimeError("Store timeline operation is already bound")
        self._operation = operation

    def start(self) -> None:
        if self._operation is None:
            raise RuntimeError("Store timeline has no bound operation")
        self._executor.start()
        self._raise_if_failed()

    def submit(self, transfer: KVTransferBatch, source_ready_event: Any) -> StoreBatch:
        self._raise_if_not_running()
        batch = StoreBatch(transfer)
        self._executor.submit(_StoreSubmission(batch, source_ready_event))
        return batch

    def wait(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]:
        while True:
            self._raise_if_failed()
            if batch.completed.wait(timeout=_STORE_FENCE_POLL_INTERVAL_S):
                break
        self._raise_if_failed()
        return tuple(batch.completions)

    def close(self) -> None:
        self._executor.close()
        self._raise_if_failed()

    def _execute(self, submission: _StoreSubmission) -> None:
        if self._operation is None:
            raise RuntimeError("Store timeline has no operation")
        submission.batch.completions.extend(
            self._operation(submission.batch.transfer, submission.source_ready_event, None)
        )

    @staticmethod
    def _complete(submission: _StoreSubmission) -> None:
        submission.batch.completed.set()

    def _raise_if_failed(self) -> None:
        if self._executor.failure is not None:
            raise RuntimeError("KVPoolStoreExecutor failed during asynchronous Store") from self._executor.failure

    def _raise_if_not_running(self) -> None:
        self._raise_if_failed()
        self._executor.check_running()
