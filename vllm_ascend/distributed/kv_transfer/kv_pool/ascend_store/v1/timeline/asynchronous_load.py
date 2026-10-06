"""Execute whole-batch Load on one background timeline."""

from __future__ import annotations

import threading
from collections.abc import Callable

from ..worker.transfer.batch import KVTransferBatch
from ..worker.transfer.evidence import LoadCompletion
from .executor import TimelineExecutor

LoadOperation = Callable[[KVTransferBatch, int | None], tuple[LoadCompletion, ...]]


class AsynchronousLoadTimeline:
    """Execute each submitted request batch on one background timeline."""

    collects_completions = True

    def __init__(self, thread_initializer: Callable[[], None], operation: LoadOperation) -> None:
        self._operation = operation
        self._completed_lock = threading.Lock()
        self._completed: list[LoadCompletion] = []
        self._executor = TimelineExecutor("KVPoolLoadExecutor", thread_initializer, self._execute)

    def start(self) -> None:
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

    @property
    def stopped(self) -> bool:
        return self._executor.stopped

    def close(self) -> None:
        self._executor.close()
        self._raise_if_failed()

    def _execute(self, batch: KVTransferBatch) -> None:
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
