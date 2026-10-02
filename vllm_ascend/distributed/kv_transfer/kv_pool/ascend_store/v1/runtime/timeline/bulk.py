"""Execute non-layerwise KV Pool transfers."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import LoadCompletion, LoadTransfer, StoreBatch, StoreCompletion, StoreTransfer
from .executor import TimelineExecutor

_STORE_FENCE_POLL_INTERVAL_S = 1.0


class LoadTimeline:
    """Execute and publish Load completions in the submitting call."""

    collects_completions = False

    def __init__(self) -> None:
        self._operation: Callable[[LoadTransfer], LoadCompletion] | None = None

    def bind_operation(self, operation: Callable[[LoadTransfer], LoadCompletion]) -> None:
        if self._operation is not None:
            raise RuntimeError("Load timeline operation is already bound")
        self._operation = operation

    def start(self) -> None:
        if self._operation is None:
            raise RuntimeError("Load timeline has no bound operation")

    def submit(self, transfers: list[LoadTransfer]) -> tuple[LoadCompletion, ...]:
        if self._operation is None:
            raise RuntimeError("Load timeline has no bound operation")
        return tuple(self._operation(transfer) for transfer in transfers)

    def collect(self) -> tuple[LoadCompletion, ...]:
        return ()

    def abort(self) -> None:
        return

    def close(self) -> None:
        return


class AsyncLoadTimeline:
    """Execute fixed Load work on one background timeline."""

    collects_completions = True

    def __init__(self, thread_initializer: Callable[[], None]) -> None:
        self._operation: Callable[[LoadTransfer], LoadCompletion] | None = None
        self._completed_lock = threading.Lock()
        self._completed: list[LoadCompletion] = []
        self._executor = TimelineExecutor("KVPoolLoadExecutor", thread_initializer, self._execute)

    def bind_operation(self, operation: Callable[[LoadTransfer], LoadCompletion]) -> None:
        if self._operation is not None:
            raise RuntimeError("Load timeline operation is already bound")
        self._operation = operation

    def start(self) -> None:
        if self._operation is None:
            raise RuntimeError("Load timeline has no bound operation")
        self._executor.start()
        self._raise_if_failed()

    def submit(self, transfers: list[LoadTransfer]) -> tuple[LoadCompletion, ...]:
        self._raise_if_not_running()
        for transfer in transfers:
            self._executor.submit(transfer)
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

    def _execute(self, transfer: LoadTransfer) -> None:
        if self._operation is None:
            raise RuntimeError("Asynchronous Load timeline has no operation")
        completion = self._operation(transfer)
        with self._completed_lock:
            self._completed.append(completion)

    def _raise_if_failed(self) -> None:
        if self._executor.failure is None:
            return
        commands = (self._executor.failed_command, *self._executor.discarded_commands)
        request_ids = sorted(command.request_id for command in commands if command is not None)
        requests = f"; unfinished requests: {request_ids}" if request_ids else ""
        error = RuntimeError(f"KVPoolLoadExecutor terminated during asynchronous Load{requests}")
        raise error from self._executor.failure

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
        self._operation: Callable[[StoreTransfer, Any], StoreCompletion] | None = None
        self._executor = TimelineExecutor("KVPoolStoreExecutor", thread_initializer, self._execute, self._complete)

    def bind_operation(self, operation: Callable[[StoreTransfer, Any], StoreCompletion]) -> None:
        if self._operation is not None:
            raise RuntimeError("Store timeline operation is already bound")
        self._operation = operation

    def start(self) -> None:
        if self._operation is None:
            raise RuntimeError("Store timeline has no bound operation")
        self._executor.start()
        self._raise_if_failed()

    def submit(self, transfers: list[StoreTransfer], source_ready_event: Any) -> StoreBatch:
        self._raise_if_not_running()
        batch = StoreBatch(tuple(transfers))
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
        for transfer in submission.batch.transfers:
            submission.batch.completions.append(self._operation(transfer, submission.source_ready_event))

    @staticmethod
    def _complete(submission: _StoreSubmission) -> None:
        submission.batch.completed.set()

    def _raise_if_failed(self) -> None:
        if self._executor.failure is not None:
            raise RuntimeError("KVPoolStoreExecutor failed during asynchronous Store") from self._executor.failure

    def _raise_if_not_running(self) -> None:
        self._raise_if_failed()
        self._executor.check_running()
