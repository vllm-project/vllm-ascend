"""Execute whole-batch Store on one background timeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..protocol.transfer import StoreCommand
from ..worker.transfer.evidence import StoreCompletion
from .executor import TimelineExecutor
from .store_batch import StoreBatch

StoreOperation = Callable[[tuple[StoreCommand, ...], Any], tuple[StoreCompletion, ...]]
_STORE_FENCE_POLL_INTERVAL_S = 1.0


@dataclass(frozen=True, slots=True)
class _StoreSubmission:
    batch: StoreBatch
    commands: tuple[StoreCommand, ...]
    source_ready_event: Any


class AsynchronousStoreTimeline:
    """Preserve FIFO whole-step Store submission and explicit completion fences."""

    def __init__(self, thread_initializer: Callable[[], None], operation: StoreOperation) -> None:
        self._operation = operation
        self._executor = TimelineExecutor("KVPoolStoreExecutor", thread_initializer, self._execute, self._complete)

    def start(self) -> None:
        self._executor.start()
        self._raise_if_failed()

    def submit(self, commands: tuple[StoreCommand, ...], source_ready_event: Any) -> StoreBatch:
        self._raise_if_not_running()
        batch = StoreBatch(
            request_ids=frozenset(command.request_id for command in commands),
            store_job_ids=frozenset(command.store_job_id for command in commands if command.store_job_id is not None),
        )
        self._executor.submit(_StoreSubmission(batch, commands, source_ready_event))
        return batch

    def wait(self, batch: StoreBatch) -> tuple[StoreCompletion, ...]:
        while True:
            self._raise_if_failed()
            if batch.completed.wait(timeout=_STORE_FENCE_POLL_INTERVAL_S):
                break
        self._raise_if_failed()
        return tuple(batch.completions)

    def collect(self, batch: StoreBatch | None) -> tuple[StoreCompletion, ...] | None:
        """Observe readiness without waiting, including fatal executor failures before any completion."""
        self._raise_if_failed()
        if batch is None or not batch.completed.is_set():
            return None
        return self.wait(batch)

    @property
    def stopped(self) -> bool:
        return self._executor.stopped

    def close(self) -> None:
        self._executor.close()
        self._raise_if_failed()

    def _execute(self, submission: _StoreSubmission) -> None:
        submission.batch.completions.extend(self._operation(submission.commands, submission.source_ready_event))

    @staticmethod
    def _complete(submission: _StoreSubmission) -> None:
        submission.batch.completed.set()

    def _raise_if_failed(self) -> None:
        if self._executor.failure is not None:
            raise RuntimeError("KVPoolStoreExecutor failed during asynchronous Store") from self._executor.failure

    def _raise_if_not_running(self) -> None:
        self._raise_if_failed()
        self._executor.check_running()
