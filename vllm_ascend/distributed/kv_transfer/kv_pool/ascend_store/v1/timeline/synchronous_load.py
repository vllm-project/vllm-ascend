"""Execute Load in the submitting thread."""

from __future__ import annotations

from collections.abc import Callable

from ..worker.transfer.batch import KVTransferBatch
from ..worker.transfer.evidence import LoadCompletion

LoadOperation = Callable[[KVTransferBatch, int | None], tuple[LoadCompletion, ...]]


class SynchronousLoadTimeline:
    """Execute and publish Load completions in the submitting call."""

    collects_completions = False

    def __init__(self, operation: LoadOperation) -> None:
        self._operation = operation

    def start(self) -> None:
        return

    def submit(self, batch: KVTransferBatch) -> tuple[LoadCompletion, ...]:
        return self._operation(batch, None)

    def collect(self) -> tuple[LoadCompletion, ...]:
        return ()

    def abort(self) -> None:
        return

    @property
    def stopped(self) -> bool:
        return True

    def close(self) -> None:
        return
