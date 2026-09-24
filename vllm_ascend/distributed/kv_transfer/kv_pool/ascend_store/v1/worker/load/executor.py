"""Synchronous execution of fully resolved classic Load tasks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

from .task import LoadTask


@dataclass(frozen=True, slots=True)
class LoadExecutionResult:
    """Raw Backend result codes aligned with one Load task's chunks."""

    request_id: str
    result_codes: tuple[int, ...] | None


LoadCompletion = tuple[LoadTask, LoadExecutionResult]


class LoadExecutor:
    """Execute Load tasks synchronously without interpreting block validity."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    def start_and_wait_ready(self) -> None:
        return

    def close(self) -> None:
        return

    def submit(self, tasks: list[LoadTask]) -> Iterable[LoadCompletion]:
        for task in tasks:
            yield task, self.execute(task)

    def take_completed(self, request_ids: set[str]) -> list[LoadCompletion]:
        return []

    def execute(self, task: LoadTask) -> LoadExecutionResult:
        if not task.chunks:
            return LoadExecutionResult(task.request_id, ())

        result_codes = self._backend.get(
            [chunk.backend_key for chunk in task.chunks],
            [list(chunk.addresses) for chunk in task.chunks],
            [list(chunk.sizes) for chunk in task.chunks],
        )
        return LoadExecutionResult(task.request_id, None if result_codes is None else tuple(result_codes))
