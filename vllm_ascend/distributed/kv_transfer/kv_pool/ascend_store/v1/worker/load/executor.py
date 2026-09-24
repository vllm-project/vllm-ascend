"""Synchronous execution of fully resolved classic Load tasks."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

from .task import LoadTask


@dataclass(frozen=True, slots=True)
class LoadExecutionResult:
    """Raw Backend result codes aligned with one Load task's chunks."""

    result_codes: tuple[int, ...] | None


class LoadExecutor:
    """Execute one Load task synchronously without interpreting block validity."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    def execute(self, task: LoadTask) -> LoadExecutionResult:
        if not task.chunks:
            return LoadExecutionResult(())

        result_codes = self._backend.get(
            [chunk.backend_key for chunk in task.chunks],
            [list(chunk.addresses) for chunk in task.chunks],
            [list(chunk.sizes) for chunk in task.chunks],
        )
        return LoadExecutionResult(None if result_codes is None else tuple(result_codes))
