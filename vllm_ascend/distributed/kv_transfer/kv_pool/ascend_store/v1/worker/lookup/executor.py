"""Execution of fully resolved classic Lookup tasks."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend

from .task import LookupTask


@dataclass(frozen=True, slots=True)
class LookupExecutionResult:
    """Raw Backend presence codes aligned with a Lookup task's keys."""

    presence: tuple[int, ...]


class LookupExecutor:
    """Execute one Lookup task without interpreting prefix availability."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    def execute(self, task: LookupTask) -> LookupExecutionResult:
        return LookupExecutionResult(tuple(self._backend.exists(list(task.backend_keys))))
