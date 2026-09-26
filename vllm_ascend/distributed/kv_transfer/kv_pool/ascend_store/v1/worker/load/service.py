"""Business entry point for Worker Load."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ...protocol.transfer import LoadRequestBatch
from ..coordinator import KVTransferCoordinator
from .executor import LoadExecutionResult, LoadExecutor, LoadTaskCompletion
from .task import LoadTask, LoadTaskBuilder, resolve_load_end_token


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Terminal Worker Load facts consumed through the Connector's split hooks."""

    completed_request_ids: frozenset[str]
    failed_request_ids: frozenset[str]
    failed_block_ids: frozenset[int]


class LoadService:
    """Build Load tasks, submit them to one Executor and retain failures."""

    def __init__(
        self,
        coordinator: KVTransferCoordinator,
        task_builder: LoadTaskBuilder,
        executor: LoadExecutor,
        cache_transfer_granularity: int,
        uses_group_scoped_block_ids: bool,
    ) -> None:
        self._coordinator = coordinator
        self._task_builder = task_builder
        self._executor = executor
        self._cache_transfer_granularity = cache_transfer_granularity
        self._uses_group_scoped_block_ids = uses_group_scoped_block_ids
        self._failed_request_ids: set[str] = set()
        self._failed_block_ids: set[int] = set()

    def start(self) -> None:
        """Prepare Load execution after KV buffers are registered."""
        self._executor.start_and_wait_ready()

    def close(self) -> None:
        self._executor.close()

    def load(self, request_batch: LoadRequestBatch) -> None:
        tasks = []
        for request in request_batch.requests:
            load_end_token = resolve_load_end_token(request, self._cache_transfer_granularity)
            selections = self._coordinator.select_load(request.block_hashes, load_end_token)
            tasks.append(self._task_builder.build(request, load_end_token, selections))
        task_completions = tuple(self._executor.submit(tasks))
        self._record_failures(task_completions)
        if task_completions and self._failed_request_ids:
            raise RuntimeError(f"Hybrid KV Load failed for requests: {sorted(self._failed_request_ids)}")

    def collect_result(self) -> LoadResult:
        task_completions = tuple(self._executor.collect())
        self._record_failures(task_completions)
        load_result = LoadResult(
            frozenset(execution_result.request_id for _, execution_result in task_completions),
            frozenset(self._failed_request_ids),
            frozenset(self._failed_block_ids),
        )
        self._failed_request_ids.clear()
        self._failed_block_ids.clear()
        return load_result

    def _record_failures(self, task_completions: Iterable[LoadTaskCompletion]) -> None:
        for task, result in task_completions:
            failed_block_ids = self._find_failed_block_ids(task, result)
            if not failed_block_ids:
                continue
            if self._uses_group_scoped_block_ids:
                self._failed_request_ids.add(task.request_id)
            else:
                self._failed_block_ids.update(failed_block_ids)

    @staticmethod
    def _find_failed_block_ids(task: LoadTask, result: LoadExecutionResult) -> set[int]:
        if result.result_codes is None or len(result.result_codes) != len(task.chunks):
            return {chunk.block_id for chunk in task.chunks}
        return {chunk.block_id for chunk, code in zip(task.chunks, result.result_codes) if code != 0}
