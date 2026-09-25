"""Business entry point for classic Load."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...metadata import LoadRequestBatch
from .executor import LoadExecution, LoadExecutionResult, LoadTaskResult
from .task import LoadTask, LoadTaskBuilder


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Completed asynchronous requests and failed blocks collected together."""

    completed_request_ids: frozenset[str]
    failed_block_ids: frozenset[int]


class LoadService:
    """Build Load tasks, submit them to one Executor and retain block failures."""

    def __init__(
        self,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        cache_transfer_granularity: int,
        tp_rank: int,
        executor: LoadExecution,
    ) -> None:
        self._task_builder = LoadTaskBuilder(
            token_database,
            block_size,
            cache_transfer_granularity,
            tp_rank,
        )
        self._executor = executor
        self._failed_block_ids: set[int] = set()

    def start(self) -> None:
        """Prepare Load execution after KV buffers are registered."""
        self._executor.start_and_wait_ready()

    def close(self) -> None:
        self._executor.close()

    def load(self, request_batch: LoadRequestBatch) -> None:
        tasks = [self._task_builder.build(request) for request in request_batch.requests]
        self._record_failed_blocks(self._executor.submit(tasks))

    def collect_result(self) -> LoadResult:
        task_results = tuple(self._executor.collect())
        self._record_failed_blocks(task_results)
        load_result = LoadResult(
            frozenset(execution_result.request_id for _, execution_result in task_results),
            frozenset(self._failed_block_ids),
        )
        self._failed_block_ids.clear()
        return load_result

    def _record_failed_blocks(self, task_results: Iterable[LoadTaskResult]) -> None:
        for task, result in task_results:
            self._failed_block_ids.update(self._find_failed_block_ids(task, result))

    @staticmethod
    def _find_failed_block_ids(task: LoadTask, result: LoadExecutionResult) -> set[int]:
        if result.result_codes is None:
            return {chunk.block_id for chunk in task.chunks}
        return {chunk.block_id for chunk, code in zip(task.chunks, result.result_codes) if code != 0}
