"""Business entry point for classic Load."""

from __future__ import annotations

from collections.abc import Iterable

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...metadata import LoadRequest
from .async_executor import AsyncLoadExecutor
from .executor import LoadCompletion, LoadExecutionResult, LoadExecutor
from .task import LoadTask, LoadTaskBuilder


class LoadService:
    """Build Load tasks, select their execution mode and retain block failures."""

    def __init__(
        self,
        backend: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        cache_transfer_granularity: int,
        tp_rank: int,
        load_async: bool = False,
    ) -> None:
        self._task_builder = LoadTaskBuilder(
            token_database,
            block_size,
            cache_transfer_granularity,
            tp_rank,
        )
        self._executor: LoadExecutor | AsyncLoadExecutor
        self._executor = AsyncLoadExecutor(backend) if load_async else LoadExecutor(backend)
        self._failed_block_ids: set[int] = set()
        self._pending_request_ids: set[str] = set()
        self._finished_requests_awaiting_load: set[str] = set()

    def start(self) -> None:
        """Start asynchronous Load execution after KV buffers are registered."""
        self._executor.start_and_wait_ready()

    def close(self) -> None:
        self._executor.close()

    def load(self, requests: list[LoadRequest]) -> None:
        tasks = [self._task_builder.build(request) for request in requests]
        self._pending_request_ids.update(task.request_id for task in tasks)
        self._consume_completions(self._executor.submit(tasks))

    def take_finished_request_ids(
        self,
        loading_request_ids: set[str],
        finished_request_ids: set[str],
    ) -> set[str]:
        # vLLM keeps a finished request's blocks alive until its asynchronous
        # Load is reported, so retain the one-shot finish notification.
        self._finished_requests_awaiting_load.update(finished_request_ids & self._pending_request_ids)
        return self._consume_completions(
            self._executor.take_completed(loading_request_ids | self._finished_requests_awaiting_load)
        )

    def _consume_completions(self, completions: Iterable[LoadCompletion]) -> set[str]:
        completed_request_ids = set()
        for task, result in completions:
            self._failed_block_ids.update(self._find_failed_block_ids(task, result))
            completed_request_ids.add(result.request_id)
        self._pending_request_ids.difference_update(completed_request_ids)
        self._finished_requests_awaiting_load.difference_update(completed_request_ids)
        return completed_request_ids

    @staticmethod
    def _find_failed_block_ids(task: LoadTask, result: LoadExecutionResult) -> set[int]:
        if result.result_codes is None:
            return {chunk.block_id for chunk in task.chunks}
        return {chunk.block_id for chunk, code in zip(task.chunks, result.result_codes) if code != 0}

    def take_failed_block_ids(self) -> set[int]:
        failed_block_ids = self._failed_block_ids.copy()
        self._failed_block_ids.clear()
        return failed_block_ids
