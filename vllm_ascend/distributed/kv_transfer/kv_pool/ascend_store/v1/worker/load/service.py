"""Business entry point for classic synchronous Load."""

from __future__ import annotations

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...metadata import LoadRequest
from .executor import LoadExecutionResult, LoadExecutor
from .task import LoadTask, LoadTaskBuilder


class LoadService:
    """Load eligible requests synchronously and retain block failures until consumed."""

    def __init__(
        self,
        backend: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        cache_transfer_granularity: int,
        tp_rank: int,
    ) -> None:
        self._task_builder = LoadTaskBuilder(
            token_database,
            block_size,
            cache_transfer_granularity,
            tp_rank,
        )
        self._executor = LoadExecutor(backend)
        self._failed_block_ids: set[int] = set()

    def load(self, requests: list[LoadRequest]) -> None:
        for request in requests:
            task = self._task_builder.build(request)
            result = self._executor.execute(task)
            self._failed_block_ids.update(self._find_failed_block_ids(task, result))

    @staticmethod
    def _find_failed_block_ids(task: LoadTask, result: LoadExecutionResult) -> set[int]:
        if result.result_codes is None:
            return {chunk.block_id for chunk in task.chunks}
        return {chunk.block_id for chunk, code in zip(task.chunks, result.result_codes) if code != 0}

    def take_failed_block_ids(self) -> set[int]:
        failed_block_ids = self._failed_block_ids.copy()
        self._failed_block_ids.clear()
        return failed_block_ids
