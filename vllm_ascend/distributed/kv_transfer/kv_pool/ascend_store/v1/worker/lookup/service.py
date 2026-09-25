"""Business entry point for classic Worker Lookup."""

from __future__ import annotations

from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import BlockHash

from .executor import LookupExecutionResult, LookupExecutor
from .task import LookupTask, LookupTaskBuilder


class LookupService:
    """Return the continuous aligned prefix present on all KV partitions."""

    def __init__(
        self,
        task_builder: LookupTaskBuilder,
        executor: LookupExecutor,
        max_model_len: int,
        cache_transfer_granularity: int,
    ) -> None:
        self._task_builder = task_builder
        self._executor = executor
        self._max_model_len = max_model_len
        self._cache_transfer_granularity = cache_transfer_granularity

    def lookup(self, token_len: int, block_hashes: list[BlockHash] | list[str]) -> int:
        try:
            task = self._task_builder.build(token_len, block_hashes)
            if not task.backend_keys:
                return 0
            result = self._executor.execute(task)
            return self._continuous_hit_end(task, result)
        except Exception as error:
            logger.error("Remote connection failed in lookup. type=%s, error=%s", type(error).__name__, error)
            return 0

    def _continuous_hit_end(self, task: LookupTask, result: LookupExecutionResult) -> int:
        hit_end = 0
        for index, end in enumerate(task.chunk_ends):
            if end > self._max_model_len:
                break
            if not all(result.presence[rank * len(task.chunk_ends) + index] == 1 for rank in range(task.num_ranks)):
                break
            if end % self._cache_transfer_granularity == 0:
                hit_end = end
        return hit_end
