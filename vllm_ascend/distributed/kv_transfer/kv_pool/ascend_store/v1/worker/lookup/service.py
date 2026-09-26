"""Business entry point for Worker Lookup."""

from __future__ import annotations

from vllm.logger import logger

from ..coordinator import KVTransferCoordinator, LookupChunkSelection, LookupObservation
from .executor import LookupExecutionResult, LookupExecutor
from .request import WorkerLookupRequest
from .task import LookupTask, LookupTaskBuilder


class LookupService:
    """Expose request-level Lookup while containing Backend failures."""

    def __init__(
        self,
        coordinator: KVTransferCoordinator,
        task_builder: LookupTaskBuilder,
        executor: LookupExecutor,
    ) -> None:
        self._coordinator = coordinator
        self._task_builder = task_builder
        self._executor = executor

    def lookup(self, request: WorkerLookupRequest) -> int:
        try:
            if request.transfer_group_ids != self._coordinator.group_ids:
                raise ValueError(
                    f"Lookup groups {request.transfer_group_ids} do not match configured groups "
                    f"{self._coordinator.group_ids}"
                )
            observations = tuple(
                self._execute_selection(request, selection)
                for selection in self._coordinator.select_lookup(request.lookup_end_token, request.local_cached_tokens)
            )
            return self._coordinator.resolve_lookup(
                request.block_hashes,
                request.lookup_end_token,
                request.local_cached_tokens,
                observations,
            )
        except Exception as error:
            logger.error("Remote connection failed in lookup. type=%s, error=%s", type(error).__name__, error)
            return 0

    def _execute_selection(self, request: WorkerLookupRequest, selection: LookupChunkSelection) -> LookupObservation:
        task = self._task_builder.build(request, selection)
        if not task.backend_keys:
            return LookupObservation(task.group_id, task.chunk_ends, task.chunk_hashes, ())
        result = self._executor.execute(task)
        return LookupObservation(
            task.group_id,
            task.chunk_ends,
            task.chunk_hashes,
            self._chunk_presence(task, result),
        )

    @staticmethod
    def _chunk_presence(task: LookupTask, result: LookupExecutionResult) -> tuple[bool, ...]:
        num_chunks = len(task.chunk_ends)
        actual_result_count = len(result.presence_codes)
        expected_result_count = task.num_ranks * num_chunks
        if actual_result_count != expected_result_count:
            raise ValueError(f"Lookup returned {actual_result_count} results for {expected_result_count} Backend keys")
        return tuple(
            all(result.presence_codes[rank * num_chunks + index] == 1 for rank in range(task.num_ranks))
            for index in range(num_chunks)
        )
