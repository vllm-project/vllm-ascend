"""Business entry point for Worker Lookup."""

from __future__ import annotations

from vllm.logger import logger

from ...protocol.lookup import LookupRequest, LookupResult
from ..projection import KVObjectProjection, KVObjectProjector
from ..region import KVRegionOperator, LookupObservation
from .executor import LookupExecutionResult, LookupExecutor
from .task import LookupTask, LookupTaskBuilder


class LookupService:
    """Expose request-level Lookup while containing Backend failures."""

    def __init__(
        self,
        region_operator: KVRegionOperator,
        object_projector: KVObjectProjector,
        task_builder: LookupTaskBuilder,
        executor: LookupExecutor,
    ) -> None:
        self._region_operator = region_operator
        self._object_projector = object_projector
        self._task_builder = task_builder
        self._executor = executor

    def lookup(self, request: LookupRequest) -> LookupResult:
        try:
            if request.transfer_group_ids != self._region_operator.group_ids:
                raise ValueError(
                    f"Lookup groups {request.transfer_group_ids} do not match configured groups "
                    f"{self._region_operator.group_ids}"
                )
            query_region = self._region_operator.lookup_region(request.query_range)
            observations = tuple(
                self._execute_projection(projection)
                for projection in self._object_projector.project(query_region, request.block_hashes)
            )
            return LookupResult(
                self._region_operator.resolve_lookup(
                    request.block_hashes,
                    query_region,
                    observations,
                )
            )
        except Exception as error:
            logger.error("Remote connection failed in lookup. type=%s, error=%s", type(error).__name__, error)
            return LookupResult(0)

    def _execute_projection(self, projection: KVObjectProjection) -> LookupObservation:
        task = self._task_builder.build(projection)
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
