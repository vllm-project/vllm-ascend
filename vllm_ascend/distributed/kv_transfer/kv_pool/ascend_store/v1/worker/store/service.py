"""Business entry point for classic asynchronous Store."""

from __future__ import annotations

from typing import cast

import torch
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import Backend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ChunkedTokenDatabase

from ...metadata import StoreRequest
from .executor import StoreExecutor
from .task import StoreTask, StoreTaskBuilder


class StoreService:
    """Turn Store requests into tasks and submit them for asynchronous execution."""

    def __init__(
        self,
        backend: Backend,
        token_database: ChunkedTokenDatabase,
        block_size: int,
        tp_rank: int,
        pcp_rank: int,
        pcp_size: int,
        dcp_size: int,
        put_step: int,
        kv_role: str,
    ) -> None:
        self._backend = backend
        self._task_builder = StoreTaskBuilder(
            token_database,
            block_size,
            tp_rank,
            pcp_rank,
            pcp_size,
            dcp_size,
            put_step,
            kv_role,
        )
        self._executor: StoreExecutor | None = None

    def start(self) -> None:
        """Start Store execution after the Worker has registered its KV buffers."""
        if self._executor is not None:
            return
        self._executor = StoreExecutor(self._backend)
        self._executor.start_and_wait_ready()

    def submit(self, requests: list[StoreRequest]) -> None:
        if not requests:
            return
        source_ready_event = torch.npu.Event()
        source_ready_event.record()
        tasks = []
        for request in requests:
            try:
                tasks.append(self._task_builder.build(request, source_ready_event))
            except Exception:
                logger.exception("Failed to prepare Store task for request %s", request.request_id)
                tasks.append(StoreTask(request.request_id, source_ready_event, ()))
        executor = cast(StoreExecutor, self._executor)
        executor.submit_batch(tasks)

    def wait_for_previous_store(self) -> None:
        executor = self._executor
        if executor is not None:
            executor.wait_for_previous_store()

    def discard_preempted_and_finished_requests(self, preempted_request_ids: set[str]) -> None:
        executor = self._executor
        if executor is not None:
            executor.discard_preempted_and_finished_requests(preempted_request_ids)
