"""Business entry point for classic asynchronous Store."""

from __future__ import annotations

import torch
from vllm.logger import logger

from ...metadata import StoreRequestBatch
from .executor import StoreExecutor
from .task import StoreTask, StoreTaskBuilder


class StoreService:
    """Turn Store requests into tasks and submit them for asynchronous execution."""

    def __init__(
        self,
        task_builder: StoreTaskBuilder,
        executor: StoreExecutor,
    ) -> None:
        self._task_builder = task_builder
        self._executor = executor

    def start(self) -> None:
        """Start Store execution after the Worker has registered its KV buffers."""
        self._executor.start_and_wait_ready()

    def close(self) -> None:
        self._executor.close()

    def submit(self, request_batch: StoreRequestBatch) -> None:
        if not request_batch.requests:
            return
        source_ready_event = torch.npu.Event()
        source_ready_event.record()
        tasks = []
        for request in request_batch.requests:
            try:
                tasks.append(self._task_builder.build(request, source_ready_event))
            except Exception:
                logger.exception("Failed to prepare Store task for request %s", request.request_id)
                tasks.append(StoreTask(request.request_id, source_ready_event, ()))
        self._executor.submit_batch(tasks)

    def wait_for_previous_store(self) -> None:
        self._executor.wait_for_previous_store()

    def finish_step(self, request_batch: StoreRequestBatch) -> None:
        self._executor.finish_step(request_batch.preempted_request_ids)
