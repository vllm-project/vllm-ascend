"""Business entry point for asynchronous Worker Store."""

from __future__ import annotations

import torch

from ...protocol.transfer import StoreRequestBatch
from ..projection import KVObjectProjector
from ..region import KVRegionOperator
from .executor import StoreExecutionResult, StoreExecutor
from .task import StoreTaskBuilder


class StoreService:
    """Turn Store requests into tasks and submit them for asynchronous execution."""

    def __init__(
        self,
        region_operator: KVRegionOperator,
        object_projector: KVObjectProjector,
        task_builder: StoreTaskBuilder,
        executor: StoreExecutor,
    ) -> None:
        self._region_operator = region_operator
        self._object_projector = object_projector
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
        tasks = [
            self._task_builder.build(
                request,
                source_ready_event,
                self._object_projector.project(
                    self._region_operator.store_region(request.store_range, request.num_prompt_tokens),
                    request.block_hashes,
                ),
            )
            for request in request_batch.requests
        ]
        self._executor.submit_batch(tasks)

    def wait_for_previous_store(self) -> tuple[StoreExecutionResult, ...]:
        return self._executor.wait_for_previous_store()
