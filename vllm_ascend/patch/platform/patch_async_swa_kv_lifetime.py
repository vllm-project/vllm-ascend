# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextvars import ContextVar
from functools import wraps
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

_prune_context: ContextVar[tuple[str, int] | None] = ContextVar("ascend_swa_prune_context", default=None)


def _handle_negative_in_flight(
    request_id: str,
    num_in_flight_tokens: int,
    where: str,
    num_scheduled_tokens: int | None = None,
) -> None:
    msg = (
        "SWA_BLOCK_DIAG negative_in_flight_tokens "
        f"where={where} request_id={request_id} "
        f"num_in_flight_tokens={num_in_flight_tokens}"
    )
    if num_scheduled_tokens is not None:
        msg += f" num_scheduled_tokens={num_scheduled_tokens}"
    logger.warning(msg)


def _safe_in_flight_tokens(request: Request, where: str) -> int:
    num_in_flight_tokens = getattr(request, "num_in_flight_tokens", 0)
    if num_in_flight_tokens < 0:
        _handle_negative_in_flight(
            request.request_id,
            num_in_flight_tokens,
            where,
        )
        request.num_in_flight_tokens = 0
        return 0
    return num_in_flight_tokens


def _max_in_flight_tokens(vllm_config: VllmConfig) -> int:
    return vllm_config.max_concurrent_batches * vllm_config.scheduler_config.max_num_batched_tokens


_original_request_init = Request.__init__


@wraps(_original_request_init)
def _patched_request_init(self: Request, *args: Any, **kwargs: Any) -> None:
    _original_request_init(self, *args, **kwargs)
    self.num_in_flight_tokens = 0


_original_update_after_schedule = Scheduler._update_after_schedule


@wraps(_original_update_after_schedule)
def _patched_update_after_schedule(self: Scheduler, scheduler_output: SchedulerOutput) -> None:
    _original_update_after_schedule(self, scheduler_output)
    for request_id, num_scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
        self.requests[request_id].num_in_flight_tokens += num_scheduled_tokens


_original_update_from_output = Scheduler.update_from_output


@wraps(_original_update_from_output)
def _patched_update_from_output(
    self: Scheduler,
    scheduler_output: SchedulerOutput,
    model_runner_output: Any,
) -> Any:
    for request_id, num_scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
        if request := self.requests.get(request_id):
            request.num_in_flight_tokens -= num_scheduled_tokens
            if request.num_in_flight_tokens < 0:
                _handle_negative_in_flight(
                    request_id,
                    request.num_in_flight_tokens,
                    "update_from_output",
                    num_scheduled_tokens,
                )
                request.num_in_flight_tokens = 0
    return _original_update_from_output(self, scheduler_output, model_runner_output)


_original_allocate_slots = KVCacheManager.allocate_slots


@wraps(_original_allocate_slots)
def _patched_allocate_slots(self: KVCacheManager, request: Request, *args: Any, **kwargs: Any) -> Any:
    token = _prune_context.set((request.request_id, _safe_in_flight_tokens(request, "allocate_slots")))
    try:
        return _original_allocate_slots(self, request, *args, **kwargs)
    finally:
        _prune_context.reset(token)


_original_connector_finished = Scheduler._connector_finished


@wraps(_original_connector_finished)
def _patched_connector_finished(self: Scheduler, request: Request) -> tuple[bool, dict[str, Any] | None]:
    token = _prune_context.set((request.request_id, _safe_in_flight_tokens(request, "connector_finished")))
    try:
        return _original_connector_finished(self, request)
    finally:
        _prune_context.reset(token)


_original_remove_skipped_blocks = SingleTypeKVCacheManager.remove_skipped_blocks


@wraps(_original_remove_skipped_blocks)
def _patched_remove_skipped_blocks(
    self: SingleTypeKVCacheManager,
    request_id: str,
    total_computed_tokens: int,
) -> None:
    context = _prune_context.get()
    if (
        context is not None
        and context[0] == request_id
        and isinstance(self.kv_cache_spec, (ChunkedLocalAttentionSpec, SlidingWindowSpec))
    ):
        num_in_flight_tokens = context[1]
        if num_in_flight_tokens < 0:
            _handle_negative_in_flight(
                request_id,
                num_in_flight_tokens,
                "remove_skipped_blocks",
            )
            num_in_flight_tokens = 0
        total_computed_tokens = max(0, total_computed_tokens - num_in_flight_tokens)
    _original_remove_skipped_blocks(self, request_id, total_computed_tokens)


def _patched_chunked_local_max_memory_usage_bytes(self: ChunkedLocalAttentionSpec, vllm_config: VllmConfig) -> int:
    max_blocks = self.max_admission_blocks_per_request(
        max_num_batched_tokens=_max_in_flight_tokens(vllm_config),
        max_model_len=vllm_config.model_config.max_model_len,
    )
    return max_blocks * self.page_size_bytes


def _patched_swa_max_memory_usage_bytes(self: SlidingWindowSpec, vllm_config: VllmConfig) -> int:
    assert vllm_config.parallel_config.decode_context_parallel_size == 1, "DCP not support sliding window."
    max_blocks = self.max_admission_blocks_per_request(
        max_num_batched_tokens=_max_in_flight_tokens(vllm_config),
        max_model_len=vllm_config.model_config.max_model_len,
    )
    return max_blocks * self.page_size_bytes


_original_scheduler_init = Scheduler.__init__


@wraps(_original_scheduler_init)
def _patched_scheduler_init(self: Scheduler, vllm_config: VllmConfig, *args: Any, **kwargs: Any) -> None:
    _original_scheduler_init(self, vllm_config, *args, **kwargs)
    max_in_flight_tokens = _max_in_flight_tokens(vllm_config)
    for manager in self.kv_cache_manager.coordinator.single_type_managers:
        spec = manager.kv_cache_spec
        if isinstance(spec, (ChunkedLocalAttentionSpec, SlidingWindowSpec)):
            manager._max_admission_blocks_per_request = spec.max_admission_blocks_per_request(
                max_num_batched_tokens=max_in_flight_tokens,
                max_model_len=self.max_model_len,
            )


Request.__init__ = _patched_request_init
Scheduler.__init__ = _patched_scheduler_init
Scheduler._update_after_schedule = _patched_update_after_schedule
Scheduler.update_from_output = _patched_update_from_output
Scheduler._connector_finished = _patched_connector_finished
KVCacheManager.allocate_slots = _patched_allocate_slots
SingleTypeKVCacheManager.remove_skipped_blocks = _patched_remove_skipped_blocks
ChunkedLocalAttentionSpec.max_memory_usage_bytes = _patched_chunked_local_max_memory_usage_bytes
SlidingWindowSpec.max_memory_usage_bytes = _patched_swa_max_memory_usage_bytes
