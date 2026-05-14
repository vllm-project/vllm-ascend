# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, TypeVar, cast
from dataclasses import dataclass

import msgspec
import zmq

from vllm.config import ParallelConfig, vllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.tracing import instrument, maybe_init_worker_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import(
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import(
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityRequest,
)
from vllm.v1.engine.utils import(
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import compute_iteration_details
from vllm.version import __version__ as VLLM_VERSION

class ErrorClassifier:
    def __init__(self):
        self.err_map: dict[str, list[str]] = {}

    def register(self, error_type: str, patterns: list[str]):
        """
        Register Error Map
        """
        self.err_map.setdefault(error_type, []).extend(patterns)

    def classify(self, error_message: str) -> str | None:
        for error_type, error_code in self.err_map.items():
            for pat in error_code:
                if pat in error_message.lower():
                    return error_type
        return None

@dataclass
class ErrorOutput:
    error_type: str
    error_message: str

def step_with_batch_queue(
        self,
) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
    """Schedule and execute batches with the batch queue.
    Note that if nothing to output in this step, None is returned.

    The execution flow is as follows:
    1. Try to schedule a new batch if the batch queue is not full.
    If a new batch is scheduled, directly return an empty engine core
    output. In other words, fulfilling the batch queue has a higher priority
    than getting model outputs.
    2. If there is no new scheduled batch, meaning that the batch queue
    is full or no other requests can be scheduled, we block until the first
    batch in the job queue is finished.
    3. Update the scheduler from the output.
    """
    # If paused, don't schedule any work.
    if self._scheduler_paused:
        return {}, False

    batch_queue = self.batch_queue
    assert batch_queue is not None

    # Try to schedule a new batch if the batch queue is not full, but
    # the scheduler may return an empty batch if all requests are scheduled.
    # Note that this is not blocking.
    assert len(batch_queue) < self.batch_queue_size

    model_executed = False
    deferred_scheduler_output = None
    if self.scheduler.has_requests():
        scheduler_output = self.scheduler.schedule()
        exec_future = self.model_executor.execute_model(
            scheduler_output, non_block=True
        )
        if not self.is_ec_producer:
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

        if self.is_pooling_model or not model_executed:
            # No sampling required (no requests scheduled).
            future = cast(Future[ModelRunnerOutput], exec_future)
        else:
            if not scheduler_output.pending_structured_output_tokens:
                # We aren't waiting for any tokens, get any grammar output
                # and sample immediately.
                grammar_output = self.scheduler.get_grammar_bitmask(
                    scheduler_output
                )
                future = self.model_executor.sample_tokens(
                    grammar_output, non_block=True
                )
            else:
                # We need to defer sampling until we have processed the model output
                # from the prior step.
                deferred_scheduler_output = scheduler_output

        if not deferred_scheduler_output:
            # Add this step's future to the queue.
            batch_queue.appendleft((future, scheduler_output, exec_future))
            if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
            ):
                # Don't block on next worker response unless the queue is full
                # or there are no more requests to schedule.
                return None, True

    elif not batch_queue:
        # Queue is empty. We should not reach here since this method should
        # only be called when the scheduler contains requests or the queue
        # is non-empty.
        return None, False

    # Block until the next result is available.
    future, scheduler_output, exec_model_fut = batch_queue.pop()
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        retry_count = 0
        while isinstance(model_output, ErrorOutput) and retry_count < 3:
            if model_output.error_type == "network_error":
                schedules_to_retry = []
                schedules_to_retry.append(scheduler_output)
                while batch_queue:
                    queued_future, queued_sched_output, queued_exec_fut = batch_queue.pop()
                    try:
                        _ = queued_future.result()
                    except Exception:
                        pass
                    if hasattr(self, "step_counter"):
                        self.step_counter += 1
                    schedules_to_retry.append(queued_sched_output)

                logger.info(f"Detected remote error, re-executing {len(schedules_to_retry)} schedules")
                for sched_output in schedules_to_retry:
                    re_exec_future = self.model_executor.execute_model(
                        sched_output, non_block=True
                    )
                    needs_sampling = not self.is_pooling_model and sched_output.total_num_scheduled_tokens > 0
                    if needs_sampling and not sched_output.pending_structured_output_tokens:
                        grammar_output = self.scheduler.get_grammar_bitmask(
                            sched_output
                        )
                        re_sample_future = self.model_executor.sample_tokens(
                            grammar_output, non_block=True
                        )
                        batch_queue.appendleft((re_sample_future, sched_output,re_exec_future))
                    else:
                        batch_queue.appendleft((re_exec_future, sched_output,re_exec_future))

                future, scheduler_output, exec_model_fut = batch_queue.pop()
                if hasattr(self, "step_counter"):
                    self.step_counter += 1
                model_output = future.result()
            else:
                raise RuntimeError(f"Unexpected error: {model_output.error_message}")

        if model_output is None:
            # None from sample_tokens() implies that the original execute_model()
            # call failed - raise that exception.
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

    # Before processing the model output, process any aborts that happened
    # during the model execution.
    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    # NOTE(nick): We can either handle the deferred tasks here or save
    # in a field and do it immediately once step_with_batch_queue is
    # re-called. The latter slightly favors TTFT over TPOT/throughput.
    if deferred_scheduler_output:
        # If we are doing speculative decoding with structured output,
        # we need to get the draft token ids from the prior step before
        # we can compute the grammar bitmask for the deferred request.
        if self.use_spec_decode:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            assert draft_token_ids is not None
            # Update the draft token ids in the scheduler output to
            # filter out the invalid spec tokens, which will be padded
            # with -1 and skipped by the grammar bitmask computation.
            self.scheduler.update_draft_token_ids_in_output(
                draft_token_ids, deferred_scheduler_output
            )
        # We now have the tokens needed to compute the bitmask for the
        # deferred request. Get the bitmask and call sample tokens.
        grammar_output = self.scheduler.get_grammar_bitmask(
            deferred_scheduler_output
        )
        future = self.model_executor.sample_tokens(grammar_output, non_block=True)
        batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

    return engine_core_outputs, model_executed

def enqueue_output(self, output: Any):
    """
    Prepares output from the worker and enqueues it to the
    worker_response_mq.If the output is an Exception, it is
    converted to a FAILURE response.
    """
    if isinstance(output, AsyncModelRunnerOutput):
        try:
            output = output.get_output()
        except Exception as e:
            error_type = self.err_classifier.classify(str(e))
            if error_type is not None:
                output = ErrorOutput(
                    error_type=error_type,
                    error_message=str(e),
                )
            else:
                output = e

    if isinstance(output, Exception):
        result = (WorkerProc.ResponseStatus.FAILURE, str(output))
    else:
        result = (WorkerProc.ResponseStatus.SUCCESS, output)
    if (response_mq := self.worker_response_mq) is not None:
        response_mq.enqueue(result)


EngineCore.step_with_batch_queue = step_with_batch_queue
vllm.v1.executor.multiproc_executor.MultiprocExecutor.enqueue_output = enqueue_output