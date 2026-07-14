#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Patches ``vllm.v1.engine.core.EngineCore`` to support Virtual Pipeline
Parallelism (VPP) scheme2.

The vLLM tree does NOT ship VPP: ``VppContinuationOutput`` is injected by
``patch_outputs`` and the per-batch ``batch_id`` stamping by
``patch_scheduler``.  This patch closes the engine-core gap by:

  * ``EngineCore.__init__`` — detect VPP from
    ``vllm_config.additional_config["virtual_pipeline_parallel_size"]``,
    record ``self.vpp_enabled``, bump ``self.batch_queue_size`` by one
    so the fold-back scheme can over-schedule, and allocate
    ``self.allow_over_batch_queue`` / ``self._deferred_scheduler_output``.
  * ``EngineCore.step_with_batch_queue`` — respect
    ``allow_over_batch_queue``, dispatch on ``VppContinuationOutput``
    (re-issue ``execute_model`` and re-enqueue), handle the ``None``-output
    case for VPP-disabled batches coming back from ``sample_tokens``, and
    replay ``_deferred_scheduler_output`` on the next visit.  When VPP is
    not enabled it delegates to the original upstream method unchanged.
  * ``DPEngineCoreProc.execute_dummy_batch`` — when a batch queue is
    active, dispatch a non-blocking collective RPC instead of
    blocking; otherwise fall back to the inherited behaviour. A blocking
    drain would destroy pipeline overlap.

This patch is intentionally engine-only; the model/executor sides are
patched separately via ``patch_vpp_make_layers`` / ``patch_multiproc_executor``.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from typing import cast

from vllm.v1.engine.core import DPEngineCoreProc, EngineCore
from vllm.v1.outputs import ModelRunnerOutput, VppContinuationOutput
from vllm.logger import logger

# Extra headroom above ``batch_queue_size`` for the deque capacity.
# VPP over-schedules by +1 for the fold-back continuation cycle, and
# transient bursts (e.g. async pre-fetching) risk silent drops if
# ``maxlen`` is too tight.  Value 10 is empirically sufficient; scale
# proportionally with ``max_num_seqs`` if needed.
_VPP_BATCH_QUEUE_HEADROOM = 10

# ---------------------------------------------------------------------------
# EngineCore.__init__ — VPP-aware setup
# ---------------------------------------------------------------------------
_original_engine_core_init = EngineCore.__init__


def _ascend_vpp_post_init(self: EngineCore) -> None:
    """Apply VPP-specific state adjustments AFTER the original __init__.

    The original ``__init__`` already builds the ``batch_queue`` according
    to ``max_concurrent_batches``; we simply resize when VPP is enabled
    and stash the auxiliary attributes.
    """
    additional_config = getattr(self.vllm_config, "additional_config", None)
    vp_size = additional_config.get("virtual_pipeline_parallel_size", 1) if isinstance(additional_config, dict) else 1
    self.vpp_enabled = isinstance(vp_size, int) and vp_size > 1
    self.vp_size = int(vp_size) if self.vpp_enabled else 1

    self.allow_over_batch_queue = False
    self._deferred_scheduler_output = None

    if self.vpp_enabled and self.batch_queue is not None:
        # VPP scheme2 needs one extra batch in flight so the fold-back
        # schedule can over-subscribe; the headroom keeps the deque from
        # dropping entries during the over-scheduled continuation cycle.
        self.batch_queue_size += 1
        self.batch_queue = deque(maxlen=self.batch_queue_size + _VPP_BATCH_QUEUE_HEADROOM)


def _ascend_vpp_engine_core_init(self: EngineCore, *args, **kwargs) -> None:
    _original_engine_core_init(self, *args, **kwargs)
    _ascend_vpp_post_init(self)


EngineCore.__init__ = _ascend_vpp_engine_core_init


# ---------------------------------------------------------------------------
# EngineCore.step_with_batch_queue — VPP-aware scheduling
# ---------------------------------------------------------------------------
# Captured before reassignment so the non-VPP path can delegate to the
# original upstream method unchanged (byte-identical behaviour for users
# who have not enabled VPP).  Only ``vpp_enabled`` engines run the
# VPP-aware loop below.
_original_step_with_batch_queue = EngineCore.step_with_batch_queue


def _ascend_vpp_step_with_batch_queue(
    self: EngineCore,
) -> tuple[dict, bool]:
    """Drop-in replacement for ``EngineCore.step_with_batch_queue``.

    When VPP is disabled, delegates to the original upstream method so that
    non-VPP users are unaffected by the VPP-aware rewrite.  When VPP is
    enabled, respects ``allow_over_batch_queue``, recognizes
    ``VppContinuationOutput``, and replays ``_deferred_scheduler_output``
    across calls.
    """
    if not getattr(self, "vpp_enabled", False):
        return _original_step_with_batch_queue(self)

    batch_queue = self.batch_queue
    assert batch_queue is not None

    # Try to schedule a new batch if the batch queue is not full, but
    # the scheduler may return an empty batch if all requests are scheduled.
    # Note that this is not blocking.
    can_schedule = len(batch_queue) < self.batch_queue_size or self.allow_over_batch_queue
    logger.info("batch queue size:%s, self.batch_queue_size:%s", len(batch_queue), self.batch_queue_size)

    if self.allow_over_batch_queue:
        self.allow_over_batch_queue = False

    model_executed = False
    deferred_scheduler_output = None
    if can_schedule and self.scheduler.has_requests():
        scheduler_output = self.scheduler.schedule()
        with self.log_error_detail(scheduler_output):
            exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
        if self.is_ec_consumer:
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

        if self.is_pooling_model or not model_executed or self.vpp_enabled:
            # No sampling required (no requests scheduled).
            future = cast(
                Future[ModelRunnerOutput | VppContinuationOutput | None],
                exec_future,
            )
        else:
            if not scheduler_output.pending_structured_output_tokens:
                # We aren't waiting for any tokens, get any grammar output
                # and sample immediately.
                grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
                future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            else:
                # We need to defer sampling until we have processed
                # the model output from the prior step.
                deferred_scheduler_output = scheduler_output

        if not deferred_scheduler_output:
            # Add this step's future to the queue.
            batch_queue.appendleft((future, scheduler_output, exec_future))
            if model_executed and len(batch_queue) < self.batch_queue_size and not batch_queue[-1][0].done():
                # Don't block on next worker response unless the queue
                # is full or there are no more requests to schedule.
                return None, True

    elif not batch_queue:
        # Queue is empty. We should not reach here since this method
        # should only be called when the scheduler contains requests
        # or the queue is non-empty.
        return None, False

    # Block until the next result is available.
    future, scheduler_output, exec_model_fut = batch_queue.pop()
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if isinstance(model_output, VppContinuationOutput):
            self.allow_over_batch_queue = True
            if model_output.kv_connector_output:
                self.scheduler._update_from_kv_xfer_finished(model_output.kv_connector_output)
            exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
            if model_output.next_vp_stage == self.vp_size - 1:
                # We aren't waiting for any tokens, get any grammar
                # output and sample immediately.
                grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
                future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            else:
                future = cast(
                    Future[ModelRunnerOutput | VppContinuationOutput | None],
                    exec_future,
                )
            batch_queue.appendleft((future, scheduler_output, exec_future))
            return None, True

        if model_output is None:
            if not self.vpp_enabled:
                # None from sample_tokens() implies that the original
                # execute_model() call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")
            if scheduler_output.pending_structured_output_tokens:
                self._deferred_scheduler_output = scheduler_output
                return None, True
            grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, scheduler_output, exec_model_fut))
            if len(batch_queue) < self.batch_queue_size and not batch_queue[-1][0].done():
                return None, True
            return None, True

    # Before processing the model output, process any aborts that
    # happened during the model execution.
    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)

    if deferred_scheduler_output is None and self._deferred_scheduler_output is not None:
        deferred_scheduler_output = self._deferred_scheduler_output
        self._deferred_scheduler_output = None

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
            self.scheduler.update_draft_token_ids_in_output(draft_token_ids, deferred_scheduler_output)
        # We now have the tokens needed to compute the bitmask for
        # the deferred request. Get the bitmask and call sample tokens.
        grammar_output = self.scheduler.get_grammar_bitmask(deferred_scheduler_output)
        future = self.model_executor.sample_tokens(grammar_output, non_block=True)
        batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

    return engine_core_outputs, model_executed


EngineCore.step_with_batch_queue = _ascend_vpp_step_with_batch_queue


# ---------------------------------------------------------------------------
# DPEngineCoreProc.execute_dummy_batch — non-blocking under batch queue
# ---------------------------------------------------------------------------
_original_dp_execute_dummy_batch = DPEngineCoreProc.execute_dummy_batch


def _ascend_vpp_dp_execute_dummy_batch(self: DPEngineCoreProc) -> None:
    """Run a dummy batch, non-blocking if a batch queue is active.

    A blocking ``execute_dummy_batch`` would drain every in-flight
    future in the worker ``futures_queue`` and destroy pipeline overlap.
    When VPP keeps a ``batch_queue`` populated we instead issue a
    non-blocking collective RPC; the dummy future is consumed the next
    time ``step_with_batch_queue`` drains the queue, and worker-side
    ordering keeps the DP all2all calls paired up correctly.
    """
    if getattr(self, "batch_queue", None) is not None:
        self.model_executor.collective_rpc(
            "execute_dummy_batch",
            unique_reply_rank=self.model_executor.output_rank,
            non_block=True,
        )
        return
    _original_dp_execute_dummy_batch(self)


DPEngineCoreProc.execute_dummy_batch = _ascend_vpp_dp_execute_dummy_batch
