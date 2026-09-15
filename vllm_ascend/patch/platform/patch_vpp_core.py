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
  * ``DPEngineCoreProc.execute_dummy_batch`` — on VPP-enabled engines with
    a batch queue active, dispatch a non-blocking collective RPC instead of
    blocking (a blocking drain would destroy pipeline overlap), bounding
    pending dummy futures so an idle DP rank cannot fill the worker's
    response ring; non-VPP engines fall back to the inherited behaviour.

This patch is intentionally engine-only; the model/executor sides are
patched separately via ``patch_vpp_make_layers`` / ``patch_multiproc_executor``.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from typing import cast

from vllm.v1.engine.core import DPEngineCoreProc, EngineCore, EngineCoreProc
from vllm.v1.outputs import ModelRunnerOutput, VppContinuationOutput

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
        # Split the single batch_queue into two semantically disjoint
        # deques: ``initial_queue`` holds batches currently in their first
        # virtual stage (worker future = execute_model), while
        # ``continuation_queue`` holds batches that have advanced and are
        # waiting on last-stage sample_tokens (worker future = sample_tokens).
        # This split lets ``step_with_batch_queue`` drain the continuation
        # side without being blocked by a not-yet-done entry sitting in
        # front of it on a single mixed deque.
        self.initial_queue = deque(
            maxlen=self.batch_queue_size + _VPP_BATCH_QUEUE_HEADROOM)
        self.continuation_queue = deque(
            maxlen=self.batch_queue_size + _VPP_BATCH_QUEUE_HEADROOM)
        # ``batch_queue`` is intentionally NOT created so that accidental
        # fall-through to the old single-queue paths raises AttributeError
        # instead of silently regressing.


def _ascend_vpp_engine_core_init(self: EngineCore, *args, **kwargs) -> None:
    _original_engine_core_init(self, *args, **kwargs)
    _ascend_vpp_post_init(self)


EngineCore.__init__ = _ascend_vpp_engine_core_init


# ---------------------------------------------------------------------------
# EngineCoreProc.has_work — VPP queues are work too
# ---------------------------------------------------------------------------
# Upstream has_work() checks ``self.batch_queue``, which VPP intentionally
# never fills (batches live in the split initial/continuation queues).
# Without this patch the engine parks in _process_input_queue as soon as the
# scheduler has no requests, leaving in-flight fold-back batches unpumped:
# PP0 then never posts the matching fold recv and PP1's already-enqueued
# isend times out on-device (~204s stars notify-wait) and the HCCL watchdog
# kills the worker.  PD prefill hits this deterministically because its
# requests finish right after prefill, while the oversubscribed batches are
# still crossing RPC boundaries.
_original_has_work = EngineCoreProc.has_work


def _ascend_vpp_has_work(self: EngineCoreProc) -> bool:
    if _original_has_work(self):
        return True
    if getattr(self, "vpp_enabled", False):
        # The split queues only exist when the upstream batch_queue was
        # created (async scheduling); fall back to empty otherwise.
        return bool(getattr(self, "initial_queue", ())) or bool(
            getattr(self, "continuation_queue", ()))
    return False


EngineCoreProc.has_work = _ascend_vpp_has_work


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
    enabled, the function operates on two semantically disjoint deques
    (``initial_queue`` and ``continuation_queue``) instead of the upstream
    single ``batch_queue``.  The exact same VPP-aware branches are preserved:
    ``is_pooling_model``, ``kv_connector_output``,
    ``pending_structured_output_tokens``, ``use_spec_decode`` and the
    deferred-replay path; only the queue plumbing differs.

    Phase order (per the VPP pipeline):

      1. **Phase 1** (pop 1 initial): when ``initial_queue`` is at capacity
         or its oldest future is already done, advance the oldest batch
         from vp_stage 0 to vp_stage 1 (``sample_tokens``) so it can land
         in ``continuation_queue`` for next-step release.
      2. **Phase 2** (admit): if ``has_requests()``, call
         ``scheduler.schedule()`` and dispatch whatever the scheduler
         returns -- a 0-token result is intentionally NOT enqueued
         anywhere (placeholder spacers were removed).
      3. **Phase 3** (release 1 continuation): pop the oldest
         ``continuation_queue`` entry.  If the corresponding
         ``ModelRunnerOutput`` is ready, run ``update_from_output`` to
         release slots.
    """
    if not getattr(self, "vpp_enabled", False):
        return _original_step_with_batch_queue(self)

    initial_queue = self.initial_queue
    continuation_queue = self.continuation_queue

    # Single engine_core_outputs / single model_executed: this function
    # performs exactly one ``update_from_output`` per step (in Phase 3).
    # Phase 2's defensive ``ModelRunnerOutput`` branch may overwrite this
    # (loss of one defensively-drained output is acceptable -- it never
    # occurs under vp_size=2 which is the only supported topology here).
    engine_core_outputs = None
    model_executed = False
    deferred_scheduler_output = None

    if self.allow_over_batch_queue:
        self.allow_over_batch_queue = False

    # ====== Phase 1 (first): pop 1 initial when full or oldest done ======
    if initial_queue and (
        len(initial_queue) >= self.batch_queue_size
        or initial_queue[-1][0].done()
    ):
        future, scheduler_output, exec_model_fut = initial_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if isinstance(model_output, VppContinuationOutput):
                # Re-dispatch into continuation_queue (never back to
                # initial_queue).  Mirrors the original line 279-313 path
                # but writes into continuation_queue instead of batch_queue.
                self.allow_over_batch_queue = True
                if model_output.kv_connector_output:
                    self.scheduler._update_from_kv_xfer_finished(
                        model_output.kv_connector_output)
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True)
                if model_output.next_vp_stage == self.vp_size - 1:
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output)
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True)
                else:
                    future = cast(
                        Future[ModelRunnerOutput | VppContinuationOutput
                               | None],
                        exec_future,
                    )
                continuation_queue.appendleft(
                    (future, scheduler_output, exec_future))
                # NOTE: do NOT early-return here (the upstream
                # line 314 return is gone) -- phase 1 admit and
                # phase 3 release still run in this same step.

            elif model_output is None:
                if not self.vpp_enabled:
                    # Defensive: only reachable when patching is bypassed.
                    exec_model_fut.result()
                    raise RuntimeError("unexpected error")
                if scheduler_output.pending_structured_output_tokens:
                    self._deferred_scheduler_output = scheduler_output
                else:
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output)
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True)
                    continuation_queue.appendleft(
                        (future, scheduler_output, exec_model_fut))
            else:
                # Defensive: ModelRunnerOutput (vp_size=2 should not reach
                # here under normal pipeline flow).  Mirrors original
                # line 337-349 for completeness.
                self._process_aborts_queue()
                engine_core_outputs = self.scheduler.update_from_output(
                    scheduler_output, model_output)
                model_executed = True

    # ====== Phase 2 (second): admit if waiting requests exist ======
    # Always enqueue scheduler_output (even if total_num_scheduled_tokens ==
    # 0).  Removing the old ``if tokens > 0: ... else: skip`` branch keeps
    # the pipeline cadence alive even when the upstream scheduler has
    # nothing ready to admit (e.g., the ``skip_gated`` window -- running
    # requests whose ``next_decode_eligible_step`` hasn't fired yet).  A
    # 0-token dispatched scheduler_output flows through initial_queue ->
    # continuation_queue -> update_from_output and is removed without
    # affecting scheduler state.  Without this the busy_loop tight-spins
    # in scenarios like a single curl test under 16seqs config.
    if self.scheduler.has_requests():
        scheduler_output = self.scheduler.schedule()
        if self.is_ec_consumer:
            # Original line 195-196: ``model_executed`` is only set under
            # ``is_ec_consumer``.  Preserve that exact branch.
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

        with self.log_error_detail(scheduler_output):
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True)

        # Original line 198-213 verbatim: under VPP enabled,
        # ``future = exec_future`` always; the else branch below is
        # defensive (only non-VPP + non-pooling + executed reaches it).
        if self.is_pooling_model or not model_executed or self.vpp_enabled:
            future = cast(
                Future[ModelRunnerOutput | VppContinuationOutput | None],
                exec_future,
            )
        else:
            if not scheduler_output.pending_structured_output_tokens:
                grammar_output = self.scheduler.get_grammar_bitmask(
                    scheduler_output)
                future = self.model_executor.sample_tokens(
                    grammar_output, non_block=True)
            else:
                deferred_scheduler_output = scheduler_output

        if not deferred_scheduler_output:
            initial_queue.appendleft(
                (future, scheduler_output, exec_future))

    elif not initial_queue and not continuation_queue:
        # Both queues are empty.  We should not reach here since this
        # method should only be called when the scheduler contains
        # requests or one of the queues is non-empty.
        return None, False

    # ====== Phase 3 (third): release 1 continuation ======
    if continuation_queue:
        # Peek at the right (oldest).  If the future is not done and
        # ``initial_queue`` is empty, block on completion (the
        # continuation-side analogue of the busy-loop throttle).  If
        # ``initial_queue`` is non-empty, simply skip -- the next
        # ``step_with_batch_queue`` call will land here again with the
        # same futures, and ``initial_queue`` will be drained first
        # before this branch fires its pop.
        if not continuation_queue[-1][0].done():
            continuation_queue[-1][0].result()
        if continuation_queue and continuation_queue[-1][0].done():
            future, scheduler_output, exec_model_fut = continuation_queue.pop()
            with (
                self.log_error_detail(scheduler_output),
                self.log_iteration_details(scheduler_output),
            ):
                model_output = future.result()
                if isinstance(model_output, ModelRunnerOutput):
                    self._process_aborts_queue()
                    engine_core_outputs = self.scheduler.update_from_output(
                        scheduler_output, model_output)
                    model_executed = True
                elif isinstance(model_output, VppContinuationOutput):
                    # Defensive: vp_size>2 (more than 2 virtual stages)
                    # only.  Should not be hit under vp_size=2.
                    self.allow_over_batch_queue = True
                    if model_output.kv_connector_output:
                        self.scheduler._update_from_kv_xfer_finished(
                            model_output.kv_connector_output)
                    exec_future = self.model_executor.execute_model(
                        scheduler_output, non_block=True)
                    if model_output.next_vp_stage == self.vp_size - 1:
                        grammar_output = self.scheduler.get_grammar_bitmask(
                            scheduler_output)
                        future = self.model_executor.sample_tokens(
                            grammar_output, non_block=True)
                    else:
                        future = cast(
                            Future[ModelRunnerOutput
                                   | VppContinuationOutput
                                   | None],
                            exec_future,
                        )
                    continuation_queue.appendleft(
                        (future, scheduler_output, exec_future))
                elif model_output is None:
                    if not self.vpp_enabled:
                        exec_model_fut.result()
                        raise RuntimeError("unexpected error")
                    if scheduler_output.pending_structured_output_tokens:
                        self._deferred_scheduler_output = scheduler_output
                    else:
                        grammar_output = self.scheduler.get_grammar_bitmask(
                            scheduler_output)
                        future = self.model_executor.sample_tokens(
                            grammar_output, non_block=True)
                        continuation_queue.appendleft(
                            (future, scheduler_output, exec_model_fut))
                # No early-return -- fall through to deferred handler.

    # ====== Deferred output replay (original line 351-373 verbatim) ======
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
            self.scheduler.update_draft_token_ids_in_output(
                draft_token_ids, deferred_scheduler_output)
        # We now have the tokens needed to compute the bitmask for the
        # deferred request. Get the bitmask and call sample tokens.
        grammar_output = self.scheduler.get_grammar_bitmask(
            deferred_scheduler_output)
        sample_future = self.model_executor.sample_tokens(
            grammar_output, non_block=True)
        # Third tuple slot is unused under VPP (see
        # patch_vpp_make_layers / model_runner_v1); ``sample_future``
        # is reused as a placeholder so the entry satisfies the
        # (future, scheduler_output, exec_future) shape.
        continuation_queue.appendleft(
            (sample_future, deferred_scheduler_output, sample_future))

    # Single engine_core_outputs returned -- matches upstream
    # ``return engine_core_outputs, model_executed`` semantics
    # byte-for-byte (one update_from_output per step call).
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

    Two guards make this safe outside the original VPP-only context:

    * Non-VPP engines (e.g. the decode instance in PD disaggregation)
      fall back to the upstream blocking behaviour.  They never run
      ``step_with_batch_queue`` while idle, so nothing would ever drain
      the dummy futures.
    * Even for VPP engines, an *idle* DP rank never calls
      ``step_with_batch_queue``, so ``futures_queue`` would grow without
      bound and the worker's response ring (10 slots) eventually fills
      up, deadlocking the worker's async-output thread ("No available
      shared memory broadcast block found").  We therefore drain the
      oldest futures once more than ``_DUMMY_MAX_PENDING`` are pending;
      their responses are already in the ring, so draining never waits
      on model execution.
    """
    if getattr(self, "batch_queue", None) is not None and getattr(
        self, "vpp_enabled", False
    ):
        futures_queue = getattr(self.model_executor, "futures_queue", None)
        if futures_queue is not None:
            while len(futures_queue) >= _DUMMY_MAX_PENDING:
                # FIFO: the rightmost entry is the oldest future; its
                # response is already in the ring, so this never blocks.
                futures_queue[-1].result()
        self.model_executor.collective_rpc(
            "execute_dummy_batch",
            unique_reply_rank=self.model_executor.output_rank,
            non_block=True,
        )
        return
    _original_dp_execute_dummy_batch(self)


# Keep well below the worker response-ring capacity (10 blocks) so an idle
# DP rank can never fill it with unread dummy-batch responses.
_DUMMY_MAX_PENDING = 8


DPEngineCoreProc.execute_dummy_batch = _ascend_vpp_dp_execute_dummy_batch