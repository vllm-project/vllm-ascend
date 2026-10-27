# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in decoder-local prefix-cache admission.

Select DecoderAppendPrefillScheduler or AsyncDecoderAppendPrefillScheduler
via scheduler_cls. Only requests with sampling_params.extra_args containing
require_decoder_cache_hit=1 are subject to admission. A usable local prefix is
required before scheduling prefill; other requests retain upstream behavior.

The schedule body is copied from vLLM 544f182d9f0c9bfe8bd7645525951d5e3692a542:
there is no upstream hook that can reject one waiting request and continue the
same scheduling iteration. Schedule deltas are per-step rejection bookkeeping,
skipping pending rejections, the admission guard, bypassing remote prefix
matching for marked requests, and assigning the draft-skipping hint only when
the output supports it. Initialization supplies disabled draft-skipping and
local-prefill-cadence defaults for v0.32.0. Keep the drift test aligned when
updating it.

DyntraLB is currently unsupported: it requires Decoder instances to execute
decode-only work, while this scheduler also executes append-prefill on Decoders.
Recompute variants share admission here and reuse RecomputeScheduler.schedule().
A preempted request with live offload state can restore its own KV without a
new local cache hit. Failed recovery preserves the no-replay-after-output
contract. First-decode padding is disabled for the combination because a batch
may contain append-prefill; speculative decoding itself remains enabled.
Other Ascend scheduler policies are not composed here. This module does not
install global patches or change the API server's terminal-event handling.
"""

import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import KVConnectorBlockState, NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.stats import KV_FETCH_IN_PROGRESS, KV_FETCH_WAITING_TO_START
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler as _SchedulerBase
else:
    _SchedulerBase = object


class DecoderCacheAdmissionMixin(_SchedulerBase):
    """Shared local admission and batch-scoped terminal output handling."""

    def __init__(self, *args, **kwargs) -> None:
        # Scheduler policy compatibility is validated in platform._check_ascend_config.
        super().__init__(*args, **kwargs)
        self._pending_decoder_cache_rejections: dict[str, Request] = {}
        # EngineCore retains this exact SchedulerOutput until update_from_output.
        # Keep references here, not on the output sent to model workers.
        self._decoder_cache_rejection_batches: dict[int, tuple[SchedulerOutput, list[Request]]] = {}

    @staticmethod
    def _requires_decoder_cache_hit(request: Request) -> bool:
        params = request.sampling_params
        return params is not None and (params.extra_args or {}).get("require_decoder_cache_hit") == 1

    def _get_local_prefix_cache_hit(self, request: Request) -> tuple[KVCacheBlocks, int, int, bool]:
        if not self._requires_decoder_cache_hit(request):
            return super()._get_local_prefix_cache_hit(request)
        # Divergent hybrid hits are only valid with connector assistance. Use
        # the common resumable boundary for a strictly local request instead.
        blocks, num_local, shared_boundary = self.kv_cache_manager.get_computed_blocks(request)
        return blocks, num_local, shared_boundary, False

    def _should_reject_decoder_cache_miss(self, request: Request, local_hits: int) -> bool:
        # Hits fully consumed by bounded SWA replay save no local prefill.
        return self._requires_decoder_cache_hit(request) and local_hits <= self.prefix_replay_tokens

    def _record_decoder_cache_rejections(self, output: SchedulerOutput, requests: list[Request]) -> None:
        if requests:
            self._decoder_cache_rejection_batches[id(output)] = (output, requests)

    def _finish_decoder_cache_rejections(self, output: SchedulerOutput) -> dict[int, list[EngineCoreOutput]]:
        batch = self._decoder_cache_rejection_batches.pop(id(output), None)
        results: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        if batch is None:
            return results
        request_ids = []
        for request in batch[1]:
            request_id = request.request_id
            if self._pending_decoder_cache_rejections.get(request_id) is request:
                del self._pending_decoder_cache_rejections[request_id]
            # Aborts may have removed the request, including reuse of its ID.
            if self.requests.get(request_id) is request:
                request_ids.append(request_id)
        for request in self.finish_requests(request_ids, RequestStatus.FINISHED_STOPPED):
            results[request.client_index].append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    stop_reason=(
                        "decoder_cache_lost_after_output" if request.num_output_tokens else "decoder_cache_miss"
                    ),
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                )
            )
        return results

    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput
    ) -> dict[int, EngineCoreOutputs]:
        # Finish before upstream collects finished IDs and scheduler stats, but
        # only when this batch completes; earlier in-flight batches keep their
        # normal output and deferred-free ordering.
        rejections = self._finish_decoder_cache_rejections(scheduler_output)
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for client_index, rejected_outputs in rejections.items():
            outputs.setdefault(client_index, EngineCoreOutputs()).outputs.extend(rejected_outputs)
        return outputs


class DecoderAppendPrefillScheduler(DecoderCacheAdmissionMixin, Scheduler):
    """Reject decoder-local cache misses without scheduling their prefill."""

    prefill_capacity_bound: bool
    last_prefill_step: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # v0.32.0 has neither draft skipping nor local prefill cadence. Keep
        # its normal draft generation and ungated prefill behavior, while
        # preserving the values initialized by newer upstream schedulers.
        self.can_skip_drafts = getattr(self, "can_skip_drafts", False)
        self.local_prefill_interval = getattr(self, "local_prefill_interval", 1)
        self.num_executed_steps = getattr(self, "num_executed_steps", 0)
        self.last_prefill_step = getattr(self, "last_prefill_step", -self.local_prefill_interval)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        rejected_requests: list[Request] = []
        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        spec = self.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
        input_budget = self.scheduler_config.max_num_batched_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Whether a request scheduled this step can use the drafts proposed after
        # it. Stays set if that request is preempted later in the step.
        # True on v0.32.0: later OR expressions short-circuit, so they never
        # call _drafts_unused(), which that upstream version does not provide.
        drafts_needed = not self.can_skip_drafts
        # Whether the running batch contains any prefill requests.
        prefill_scheduled = False
        # Whether a waiting request was admitted with prompt left to compute.
        # Separate from `prefill_scheduled`, which covers only the running loop.
        prefill_admitted = False
        # Whether any scheduled request has a synchronous connector KV load.
        has_sync_kv_loads = False

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.
        defer_prefills = (throttle_prefills and not self.prefill_capacity_bound) and any(
            not r.is_prefill_chunk for r in self.running
        )

        # `long_prefill_token_threshold` exists to stop a long prefill from
        # starving other requests of the token budget. When it is the only
        # request there is nobody to starve, so let it use the whole budget.
        num_running, num_waiting = self.get_request_counts()
        num_eligible_reqs = num_running + num_waiting
        long_prefill_token_threshold = (
            self.scheduler_config.long_prefill_token_threshold if num_eligible_reqs > 1 else 0
        )
        if long_prefill_token_threshold > 0 and self.adaptive_long_prefill_threshold:
            # Floor the cap at a fair share of the input budget so it never
            # cuts a request below max_num_batched_tokens / num requests.
            long_prefill_token_threshold = max(long_prefill_token_threshold, input_budget // num_eligible_reqs)

        # Local prefill cadence: hold prefill until the interval since the last
        # prefill-carrying step has elapsed, so decode runs in longer
        # uninterrupted stretches. Mirrors the DP gate above, including its
        # capacity escape: once a release step cannot drain the waiting queue,
        # throttling stops (#44558, test_throttle_capacity_bound_guard_admits).
        # Without that escape a backed-up engine holds both admissions and
        # in-flight chunks for interval-1 of every interval steps, capping
        # prefill at roughly one budget per interval and growing the waiting
        # queue without bound.
        if (
            not defer_prefills
            and self.local_prefill_interval > 1
            and not self.prefill_capacity_bound
            # This call becomes executed step num_executed_steps + 1 if it
            # schedules anything, and last_prefill_step is stamped on the same
            # convention below.
            and (self.num_executed_steps + 1) - self.last_prefill_step < self.local_prefill_interval
            and any(not r.is_prefill_chunk for r in self.running)
        ):
            defer_prefills = True

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if input_budget <= draft_slots:
                break

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue

            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            if (
                self.ec_connector is not None
                and request.mm_features
                and not self.ec_connector.ensure_cache_available(
                    request,
                    request.num_computed_tokens - request.num_output_placeholders,
                )
            ):
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
            )
            if 0 < long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens - self.num_sampled_tokens_per_step,
            )

            # Apply Mamba alignment before encoder caps.
            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            duplicate_encoder_inputs: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                    duplicate_encoder_inputs,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=self.num_prefill_lookahead,
                )

            # Multi-module MTP: avoid ending a prefill chunk within
            # num_prefill_lookahead of the prefill end.
            num_new_tokens = self._reserve_prefill_lookahead(request, request.num_computed_tokens, num_new_tokens)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # 5. Insufficient budget to keep a multi-module MTP prefill
                #    chunk out of the prefill-lookahead window.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    if self.connector is not None and self.connector.has_pending_block_frees():
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                    else:
                        preempted_req = self.running[-1]

                    # A deferred free will not help with immediate allocation.
                    if not self._request_blocks_can_be_freed(preempted_req):
                        break

                    if self.policy == SchedulingPolicy.PRIORITY:
                        victim_index = self.running.index(preempted_req)
                        del self.running[victim_index]
                        if victim_index < req_index:
                            req_index -= 1

                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            restored = num_scheduled_tokens.pop(preempted_req_id)
                            token_budget += restored
                            input_budget += restored + draft_slots
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(preempted_req_id, None)
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i) for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(
                        preempted_req,
                        scheduled_timestamp,
                        drop_stale_output=self.requires_kv_delivery,
                    )
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            req_index += 1
            drafts_needed = drafts_needed or not self._drafts_unused(
                request, request.num_computed_tokens, num_new_tokens
            )

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens + request.num_computed_tokens - request.num_tokens - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                encoder_compute_budget = new_encoder_compute_budget
            if encoder_inputs_to_schedule or external_load_encoder_input:
                self._allocate_encoder_inputs(
                    request,
                    (encoder_inputs_to_schedule or []) + external_load_encoder_input,
                    duplicate_encoder_inputs,
                )

        # Record the LoRAs of all running requests, including those not
        # scheduled in this step, which can join a later batch.
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in self.running
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state != PauseState.PAUSED_ALL:
            step_skipped_waiting: deque[Request] = deque()
            step_skipped_kv_holding: deque[Request] = deque()

            def skip_request(from_queue: RequestQueue) -> None:
                request_to_skip = from_queue.pop_request()
                self.deferred_waiting.add(request_to_skip)
                if self._holds_kv_blocks(request_to_skip):
                    step_skipped_kv_holding.appendleft(request_to_skip)
                else:
                    step_skipped_waiting.appendleft(request_to_skip)

            while token_budget > 0:
                # Requests holding KV blocks are always drained first, and are
                # the only ones admitted under PAUSED_NEW (e.g. async KV loads).
                request_queue = self.kv_holding_waiting
                if not request_queue and self._pause_state == PauseState.UNPAUSED:
                    request_queue = self.waiting
                if not request_queue or input_budget <= draft_slots:
                    break
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_active_reqs:
                    break

                request = request_queue.peek_request()
                request_id = request.request_id
                if self._pending_decoder_cache_rejections.get(request_id) is request:
                    skip_request(request_queue)
                    continue

                ready_to_schedule = self._handle_blocked_waiting_request(request)
                if not ready_to_schedule:
                    skip_request(request_queue)
                    continue

                if request.num_stale_output_tokens and not request.drop_stale_output:
                    # Deliverable stale output still in flight: resuming now
                    # could resample a position that output later delivers.
                    # It drains within the pipeline depth.
                    skip_request(request_queue)
                    continue

                # Check that adding the request still respects the max_loras constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    skip_request(request_queue)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
                did_prefix_cache_lookup = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    did_prefix_cache_lookup = True
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                        request.shared_prefix_boundary,
                        hit_diverged,
                    ) = self._get_local_prefix_cache_hit(request)

                    if self._should_reject_decoder_cache_miss(request, num_new_local_computed_tokens):
                        self._pending_decoder_cache_rejections[request_id] = request
                        rejected_requests.append(request)
                        skip_request(request_queue)
                        continue

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None and not self._requires_decoder_cache_hit(request):
                        # Present a block-aligned local hit to the connector so
                        # a strictly longer remote hit can supersede a local
                        # sub-block tail without racing its copy-on-write.
                        partial_tail = num_new_local_computed_tokens % self.block_size
                        block_aligned_local = num_new_local_computed_tokens - partial_tail
                        ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                            request, block_aligned_local
                        )
                        if request.skip_reading_prefix_cache:
                            ext_tokens, load_kv_async = 0, False

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            skip_request(request_queue)
                            continue

                        if self.prefix_replay_tokens and not (
                            ext_tokens and self._load_restores_replay_window(request)
                        ):
                            # DSV41 SWA bounded replay recomputes the hit's last
                            # window, from the block holding its first
                            # token; the sliding-window groups retire
                            # whole blocks below the window of the hit's
                            # next token. Hits end on a block boundary, or
                            # a hit ending one token short of one would
                            # retire the block the replay starts in. A load
                            # that carries the window replays nothing.
                            ext_tokens -= ext_tokens % self.block_size
                            load_kv_async = load_kv_async and ext_tokens > 0

                        if partial_tail and ext_tokens > partial_tail:
                            # Remote strictly exceeds the full local hit: drop the
                            # sub-block tail so no CoW is needed, and let the load
                            # cover it. Trim the partial block out of the local
                            # computed blocks so it is not adopted from the cache.
                            new_computed_blocks = self.kv_cache_manager.truncate_computed_blocks(
                                new_computed_blocks, block_aligned_local
                            )
                            num_new_local_computed_tokens = block_aligned_local
                            num_external_computed_tokens = ext_tokens
                        elif partial_tail:
                            # Remote does not exceed the full local hit: keep the
                            # local sub-block tail and load nothing external.
                            num_external_computed_tokens = 0
                            # Nothing to load remotely -> not an async-load step;
                            # clearing avoids the `load_kv_async` assert below.
                            load_kv_async = False
                        else:
                            num_external_computed_tokens = ext_tokens

                        if hit_diverged and num_external_computed_tokens == 0:
                            # No external tokens back the deeper local hit, so its
                            # resume boundary would have no valid Mamba state.
                            # Reconcile to the boundary every group agrees on.
                            (
                                new_computed_blocks,
                                num_new_local_computed_tokens,
                                request.shared_prefix_boundary,
                            ) = self.kv_cache_manager.get_computed_blocks(request)

                        connector_prefix_cache_queries = request.num_tokens - num_new_local_computed_tokens
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
                    assert num_computed_tokens <= request.num_tokens
                    # DSV41 SWA bounded replay: a KV load that carries the
                    # window is taken as is; any other hit no longer than the
                    # replayed window would be recomputed in full and save
                    # nothing.
                    window_loaded = bool(num_external_computed_tokens) and self._load_restores_replay_window(request)
                    if not window_loaded and 0 < num_computed_tokens <= self.prefix_replay_tokens:
                        new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                        num_new_local_computed_tokens = 0
                        num_external_computed_tokens = 0
                        connector_prefix_cache_hits = 0
                        num_computed_tokens = 0
                        load_kv_async = False

                    # Include encoder inputs needed to replay the prefix tail.
                    ec_computed_tokens = num_computed_tokens
                    if not window_loaded:
                        ec_computed_tokens = max(0, num_computed_tokens - self.prefix_replay_tokens)
                    if self._ec_transfer_pending(request, ec_computed_tokens):
                        skip_request(request_queue)
                        continue

                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats and request.num_preemptions <= 0:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                    self._set_kv_fetch_stage(request, KV_FETCH_WAITING_TO_START if load_kv_async else None)
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed. A streaming-input
                    # session resumes here too, carrying whatever media its
                    # latest chunk added, so this branch needs the same gate.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                    if self._ec_transfer_pending(request, num_computed_tokens):
                        skip_request(request_queue)
                        continue

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                duplicate_encoder_inputs = []
                new_encoder_compute_budget = encoder_compute_budget
                pad_spec_decode = False

                # SWA bounded replay: recompute the tail of the hit without
                # rewriting its cached KV. An async load replays once the KV
                # has arrived (_update_waiting_for_remote_kv).
                num_replay_tokens = 0

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                    # DP prefill balancing: defer this step's local prefill
                    # compute to a cadence-aligned step.
                    break
                else:
                    request_token_budget = min(token_budget, input_budget - draft_slots)
                    if did_prefix_cache_lookup:
                        # Fresh admission: local and/or sync external hit.
                        num_replay_tokens = self._mark_prefix_replay(request, num_computed_tokens, window_loaded)
                        num_computed_tokens -= num_replay_tokens
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens

                    # Pad new decode requests to uniform spec decoding size to
                    # preserve full cudagraph for this step.
                    # Not for diffusion where draft tokens can't be padded.
                    if (
                        (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                        and self.num_sampled_tokens_per_step > 0
                        and num_new_tokens == 1
                        and not prefill_scheduled
                        and (scheduled_running_reqs or num_computed_tokens > 0)
                    ):
                        padded_num_tokens = 1 + self.num_spec_tokens
                        # Pad only when there is room for the sampled token(s).
                        if (
                            num_computed_tokens + padded_num_tokens + self.num_sampled_tokens_per_step
                            <= self.max_model_len
                        ):
                            if padded_num_tokens > request_token_budget:
                                # Prefer to not schedule than schedule un-padded.
                                break
                            num_new_tokens = padded_num_tokens
                            pad_spec_decode = True

                    if 0 < long_prefill_token_threshold < num_new_tokens:
                        num_new_tokens = long_prefill_token_threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > request_token_budget
                        and not self.is_mm_encoder_only
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, request_token_budget)
                    assert num_new_tokens > 0

                    # Apply Mamba alignment before encoder caps.
                    if self.need_mamba_block_aligned_split:
                        num_new_tokens = self._mamba_block_aligned_split(
                            request,
                            num_new_tokens,
                            num_new_local_computed_tokens,
                            num_external_computed_tokens,
                        )
                        if num_new_tokens == 0:
                            break
                        if pad_spec_decode and num_new_tokens != 1 + self.num_spec_tokens:
                            # Alignment clipped the placeholder rows. The split
                            # aligns prefill chunks, but the padded tail rows are
                            # speculative positions, not prefill tokens. A padded
                            # request must keep all 1 + num_spec rows or the
                            # sampler's row count stops matching its query rows,
                            # so drop the padding instead of shortening it.
                            num_new_tokens = 1
                            pad_spec_decode = False

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                            duplicate_encoder_inputs,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=self.num_prefill_lookahead,
                        )

                    # Multi-module MTP: avoid ending a prefill chunk within
                    # num_prefill_lookahead of the prefill end.
                    num_new_tokens = self._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)

                    if num_new_tokens == 0:
                        # The request cannot be scheduled. Requeue it so it does
                        # not stall the requests behind it.
                        skip_request(request_queue)
                        continue

                # During async KV load, no forward pass is run yet.
                # Allocate speculative lookahead slots later to avoid
                # mismatching local and remote block counts.
                limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
                effective_lookahead_tokens = 0 if limit_lookahead_tokens else self.num_lookahead_tokens

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if self.is_encoder_decoder and request.has_encoder_inputs and encoder_inputs_to_schedule:
                    num_encoder_tokens = sum(request.get_num_encoder_embeds(i) for i in encoder_inputs_to_schedule)

                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations)
                    # plus its own spec decode step blocks, to avoid deadlock and
                    # predictable preemptions.
                    reserved_blocks = self._inflight_prefill_reserved_blocks() + self._spec_decode_step_blocks

                # Replayed tokens are already counted in the adopted hit; a
                # chunk that ends inside the replayed range needs no new slots.
                num_tokens_past_hit = max(num_new_tokens - num_replay_tokens, 0)
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_tokens_past_hit,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    skip_zeroing_group_ids=(
                        self.connector.get_loaded_kv_cache_group_ids(request)
                        if load_kv_async and self.connector is not None
                        else ()
                    ),
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                    reserved_blocks=reserved_blocks,
                    has_scheduled_reqs=bool(self.running),
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if self.connector_prefix_cache_stats is not None and connector_prefix_cache_queries != 0:
                        preempted = request.num_preemptions > 0
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=preempted,
                            # The connector knows the tier only after the load
                            # plan above exists.
                            hits_by_source=(
                                self.connector.get_external_cache_hit_sources(request, connector_prefix_cache_hits)
                                if connector_prefix_cache_hits and not preempted
                                else None
                            ),
                        )

                # Record at admission so unscheduled lookups are not counted.
                if did_prefix_cache_lookup:
                    self.kv_cache_manager.record_prefix_cache_stats(request, num_new_local_computed_tokens)

                if load_kv_async:
                    # This path deliberately does not reach `prefill_admitted`
                    # below, so it does not restart the prefill cadence:
                    # `num_new_tokens` was set to 0 above, so the step carries
                    # no prefill compute for this request. The interval is
                    # timed from steps that actually prefilled; refreshing it
                    # here would start the cadence from a step that did none.
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    self._inflight_prefills.add(request)
                    self._set_kv_fetch_stage(request, KV_FETCH_IN_PROGRESS)
                    skip_request(request_queue)
                    continue

                request = request_queue.pop_request()
                self.deferred_waiting.discard(request)
                self.running.append(request)
                if num_external_computed_tokens > 0:
                    # load_kv_async is False here
                    has_sync_kv_loads = True
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(request_id)
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                input_budget -= num_new_tokens + draft_slots
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                self._set_kv_fetch_stage(request, None)
                drafts_needed = drafts_needed or not self._drafts_unused(request, num_computed_tokens, num_new_tokens)
                if pad_spec_decode:
                    assert num_new_tokens == 1 + self.num_spec_tokens
                    scheduled_spec_decode_tokens[request_id] = [-1] * self.num_spec_tokens
                # Must use the SAME predicate as the deferral gate in the
                # waiting loop (`num_computed_tokens < num_tokens - 1`).
                # Stamping on `< num_prompt_tokens` instead let a request with
                # exactly one token left pass the gate on a held step and still
                # restart the interval -- which is every full prefix-cache hit,
                # and every remote-KV resume, since those set
                # num_computed_tokens = num_tokens - 1.
                prefill_admitted |= num_computed_tokens < request.num_tokens - 1
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    encoder_compute_budget = new_encoder_compute_budget
                if encoder_inputs_to_schedule or external_load_encoder_input:
                    self._allocate_encoder_inputs(
                        request,
                        (encoder_inputs_to_schedule or []) + external_load_encoder_input,
                        duplicate_encoder_inputs,
                    )

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_kv_holding:
                self.kv_holding_waiting.prepend_requests(step_skipped_kv_holding)
            if step_skipped_waiting:
                self.waiting.prepend_requests(step_skipped_waiting)

            # DP prefill balancing: on a step that admitted prefills (release),
            # record whether it was capacity-bound.
            if not defer_prefills:
                _, num_waiting = self.get_request_counts()
                self.prefill_capacity_bound = num_waiting > len(self.deferred_waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        # Count this step before stamping the cadence below, so that
        # last_prefill_step includes the step that carried the prefill.
        if total_num_scheduled_tokens > 0:
            self.num_executed_steps += 1

        # Restart the cadence interval from any step that carried prefill,
        # whether a new admission or a running chunk.
        if self.local_prefill_interval > 1 and (prefill_scheduled or prefill_admitted):
            self.last_prefill_step = self.num_executed_steps

        assert token_budget >= 0
        assert input_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) <= len(self.running)

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                    uses_mrope=self.model_uses_mrope,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    uses_mrope=self.model_uses_mrope,
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step (MRV1-only).
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        # Mamba "align" boundary states must be handed off with exact block ids;
        # they cannot be reconstructed from a connector's append-only block
        # table. Drained every step so stale offers cannot accumulate.
        boundary_state_offloads = self.kv_cache_manager.take_boundary_state_offloads()

        kv_connector_block_state = None
        if self.connector is not None:
            # Any request scheduled this step can become a connector job now,
            # not only the ones that were allocated blocks: a store save lands
            # on the step that fills a block, which allocated none.
            block_state_req_ids = set(num_scheduled_tokens)
            block_state_req_ids.update(req_id for req_id in boundary_state_offloads if req_id in self.requests)
            kv_connector_block_state = KVConnectorBlockState(
                req_ids=block_state_req_ids,
                resolve_block_ids=self.kv_cache_manager.get_block_ids,
                boundary_state_offloads=boundary_state_offloads,
            )

        kv_cache_block_copies, cow_retained_blocks = self.kv_cache_manager.take_kv_cache_block_copies()
        if kv_cache_block_copies:
            # The copies run with this step's execution; the first non-empty
            # step at or after it gets seq `sched_step_seq + 1` (0-token steps
            # do not advance the seq), and its completion implies the copies
            # have run.
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
        pending_kv_cache_block_copies = kv_cache_block_copies or None

        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]

        scheduled_encoder_input_stats = None
        if self.log_stats and self.observability_config.enable_logging_iteration_details:
            scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(scheduled_encoder_inputs)

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            scheduled_encoder_input_stats=scheduled_encoder_input_stats,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
            has_sync_kv_loads=has_sync_kv_loads,
            kv_cache_block_copies=pending_kv_cache_block_copies,
            kv_connector_block_state=kv_connector_block_state,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
            ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata(),
        )
        # v0.32.0's output dataclass does not carry this optional worker hint.
        if hasattr(scheduler_output, "skip_drafts"):
            scheduler_output.skip_drafts = not drafts_needed

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        if self.aux_output_connector is not None:
            scheduler_output.aux_output_connector_metadata = self.aux_output_connector.build_connector_meta(
                scheduler_output, self.requests
            )

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        # Connector-only block state must not be dispatched to workers.
        scheduler_output.kv_connector_block_state = None

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        self._record_decoder_cache_rejections(scheduler_output, rejected_requests)
        return scheduler_output


class AsyncDecoderAppendPrefillScheduler(AsyncScheduler, DecoderAppendPrefillScheduler):
    """Admission with upstream async placeholders and stale-output handling."""


class DecoderAppendPrefillRecomputeScheduler(DecoderCacheAdmissionMixin, RecomputeScheduler):
    """Apply admission to new requests and preserve same-request KV restore."""

    _allow_decode_padding = False

    def _is_restoring_decoder_request(self, request: Request) -> bool:
        if not request.num_preemptions:
            return False
        has_state = getattr(self.connector, "has_preempted_request", None)
        if has_state is not None:
            return bool(has_state(request.request_id))
        # MultiConnector exposes its children without requiring a recovery hook
        # on the wrapper. Only live state for this same request permits restore.
        return any(
            hook(request.request_id)
            for connector in getattr(self.connector, "sub_connectors", ())
            if (hook := getattr(connector, "has_preempted_request", None)) is not None
        )

    def _can_admit_waiting_request(self, request: Request) -> bool:
        return self._pending_decoder_cache_rejections.get(
            request.request_id
        ) is not request and super()._can_admit_waiting_request(request)

    def _can_admit_local_prefix(self, request: Request, local_hits: int) -> bool:
        if self._is_restoring_decoder_request(request):
            return True
        if self._should_reject_decoder_cache_miss(request, local_hits):
            self._pending_decoder_cache_rejections[request.request_id] = request
            self._step_decoder_rejections.append(request)
            return False
        return True

    def _get_admission_replay_tokens(self, request: Request, local_hits: int) -> int:
        if self._requires_decoder_cache_hit(request):
            return self._mark_prefix_replay(request, local_hits, window_loaded=False)
        return 0

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        failed = request.request_id in self.failed_recving_kv_req_ids
        super()._update_waiting_for_remote_kv(request)
        if failed and self._requires_decoder_cache_hit(request):
            # Never turn a failed restore into an implicit local full prefill.
            self._pending_decoder_cache_rejections[request.request_id] = request
            self._step_decoder_rejections.append(request)

    def _get_external_prefix_cache_hit(self, request: Request, local_hits: int) -> tuple[int | None, bool]:
        if self._requires_decoder_cache_hit(request) and not self._is_restoring_decoder_request(request):
            return 0, False
        return super()._get_external_prefix_cache_hit(request, local_hits)

    def _finish_recomputed_request(self, request: Request) -> None:
        if not self._requires_decoder_cache_hit(request):
            return super()._finish_recomputed_request(request)
        # The victim was removed from running by Recompute. Preserve it until
        # this batch completes, so older in-flight output drains before deciding
        # whether replay is safe. Upstream preemption fences KV block freeing.
        Scheduler._preempt_request(self, request, time.monotonic(), drop_stale_output=self.requires_kv_delivery)
        self._pending_decoder_cache_rejections[request.request_id] = request
        self._step_decoder_rejections.append(request)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self._step_decoder_rejections: list[Request] = []
        output = super().schedule(throttle_prefills)
        self._record_decoder_cache_rejections(output, self._step_decoder_rejections)
        return output


class AsyncDecoderAppendPrefillRecomputeScheduler(AsyncScheduler, DecoderAppendPrefillRecomputeScheduler):
    pass
