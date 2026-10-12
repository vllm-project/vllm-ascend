# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Defer-aware preemption retry: backport of vLLM PR #49675.

Upstream vLLM (and vllm-ascend's pinned release tag) can hit a
zero-reclaim preemption cascade when deferred block freeing is enabled
(async scheduling + PD KV-consumer): the ``schedule()`` allocation retry
loop preempts a request whose blocks cannot be returned to the pool yet
(its last scheduled step is still in flight), so retrying
``allocate_slots`` keeps failing, preempts the running requests one by
one, and stalls the engine.

vLLM PR #49675 fixes the cascade with three small changes:

1. ``Scheduler.schedule()`` -- the preemption retry loop now picks the
   victim *before* removing it and bails out of the retry loop when the
   victim's blocks are still fenced by an in-flight step
   (``_should_defer_request_block_free``); the allocation is retried on
   a later step instead of cascading through the whole running batch.
2. ``Scheduler._should_defer_request_block_free()`` -- new helper shared
   by the retry loop and the free path.
3. ``Scheduler._free_request_blocks()`` -- refactored to use the helper
   (no behavior change by itself).

vllm-ascend porting notes (see ``patch/__init__.py`` for the general
strategy):

* the two small methods are applied verbatim from the PR;
* ``schedule()`` is an ~800-line method, so (as in
  ``patch_balance_schedule.py``) the whole body is copied verbatim from
  the pinned vLLM release tag (``.github/vllm-release-tag.commit`` --
  v0.27.1 at the time of writing) with exactly one delta: the PR's
  preemption-loop restructure, adapted to the pin's older bookkeeping
  (``self.running.remove()`` + the in-``scheduled_running_reqs`` budget
  restore instead of main's ``victim_index`` tracking);
* the copy is guarded at import time: if the installed vLLM's
  ``Scheduler.schedule`` signature no longer matches the pin, or if the
  installed vLLM already carries the fix, we fail loudly / skip instead
  of silently shadowing a drifted upstream.

Behavior is unchanged when ``defer_block_free`` is disabled: the helper
returns ``False`` and the retry loop behaves exactly as upstream.
"""

import inspect
import time

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Small method patches -- verbatim from vLLM PR #49675.
# ---------------------------------------------------------------------------


def _should_defer_request_block_free(self: Scheduler, request: Request) -> bool:
    return self.defer_block_free and request.last_sched_seq > self.processed_step_seq


def _free_request_blocks(self: Scheduler, request: Request):
    """Free the request's KV blocks, deferring the return to the block
    pool when an in-flight GPU step may still write them.
    """
    if not self._should_defer_request_block_free(request):
        self.kv_cache_manager.free(request)
        return
    blocks = self.kv_cache_manager.pop_blocks_for_free(request)
    if blocks:
        self.deferred_frees.append((self.sched_step_seq, blocks))


# ---------------------------------------------------------------------------
# ``schedule()`` -- verbatim copy of the pinned release tag's
# ``Scheduler.schedule()`` (see module docstring), modulo the PR #49675
# preemption-loop delta.
# ---------------------------------------------------------------------------


def _defer_free_aware_schedule(self: Scheduler, throttle_prefills: bool = False) -> SchedulerOutput:
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
    if self._pause_state == PauseState.PAUSED_ALL:
        # Do not schedule any requests when paused.
        token_budget = 0

    # Encoder-related.
    scheduled_encoder_inputs: dict[str, list[int]] = {}
    encoder_compute_budget = self.max_num_encoder_input_tokens
    # Spec decode-related.
    scheduled_spec_decode_tokens: dict[str, list[int]] = {}
    # Whether the running batch contains any prefill requests.
    prefill_scheduled = False

    # For logging.
    scheduled_timestamp = time.monotonic()

    self.kv_cache_manager.new_step_starts()

    # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
    # all prefill compute unless saturated.
    defer_prefills = (throttle_prefills and not self.prefill_capacity_bound) and any(
        not r.is_prefill_chunk for r in self.running
    )

    # First, schedule the RUNNING requests.
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]

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

        num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(num_new_tokens, token_budget)

        # Make sure the input position does not exceed the max model len.
        # This is necessary when using spec decoding.
        num_new_tokens = min(
            num_new_tokens,
            self.max_model_len - request.num_computed_tokens - self.num_sampled_tokens_per_step,
        )

        # Schedule encoder inputs.
        encoder_inputs_to_schedule = None
        external_load_encoder_input: list[int] = []
        new_encoder_compute_budget = encoder_compute_budget
        if request.has_encoder_inputs:
            (
                encoder_inputs_to_schedule,
                num_new_tokens,
                new_encoder_compute_budget,
                external_load_encoder_input,
            ) = self._try_schedule_encoder_inputs(
                request,
                request.num_computed_tokens,
                num_new_tokens,
                encoder_compute_budget,
                shift_computed_tokens=1 if self.use_eagle else 0,
            )

        if self.need_mamba_block_aligned_split:
            num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)

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

                # The request cannot be scheduled.
                # Preempt the lowest-priority request.
                if self.policy == SchedulingPolicy.PRIORITY:
                    preempted_req = max(
                        self.running,
                        key=lambda r: (r.priority, r.arrival_time),
                    )
                else:
                    preempted_req = self.running[-1]

                # A deferred free cannot satisfy this allocation retry.
                if self._should_defer_request_block_free(preempted_req):
                    break

                if self.policy == SchedulingPolicy.PRIORITY:
                    self.running.remove(preempted_req)
                    if preempted_req in scheduled_running_reqs:
                        preempted_req_id = preempted_req.request_id
                        scheduled_running_reqs.remove(preempted_req)
                        token_budget += num_scheduled_tokens.pop(preempted_req_id)
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
                        req_index -= 1
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
        req_index += 1

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
            # Allocate the encoder cache.
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)
            encoder_compute_budget = new_encoder_compute_budget
        if external_load_encoder_input:
            for i in external_load_encoder_input:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)

    # Record the LoRAs in scheduled_running_reqs
    scheduled_loras: set[int] = set()
    if self.lora_config:
        scheduled_loras = set(
            req.lora_request.lora_int_id
            for req in scheduled_running_reqs
            if req.lora_request and req.lora_request.lora_int_id > 0
        )
        assert len(scheduled_loras) <= self.lora_config.max_loras

    # Next, schedule the WAITING requests.
    if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
        step_skipped_waiting = create_request_queue(self.policy)

        while (self.waiting or self.skipped_waiting) and token_budget > 0:
            # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
            # in `running` but still hold a model-runner request slot.
            num_running = len(self.running) + self.num_waiting_for_streaming_input
            if num_running >= self.max_num_running_reqs:
                break

            request_queue = self._select_waiting_queue_for_scheduling()
            assert request_queue is not None

            request = request_queue.peek_request()
            request_id = request.request_id

            # try to promote blocked statuses while traversing skipped queue.
            if self._is_blocked_waiting_status(request.status) and not self._try_promote_blocked_waiting_request(
                request
            ):
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    logger.debug(
                        "%s is still in WAITING_FOR_REMOTE_KVS state.",
                        request_id,
                    )
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue

            if request.num_stale_output_tokens > 0 and not request.drop_stale_output:
                # Deliverable stale output still in flight: resuming now
                # could resample a position that output later delivers.
                # It drains within the pipeline depth.
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue

            # Check that adding the request still respects the max_loras
            # constraint.
            if (
                self.lora_config
                and request.lora_request
                and (
                    len(scheduled_loras) == self.lora_config.max_loras
                    and request.lora_request.lora_int_id not in scheduled_loras
                )
            ):
                # Scheduling would exceed max_loras, skip.
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue

            num_external_computed_tokens = 0
            load_kv_async = False
            connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
            did_prefix_cache_lookup = False

            # Get already-cached tokens.
            if request.num_computed_tokens == 0:
                did_prefix_cache_lookup = True
                hit_diverged = False
                # Get locally-cached tokens.
                if self.connector is not None:
                    # A KV connector transfers the missing suffix, which needs a
                    # hybrid-aware lookup that can diverge across groups.
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                        request.shared_prefix_boundary,
                        hit_diverged,
                    ) = self.kv_cache_manager.get_computed_blocks_for_connector(request)
                else:
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                        # Marconi shared-prefix junction to pin; 0 if none.
                        request.shared_prefix_boundary,
                    ) = self.kv_cache_manager.get_computed_blocks(request)

                # Get externally-cached tokens if using a KVConnector.
                if self.connector is not None:
                    # Present a block-aligned local hit to the connector so
                    # a strictly longer remote hit can supersede a local
                    # sub-block tail without racing its copy-on-write.
                    partial_tail = num_new_local_computed_tokens % self.block_size
                    block_aligned_local = num_new_local_computed_tokens - partial_tail
                    ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(request, block_aligned_local)

                    if ext_tokens is None:
                        # The request cannot be scheduled because
                        # the KVConnector couldn't determine
                        # the number of matched tokens.
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

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

                # Skip request with pending mm encoding prefetches
                if (
                    self.ec_connector is not None
                    and request.mm_features
                    and not self.ec_connector.ensure_cache_available(request, num_computed_tokens)
                ):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Track first scheduled prefill, not post-preemption repeat prefills
                if request.prefill_stats and request.num_preemptions <= 0:
                    assert num_computed_tokens <= request.num_prompt_tokens
                    request.prefill_stats.set(
                        num_prompt_tokens=request.num_prompt_tokens,
                        num_local_cached_tokens=num_new_local_computed_tokens,
                        num_external_cached_tokens=num_external_computed_tokens,
                    )
            else:
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                num_new_local_computed_tokens = 0
                num_computed_tokens = request.num_computed_tokens

            encoder_inputs_to_schedule = None
            external_load_encoder_input = []
            new_encoder_compute_budget = encoder_compute_budget
            pad_spec_decode = False

            if load_kv_async:
                # KVTransfer: loading remote KV, do not allocate for new work.
                assert num_external_computed_tokens > 0
                num_new_tokens = 0
            elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                # DP prefill balancing: defer this step's local prefill
                # compute to a cadence-aligned step.
                break
            else:
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
                    and (scheduled_running_reqs and not prefill_scheduled)
                ):
                    num_new_tokens = 1 + self.num_spec_tokens
                    if num_new_tokens > token_budget or num_computed_tokens + num_new_tokens > self.max_model_len:
                        # Prefer to not schedule than schedule un-padded here.
                        break
                    pad_spec_decode = True

                threshold = self.scheduler_config.long_prefill_token_threshold
                if 0 < threshold < num_new_tokens:
                    num_new_tokens = threshold

                # chunked prefill has to be enabled explicitly to allow
                # pooling requests to be chunked
                if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
                    # If chunked_prefill is disabled,
                    # we can stop the scheduling here.
                    break

                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                if request.has_encoder_inputs:
                    (
                        encoder_inputs_to_schedule,
                        num_new_tokens,
                        new_encoder_compute_budget,
                        external_load_encoder_input,
                    ) = self._try_schedule_encoder_inputs(
                        request,
                        num_computed_tokens,
                        num_new_tokens,
                        encoder_compute_budget,
                        shift_computed_tokens=1 if self.use_eagle else 0,
                    )
                    if num_new_tokens == 0:
                        # The request cannot be scheduled.
                        break

            # Skip block alignment when setting up async receive (no local work).
            if self.need_mamba_block_aligned_split and not load_kv_async:
                num_new_tokens = self._mamba_block_aligned_split(
                    request,
                    num_new_tokens,
                    num_new_local_computed_tokens,
                    num_external_computed_tokens,
                )
                if num_new_tokens == 0:
                    break

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
                # only if it fits in (free - other in-flight reservations), to
                # avoid deadlock and predictable preemptions.
                reserved_blocks = self._inflight_prefill_reserved_blocks()

            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_local_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=effective_lookahead_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=load_kv_async,
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
                    self.connector_prefix_cache_stats.record(
                        num_tokens=connector_prefix_cache_queries,
                        num_hits=connector_prefix_cache_hits,
                        preempted=request.num_preemptions > 0,
                    )

            # Record at admission so unscheduled lookups are not counted.
            if did_prefix_cache_lookup:
                self.kv_cache_manager.record_prefix_cache_stats(request, num_new_local_computed_tokens)

            request = request_queue.pop_request()
            if load_kv_async:
                # If loading async, allocate memory and put request
                # into the WAITING_FOR_REMOTE_KV state.
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                step_skipped_waiting.prepend_request(request)
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
                if self.needs_kv_cache_zeroing:
                    # Skip zeroing of the blocks the async load will
                    # overwrite; the zeroing could race the write.
                    self._skip_zero_block_ids.update(
                        self.kv_cache_manager.get_zeroing_block_ids_in_range(
                            request.request_id,
                            num_new_local_computed_tokens,
                            num_computed_tokens,
                        )
                    )
                continue

            self.running.append(request)
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
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            if pad_spec_decode:
                scheduled_spec_decode_tokens[request_id] = [-1] * self.num_spec_tokens
            # Only track requests that will still be prefilling after this chunk.
            if num_computed_tokens + num_new_tokens < request.num_tokens:
                self._inflight_prefills.add(request)
            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            # Allocate for external load encoder cache
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # re-queue requests skipped in this pass ahead of older skipped items.
        if step_skipped_waiting:
            self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # DP prefill balancing: on a step that admitted prefills (release),
        # record whether it was capacity-bound.
        if not defer_prefills:
            self.prefill_capacity_bound = bool(self.waiting)

    # Check if the scheduling constraints are satisfied.
    total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
    assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

    assert token_budget >= 0
    assert len(self.running) <= self.max_num_running_reqs
    # Since some requests in the RUNNING queue may not be scheduled in
    # this step, the total number of scheduled requests can be smaller than
    # len(self.running).
    assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) <= len(self.running)

    # Get the longest common prefix among all requests in the running queue.
    # This can be potentially used for cascade attention.
    num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
    with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
        if self.running:
            any_request_id = self.running[0].request_id
            num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)

    # Construct the scheduler output.
    if self.use_v2_model_runner:
        scheduled_new_reqs.extend(scheduled_resumed_reqs)
        scheduled_resumed_reqs.clear()
        new_reqs_data = [
            NewRequestData.from_request(
                req,
                req_to_new_blocks[req.request_id].get_block_ids(),
                req._all_token_ids,
            )
            for req in scheduled_new_reqs
        ]
    else:
        new_reqs_data = [
            NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
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

    # Producer partial-tail hand-off for external KV connectors. Drained
    # before the CoW retentions are released below, so the pin lands while
    # the cow block still holds a retention ref. Without a producer-side
    # connector nothing consumes the hand-off, so skip the drain (and its
    # pin); the manager drops stale entries when the request's blocks are
    # popped for free.
    pending_partial_tail_offloads = None
    if (
        self.connector is not None
        and self.vllm_config.kv_transfer_config is not None
        and self.vllm_config.kv_transfer_config.is_kv_producer
    ):
        pending_partial_tail_offloads = self.kv_cache_manager.take_partial_tail_offloads() or None

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
        num_common_prefix_blocks=num_common_prefix_blocks,
        preempted_req_ids=self.reset_preempted_req_ids,
        # finished_req_ids is an existing state in the scheduler,
        # instead of being newly scheduled in this step.
        # It contains the request IDs that are finished in between
        # the previous and the current steps.
        finished_req_ids=self.finished_req_ids,
        free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
        kv_cache_block_copies=pending_kv_cache_block_copies,
        partial_tail_offloads=pending_partial_tail_offloads,
        num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
        ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata(),
    )

    # NOTE(Kuntai): this function is designed for multiple purposes:
    # 1. Plan the KV cache store
    # 2. Wrap up all the KV cache load / save ops into an opaque object
    # 3. Clear the internal states of the connector
    if self.connector is not None:
        meta = self._build_kv_connector_meta(self.connector, scheduler_output)
        scheduler_output.kv_connector_metadata = meta

    # Build the connector meta for ECConnector
    if self.ec_connector is not None:
        ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(scheduler_output)
        scheduler_output.ec_connector_metadata = ec_meta

    # Advance the fence only for non-empty steps (those that actually
    # write KV and have their output processed later in update_from_output).
    if self.defer_block_free and total_num_scheduled_tokens > 0:
        self.sched_step_seq += 1

    with record_function_or_nullcontext("schedule: update_after_schedule"):
        self._update_after_schedule(scheduler_output)
    return scheduler_output


# ---------------------------------------------------------------------------
# Install (idempotent, drift-guarded).
# ---------------------------------------------------------------------------

# The pinned ``schedule()`` signature shared by the release tag and the
# main-verified ref. A mismatch means upstream drifted and the verbatim
# copy above no longer tracks the pin: fail loudly instead of shadowing.
_PINNED_SCHEDULE_PARAMETERS = ("self", "throttle_prefills")

if hasattr(Scheduler, "_should_defer_request_block_free"):
    # The installed vLLM already carries PR #49675 (or a successor):
    # nothing to patch. Keeps the module idempotent and pin-advance-safe.
    logger.debug(
        "vLLM Scheduler already implements _should_defer_request_block_free; skipping the PR #49675 backport patch."
    )
else:
    _installed_parameters = tuple(inspect.signature(Scheduler.schedule).parameters)
    if _installed_parameters != _PINNED_SCHEDULE_PARAMETERS:
        raise RuntimeError(
            "vllm_ascend patch_defer_block_free_schedule: upstream "
            f"Scheduler.schedule signature {_installed_parameters} no longer "
            f"matches the pinned {_PINNED_SCHEDULE_PARAMETERS}; the verbatim "
            "schedule() copy in this patch is stale -- re-sync it with the "
            "new pin before proceeding."
        )
    Scheduler._should_defer_request_block_free = _should_defer_request_block_free
    Scheduler._free_request_blocks = _free_request_blocks
    Scheduler.schedule = _defer_free_aware_schedule
    logger.info("Patched Scheduler.schedule/_free_request_blocks with the PR #49675 defer-aware preemption backport.")
