# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import (
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend.ascend_config import NonBSPConfig
from vllm_ascend.core.scheduler_diagnostics import print_scheduler_summary


class NonBSPScheduler(Scheduler):
    running: list[Request]
    prefill_capacity_bound: bool
    connector_prefix_cache_stats: PrefixCacheStats | None
    finished_req_ids: set[str]
    reset_preempted_req_ids: set[str]

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        additional_config = vllm_config.additional_config or {}
        scheduler_extension_config = additional_config.get("scheduler_config") or {}
        nonbsp_user_config = scheduler_extension_config.get("nonbsp_config") or {}
        self._enable_diagnostics = NonBSPConfig(nonbsp_user_config).enable_diagnostics
        self.modifications: dict | None = None
        self.lb_freeze: bool = False
        self._lb_paused_req_ids: set[str] = set()
        self._lb_kv_prefetch_enabled: bool = False
        self._lb_admission_candidates: list[Request] = []
        self._lb_admit_req_ids: set[str] | None = None
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

    def _waiting_requests_in_schedule_order(self) -> list[Request]:
        if self.policy == SchedulingPolicy.FCFS:
            return [*self.skipped_waiting, *self.waiting]

        waiting = list(self.waiting)
        skipped = list(self.skipped_waiting)
        ordered: list[Request] = []
        waiting_idx = skipped_idx = 0
        while waiting_idx < len(waiting) and skipped_idx < len(skipped):
            if waiting[waiting_idx] < skipped[skipped_idx]:
                ordered.append(waiting[waiting_idx])
                waiting_idx += 1
            else:
                ordered.append(skipped[skipped_idx])
                skipped_idx += 1
        ordered.extend(waiting[waiting_idx:])
        ordered.extend(skipped[skipped_idx:])
        return ordered

    def _refresh_blocked_waiting_requests(self) -> None:
        for request in list(self.skipped_waiting):
            if self._is_blocked_waiting_status(request.status):
                self._try_promote_blocked_waiting_request(request)

    def _run_lb_kv_prefetch(self) -> set[str]:
        if self.connector is None or not self._lb_kv_prefetch_enabled:
            return set()

        kv_prefetch_limit = 2 * self.max_num_running_reqs - len(self.running)
        kv_prefetch_count = 0
        kv_not_ready_req_ids: set[str] = set()
        requests_to_move: list[Request] = []
        for request in self._waiting_requests_in_schedule_order():
            if request.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
                continue
            if request.num_computed_tokens != 0:
                continue
            if kv_prefetch_count >= kv_prefetch_limit:
                kv_not_ready_req_ids.add(request.request_id)
                continue
            kv_prefetch_count += 1

            new_computed_blocks, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)
            ext_tokens, load_async = self.connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)
            if ext_tokens is None:
                kv_not_ready_req_ids.add(request.request_id)
                continue
            if not load_async or not ext_tokens:
                continue

            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                0,
                num_new_computed_tokens=num_new_local_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=0,
                num_external_computed_tokens=ext_tokens,
                delay_cache_blocks=True,
                full_sequence_must_fit=self.scheduler_reserve_full_isl,
            )
            if new_blocks is None:
                kv_not_ready_req_ids.add(request.request_id)
                continue

            self.connector.update_state_after_alloc(
                request,
                self.kv_cache_manager.get_blocks(request.request_id),
                ext_tokens,
            )
            request.num_computed_tokens = num_new_local_computed_tokens + ext_tokens
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            self._inflight_prefills.add(request)
            if request in self.waiting:
                requests_to_move.append(request)

        if requests_to_move:
            self.waiting.remove_requests(requests_to_move)
            for request in requests_to_move:
                self.skipped_waiting.add_request(request)
        return kv_not_ready_req_ids

    def prepare_nonbsp_step(self) -> list[Request]:
        self._refresh_blocked_waiting_requests()
        kv_not_ready_req_ids = self._run_lb_kv_prefetch()
        self._lb_admission_candidates = [
            request
            for request in self._waiting_requests_in_schedule_order()
            if request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)
            and request.request_id not in kv_not_ready_req_ids
        ]
        return self._lb_admission_candidates

    def _apply_load_balance_modifications(self) -> None:
        self._lb_admit_req_ids = None
        if self.modifications is None:
            self.lb_freeze = False
            return

        out_blks = self.modifications["out_blk"]
        in_blks = self.modifications["in_blk"]
        self.lb_freeze = self.modifications["freeze"]
        scheduled_timestamp = time.monotonic()
        block_size = self.block_size

        def req_blk_num(req: Request) -> int:
            return (len(req.all_token_ids) + block_size - 1) // block_size

        remaining_out = list(out_blks)
        for priority_filter in (
            lambda r: getattr(r, "lb_newly_added", False),
            lambda r: True,
        ):
            if not remaining_out:
                break
            for req in list(self.running):
                if not remaining_out:
                    break
                blk_num = req_blk_num(req)
                if blk_num in remaining_out and priority_filter(req):
                    remaining_out.remove(blk_num)
                    self.running.remove(req)
                    self._lb_pause_request(req, scheduled_timestamp)

        remaining_in = [] if self.lb_freeze else list(in_blks)
        admit_req_ids: set[str] = set()
        for req in self._lb_admission_candidates:
            if not remaining_in:
                break
            blk_num = req_blk_num(req)
            if blk_num in remaining_in:
                remaining_in.remove(blk_num)
                admit_req_ids.add(req.request_id)
        self._lb_admit_req_ids = admit_req_ids

    def _can_admit_waiting_request(self, request: Request) -> bool:
        return self._lb_admit_req_ids is None or request.request_id in self._lb_admit_req_ids

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
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

        self._apply_load_balance_modifications()

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

            num_new_tokens = (
                request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
            )
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

                    self._preempt_request(preempted_req, scheduled_timestamp)
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

                # Try to promote blocked statuses while traversing waiting.
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

                if not self._can_admit_waiting_request(request):
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
                num_uncached_common_prefix_tokens = 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    if (
                        self.connector is not None
                        and self.has_mamba_layers
                        and isinstance(
                            self.kv_cache_manager.coordinator,
                            HybridKVCacheCoordinator,
                        )
                    ):
                        computed, per_group_hits = self.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
                            request.block_hashes,
                            request.num_tokens - 1,
                        )
                        new_computed_blocks = self.kv_cache_manager.create_kv_cache_blocks(computed)
                        # NOTE(ZhanqiuHu): For Mamba hybrid models,
                        # num_new_local_computed_tokens should be the FA hit
                        # length. This value is passed to the connector's
                        # get_num_new_matched_tokens which computes:
                        # external = total - local_computed.
                        # Using the FA hit skips re-transferring FA blocks
                        # already cached on D-side. The Mamba state (always
                        # the last block) is transferred unconditionally by
                        # _apply_prefix_caching in nixl/worker.py.
                        num_new_local_computed_tokens = max(per_group_hits)
                        if self.kv_cache_manager.log_stats:
                            assert self.kv_cache_manager.prefix_cache_stats is not None
                            self.kv_cache_manager.prefix_cache_stats.record(
                                num_tokens=request.num_tokens,
                                num_hits=num_new_local_computed_tokens,
                                preempted=request.num_preemptions > 0,
                            )
                    else:
                        new_computed_blocks, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_blocks(
                            request
                        )

                    # In case of hybrid models, obtain hint for Marconi-style APC logic
                    if self.has_mamba_layers:
                        num_uncached_common_prefix_tokens = getattr(
                            self.kv_cache_manager.coordinator,
                            "num_uncached_common_prefix_tokens",
                            0,
                        )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                            request, num_new_local_computed_tokens
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        num_external_computed_tokens = ext_tokens

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
                    if request.prefill_stats is not None:
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
                        num_uncached_common_prefix_tokens,
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

        # Drain new attention block ids every step so the manager-side list
        # does not grow unbounded; only kv-cache zeroing consumes them.
        new_attn_block_ids = self.kv_cache_manager.take_new_block_ids()
        new_block_ids_to_zero = (new_attn_block_ids or None) if self.needs_kv_cache_zeroing else None

        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
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

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        self._lb_paused_req_ids.discard(request.request_id)
        super()._preempt_request(request, timestamp)

    def _lb_pause_request(self, request: Request, timestamp: float) -> None:
        """Pause a request for NonBSP load balancing without freeing KV."""
        assert request.status == RequestStatus.RUNNING, "Only running requests can be lb-paused"
        request.status = RequestStatus.PREEMPTED
        self._lb_paused_req_ids.add(request.request_id)
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
        self.waiting.prepend_request(request)

    def is_lb_paused(self, request: Request) -> bool:
        return request.request_id in self._lb_paused_req_ids

    def _handle_stopped_request(self, request: Request) -> bool:
        was_preempted = request.status != RequestStatus.RUNNING
        finished = super()._handle_stopped_request(request)
        if was_preempted:
            self.skipped_waiting.remove_requests((request,))
        return finished

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id in num_scheduled_tokens:
            self._lb_paused_req_ids.discard(req_id)

        if self.scheduler_config.async_scheduling:
            # Match vLLM's AsyncScheduler semantics. num_computed_tokens has
            # already advanced above, while the model output has not necessarily
            # arrived yet. Output placeholders keep decode requests schedulable in
            # the next engine step instead of making them alternate between one
            # scheduled step and one num_new_tokens == 0 step.
            spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
            self._spec_token_placeholders = [-1] * scheduler_output.num_spec_tokens_to_schedule
            for req_id in num_scheduled_tokens:
                request = self.requests[req_id]
                if request.is_prefill_chunk:
                    continue

                scheduler_output.pending_structured_output_tokens |= (
                    request.use_structured_output and request.num_output_placeholders > 0
                )
                cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
                request.num_output_placeholders += self.num_sampled_tokens_per_step + cur_num_spec_tokens
                request.spec_token_ids = self._spec_token_placeholders
                if self.use_v2_model_runner:
                    request.next_decode_eligible_step = self.current_step + self.pp_size

        if self._enable_diagnostics:
            print_scheduler_summary(self, scheduler_output)

    def _update_request_with_output(self, request: Request, new_token_ids: list[int]) -> tuple[list[int], bool]:
        if self.scheduler_config.async_scheduling and request.async_tokens_to_discard > 0:
            # reset_prefix_cache() force-preempted this request. Drop stale
            # in-flight output frames until the recorded count is drained.
            request.async_tokens_to_discard -= 1
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(request, new_token_ids)

        if self.scheduler_config.async_scheduling:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

            # Match AsyncScheduler: true preemption frees the request's KV
            # blocks, so only cache confirmed tokens while it remains running.
            if status_before_update == RequestStatus.RUNNING:
                self.kv_cache_manager.cache_blocks(
                    request,
                    request.num_computed_tokens - request.num_output_placeholders,
                )
        return new_token_ids, stopped

    def _free_request(self, request: Request, delay_free_blocks: bool = False) -> dict[str, Any] | None:
        self._lb_paused_req_ids.discard(request.request_id)
        return super()._free_request(request, delay_free_blocks)
