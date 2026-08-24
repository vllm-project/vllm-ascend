##
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/v1/core/sched/scheduler.py
#

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import cast

from vllm.config import SchedulerConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.logger import logger
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import record_function_or_nullcontext

# delta for dyntra_lb: reuse the load-balancing policy mixin and scheduler diagnostics.
from vllm_ascend.core.dyntra_lb_scheduler import (
    DyntraLBPolicyMixin,
    diagnostics_enabled,
    print_scheduler_summary,
)


@dataclass
class RecomputeSchedulerConfig(SchedulerConfig):
    scheduler_cls: str | type[object] = "vllm_ascend.core.recompute_scheduler.RecomputeScheduler"

    @classmethod
    def initialize_from_config(cls, vllm_config: VllmConfig):
        vllm_scheduler_config = vllm_config.scheduler_config
        scheduler_config = {
            field.name: getattr(vllm_scheduler_config, field.name)
            for field in fields(vllm_scheduler_config)
            if field.init
        }
        if vllm_scheduler_config.async_scheduling:
            scheduler_config["scheduler_cls"] = "vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler"
        else:
            scheduler_config["scheduler_cls"] = "vllm_ascend.core.recompute_scheduler.RecomputeScheduler"
        scheduler_config["max_model_len"] = vllm_config.model_config.max_model_len
        scheduler_config["is_encoder_decoder"] = vllm_config.model_config.is_encoder_decoder
        return cls(**scheduler_config)


@dataclass
class PreemptedRequestData:
    req_id: str
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int


@dataclass
class RecomputeSchedulerOutput(SchedulerOutput):
    preempted_reqs: list[PreemptedRequestData] | None = None


class RecomputeScheduler(Scheduler):
    running: list[Request]

    def _get_computed_blocks_for_connector(self, request: Request) -> tuple[KVCacheBlocks, int, int, bool]:
        kv_cache_manager = self.kv_cache_manager
        coordinator = kv_cache_manager.coordinator
        if (
            request.kv_transfer_params
            and request.kv_transfer_params.get("do_remote_prefill")
            and isinstance(coordinator, HybridKVCacheCoordinator)
            and getattr(coordinator, "full_attention_group_id", None) is not None
        ):
            prefix_cache_lookup_enabled = kv_cache_manager.enable_caching and not request.skip_reading_prefix_cache
            if not prefix_cache_lookup_enabled:
                return kv_cache_manager.empty_kv_cache_blocks, 0, 0, False

            fa_group_id = coordinator.full_attention_group_id
            assert fa_group_id is not None
            computed, per_group_hits = coordinator.find_longest_cache_hit_per_group(
                request.block_hashes,
                request.num_tokens - 1,
            )
            if not any(hit > per_group_hits[fa_group_id] for hit in per_group_hits):
                num_local = per_group_hits[fa_group_id]
                hit_diverged = min(per_group_hits) < num_local
                if hit_diverged:
                    blocks = kv_cache_manager.create_kv_cache_blocks(computed)
                    logger.info(
                        "[RecomputeScheduler] Partial-group cache candidate "
                        "request_id=%s per_group_hits=%s full_attention_group_id=%d "
                        "selected_local_tokens=%d",
                        request.request_id,
                        per_group_hits,
                        fa_group_id,
                        num_local,
                    )
                    return blocks, num_local, 0, True

        blocks, num_local, shared_prefix_boundary = kv_cache_manager.get_computed_blocks(request)
        return blocks, num_local, shared_prefix_boundary, False

    # delta for dyntra_lb: provide an overridable no-op hook for applying a cross-rank plan.
    def _apply_load_balance_modifications(self) -> None:
        """Apply optional scheduling-policy changes before running requests."""
        return

    # delta for dyntra_lb: provide an overridable no-op hook for filtering waiting-request admission.
    def _can_admit_waiting_request(self, request: Request) -> bool:
        """Return whether an optional policy allows this waiting request."""
        return True

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            if request.num_computed_tokens:
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
                if self.needs_kv_cache_zeroing:
                    self.kv_cache_manager.record_blocks_for_zeroing(request.request_id, request.num_computed_tokens)
            else:
                self.kv_cache_manager.free(request)
            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            num_computed_tokens = min(request.num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
            request.num_computed_tokens = num_computed_tokens

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def schedule(self, throttle_prefills: bool = False) -> RecomputeSchedulerOutput:
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
        preempted_req_data: list[PreemptedRequestData] = []

        # delta for dyntra_lb: apply cross-rank request migrations before allocating this scheduling step.
        self._apply_load_balance_modifications()

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
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

        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.
        defer_prefills = (throttle_prefills and not self.prefill_capacity_bound) and any(  # type: ignore
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
                or request.num_computed_tokens >= self.max_model_len
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
                    transfer_config = self.vllm_config.kv_transfer_config
                    if transfer_config is not None and not transfer_config.is_kv_producer:
                        preempted_req = self.running.pop()
                        preempted_req_id = preempted_req.request_id
                        preempted_block_ids = self.kv_cache_manager.get_block_ids(preempted_req_id)
                        preempted_num_computed_tokens = preempted_req.num_computed_tokens
                        preempt_hook = (
                            getattr(self.connector, "update_state_before_preempt", None)
                            if self.connector is not None
                            else None
                        )
                        offloaded = False
                        if preempt_hook is not None:
                            offloaded = bool(
                                preempt_hook(
                                    preempted_req,
                                    preempted_block_ids,
                                    preempted_num_computed_tokens,
                                )
                            )
                        if offloaded:
                            logger.info(
                                "[RecomputeScheduler] Recompute preemption offload "
                                "enabled for request %s, computed_tokens=%d.",
                                preempted_req_id,
                                preempted_num_computed_tokens,
                            )
                            preempted_req_data.append(
                                PreemptedRequestData(
                                    req_id=preempted_req_id,
                                    block_ids=preempted_block_ids,
                                    num_computed_tokens=preempted_num_computed_tokens,
                                )
                            )
                        else:
                            logger.info(
                                "[RecomputeScheduler] Preempted request %s without recompute offload. "
                                "computed_tokens=%d.",
                                preempted_req_id,
                                preempted_num_computed_tokens,
                            )
                        self._preempt_request(
                            preempted_req,
                            scheduled_timestamp,
                            drop_stale_output=self.requires_kv_delivery,
                        )
                        preempted_reqs.append(preempted_req)
                        if preempted_req == request:
                            break
                    else:
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

                        self._preempt_request(
                            preempted_req,
                            scheduled_timestamp,
                            drop_stale_output=self.requires_kv_delivery,
                        )
                        preempted_reqs.append(preempted_req)
                        logger.info(
                            "[RecomputeScheduler] Preempted request %s. running_count=%s, token_budget=%s",
                            preempted_req.request_id,
                            len(self.running),
                            token_budget,
                        )
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
                            "[RecomputeScheduler] %s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # delta for dyntra_lb: admit only waiting requests selected by the current load-balancing plan.
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
                hit_diverged = False
                partial_group_hit_selected = False
                did_prefix_cache_lookup = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    did_prefix_cache_lookup = True
                    # Get locally-cached tokens.
                    if self.connector is not None:
                        (
                            new_computed_blocks,
                            num_new_local_computed_tokens,
                            request.shared_prefix_boundary,
                            hit_diverged,
                        ) = self._get_computed_blocks_for_connector(request)
                        partial_group_hit_selected = hit_diverged
                    else:
                        computed_result = self.kv_cache_manager.get_computed_blocks(request)
                        (
                            new_computed_blocks,
                            num_new_local_computed_tokens,
                            request.shared_prefix_boundary,
                        ) = cast(tuple[KVCacheBlocks, int, int], computed_result)

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
                        selected_local_computed_tokens = num_new_local_computed_tokens
                        if hit_diverged and num_external_computed_tokens == 0:
                            (
                                new_computed_blocks,
                                num_new_local_computed_tokens,
                                request.shared_prefix_boundary,
                            ) = self.kv_cache_manager.get_computed_blocks(request)
                            partial_group_hit_selected = False
                        if hit_diverged:
                            logger.info(
                                "[RecomputeScheduler] Partial-group cache decision "
                                "request_id=%s selected_local_tokens=%d "
                                "external_tokens=%d final_local_tokens=%d fallback=%s",
                                request_id,
                                selected_local_computed_tokens,
                                num_external_computed_tokens,
                                num_new_local_computed_tokens,
                                num_external_computed_tokens == 0,
                            )
                        connector_prefix_cache_queries = request.num_tokens - num_new_local_computed_tokens
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
                    assert num_computed_tokens <= request.num_tokens

                    # Skip requests with pending multimodal encoding prefetches.
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
                    )
                    if num_new_tokens == 0:
                        break

                # During async KV load, no forward pass is run yet. Allocate
                # speculative lookahead slots later to avoid mismatching local
                # and remote block counts.
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

                record_prefix_cache_stats = getattr(self.kv_cache_manager, "record_prefix_cache_stats", None)
                if did_prefix_cache_lookup and record_prefix_cache_stats is not None:
                    record_prefix_cache_stats(
                        request,
                        num_new_local_computed_tokens,
                    )
                elif partial_group_hit_selected and self.kv_cache_manager.log_stats:
                    assert self.kv_cache_manager.prefix_cache_stats is not None
                    self.kv_cache_manager.prefix_cache_stats.record(
                        num_tokens=request.num_tokens,
                        num_hits=num_new_local_computed_tokens,
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
                    if self.needs_kv_cache_zeroing:
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

        kv_cache_block_copies, cow_retained_blocks = self.kv_cache_manager.take_kv_cache_block_copies()
        if kv_cache_block_copies:
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
        pending_kv_cache_block_copies = kv_cache_block_copies or None

        # Dynamic speculative decoding: compute optimal K.
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and num_scheduled_tokens:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]

        scheduler_output = RecomputeSchedulerOutput(
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
            new_block_ids_to_zero=self._get_new_block_ids_to_zero(),
            kv_cache_block_copies=pending_kv_cache_block_copies,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
            preempted_reqs=preempted_req_data,
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

        # Advance the fence only for non-empty steps that will later be
        # processed by update_from_output.
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        # delta for dyntra_lb: emit per-step queue and KV-block diagnostics when explicitly enabled.
        if diagnostics_enabled(self.vllm_config):
            print_scheduler_summary(self, scheduler_output)
        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)


class AsyncRecomputeScheduler(AsyncScheduler, RecomputeScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# delta for dyntra_lb: combine the synchronous recompute scheduler with the DyntraLB policy.
class DyntraLBRecomputeScheduler(DyntraLBPolicyMixin, RecomputeScheduler):
    pass


# delta for dyntra_lb: combine async recompute scheduling with the same DyntraLB policy.
class AsyncDyntraLBRecomputeScheduler(DyntraLBPolicyMixin, AsyncRecomputeScheduler):
    pass
