# mypy: ignore-errors
#
# custom Scheduler.schedule and Scheduler.update_from_output patch for
# vllm 0.30.0
# vllm Ascend rfc/vllm_cann
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import KVEventsConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    KVConnectorBlockState,
    NewRequestData,
    ScheduledEncoderInputStats,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    get_mamba_prefill_checkpoint_position,
    is_mamba_prefill_checkpoint_valid,
)
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import (
    PrefixCacheStats,
    RequestSpecDecodeMetrics,
    SchedulerStats,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.core.sched import scheduler as scheduler_mod

logger = init_logger(__name__)
SCHED_PATCHED = False


def custom_schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
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
    # Whether the running batch contains any prefill requests.
    prefill_scheduled = False
    # Whether any scheduled request has a synchronous connector KV load.
    has_sync_kv_loads = False

    # For logging.
    scheduled_timestamp = time.monotonic()

    self.kv_cache_manager.new_step_starts()

    # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
    # all prefill compute unless saturated.
    defer_prefills = (
        throttle_prefills and not self.prefill_capacity_bound
    ) and any(not r.is_prefill_chunk for r in self.running)

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
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(
            num_new_tokens, token_budget, input_budget - draft_slots
        )

        # Make sure the input position does not exceed the max model len.
        # This is necessary when using spec decoding.
        num_new_tokens = min(
            num_new_tokens,
            self.max_model_len
            - request.num_computed_tokens
            - self.num_sampled_tokens_per_step,
        )

        # Apply Mamba alignment before encoder caps.
        if self.need_mamba_block_aligned_split:
            num_new_tokens = self._mamba_block_aligned_split(
                request, num_new_tokens
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
                shift_computed_tokens=self.num_prefill_lookahead,
            )

        # Multi-module MTP: avoid ending a prefill chunk within
        # num_prefill_lookahead of the prefill end.
        num_new_tokens = self._reserve_prefill_lookahead(
            request, request.num_computed_tokens, num_new_tokens
        )

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

                if (
                    self.connector is not None
                    and self.connector.has_pending_block_frees()
                ):
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
                        preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                            preempted_req_id, None
                        )
                        if preempted_encoder_inputs:
                            # Restore encoder compute budget if the preempted
                            # request had encoder inputs scheduled in this step.
                            num_embeds_to_restore = sum(
                                preempted_req.get_num_encoder_embeds(i)
                                for i in preempted_encoder_inputs
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

        # Speculative decode related.
        if request.spec_token_ids:
            num_scheduled_spec_tokens = (
                num_new_tokens
                + request.num_computed_tokens
                - request.num_tokens
                - request.num_output_placeholders
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
            if input_budget <= draft_slots:
                break
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
            if self._is_blocked_waiting_status(
                request.status
            ) and not self._try_promote_blocked_waiting_request(request):
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    logger.debug(
                        "%s is still in WAITING_FOR_REMOTE_KVS state.",
                        request_id,
                    )
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue

            if (
                request.num_stale_output_tokens > 0
                and not request.drop_stale_output
            ):
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
                (
                    new_computed_blocks,
                    num_new_local_computed_tokens,
                    request.shared_prefix_boundary,
                    hit_diverged,
                ) = self._get_local_prefix_cache_hit(request)

                # Get externally-cached tokens if using a KVConnector.
                if self.connector is not None:
                    # Present a block-aligned local hit to the connector so
                    # a strictly longer remote hit can supersede a local
                    # sub-block tail without racing its copy-on-write.
                    partial_tail = num_new_local_computed_tokens % self.block_size
                    block_aligned_local = (
                        num_new_local_computed_tokens - partial_tail
                    )
                    ext_tokens, load_kv_async = (
                        self.connector.get_num_new_matched_tokens(
                            request, block_aligned_local
                        )
                    )

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
                        new_computed_blocks = (
                            self.kv_cache_manager.truncate_computed_blocks(
                                new_computed_blocks, block_aligned_local
                            )
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

                    connector_prefix_cache_queries = (
                        request.num_tokens - num_new_local_computed_tokens
                    )
                    connector_prefix_cache_hits = num_external_computed_tokens

                # Total computed tokens (local + external).
                num_computed_tokens = (
                    num_new_local_computed_tokens + num_external_computed_tokens
                )
                assert num_computed_tokens <= request.num_tokens

                # Skip request with pending mm encoding prefetches
                if self._ec_transfer_pending(request, num_computed_tokens):
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
                    params = request.kv_transfer_params or {}
                    if params.get("do_remote_prefill"):
                        request.prefill_stats.num_cached_tokens = (
                            params.get("remote_num_cached_tokens", 0)
                        )
            else:
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed. A streaming-input
                # session resumes here too, carrying whatever media its
                # latest chunk added, so this branch needs the same gate.
                new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                num_new_local_computed_tokens = 0
                num_computed_tokens = request.num_computed_tokens

                if self._ec_transfer_pending(request, num_computed_tokens):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

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
                request_token_budget = min(token_budget, input_budget - draft_slots)
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
                        num_computed_tokens
                        + padded_num_tokens
                        + self.num_sampled_tokens_per_step
                        <= self.max_model_len
                    ):
                        if padded_num_tokens > request_token_budget:
                            # Prefer to not schedule than schedule un-padded.
                            break
                        num_new_tokens = padded_num_tokens
                        pad_spec_decode = True

                threshold = self.scheduler_config.long_prefill_token_threshold
                if 0 < threshold < num_new_tokens:
                    num_new_tokens = threshold

                # chunked prefill has to be enabled explicitly to allow
                # pooling requests to be chunked
                if (
                    not self.scheduler_config.enable_chunked_prefill
                    and num_new_tokens > request_token_budget
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
                    if (
                        pad_spec_decode
                        and num_new_tokens != 1 + self.num_spec_tokens
                    ):
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
                    ) = self._try_schedule_encoder_inputs(
                        request,
                        num_computed_tokens,
                        num_new_tokens,
                        encoder_compute_budget,
                        shift_computed_tokens=self.num_prefill_lookahead,
                    )

                # Multi-module MTP: avoid ending a prefill chunk within
                # num_prefill_lookahead of the prefill end.
                num_new_tokens = self._reserve_prefill_lookahead(
                    request, num_computed_tokens, num_new_tokens
                )

                if num_new_tokens == 0:
                    # The request cannot be scheduled.
                    break

            # During async KV load, no forward pass is run yet.
            # Allocate speculative lookahead slots later to avoid
            # mismatching local and remote block counts.
            limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
            effective_lookahead_tokens = (
                0 if limit_lookahead_tokens else self.num_lookahead_tokens
            )

            # Determine if we need to allocate cross-attention blocks.
            num_encoder_tokens = 0
            if (
                self.is_encoder_decoder
                and request.has_encoder_inputs
                and encoder_inputs_to_schedule
            ):
                num_encoder_tokens = sum(
                    request.get_num_encoder_embeds(i)
                    for i in encoder_inputs_to_schedule
                )

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
                if (
                    self.connector_prefix_cache_stats is not None
                    and connector_prefix_cache_queries != 0
                ):
                    self.connector_prefix_cache_stats.record(
                        num_tokens=connector_prefix_cache_queries,
                        num_hits=connector_prefix_cache_hits,
                        preempted=request.num_preemptions > 0,
                    )

            # Record at admission so unscheduled lookups are not counted.
            if did_prefix_cache_lookup:
                self.kv_cache_manager.record_prefix_cache_stats(
                    request, num_new_local_computed_tokens
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
            if num_external_computed_tokens > 0:
                # load_kv_async is False here
                has_sync_kv_loads = True
            if self.log_stats:
                request.record_event(
                    EngineCoreEventType.SCHEDULED, scheduled_timestamp
                )
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)
            req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                request_id
            )
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            if pad_spec_decode:
                assert num_new_tokens == 1 + self.num_spec_tokens
                scheduled_spec_decode_tokens[request_id] = [
                    -1
                ] * self.num_spec_tokens
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
    assert input_budget >= 0
    assert len(self.running) <= self.max_num_running_reqs
    # Since some requests in the RUNNING queue may not be scheduled in
    # this step, the total number of scheduled requests can be smaller than
    # len(self.running).
    assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
        scheduled_running_reqs
    ) <= len(self.running)

    # Get the longest common prefix among all requests in the running queue.
    # This can be potentially used for cascade attention.
    num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
    with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
        if self.running:
            any_request_id = self.running[0].request_id
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
            )

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
        block_state_req_ids.update(
            req_id for req_id in boundary_state_offloads if req_id in self.requests
        )
        kv_connector_block_state = KVConnectorBlockState(
            req_ids=block_state_req_ids,
            resolve_block_ids=self.kv_cache_manager.get_block_ids,
            boundary_state_offloads=boundary_state_offloads,
        )

    kv_cache_block_copies, cow_retained_blocks = (
        self.kv_cache_manager.take_kv_cache_block_copies()
    )
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
        num_spec_tokens_to_schedule = self.dynamic_sd_lookup[
            len(num_scheduled_tokens)
        ]

    scheduled_encoder_input_stats = None
    if (
        self.log_stats
        and self.observability_config.enable_logging_iteration_details
    ):
        scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(
            scheduled_encoder_inputs
        )

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
        has_sync_kv_loads=has_sync_kv_loads,
        kv_cache_block_copies=pending_kv_cache_block_copies,
        kv_connector_block_state=kv_connector_block_state,
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
        ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
            scheduler_output
        )
        scheduler_output.ec_connector_metadata = ec_meta

    # Connector-only block state must not be dispatched to workers.
    scheduler_output.kv_connector_block_state = None

    # Advance the fence only for non-empty steps (those that actually
    # write KV and have their output processed later in update_from_output).
    if self.defer_block_free and total_num_scheduled_tokens > 0:
        self.sched_step_seq += 1

    with record_function_or_nullcontext("schedule: update_after_schedule"):
        self._update_after_schedule(scheduler_output)
    return scheduler_output

def custom_update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
    sampled_token_ids = model_runner_output.sampled_token_ids
    logprobs = model_runner_output.logprobs
    prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    pooler_outputs = model_runner_output.pooler_output
    num_nans_in_logits = model_runner_output.num_nans_in_logits
    kv_connector_output = model_runner_output.kv_connector_output
    ec_connector_output = model_runner_output.ec_connector_output
    cudagraph_stats = model_runner_output.cudagraph_stats

    # Every GPU write enqueued by this and earlier steps has completed, so it is
    # safe to return deferred-free blocks to the pool.
    if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
        self.processed_step_seq += 1
        self._drain_deferred_frees()

    perf_stats: PerfStats | None = None
    if self.perf_metrics and self.perf_metrics.is_enabled():
        perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
    spec_decoding_stats: SpecDecodingStats | None = None

    failed_kv_load_req_ids: set[str] = set()
    if kv_connector_output and kv_connector_output.failed_recving:
        failed_kv_load_req_ids.update(
            self._handle_failed_recving(kv_connector_output.failed_recving)
        )
    if kv_connector_output and kv_connector_output.invalid_block_ids:
        # These blocks contain externally computed tokens that failed to
        # load. Identify affected requests and adjust their computed token
        # count to trigger recomputation of the invalid blocks.
        failed_kv_load_req_ids.update(
            self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )
        )

    # Persist per-step routed experts into the scheduler-side slot
    # buffer (CPU->CPU fancy-index assign; ~few MB per step).
    # MUST precede the per-request routing reads below: stopped
    # requests may terminate on tokens generated in this very step,
    # whose routing was just D2H'd into model_runner_output.
    routing_data = None
    routing_offsets: dict[str, int] = {}
    if model_runner_output.routed_experts is not None:
        re = model_runner_output.routed_experts
        self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
        routing_data = re.routing_data.astype(
            self.routed_experts_mgr.routed_experts_by_slot.dtype,
            copy=False,
        )
        # Build offset map using model runner's request order
        # (input_batch ordering), NOT scheduler dict order.
        offset = 0
        for rid in model_runner_output.req_ids:
            routing_offsets[rid] = offset
            offset += num_scheduled_tokens[rid]

    # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
    # the below loop can be a performance bottleneck. We should do our best
    # to avoid expensive operations inside the loop.
    stopped_running_reqs: set[Request] = set()
    stopped_preempted_reqs: set[Request] = set()
    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
        assert num_tokens_scheduled > 0
        request = self.requests.get(req_id)
        output_is_stale = False
        if request is not None:
            request.num_in_flight_tokens -= num_tokens_scheduled
            # Drain any stale share (see _preempt_request) in lockstep.
            if request.num_stale_output_tokens > 0:
                output_is_stale = True
                request.num_stale_output_tokens -= num_tokens_scheduled
                assert request.num_stale_output_tokens >= 0
        if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
            # skip failed or rescheduled requests from KV load failure
            continue
        if request is None or request.is_finished():
            # The request is already finished. This can happen if the
            # request is aborted while the model is executing it (e.g.,
            # in pipeline parallelism or in async scheduling).
            # NOTE(Kuntai): When delay_free_blocks=True (for async KV
            # cache transfer in KV connector), the aborted request will not
            # be set to None (in order to finish async KV transfer).
            # In this case, we use is_finished() to check.
            continue

        # Drop-mode stale output (same-step resume) is discarded entirely.
        if output_is_stale and request.drop_stale_output:
            continue

        req_index = model_runner_output.req_id_to_index[req_id]
        generated_token_ids = (
            sampled_token_ids[req_index] if sampled_token_ids else []
        )

        scheduled_spec_token_ids = (
            scheduler_output.scheduled_spec_decode_tokens.get(req_id)
        )
        if scheduled_spec_token_ids and (
            generated_token_ids or self.num_sampled_tokens_per_step == 0
        ):
            num_draft_tokens = len(scheduled_spec_token_ids)
            num_sampled = self.num_sampled_tokens_per_step
            num_accepted = max(len(generated_token_ids) - num_sampled, 0)
            num_rejected = num_draft_tokens - num_accepted
            # Rejections roll back num_computed_tokens (and, under async
            # scheduling, num_output_placeholders, which covers the spec
            # tokens). A stale rejection count predates the preemption
            # rollback and must not apply.
            if not output_is_stale:
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
            spec_decoding_stats = self.make_spec_decoding_stats(
                spec_decoding_stats,
                num_draft_tokens=num_draft_tokens,
                num_accepted_tokens=num_accepted,
                num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                request_id=req_id,
            )
            if request.spec_decode_metrics is not None:
                # Exclude grammar-invalidated drafts from the proposed
                # count, mirroring make_spec_decoding_stats; the accepted
                # bucket (j) is unaffected.
                adj_draft_tokens = num_draft_tokens
                if scheduler_output.num_invalid_spec_tokens:
                    adj_draft_tokens -= (
                        scheduler_output.num_invalid_spec_tokens.get(req_id, 0)
                    )
                request.spec_decode_metrics.observe(
                    num_draft_tokens=adj_draft_tokens,
                    num_accepted=num_accepted,
                    detailed=self.spec_decode_metrics_level == "detailed",
                )

        # Free encoder inputs only after the step has actually executed.
        if request.has_encoder_inputs:
            self._free_encoder_inputs(request)

        stopped = False
        new_logprobs = None
        new_sampling_mask = None
        new_token_ids = generated_token_ids
        pooler_output = pooler_outputs[req_index] if pooler_outputs else None
        kv_transfer_params = None
        ec_transfer_params = None
        prefill_stats = None
        status_before_stop = request.status
        num_output_tokens_before = len(request._output_token_ids)

        # Check for stop and update request status.
        if new_token_ids:
            new_token_ids, stopped = self._update_request_with_output(
                request, new_token_ids, is_stale=output_is_stale
            )
        elif request.pooling_params and pooler_output is not None:
            # Pooling stops as soon as there is output.
            request.status = RequestStatus.FINISHED_STOPPED
            stopped = True
        elif (
            self.is_mm_encoder_only
            and request.num_computed_tokens >= request.num_prompt_tokens
        ):
            # An encoder instance runs the encoder and publishes the
            # embeddings instead of sampling, so it stops as soon as the
            # whole prompt is consumed. Encoder inputs are never scheduled
            # past a multi-modal item the encoder cache could not admit, so
            # a consumed prompt also means every item in it was encoded.
            request.status = RequestStatus.FINISHED_STOPPED
            stopped = True

        if new_token_ids and not self.structured_output_manager.accept_tokens(
            request, new_token_ids
        ):
            logger.error(
                "Unexpected: grammar rejected tokens %s for request %s. "
                "Terminating request.",
                new_token_ids,
                req_id,
            )
            request.status = RequestStatus.FINISHED_ERROR
            request.resumable = False
            stopped = True

        routed_experts = None
        if (
            self.enable_return_routed_experts
            and routing_data is not None
            and new_token_ids
        ):
            req_offset = routing_offsets[req_id]
            end = req_offset + num_tokens_scheduled
            block_ids = self._re_block_ids.pop(req_id, [])
            if num_output_tokens_before == 0:
                # Prefill completed: read full prompt routing from
                # slot buffer using the block-ID snapshot taken at
                # schedule time (immune to async preemption).
                if (
                    request.sampling_params is not None
                    and request.sampling_params.routed_experts_prompt_start
                    is not None
                ):
                    prompt_start = (
                        request.sampling_params.routed_experts_prompt_start
                    )
                    assert prompt_start <= request.num_prompt_tokens
                else:
                    prompt_start = 0
                routed_experts = self.routed_experts_mgr.get(
                    block_ids,
                    request.num_prompt_tokens,
                    token_start=prompt_start,
                )
            else:
                if scheduled_spec_token_ids:
                    # Spec decode: accepted tokens at the START of
                    # the scheduled range, rejected at the end.
                    routed_experts = routing_data[
                        req_offset : req_offset + len(new_token_ids)
                    ]
                else:
                    # Normal decode / re-prefill: token(s) at the END.
                    routed_experts = routing_data[end - len(new_token_ids) : end]

        should_emit_output = bool(
            new_token_ids or pooler_output is not None or stopped
        )
        if should_emit_output:
            prefill_stats = request.take_prefill_stats()
            if prefill_stats is not None:
                params = request.kv_transfer_params
                if params is not None:
                    params["_remote_num_cached_tokens"] = int(
                        prefill_stats.num_cached_tokens
                    )
                prefill_stats.finalize(
                    self.kv_cache_manager.estimate_cached_tokens(request)
                )

        finish_reason = None
        if stopped:
            # Capture finish_reason BEFORE _handle_stopped_request, which may
            # reset the status to WAITING for streaming requests that continue.
            finish_reason = request.get_finished_reason()
            finished = self._handle_stopped_request(request)
            if finished:
                kv_transfer_params, ec_transfer_params = self._free_request(request)

            if status_before_stop == RequestStatus.RUNNING:
                stopped_running_reqs.add(request)
            else:
                stopped_preempted_reqs.add(request)

        # Extract sample logprobs if needed.
        if (
            request.sampling_params is not None
            and request.sampling_params.num_logprobs is not None
            and logprobs
        ):
            new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

        if self.return_sampling_mask:
            sampling_masks = model_runner_output.sampling_masks
            if new_token_ids and sampling_masks is not None:
                new_sampling_mask = sampling_masks.slice_request(
                    req_index, len(new_token_ids)
                )

        if num_nans_in_logits is not None and req_id in num_nans_in_logits:
            request.num_nans_in_logits = num_nans_in_logits[req_id]

        # Get prompt logprobs for this request.
        prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
        if should_emit_output:
            # Add EngineCoreOutput for this Request.
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=new_token_ids,
                    finish_reason=finish_reason,
                    new_logprobs=new_logprobs,
                    new_sampling_mask=new_sampling_mask,
                    new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                    pooling_output=pooler_output,
                    stop_reason=request.stop_reason,
                    events=request.take_events(),
                    prefill_stats=prefill_stats,
                    spec_decode_metrics=(
                        request.spec_decode_metrics
                        if finish_reason is not None
                        else None
                    ),
                    kv_transfer_params=kv_transfer_params,
                    ec_transfer_params=ec_transfer_params,
                    trace_headers=request.trace_headers,
                    routed_experts=routed_experts,
                    num_nans_in_logits=request.num_nans_in_logits,
                )
            )
        else:
            # Invariant: EngineCore returns no partial prefill outputs.
            assert not prompt_logprobs_tensors

    # Remove the stopped requests from the running and waiting queues.
    if stopped_running_reqs:
        self.running = remove_all(self.running, stopped_running_reqs)
    if stopped_preempted_reqs:
        # This is a rare case and unlikely to impact performance.
        self.waiting.remove_requests(stopped_preempted_reqs)
        self.skipped_waiting.remove_requests(stopped_preempted_reqs)

    error_req_ids = set(self.grammar_compile_error_reqs)
    self.grammar_compile_error_reqs.clear()
    if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
        error_req_ids.update(failed_kv_load_req_ids)
    if self.ec_connector is not None:
        # An encoder input the connector can no longer obtain. Failing is
        # retryable: re-issuing the request re-runs the encode.
        error_req_ids.update(self.ec_connector.take_unavailable_requests())

    if error_req_ids:
        error_reqs = self.finish_requests(
            error_req_ids, RequestStatus.FINISHED_ERROR
        )
        for request in error_reqs:
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                )
            )

    # KV Connector: update state for finished KV Transfers.
    if kv_connector_output:
        self._update_from_kv_xfer_finished(kv_connector_output)

    # EC Connector: update state from worker-side EC connector output.
    if self.ec_connector is not None and ec_connector_output:
        self.ec_connector.update_connector_output(ec_connector_output)

    # Worker-side KV connector stats from the model runner output.
    kv_connector_stats: KVConnectorStats | None = (
        kv_connector_output.kv_connector_stats if kv_connector_output else None
    )
    if self.connector:
        # Scheduler-side KV connector stats collected after connector update.
        scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
        if (
            scheduler_kv_connector_stats is not None
            and not scheduler_kv_connector_stats.is_empty()
        ):
            kv_connector_stats = (
                kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                if kv_connector_stats is not None
                else scheduler_kv_connector_stats
            )

    # collect KV cache events from KV cache manager
    events = self.kv_cache_manager.take_events()

    # collect KV cache events from connector
    if self.connector is not None:
        connector_events = self.connector.take_events()
        if connector_events:
            if events is None:
                events = list(connector_events)
            else:
                events.extend(connector_events)

    # publish collected KV cache events
    if events:
        batch = KVEventBatch(ts=time.time(), events=events)
        self.kv_event_publisher.publish(batch)

    # Create EngineCoreOutputs for all clients that have requests with
    # outputs in this step.
    engine_core_outputs = {
        client_index: EngineCoreOutputs(outputs=outs)
        for client_index, outs in outputs.items()
    }

    finished_req_ids = self.finished_req_ids_dict
    if finished_req_ids:
        # Include ids of requests that finished since last outputs
        # were sent.
        for client_index, finished_set in finished_req_ids.items():
            # Set finished request set in EngineCoreOutputs for this client.
            if (eco := engine_core_outputs.get(client_index)) is not None:
                eco.finished_requests = finished_set
            else:
                engine_core_outputs[client_index] = EngineCoreOutputs(
                    finished_requests=finished_set
                )
        finished_req_ids.clear()

    if (
        stats := self.make_stats(
            spec_decoding_stats,
            kv_connector_stats,
            cudagraph_stats,
            perf_stats,
        )
    ) is not None:
        # Return stats to only one of the front-ends.
        if (eco := next(iter(engine_core_outputs.values()), None)) is None:
            # We must return the stats even if there are no request
            # outputs this step.
            engine_core_outputs[0] = eco = EngineCoreOutputs()
        eco.scheduler_stats = stats

    return engine_core_outputs

def apply_patch() -> None:
    global SCHED_PATCHED
    if SCHED_PATCHED:
        return
    scheduler_mod.Scheduler.schedule = custom_schedule
    scheduler_mod.Scheduler.update_from_output = custom_update_from_output

apply_patch()