#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Scheduler subclass with profiling-based dynamic chunk sizing.

The ``schedule()`` override below is re-based on the ``Scheduler.schedule()``
of vLLM v0.29.0 and kept compatible with vLLM v0.30.0 and main through
``vllm_version_is`` branches and capability checks.  When the upstream
``schedule()`` method is refactored, this override must be updated accordingly.
"""

import inspect
import time

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import (
    KVConnectorBlockState,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend.core.profiling_chunk_predictor import ProfilingChunkManager
from vllm_ascend.utils import vllm_version_is


class ProfilingChunkScheduler(Scheduler):
    """Scheduler with profiling-based dynamic chunk sizing.

    During initialization, the scheduler profiles prefill latency at various
    chunk sizes by calling ``profile_prefill_latency`` on each worker via
    ``collective_rpc``.  A quadratic latency model is then fitted, and during
    scheduling the model predicts the optimal chunk size for each waiting
    request based on its ``num_computed_tokens``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        # `hash_block_size` was added in vLLM #40946; keep it optional so the
        # subclass works on both pinned vllm and main.
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config

        init_ascend_config(vllm_config)
        scheduler_extension_config = get_ascend_config().scheduler_config

        profiling_cfg = scheduler_extension_config.profiling_chunk_config
        self.profiling_chunk_config = profiling_cfg

        short_request_first_config = scheduler_extension_config.short_request_first_config
        self._short_request_first_enabled = short_request_first_config.enabled

        if self._short_request_first_enabled:
            from vllm_ascend.core.short_request_first_scheduler import (
                install_short_request_first_waiting_queue,
            )

            install_short_request_first_waiting_queue(
                self,
                threshold=short_request_first_config.threshold,
                long_max_wait_ms=short_request_first_config.long_max_wait_ms,
            )
        base_chunk = self.max_num_scheduled_tokens

        self.profiling_chunk_manager = ProfilingChunkManager(
            base_chunk_size=base_chunk,
            page_size=self.cache_config.block_size,
            smooth_factor=profiling_cfg.smooth_factor,
            min_chunk=profiling_cfg.min_chunk,
            max_fit_chunk=profiling_cfg.max_fit_chunk,
        )
        self._profiling_initialized = False

        logger.info(
            "[ProfilingChunk] Scheduler initialized. base_chunk=%d, page_size=%d, smooth_factor=%.2f, min_chunk=%d",
            base_chunk,
            self.cache_config.block_size,
            profiling_cfg.smooth_factor,
            profiling_cfg.min_chunk,
        )

    # ------------------------------------------------------------------
    # Profiling initialization
    # ------------------------------------------------------------------

    def run_profiling_chunk_init(self, model_executor) -> None:
        """Profile prefill latency using real model forward passes.

        Called by EngineCore after model_executor is ready.  Collects latency
        samples at different chunk sizes and fits the quadratic model.
        """
        if self._profiling_initialized:
            return
        self._profiling_initialized = True

        if model_executor is None:
            logger.warning("[ProfilingChunk] No model_executor provided, skipping profiling")
            return

        logger.info("[ProfilingChunk] Running startup profiling with real model forward...")

        seq_lens: list[int] = []
        latencies: list[float] = []

        base_chunk_size = self.profiling_chunk_manager.base_chunk_size
        num_samples = 64

        # Determine unique_reply_rank for PP setups
        rpc_kwargs = self._build_rpc_kwargs(model_executor)

        total_steps = num_samples + 1
        log_interval = max(1, total_steps // 10)
        t_start = time.perf_counter()

        for i in range(total_steps):
            chunk_size = int(base_chunk_size - (i - 1) * (base_chunk_size / num_samples))
            if chunk_size <= 0:
                break

            if i % log_interval == 0 or i == total_steps - 1:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "[ProfilingChunk] Profiling prefill latency: %d/%d samples done (chunk=%d, elapsed=%.1fs)",
                    max(i - 1, 0),
                    num_samples,
                    chunk_size,
                    elapsed,
                )

            try:
                result = model_executor.collective_rpc(
                    "profile_prefill_latency",
                    args=(chunk_size,),
                    **rpc_kwargs,
                )

                # First iteration is warm-up
                if i == 0:
                    continue

                latency_ms = self._extract_latency(result)
                if latency_ms is None:
                    continue

                seq_lens.append(chunk_size)
                latencies.append(latency_ms)

            except Exception as e:
                logger.debug(
                    "[ProfilingChunk] Forward failed for chunk=%d: %s",
                    chunk_size,
                    e,
                )
                continue

        if len(seq_lens) < 8:
            logger.warning(
                "[ProfilingChunk] Profiling failed: only %d/8 samples collected",
                len(seq_lens),
            )
            return

        logger.info(
            "[ProfilingChunk] Collected %d samples. Latency range: [%.2f, %.2f] ms",
            len(seq_lens),
            min(latencies),
            max(latencies),
        )

        predictor = self.profiling_chunk_manager.predictor
        if not predictor.fit(seq_lens, latencies):
            return

        predictor.set_target_latency(base_chunk_size)
        predictor.is_ready = True
        self.profiling_chunk_manager._profiling_done = True

        logger.info("[ProfilingChunk] Profiling completed successfully")

    @staticmethod
    def _build_rpc_kwargs(model_executor) -> dict:
        """Build kwargs for collective_rpc, handling PP unique_reply_rank."""
        kwargs: dict = {}
        if not hasattr(model_executor, "collective_rpc"):
            return kwargs

        sig = inspect.signature(model_executor.collective_rpc)
        if "unique_reply_rank" not in sig.parameters:
            return kwargs

        try:
            pc = model_executor.vllm_config.parallel_config
            output_rank = pc.world_size - pc.tensor_parallel_size * pc.prefill_context_parallel_size
            kwargs["unique_reply_rank"] = output_rank
        except AttributeError:
            pass

        return kwargs

    @staticmethod
    def _extract_latency(result) -> float | None:
        """Extract latency value from collective_rpc result."""
        if isinstance(result, (int, float)):
            return float(result)
        if isinstance(result, list) and len(result) > 0:
            return float(result[0])
        return None

    # ------------------------------------------------------------------
    # schedule() override
    # ------------------------------------------------------------------
    # The method below is based on the upstream Scheduler.schedule()
    # with profiling-based chunk sizing applied to both RUNNING requests
    # (chunked prefill continuation) and WAITING requests (new prefill).
    # Modified sections are marked with ">>> PROFILING CHUNK" comments.
    # ------------------------------------------------------------------

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:  # noqa: C901
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
        # >>> PROFILING CHUNK >>>
        target_latency = self.profiling_chunk_manager.predictor.target_latency
        time_budget = target_latency if target_latency is not None else float("inf")
        # <<< PROFILING CHUNK <<<
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
        defer_prefills = (throttle_prefills and not self.prefill_capacity_bound) and any(
            not r.is_prefill_chunk for r in self.running
        )

        # First, schedule the RUNNING requests.
        req_index = 0
        # >>> PROFILING CHUNK >>>
        while req_index < len(self.running) and token_budget > 0 and time_budget > 0:
            # <<< PROFILING CHUNK <<<
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

            if not vllm_version_is("0.29.0") and (
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
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
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

            # >>> PROFILING CHUNK: dynamic chunk sizing for RUNNING >>>
            if (
                self.profiling_chunk_manager is not None
                and self.profiling_chunk_manager.is_ready
                and request.num_computed_tokens < request.num_prompt_tokens
                and (request.num_computed_tokens > 0 or not self.profiling_chunk_config.need_timing)
            ):
                predicted_chunk = self.profiling_chunk_manager.predict_chunk_size(
                    num_computed_tokens=request.num_computed_tokens,
                    target_time=time_budget,
                )
                if predicted_chunk is not None and predicted_chunk > 0:
                    logger.debug(
                        "[ProfilingChunk] Dynamic chunk for %s: %s -> %s (predicted=%s)",
                        request.request_id,
                        num_new_tokens,
                        min(predicted_chunk, num_new_tokens),
                        predicted_chunk,
                    )
                    num_new_tokens = min(predicted_chunk, num_new_tokens)
                elif self.profiling_chunk_config.need_timing:
                    logger.info("[Dynamic Chunk] Online calibration stage. Long requests are better")
                elif time_budget == target_latency:
                    logger.warning_once(
                        "[Dynamic Chunk] Profiling Failed. Degenerated to a fixed chunk size"
                        "Please increase the `max_fit_chunk` to profile more data"
                    )
                else:
                    break
            # <<< PROFILING CHUNK <<<

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

                    # vLLM 0.30.0 can temporarily fail allocation while a KV
                    # connector still owns blocks pending deferred release.
                    # Preempting another request cannot make progress in that
                    # state. Use capability detection to retain compatibility
                    # with older connector implementations.
                    has_pending_block_frees = (
                        getattr(self.connector, "has_pending_block_frees", None) if self.connector is not None else None
                    )
                    if has_pending_block_frees is not None and has_pending_block_frees():
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if vllm_version_is("0.29.0"):
                        if self.policy == SchedulingPolicy.PRIORITY:
                            preempted_req = max(
                                self.running,
                                key=lambda r: (r.priority, r.arrival_time),
                            )
                            # Record the index of the preemption victim to
                            # maintain accurate loop state.
                            victim_index = self.running.index(preempted_req)
                            del self.running[victim_index]
                            # Decrement the loop cursor if the removed request
                            # preceded the current iteration, preventing the
                            # silent omission of the subsequent request.
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
                    else:
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
            # >>> PROFILING CHUNK >>>
            # Decode requests (num_new_tokens == 1) have negligible latency;
            # skip time_budget accounting so they don't starve other requests.
            if request.num_computed_tokens < request.num_prompt_tokens:
                time_budget -= self.profiling_chunk_manager.predict_time(num_new_tokens, request.num_computed_tokens)
            # <<< PROFILING CHUNK <<<
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

            # >>> PROFILING CHUNK >>>
            while (self.waiting or self.skipped_waiting) and token_budget > 0 and time_budget > 0:
                # <<< PROFILING CHUNK <<<
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
                        block_aligned_local = num_new_local_computed_tokens - partial_tail
                        ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                            request, block_aligned_local
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
                    if vllm_version_is("0.29.0"):
                        if (
                            self.ec_connector is not None
                            and request.mm_features
                            and not self.ec_connector.ensure_cache_available(request, num_computed_tokens)
                        ):
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                    elif self._ec_transfer_pending(request, num_computed_tokens):
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

                    if not vllm_version_is("0.29.0") and self._ec_transfer_pending(request, num_computed_tokens):
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
                    if vllm_version_is("0.29.0"):
                        if (
                            (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                            and self.num_sampled_tokens_per_step > 0
                            and num_new_tokens == 1
                            and (scheduled_running_reqs and not prefill_scheduled)
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
                    elif (
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

                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # >>> PROFILING CHUNK: dynamic chunk sizing >>>
                    if (
                        self.profiling_chunk_manager is not None
                        and self.profiling_chunk_manager.is_ready
                        and request.num_computed_tokens < request.num_prompt_tokens
                        and (request.num_computed_tokens > 0 or not self.profiling_chunk_config.need_timing)
                    ):
                        predicted_chunk = self.profiling_chunk_manager.predict_chunk_size(
                            num_computed_tokens=num_computed_tokens,
                            target_time=time_budget,
                        )
                        if predicted_chunk is not None and predicted_chunk > 0:
                            num_new_tokens = min(num_new_tokens, predicted_chunk)
                        elif self.profiling_chunk_config.need_timing:
                            logger.info("[Dynamic Chunk] Online calibration stage. Long requests are better")
                        elif time_budget == target_latency:
                            logger.warning_once(
                                "[Dynamic Chunk] Profiling Failed. Degenerated to a fixed chunk size"
                                "Please increase the `max_fit_chunk` to profile more data"
                            )
                        else:
                            break
                    # <<< PROFILING CHUNK <<<

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > request_token_budget:
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
                        # The request cannot be scheduled.
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
                # >>> PROFILING CHUNK >>>
                # Decode requests (num_new_tokens == 1) have negligible latency;
                # skip time_budget accounting so they don't starve other requests.
                if request.num_computed_tokens < request.num_prompt_tokens:
                    time_budget -= self.profiling_chunk_manager.predict_time(
                        num_new_tokens, request.num_computed_tokens
                    )
                # <<< PROFILING CHUNK <<<
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if pad_spec_decode:
                    assert num_new_tokens == 1 + self.num_spec_tokens
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
        assert input_budget >= 0
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
            if vllm_version_is("0.29.0"):
                new_reqs_data = [
                    NewRequestData.from_request(
                        req,
                        req_to_new_blocks[req.request_id].get_block_ids(),
                        req._all_token_ids,
                        uses_mrope=self.model_uses_mrope,
                        uses_xdrope=self.model_uses_xdrope,
                    )
                    for req in scheduled_new_reqs
                ]
            else:
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
            if vllm_version_is("0.29.0"):
                new_reqs_data = [
                    NewRequestData.from_request(
                        req,
                        req_to_new_blocks[req.request_id].get_block_ids(),
                        uses_mrope=self.model_uses_mrope,
                        uses_xdrope=self.model_uses_xdrope,
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
            if vllm_version_is("0.29.0"):
                snapshot_req_ids = {req.req_id for req in new_reqs_data}
                snapshot_req_ids.update(
                    req_id
                    for req_id, block_ids in zip(
                        cached_reqs_data.req_ids,
                        cached_reqs_data.new_block_ids,
                        strict=True,
                    )
                    if block_ids
                )
                snapshot_req_ids.update(req_id for req_id in boundary_state_offloads if req_id in self.requests)
                kv_connector_block_state = KVConnectorBlockState(
                    block_ids={req_id: self.kv_cache_manager.get_block_ids(req_id) for req_id in snapshot_req_ids},
                    boundary_state_offloads=boundary_state_offloads,
                )
            else:
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
        return scheduler_output


class ProfilingChunkAsyncScheduler(AsyncScheduler, ProfilingChunkScheduler):
    """Profiling-chunk scheduler variant for async scheduling.

    MRO: ``AsyncScheduler`` contributes the output-placeholder accounting
    (``_update_after_schedule`` / ``_update_request_with_output``);
    ``ProfilingChunkScheduler`` contributes ``__init__`` and the copied
    ``schedule()``, which dispatches ``self._update_after_schedule()`` to
    the async implementation.

    Dynamic chunk sizing stays exact under async scheduling: upstream does
    not add output placeholders to prefill-chunk requests, so the
    ``num_computed_tokens`` read by the chunk predictor is the true prefix
    length for every request still in prefill.
    """
