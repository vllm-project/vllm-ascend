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

Compatible with vLLM v0.15.x scheduler.  When the upstream ``schedule()``
method is refactored, this override should be updated accordingly.
"""
import inspect
import time
from typing import List, Optional

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import (
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


class ProfilingChunkScheduler(Scheduler):
    """Scheduler with profiling-based dynamic chunk sizing.

    During initialization, the scheduler profiles prefill latency at various
    chunk sizes by calling ``profile_prefill_latency`` on each worker via
    ``collective_rpc``.  A quadratic latency model is then fitted, and during
    scheduling the model predicts the optimal chunk size for each waiting
    request based on its ``history_len`` (already-computed tokens).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        base_chunk = self.scheduler_config.long_prefill_token_threshold
        if base_chunk <= 0:
            base_chunk = self.max_num_scheduled_tokens

        self.profiling_chunk_manager = ProfilingChunkManager(
            base_chunk_size=base_chunk,
            page_size=self.block_size,
            context_len=self.max_model_len,
            max_prefill_tokens=self.max_num_scheduled_tokens,
        )
        self._profiling_initialized = False

        logger.info(
            "[ProfilingChunk] Scheduler initialized. "
            "base_chunk=%d, page_size=%d",
            base_chunk,
            self.block_size,
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
            logger.warning(
                "[ProfilingChunk] No model_executor provided, "
                "skipping profiling"
            )
            return

        logger.info(
            "[ProfilingChunk] Running startup profiling with real "
            "model forward..."
        )

        seq_lens: List[int] = []
        latencies: List[float] = []

        base_chunk_size = self.profiling_chunk_manager.base_chunk_size
        num_samples = 64

        # Determine unique_reply_rank for PP setups
        rpc_kwargs = self._build_rpc_kwargs(model_executor)

        for i in range(num_samples + 1):
            chunk_size = int(
                base_chunk_size - (i - 1) * (base_chunk_size / num_samples)
            )
            if chunk_size <= 0:
                break

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
                "[ProfilingChunk] Profiling failed: only %d samples "
                "collected",
                len(seq_lens),
            )
            return

        logger.info(
            "[ProfilingChunk] Collected %d samples. "
            "Latency range: [%.2f, %.2f] ms",
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
            output_rank = (
                pc.world_size
                - pc.tensor_parallel_size
                * pc.prefill_context_parallel_size
            )
            kwargs["unique_reply_rank"] = output_rank
        except AttributeError:
            pass

        return kwargs

    @staticmethod
    def _extract_latency(result) -> Optional[float]:
        """Extract latency value from collective_rpc result."""
        if isinstance(result, (int, float)):
            return float(result)
        if isinstance(result, list) and len(result) > 0:
            return float(result[0])
        return None

    # ------------------------------------------------------------------
    # schedule() override
    # ------------------------------------------------------------------
    # The method below is copied from the upstream Scheduler.schedule()
    # (vLLM v0.15.x) with profiling-based chunk sizing applied to both
    # RUNNING requests (chunked prefill continuation) and WAITING
    # requests (new prefill).  Modified sections are marked with
    # ">>> PROFILING CHUNK" comments.
    # ------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:  # noqa: C901
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        dynamic_chunking_full = False
        token_budget = self.max_num_scheduled_tokens
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        scheduled_timestamp = time.monotonic()

        # ---- Schedule RUNNING requests (unchanged from upstream) ----
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            if dynamic_chunking_full:
                break
            request = self.running[req_index]

            if self.use_pp and request.num_output_placeholders > 0:
                req_index += 1
                continue

            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens
                + 2
                - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if (
                0
                < self.scheduler_config.long_prefill_token_threshold
                < num_new_tokens
            ):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold
                )
            num_new_tokens = min(num_new_tokens, token_budget)

            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens,
            )

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

            # >>> PROFILING CHUNK: dynamic chunk sizing for RUNNING >>>
            if (
                self.profiling_chunk_manager is not None
                and self.profiling_chunk_manager.is_ready
                and num_new_tokens > 1
                and request.num_computed_tokens > 0
            ):
                predicted_chunk = (
                    self.profiling_chunk_manager
                    .predict_chunk_size(
                        history_len=request.num_computed_tokens,
                    )
                )
                if (
                    predicted_chunk is not None
                    and predicted_chunk > 0
                ):
                    if predicted_chunk <= num_new_tokens:
                        dynamic_chunking_full = True
                        num_new_tokens = predicted_chunk
            # <<< PROFILING CHUNK <<<

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                req_index += 1
                continue

            with record_function_or_nullcontext(
                "schedule: allocate_slots"
            ):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        break

                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens[
                                preempted_req.request_id
                            ]
                            req_to_new_blocks.pop(
                                preempted_req.request_id
                            )
                            num_scheduled_tokens.pop(
                                preempted_req.request_id
                            )
                            scheduled_spec_decode_tokens.pop(
                                preempted_req.request_id, None
                            )
                            preempted_encoder_inputs = (
                                scheduled_encoder_inputs.pop(
                                    preempted_req.request_id, None
                                )
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += (
                                    num_embeds_to_restore
                                )
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(
                        preempted_req, scheduled_timestamp
                    )
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        break

            if new_blocks is None:
                break

            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    del request.spec_token_ids[
                        num_scheduled_spec_tokens:
                    ]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )
                request.spec_token_ids = []

            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule
                )
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(
                            request, i
                        )

        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request
                and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        skipped_waiting_requests = create_request_queue(self.policy)

        # ---- Schedule WAITING requests ----
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs or dynamic_chunking_full:
                    break

                request = self.waiting.peek_request()

                if (
                    request.status
                    == RequestStatus.WAITING_FOR_REMOTE_KVS
                ):
                    is_ready = self._update_waiting_for_remote_kv(
                        request
                    )
                    if is_ready:
                        if request.num_preemptions:
                            request.status = RequestStatus.PREEMPTED
                        else:
                            request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in "
                            "WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(
                            request
                        )
                        continue

                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = (
                        request.structured_output_request
                    )
                    if (
                        structured_output_req
                        and structured_output_req.grammar
                    ):
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(
                            request
                        )
                        continue

                if (
                    request.status
                    == RequestStatus.WAITING_FOR_STREAMING_REQ
                ):
                    assert not request.streaming_queue
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras)
                        == self.lora_config.max_loras
                        and request.lora_request.lora_int_id
                        not in scheduled_loras
                    )
                ):
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                if request.num_computed_tokens == 0:
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request,
                                num_new_local_computed_tokens,
                            )
                        )

                        if ext_tokens is None:
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(
                                request
                            )
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                    num_computed_tokens = (
                        num_new_local_computed_tokens
                        + num_external_computed_tokens
                    )
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.empty_kv_cache_blocks
                    )
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    num_new_tokens = (
                        request.num_tokens - num_computed_tokens
                    )
                    threshold = (
                        self.scheduler_config.long_prefill_token_threshold
                    )
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # >>> PROFILING CHUNK: dynamic chunk sizing >>>
                    if (
                        self.profiling_chunk_manager is not None
                        and self.profiling_chunk_manager.is_ready
                        and num_new_tokens > 1
                        and request.num_computed_tokens > 0
                    ):
                        predicted_chunk = (
                            self.profiling_chunk_manager
                            .predict_chunk_size(
                                history_len=num_computed_tokens,
                            )
                        )
                        if (
                            predicted_chunk is not None
                            and predicted_chunk > 0
                        ):
                            num_new_tokens = min(
                                num_new_tokens, predicted_chunk
                            )
                    # <<< PROFILING CHUNK <<<

                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

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
                            shift_computed_tokens=(
                                1 if self.use_eagle else 0
                            ),
                        )
                        if num_new_tokens == 0:
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                effective_lookahead_tokens = (
                    0
                    if request.num_computed_tokens == 0
                    else self.num_lookahead_tokens
                )

                num_encoder_tokens = (
                    self._num_encoder_max_input_tokens
                    if self.is_encoder_decoder
                    and request.has_encoder_inputs
                    else 0
                )

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=(
                        num_external_computed_tokens
                    ),
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(
                            request.request_id
                        ),
                        num_external_computed_tokens,
                    )

                request = self.waiting.pop_request()
                if load_kv_async:
                    skipped_waiting_requests.prepend_request(request)
                    request.status = (
                        RequestStatus.WAITING_FOR_REMOTE_KVS
                    )
                    continue

                self._update_connector_prefix_cache_stats(request)

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED,
                        scheduled_timestamp,
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}"
                    )

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(
                        request.lora_request.lora_int_id
                    )
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = (
                    num_new_tokens
                )
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(
                                request, i
                            )

        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # ---- Assertions and output construction (unchanged) ----
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert (
            total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        )

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert len(scheduled_new_reqs) + len(
            scheduled_resumed_reqs
        ) + len(scheduled_running_reqs) <= len(self.running)

        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups
        )
        with record_function_or_nullcontext(
            "schedule: get_num_common_prefix_blocks"
        ):
            if self.running:
                any_request = self.running[0]
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request.request_id
                    )
                )

        if self.use_v2_model_runner:
            scheduled_new_reqs = (
                scheduled_new_reqs + scheduled_resumed_reqs
            )
            scheduled_resumed_reqs = []
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
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext(
            "schedule: make_cached_request_data"
        ):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(
            num_scheduled_tokens.keys()
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={
                req.request_id for req in preempted_reqs
            },
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=(
                self.encoder_cache_manager.get_freed_mm_hashes()
            ),
        )

        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        if self.ec_connector is not None:
            ec_meta = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext(
            "schedule: update_after_schedule"
        ):
            self._update_after_schedule(scheduler_output)
        return scheduler_output
