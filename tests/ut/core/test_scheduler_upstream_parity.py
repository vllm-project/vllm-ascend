# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lock Ascend copied schedule() bodies to upstream vLLM Scheduler.schedule().

Each copied scheduler (BalanceScheduler, DyntraLBScheduler, RecomputeScheduler)
must match the installed upstream ``vllm.v1.core.sched.scheduler.Scheduler.schedule``
modulo only its explicitly tracked expected normalized diff below.
"""

from __future__ import annotations

import ast
import difflib
import importlib.util
import inspect
import textwrap
from pathlib import Path
from typing import Any

import pytest

from vllm_ascend.core.dyntra_lb_scheduler import DyntraLBScheduler
from vllm_ascend.core.recompute_scheduler import RecomputeScheduler
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceScheduler
from vllm_ascend.utils import vllm_version_is

_EXPECTED_BALANCE_DIFF: tuple[str, ...] = (
    "@@",
    " def schedule(self, throttle_prefills: bool=False) -> SchedulerOutput:",
    "+    if not self._balance_enabled:",
    "+        return super().schedule(throttle_prefills)",
    "     self.current_step += 1",
    "     scheduled_new_reqs: list[Request] = []",
    "     scheduled_running_reqs: list[Request] = []",
    "     preempted_reqs: list[Request] = []",
    "     iteration_details = None",
    (
        "-    if self.log_stats and self.observability_config.enable_logging_iteration_details or "
        "self.vllm_config.profiler_config.profiler is not None:"
    ),
    (
        "+    if not vllm_version_is('0.33.0') and (self.log_stats and "
        "self.observability_config.enable_logging_iteration_details or "
        "self.vllm_config.profiler_config.profiler is not None):"
    ),
    "         iteration_details = IterationDetails()",
    "     req_to_new_blocks: dict[str, KVCacheBlocks] = {}",
    "     num_scheduled_tokens: dict[str, int] = {}",
    "@@",
    "             else:",
    "                 step_skipped_waiting.appendleft(request_to_skip)",
    "         num_async_kv_loads = 0",
    "-        max_async_kv_loads = self.scheduler_config.max_async_kv_loads_per_step or float('inf')",
    "+        max_async_kv_loads = get_max_async_kv_loads_per_step(self.scheduler_config) or float('inf')",
    "         take_async_kv = True",
    "         while True:",
    "             num_running = len(self.running) + self.num_waiting_for_streaming_input",
    "@@",
    "                 request_queue = self.kv_holding_waiting",
    "                 if not request_queue and self._pause_state == PauseState.UNPAUSED:",
    "                     request_queue = self._next_waiting_queue(take_async_kv)",
    (
        "-            elif take_async_kv and self.scheduler_config.max_num_inflight_kv_loads and "
        "(self._pause_state == PauseState.UNPAUSED):"
    ),
    (
        "+            elif take_async_kv and get_max_num_inflight_kv_loads(self.scheduler_config) and "
        "(self._pause_state == PauseState.UNPAUSED):"
    ),
    "                 request_queue = self.async_kv_waiting",
    "             else:",
    "                 break",
    "             if not request_queue or (not loads_only and input_budget <= draft_slots):",
    "+                break",
    "+            if max((t.item() for t in self.balance_queue)) == self.max_num_running_reqs:",
    "                 break",
    "             request = request_queue.peek_request()",
    "             request_id = request.request_id",
    "@@",
    "                     partial_tail = num_new_local_computed_tokens % self.block_size",
    "                     block_aligned_local = num_new_local_computed_tokens - partial_tail",
    (
        "                     ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(request, "
        "block_aligned_local)"
    ),
    "-                    if request.skip_reading_external_kv:",
    "+                    if should_skip_external_kv(request):",
    "                         ext_tokens, load_kv_async = (0, False)",
    "                     if ext_tokens is None:",
    "                         skip_request(request_queue)",
    "@@",
    "     assert input_budget >= 0",
    "     assert len(self.running) <= self.max_num_running_reqs",
    "     assert len(scheduled_new_reqs) + len(scheduled_running_reqs) <= len(self.running)",
    (
        "-    new_reqs_data = [NewRequestData.from_request(req, "
        "req_to_new_blocks[req.request_id].get_block_ids(), req.prefill_token_ids, "
        "uses_mrope=self.model_uses_mrope) for req in scheduled_new_reqs]"
    ),
    (
        "+    new_reqs_data = [NewRequestData.from_request(req, "
        "req_to_new_blocks[req.request_id].get_block_ids(), get_new_request_prefill_token_ids(req), "
        "uses_mrope=self.model_uses_mrope) for req in scheduled_new_reqs]"
    ),
    "     with record_function_or_nullcontext('schedule: make_cached_request_data'):",
    (
        "-        cached_reqs_data = self._make_cached_request_data(scheduled_running_reqs, "
        "num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_new_blocks, "
        "iteration_details=iteration_details)"
    ),
    (
        "+        cached_reqs_data = make_cached_request_data(self, scheduled_running_reqs, "
        "num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_new_blocks, "
        "iteration_details=iteration_details)"
    ),
    "     if iteration_details is not None:",
    (
        "         iteration_details.num_prefill_requests = len(num_scheduled_tokens) - "
        "iteration_details.num_decode_requests"
    ),
    (
        "         iteration_details.num_prefill_tokens = total_num_scheduled_tokens - "
        "iteration_details.num_decode_tokens"
    ),
    "@@",
    "     num_spec_tokens_to_schedule = self.num_spec_tokens",
    "     if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:",
    "         num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]",
    (
        "-    scheduler_output = SchedulerOutput(scheduled_new_reqs=new_reqs_data, "
        "scheduled_cached_reqs=cached_reqs_data, num_scheduled_tokens=num_scheduled_tokens, "
        "total_num_scheduled_tokens=total_num_scheduled_tokens, "
        "scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, "
        "scheduled_encoder_inputs=scheduled_encoder_inputs, iteration_details=iteration_details, "
        "preempted_req_ids=self.reset_preempted_req_ids, finished_req_ids=self.finished_req_ids, "
        "free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(), "
        "encoder_input_updates=self._get_encoder_input_updates(num_scheduled_tokens, new_reqs_data, "
        "scheduled_encoder_inputs), new_block_ids_to_zero=self._get_new_block_ids_to_zero(), "
        "has_sync_kv_loads=has_sync_kv_loads, kv_cache_block_copies=pending_kv_cache_block_copies, "
        "kv_connector_block_state=kv_connector_block_state, "
        "num_spec_tokens_to_schedule=num_spec_tokens_to_schedule, skip_drafts=not drafts_needed)"
    ),
    (
        "+    scheduler_output = SchedulerOutput(scheduled_new_reqs=new_reqs_data, "
        "scheduled_cached_reqs=cached_reqs_data, num_scheduled_tokens=num_scheduled_tokens, "
        "total_num_scheduled_tokens=total_num_scheduled_tokens, "
        "scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, "
        "scheduled_encoder_inputs=scheduled_encoder_inputs, **scheduler_output_version_kwargs(self, "
        "iteration_details, scheduled_encoder_inputs), preempted_req_ids=self.reset_preempted_req_ids, "
        "finished_req_ids=self.finished_req_ids, "
        "free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(), "
        "encoder_input_updates=self._get_encoder_input_updates(num_scheduled_tokens, new_reqs_data, "
        "scheduled_encoder_inputs), new_block_ids_to_zero=self._get_new_block_ids_to_zero(), "
        "has_sync_kv_loads=has_sync_kv_loads, kv_cache_block_copies=pending_kv_cache_block_copies, "
        "kv_connector_block_state=kv_connector_block_state, "
        "num_spec_tokens_to_schedule=num_spec_tokens_to_schedule, skip_drafts=not drafts_needed)"
    ),
    "     if self.connector is not None:",
    "         meta = self._build_kv_connector_meta(self.connector, scheduler_output)",
    "         scheduler_output.kv_connector_metadata = meta",
)

_EXPECTED_DYNTRA_LB_DIFF: tuple[str, ...] = (
    "@@",
    "     scheduled_running_reqs: list[Request] = []",
    "     preempted_reqs: list[Request] = []",
    "     iteration_details = None",
    (
        "-    if self.log_stats and self.observability_config.enable_logging_iteration_details or "
        "self.vllm_config.profiler_config.profiler is not None:"
    ),
    (
        "+    if not vllm_version_is('0.33.0') and (self.log_stats and "
        "self.observability_config.enable_logging_iteration_details or "
        "self.vllm_config.profiler_config.profiler is not None):"
    ),
    "         iteration_details = IterationDetails()",
    "+    self._apply_load_balance_modifications()",
    "     req_to_new_blocks: dict[str, KVCacheBlocks] = {}",
    "     num_scheduled_tokens: dict[str, int] = {}",
    "     token_budget = self.max_num_scheduled_tokens",
    "@@",
    "             else:",
    "                 step_skipped_waiting.appendleft(request_to_skip)",
    "         num_async_kv_loads = 0",
    "-        max_async_kv_loads = self.scheduler_config.max_async_kv_loads_per_step or float('inf')",
    "+        max_async_kv_loads = get_max_async_kv_loads_per_step(self.scheduler_config) or float('inf')",
    "         take_async_kv = True",
    "         while True:",
    "             num_running = len(self.running) + self.num_waiting_for_streaming_input",
    "@@",
    "                 request_queue = self.kv_holding_waiting",
    "                 if not request_queue and self._pause_state == PauseState.UNPAUSED:",
    "                     request_queue = self._next_waiting_queue(take_async_kv)",
    (
        "-            elif take_async_kv and self.scheduler_config.max_num_inflight_kv_loads and "
        "(self._pause_state == PauseState.UNPAUSED):"
    ),
    (
        "+            elif take_async_kv and get_max_num_inflight_kv_loads(self.scheduler_config) and "
        "(self._pause_state == PauseState.UNPAUSED):"
    ),
    "                 request_queue = self.async_kv_waiting",
    "             else:",
    "                 break",
    "@@",
    "                 skip_request(request_queue)",
    "                 continue",
    "             if request.num_stale_output_tokens and (not request.drop_stale_output):",
    "+                skip_request(request_queue)",
    "+                continue",
    ("+            if request_queue is not self.async_kv_waiting and (not self._can_admit_waiting_request(request)):"),
    "                 skip_request(request_queue)",
    "                 continue",
    (
        "             if self.lora_config and request.lora_request and (len(scheduled_loras) == "
        "self.lora_config.max_loras and request.lora_request.lora_int_id not in scheduled_loras):"
    ),
    "@@",
    "                     partial_tail = num_new_local_computed_tokens % self.block_size",
    "                     block_aligned_local = num_new_local_computed_tokens - partial_tail",
    (
        "                     ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(request, "
        "block_aligned_local)"
    ),
    "-                    if request.skip_reading_external_kv:",
    "+                    if should_skip_external_kv(request):",
    "                         ext_tokens, load_kv_async = (0, False)",
    "                     if ext_tokens is None:",
    "                         skip_request(request_queue)",
    "@@",
    "                     break",
    "                 take_async_kv = False",
    "                 continue",
    "+            if not load_kv_async and (not self._can_admit_waiting_request(request)):",
    "+                skip_request(request_queue)",
    "+                continue",
    "             if load_kv_async:",
    "                 assert num_external_computed_tokens > 0",
    "                 num_new_tokens = 0",
    "@@",
    ("                     num_replay_tokens = self._mark_prefix_replay(request, num_computed_tokens, window_loaded)"),
    "                     num_computed_tokens -= num_replay_tokens",
    "                 num_new_tokens = request.num_tokens - num_computed_tokens",
    (
        "-                if (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None) and "
        "self.num_sampled_tokens_per_step > 0 and (num_new_tokens == 1) and (not prefill_scheduled) and "
        "(scheduled_running_reqs or num_computed_tokens > 0):"
    ),
    (
        "+                if (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None) and "
        "self.num_sampled_tokens_per_step > 0 and (num_new_tokens == 1) and (not prefill_scheduled):"
    ),
    "                     padded_num_tokens = 1 + self.num_spec_tokens",
    (
        "                     if num_computed_tokens + padded_num_tokens + self.num_sampled_tokens_per_step "
        "<= self.max_model_len:"
    ),
    "                         if padded_num_tokens > request_token_budget:",
    "@@",
    "     assert input_budget >= 0",
    "     assert len(self.running) <= self.max_num_running_reqs",
    "     assert len(scheduled_new_reqs) + len(scheduled_running_reqs) <= len(self.running)",
    (
        "-    new_reqs_data = [NewRequestData.from_request(req, "
        "req_to_new_blocks[req.request_id].get_block_ids(), req.prefill_token_ids, "
        "uses_mrope=self.model_uses_mrope) for req in scheduled_new_reqs]"
    ),
    (
        "+    new_reqs_data = [NewRequestData.from_request(req, "
        "req_to_new_blocks[req.request_id].get_block_ids(), get_new_request_prefill_token_ids(req), "
        "uses_mrope=self.model_uses_mrope) for req in scheduled_new_reqs]"
    ),
    "     with record_function_or_nullcontext('schedule: make_cached_request_data'):",
    (
        "-        cached_reqs_data = self._make_cached_request_data(scheduled_running_reqs, "
        "num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_new_blocks, "
        "iteration_details=iteration_details)"
    ),
    (
        "+        cached_reqs_data = make_cached_request_data(self, scheduled_running_reqs, "
        "num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_new_blocks, "
        "iteration_details=iteration_details)"
    ),
    "     if iteration_details is not None:",
    (
        "         iteration_details.num_prefill_requests = len(num_scheduled_tokens) - "
        "iteration_details.num_decode_requests"
    ),
    (
        "         iteration_details.num_prefill_tokens = total_num_scheduled_tokens - "
        "iteration_details.num_decode_tokens"
    ),
    "@@",
    "     num_spec_tokens_to_schedule = self.num_spec_tokens",
    "     if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:",
    "         num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]",
    (
        "-    scheduler_output = SchedulerOutput(scheduled_new_reqs=new_reqs_data, "
        "scheduled_cached_reqs=cached_reqs_data, num_scheduled_tokens=num_scheduled_tokens, "
        "total_num_scheduled_tokens=total_num_scheduled_tokens, "
        "scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, "
        "scheduled_encoder_inputs=scheduled_encoder_inputs, iteration_details=iteration_details, "
        "preempted_req_ids=self.reset_preempted_req_ids, finished_req_ids=self.finished_req_ids, "
        "free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(), "
        "encoder_input_updates=self._get_encoder_input_updates(num_scheduled_tokens, new_reqs_data, "
        "scheduled_encoder_inputs), new_block_ids_to_zero=self._get_new_block_ids_to_zero(), "
        "has_sync_kv_loads=has_sync_kv_loads, kv_cache_block_copies=pending_kv_cache_block_copies, "
        "kv_connector_block_state=kv_connector_block_state, "
        "num_spec_tokens_to_schedule=num_spec_tokens_to_schedule, skip_drafts=not drafts_needed)"
    ),
    (
        "+    scheduler_output = SchedulerOutput(scheduled_new_reqs=new_reqs_data, "
        "scheduled_cached_reqs=cached_reqs_data, num_scheduled_tokens=num_scheduled_tokens, "
        "total_num_scheduled_tokens=total_num_scheduled_tokens, "
        "scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, "
        "scheduled_encoder_inputs=scheduled_encoder_inputs, **scheduler_output_version_kwargs(self, "
        "iteration_details, scheduled_encoder_inputs), preempted_req_ids=self.reset_preempted_req_ids, "
        "finished_req_ids=self.finished_req_ids, "
        "free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(), "
        "encoder_input_updates=self._get_encoder_input_updates(num_scheduled_tokens, new_reqs_data, "
        "scheduled_encoder_inputs), new_block_ids_to_zero=self._get_new_block_ids_to_zero(), "
        "has_sync_kv_loads=has_sync_kv_loads, kv_cache_block_copies=pending_kv_cache_block_copies, "
        "kv_connector_block_state=kv_connector_block_state, "
        "num_spec_tokens_to_schedule=num_spec_tokens_to_schedule, skip_drafts=not drafts_needed)"
    ),
    "     if self.connector is not None:",
    "         meta = self._build_kv_connector_meta(self.connector, scheduler_output)",
    "         scheduler_output.kv_connector_metadata = meta",
    "@@",
    "         self.sched_step_seq += 1",
    "     with record_function_or_nullcontext('schedule: update_after_schedule'):",
    "         self._update_after_schedule(scheduler_output)",
    "+    if self._enable_diagnostics:",
    "+        print_scheduler_summary(self, scheduler_output)",
    "     return scheduler_output",
)

_EXPECTED_RECOMPUTE_DIFF: tuple[str, ...] = (
    "@@",
    "     scheduled_running_reqs: list[Request] = []",
    "     preempted_reqs: list[Request] = []",
    "     iteration_details = None",
    (
        "-    if self.log_stats and self.observability_config.enable_logging_iteration_details or "
        "self.vllm_config.profiler_config.profiler is not None:"
    ),
    (
        "+    if not vllm_version_is('0.33.0') and (self.log_stats and "
        "self.observability_config.enable_logging_iteration_details or "
        "self.vllm_config.profiler_config.profiler is not None):"
    ),
    "         iteration_details = IterationDetails()",
    "+    self._recomputed_reqs: list[RecomputeReqInfo] = []",
    "+    self._apply_load_balance_modifications()",
    "     req_to_new_blocks: dict[str, KVCacheBlocks] = {}",
    "     num_scheduled_tokens: dict[str, int] = {}",
    "     token_budget = self.max_num_scheduled_tokens",
    "@@",
    "                                 num_scheduled_encoder_inputs -= len(preempted_encoder_inputs)",
    "                 else:",
    "                     preempted_req = self.running.pop()",
    (
        "-                self._preempt_request(preempted_req, scheduled_timestamp, "
        "drop_stale_output=self.requires_kv_delivery)"
    ),
    "-                preempted_reqs.append(preempted_req)",
    (
        "+                locally_preempted = self._preempt_or_recompute(preempted_req, scheduled_timestamp, "
        "drop_stale_output=self.requires_kv_delivery)"
    ),
    "+                if locally_preempted:",
    "+                    preempted_reqs.append(preempted_req)",
    "                 if preempted_req == request:",
    "                     break",
    "         if new_blocks is None:",
    "@@",
    "     if self.lora_config:",
    (
        "         scheduled_loras = set((req.lora_request.lora_int_id for req in self.running if "
        "req.lora_request and req.lora_request.lora_int_id > 0))"
    ),
    "         assert len(scheduled_loras) <= self.lora_config.max_loras",
    "-    if not preempted_reqs and self._pause_state != PauseState.PAUSED_ALL:",
    ("+    if not preempted_reqs and (not self._recomputed_reqs) and (self._pause_state != PauseState.PAUSED_ALL):"),
    "         if self.cache_aware_window and (not defer_prefills):",
    "             self._reorder_waiting_by_cached_prefix()",
    "         step_skipped_waiting: deque[Request] = deque()",
    "@@",
    "             else:",
    "                 step_skipped_waiting.appendleft(request_to_skip)",
    "         num_async_kv_loads = 0",
    "-        max_async_kv_loads = self.scheduler_config.max_async_kv_loads_per_step or float('inf')",
    "+        max_async_kv_loads = get_max_async_kv_loads_per_step(self.scheduler_config) or float('inf')",
    "         take_async_kv = True",
    "         while True:",
    "             num_running = len(self.running) + self.num_waiting_for_streaming_input",
    "@@",
    "                 request_queue = self.kv_holding_waiting",
    "                 if not request_queue and self._pause_state == PauseState.UNPAUSED:",
    "                     request_queue = self._next_waiting_queue(take_async_kv)",
    (
        "-            elif take_async_kv and self.scheduler_config.max_num_inflight_kv_loads and "
        "(self._pause_state == PauseState.UNPAUSED):"
    ),
    (
        "+            elif take_async_kv and get_max_num_inflight_kv_loads(self.scheduler_config) and "
        "(self._pause_state == PauseState.UNPAUSED):"
    ),
    "                 request_queue = self.async_kv_waiting",
    "             else:",
    "                 break",
    "@@",
    "                 skip_request(request_queue)",
    "                 continue",
    "             if request.num_stale_output_tokens and (not request.drop_stale_output):",
    "+                skip_request(request_queue)",
    "+                continue",
    ("+            if request_queue is not self.async_kv_waiting and (not self._can_admit_waiting_request(request)):"),
    "                 skip_request(request_queue)",
    "                 continue",
    (
        "             if self.lora_config and request.lora_request and (len(scheduled_loras) == "
        "self.lora_config.max_loras and request.lora_request.lora_int_id not in scheduled_loras):"
    ),
    "@@",
    "                     partial_tail = num_new_local_computed_tokens % self.block_size",
    "                     block_aligned_local = num_new_local_computed_tokens - partial_tail",
    (
        "                     ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(request, "
        "block_aligned_local)"
    ),
    "-                    if request.skip_reading_external_kv:",
    "+                    if should_skip_external_kv(request):",
    "                         ext_tokens, load_kv_async = (0, False)",
    "                     if ext_tokens is None:",
    "                         skip_request(request_queue)",
    "@@",
    "                     break",
    "                 take_async_kv = False",
    "                 continue",
    "+            if not load_kv_async and (not self._can_admit_waiting_request(request)):",
    "+                skip_request(request_queue)",
    "+                continue",
    "             if load_kv_async:",
    "                 assert num_external_computed_tokens > 0",
    "                 num_new_tokens = 0",
    "@@",
    ("                     num_replay_tokens = self._mark_prefix_replay(request, num_computed_tokens, window_loaded)"),
    "                     num_computed_tokens -= num_replay_tokens",
    "                 num_new_tokens = request.num_tokens - num_computed_tokens",
    (
        "-                if (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None) and "
        "self.num_sampled_tokens_per_step > 0 and (num_new_tokens == 1) and (not prefill_scheduled) and "
        "(scheduled_running_reqs or num_computed_tokens > 0):"
    ),
    (
        "+                if (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None) and "
        "self.num_sampled_tokens_per_step > 0 and (num_new_tokens == 1) and (not prefill_scheduled):"
    ),
    "                     padded_num_tokens = 1 + self.num_spec_tokens",
    (
        "                     if num_computed_tokens + padded_num_tokens + self.num_sampled_tokens_per_step "
        "<= self.max_model_len:"
    ),
    "                         if padded_num_tokens > request_token_budget:",
    "@@",
    "     assert input_budget >= 0",
    "     assert len(self.running) <= self.max_num_running_reqs",
    "     assert len(scheduled_new_reqs) + len(scheduled_running_reqs) <= len(self.running)",
    (
        "-    new_reqs_data = [NewRequestData.from_request(req, "
        "req_to_new_blocks[req.request_id].get_block_ids(), req.prefill_token_ids, "
        "uses_mrope=self.model_uses_mrope) for req in scheduled_new_reqs]"
    ),
    (
        "+    new_reqs_data = [NewRequestData.from_request(req, "
        "req_to_new_blocks[req.request_id].get_block_ids(), get_new_request_prefill_token_ids(req), "
        "uses_mrope=self.model_uses_mrope) for req in scheduled_new_reqs]"
    ),
    "     with record_function_or_nullcontext('schedule: make_cached_request_data'):",
    (
        "-        cached_reqs_data = self._make_cached_request_data(scheduled_running_reqs, "
        "num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_new_blocks, "
        "iteration_details=iteration_details)"
    ),
    (
        "+        cached_reqs_data = make_cached_request_data(self, scheduled_running_reqs, "
        "num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_new_blocks, "
        "iteration_details=iteration_details)"
    ),
    "     if iteration_details is not None:",
    (
        "         iteration_details.num_prefill_requests = len(num_scheduled_tokens) - "
        "iteration_details.num_decode_requests"
    ),
    (
        "         iteration_details.num_prefill_tokens = total_num_scheduled_tokens - "
        "iteration_details.num_decode_tokens"
    ),
    "@@",
    "     num_spec_tokens_to_schedule = self.num_spec_tokens",
    "     if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:",
    "         num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]",
    (
        "-    scheduler_output = SchedulerOutput(scheduled_new_reqs=new_reqs_data, "
        "scheduled_cached_reqs=cached_reqs_data, num_scheduled_tokens=num_scheduled_tokens, "
        "total_num_scheduled_tokens=total_num_scheduled_tokens, "
        "scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, "
        "scheduled_encoder_inputs=scheduled_encoder_inputs, iteration_details=iteration_details, "
        "preempted_req_ids=self.reset_preempted_req_ids, finished_req_ids=self.finished_req_ids, "
        "free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(), "
        "encoder_input_updates=self._get_encoder_input_updates(num_scheduled_tokens, new_reqs_data, "
        "scheduled_encoder_inputs), new_block_ids_to_zero=self._get_new_block_ids_to_zero(), "
        "has_sync_kv_loads=has_sync_kv_loads, kv_cache_block_copies=pending_kv_cache_block_copies, "
        "kv_connector_block_state=kv_connector_block_state, "
        "num_spec_tokens_to_schedule=num_spec_tokens_to_schedule, skip_drafts=not drafts_needed)"
    ),
    (
        "+    scheduler_output = RecomputeSchedulerOutput(scheduled_new_reqs=new_reqs_data, "
        "scheduled_cached_reqs=cached_reqs_data, num_scheduled_tokens=num_scheduled_tokens, "
        "total_num_scheduled_tokens=total_num_scheduled_tokens, "
        "scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, "
        "scheduled_encoder_inputs=scheduled_encoder_inputs, **scheduler_output_version_kwargs(self, "
        "iteration_details, scheduled_encoder_inputs), preempted_req_ids=self.reset_preempted_req_ids, "
        "finished_req_ids=self.finished_req_ids, "
        "free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(), "
        "encoder_input_updates=self._get_encoder_input_updates(num_scheduled_tokens, new_reqs_data, "
        "scheduled_encoder_inputs), new_block_ids_to_zero=self._get_new_block_ids_to_zero(), "
        "has_sync_kv_loads=has_sync_kv_loads, kv_cache_block_copies=pending_kv_cache_block_copies, "
        "kv_connector_block_state=kv_connector_block_state, "
        "num_spec_tokens_to_schedule=num_spec_tokens_to_schedule, skip_drafts=not drafts_needed, "
        "recomputed_reqs=self._recomputed_reqs or None)"
    ),
    "     if self.connector is not None:",
    "         meta = self._build_kv_connector_meta(self.connector, scheduler_output)",
    "         scheduler_output.kv_connector_metadata = meta",
    "@@",
    "         self.sched_step_seq += 1",
    "     with record_function_or_nullcontext('schedule: update_after_schedule'):",
    "         self._update_after_schedule(scheduler_output)",
    "+    if getattr(self, '_enable_diagnostics', False):",
    "+        print_scheduler_summary(self, scheduler_output)",
    "     return scheduler_output",
)


def _strip_docstring_and_unparse(func: ast.FunctionDef) -> list[str]:
    body = func.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        func.body = body[1:]
    return ast.unparse(func).splitlines()


def _upstream_schedule_lines() -> list[str]:
    # Resolve the installed upstream file directly via find_spec because
    # importing vllm.v1.core.sched.scheduler.Scheduler returns BalanceScheduler
    # once Ascend's platform patch is active.
    spec = importlib.util.find_spec("vllm.v1.core.sched.scheduler")
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "schedule":
                    return _strip_docstring_and_unparse(item)
    raise AssertionError(f"Scheduler.schedule not found in {spec.origin}")


def _ascend_schedule_lines(cls: type[Any]) -> list[str]:
    source = textwrap.dedent(inspect.getsource(cls.schedule))
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "schedule":
            return _strip_docstring_and_unparse(node)
    raise AssertionError(f"{cls.__name__}.schedule not found")


def _normalized_schedule_diff(cls: type[Any]) -> list[str]:
    upstream = _upstream_schedule_lines()
    ascend = _ascend_schedule_lines(cls)
    diff_lines: list[str] = []
    for line in difflib.unified_diff(upstream, ascend, "upstream", "ascend", n=3, lineterm=""):
        if line.startswith(("---", "+++")):
            continue
        diff_lines.append("@@" if line.startswith("@@") else line)
    return diff_lines


@pytest.mark.skipif(
    vllm_version_is("0.33.0"),
    reason="upstream parity diff targets the pinned vLLM main commit",
)
@pytest.mark.parametrize(
    ("scheduler_cls", "expected_diff"),
    [
        (BalanceScheduler, _EXPECTED_BALANCE_DIFF),
        (DyntraLBScheduler, _EXPECTED_DYNTRA_LB_DIFF),
        (RecomputeScheduler, _EXPECTED_RECOMPUTE_DIFF),
    ],
)
def test_copied_schedule_matches_upstream_modulo_expected_diff(
    scheduler_cls: type[Any],
    expected_diff: tuple[str, ...],
) -> None:
    expected_lines = list(expected_diff)
    actual_lines = _normalized_schedule_diff(scheduler_cls)

    if actual_lines != expected_lines:
        actual_text = "\n".join(actual_lines)
        pytest.fail(
            f"{scheduler_cls.__name__}.schedule() drifted from upstream or "
            f"its expected normalized diff.\n--- ACTUAL NORMALIZED DIFF ---\n{actual_text}"
        )
