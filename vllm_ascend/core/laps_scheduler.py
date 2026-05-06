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

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from typing import Callable, cast

from vllm.logger import logger
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request
from vllm.v1.request import RequestStatus
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend import envs


class LAPSRequestQueue(RequestQueue):
    """Two-level waiting queue for short and long prefills."""

    def __init__(
        self,
        policy: SchedulingPolicy,
        threshold: int,
        wait_window_ms: float,
        wait_max_batch: int,
        immediate_predicate: Callable[[Request], bool] | None = None,
    ) -> None:
        self.policy = policy
        self.threshold = threshold
        self.wait_window_ms = max(wait_window_ms, 0.0)
        self.wait_max_batch = max(wait_max_batch, 1)
        self.immediate_predicate = immediate_predicate
        self._immediate_queue = create_request_queue(policy)
        self._short_queue = create_request_queue(policy)
        self._long_queue = create_request_queue(policy)
        self._short_wait_started_at: float | None = None
        self._short_ready_to_dispatch = False
        self._stats_log_interval_s = max(
            envs.VLLM_ASCEND_LAPS_STATS_LOG_INTERVAL_S, 0.0
        )
        self._last_stats_log_at = time.monotonic()
        self._enqueue_counters = {"immediate": 0, "short": 0, "long": 0}
        self._dispatch_counters = {"immediate": 0, "short": 0, "long": 0}
        self._remove_counters = {"immediate": 0, "short": 0, "long": 0}
        self._short_ready_reason_counters = {
            "no_wait_window": 0,
            "max_batch": 0,
            "wait_window_elapsed": 0,
        }
        self._last_long_capped_count = 0
        self._last_short_reserved_tokens = 0
        self._last_short_actual_used_tokens = 0
        self._last_long_actual_used_tokens = 0
        self._debug_logging_enabled = logger.isEnabledFor(logging.DEBUG)
        self._force_immediate_request_ids: set[str] = set()

    def _queues(self) -> tuple[RequestQueue, ...]:
        return (self._immediate_queue, self._short_queue, self._long_queue)

    def _queue_name(self, queue: RequestQueue) -> str:
        if queue is self._immediate_queue:
            return "immediate"
        if queue is self._short_queue:
            return "short"
        if queue is self._long_queue:
            return "long"
        return "unknown"

    def _short_wait_elapsed_ms(self) -> float | None:
        if self._short_wait_started_at is None:
            return None
        return (time.monotonic() - self._short_wait_started_at) * 1000.0

    def _short_wait_state(self) -> str:
        if not self._short_queue:
            return "empty"
        if self._short_ready_to_dispatch:
            return "ready"
        if self.wait_window_ms <= 0:
            return "no_wait"
        elapsed_ms = self._short_wait_elapsed_ms()
        if elapsed_ms is None:
            return "pending"
        return f"waiting({elapsed_ms:.3f}/{self.wait_window_ms:.3f}ms)"

    def _maybe_log_stats(self, force: bool = False) -> None:
        if self._stats_log_interval_s <= 0:
            return
        now = time.monotonic()
        if not force and (now - self._last_stats_log_at) < self._stats_log_interval_s:
            return
        self._last_stats_log_at = now
        logger.info(
            "LAPS stats: threshold=%d wait_window_ms=%.3f wait_max_batch=%d "
            "sizes=(immediate=%d short=%d long=%d) short_state=%s "
            "enqueues=%s dispatches=%s removals=%s short_ready_reasons=%s "
            "long_capped_count=%d short_reserved_tokens=%d "
            "short_actual_used_tokens=%d long_actual_used_tokens=%d",
            self.threshold,
            self.wait_window_ms,
            self.wait_max_batch,
            len(self._immediate_queue),
            len(self._short_queue),
            len(self._long_queue),
            self._short_wait_state(),
            self._enqueue_counters,
            self._dispatch_counters,
            self._remove_counters,
            self._short_ready_reason_counters,
            self._last_long_capped_count,
            self._last_short_reserved_tokens,
            self._last_short_actual_used_tokens,
            self._last_long_actual_used_tokens,
        )

    def record_schedule_step_stats(
        self,
        *,
        long_capped_count: int,
        short_reserved_tokens: int,
        short_actual_used_tokens: int,
        long_actual_used_tokens: int,
    ) -> None:
        self._last_long_capped_count = long_capped_count
        self._last_short_reserved_tokens = short_reserved_tokens
        self._last_short_actual_used_tokens = short_actual_used_tokens
        self._last_long_actual_used_tokens = long_actual_used_tokens

    def _record_short_ready_reason(self, reason: str) -> None:
        if reason in self._short_ready_reason_counters:
            self._short_ready_reason_counters[reason] += 1

    def _debug_state(
        self,
        event: str,
        request: Request | None = None,
        queue: RequestQueue | None = None,
        extra: str = "",
    ) -> None:
        if not self._debug_logging_enabled:
            return
        request_id = "-" if request is None else request.request_id
        prompt_tokens = -1 if request is None else request.num_prompt_tokens
        queue_name = "-" if queue is None else self._queue_name(queue)
        extra_suffix = f", {extra}" if extra else ""
        logger.debug(
            "LAPS queue %s: request_id=%s, prompt_tokens=%d, target_queue=%s, "
            "sizes=(immediate=%d, short=%d, long=%d), short_state=%s%s",
            event,
            request_id,
            prompt_tokens,
            queue_name,
            len(self._immediate_queue),
            len(self._short_queue),
            len(self._long_queue),
            self._short_wait_state(),
            extra_suffix,
        )

    @property
    def num_immediate_requests(self) -> int:
        return len(self._immediate_queue)

    @property
    def num_short_requests(self) -> int:
        return len(self._short_queue)

    @property
    def num_long_requests(self) -> int:
        return len(self._long_queue)

    def has_short_requests(self) -> bool:
        return len(self._short_queue) > 0

    def has_long_requests(self) -> bool:
        return len(self._long_queue) > 0

    def has_immediate_requests(self) -> bool:
        return len(self._immediate_queue) > 0

    def _classify_queue(
        self, request: Request, *, force_immediate: bool = False
    ) -> RequestQueue:
        if force_immediate or request.request_id in self._force_immediate_request_ids:
            return self._immediate_queue
        if self.immediate_predicate is not None and self.immediate_predicate(request):
            return self._immediate_queue
        if request.num_prompt_tokens <= self.threshold:
            return self._short_queue
        return self._long_queue

    def _on_short_queue_changed(self) -> None:
        if not self._short_queue:
            self._short_wait_started_at = None
            self._short_ready_to_dispatch = False
            self._debug_state("short_reset")
            return
        if self._short_ready_to_dispatch:
            return
        if self.wait_window_ms <= 0 or len(self._short_queue) >= self.wait_max_batch:
            self._short_wait_started_at = None
            self._short_ready_to_dispatch = True
            reason = "no_wait_window" if self.wait_window_ms <= 0 else "max_batch"
            self._record_short_ready_reason(reason)
            self._debug_state("short_ready", extra=f"reason={reason}")
            self._maybe_log_stats()
            return
        if self._short_wait_started_at is None:
            self._short_wait_started_at = time.monotonic()
            self._debug_state("short_wait_started")

    def _short_batch_ready(self) -> bool:
        if not self._short_queue:
            self._short_wait_started_at = None
            self._short_ready_to_dispatch = False
            return False
        if self._short_ready_to_dispatch:
            return True
        if self.wait_window_ms <= 0:
            self._short_wait_started_at = None
            self._short_ready_to_dispatch = True
            self._record_short_ready_reason("no_wait_window")
            self._debug_state("short_ready", extra="reason=no_wait_window")
            self._maybe_log_stats()
            return True
        if len(self._short_queue) >= self.wait_max_batch:
            self._short_wait_started_at = None
            self._short_ready_to_dispatch = True
            self._record_short_ready_reason("max_batch")
            self._debug_state("short_ready", extra="reason=max_batch")
            self._maybe_log_stats()
            return True
        if self._short_wait_started_at is None:
            self._short_wait_started_at = time.monotonic()
            self._debug_state("short_wait_started")
            return False
        elapsed_ms = (time.monotonic() - self._short_wait_started_at) * 1000.0
        if elapsed_ms >= self.wait_window_ms:
            self._short_wait_started_at = None
            self._short_ready_to_dispatch = True
            self._record_short_ready_reason("wait_window_elapsed")
            self._debug_state(
                "short_ready",
                extra=f"reason=wait_window_elapsed, elapsed_ms={elapsed_ms:.3f}",
            )
            self._maybe_log_stats()
            return True
        return False

    def _select_schedulable_queue(self) -> RequestQueue | None:
        if self._immediate_queue:
            return self._immediate_queue
        if self._short_batch_ready():
            return self._short_queue
        if self._long_queue:
            return self._long_queue
        return None

    def select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        return self._select_schedulable_queue()

    @staticmethod
    def _request_id(request: Request | object) -> str | None:
        return getattr(request, "request_id", None)

    def _find_matching_request(
        self, queue: RequestQueue, request: Request | object
    ) -> Request | None:
        request_id = self._request_id(request)
        if request_id is None:
            return None
        for candidate in queue:
            if candidate.request_id == request_id:
                return cast(Request, candidate)
        return None

    def add_request(self, request: Request) -> None:
        queue = self._classify_queue(request)
        queue.add_request(request)
        self._enqueue_counters[self._queue_name(queue)] += 1
        if queue is not self._immediate_queue:
            self._force_immediate_request_ids.discard(request.request_id)
        if queue is self._short_queue:
            self._on_short_queue_changed()
        self._debug_state("enqueue", request=request, queue=queue)
        self._maybe_log_stats()

    def pop_request(self) -> Request:
        queue = self._select_schedulable_queue()
        if queue is None:
            raise IndexError("pop from empty LAPS queue")
        request = queue.pop_request()
        self._dispatch_counters[self._queue_name(queue)] += 1
        self._force_immediate_request_ids.discard(request.request_id)
        if queue is self._short_queue:
            self._on_short_queue_changed()
        self._debug_state("dispatch", request=request, queue=queue)
        self._maybe_log_stats()
        return request

    def peek_request(self) -> Request:
        queue = self._select_schedulable_queue()
        if queue is None:
            raise IndexError("peek from an empty LAPS queue")
        return queue.peek_request()

    def prepend_request(self, request: Request, force_immediate: bool = False) -> None:
        if force_immediate:
            self._force_immediate_request_ids.add(request.request_id)
        queue = self._classify_queue(request, force_immediate=force_immediate)
        queue.prepend_request(request)
        self._enqueue_counters[self._queue_name(queue)] += 1
        if queue is self._short_queue:
            self._on_short_queue_changed()
        self._debug_state("prepend", request=request, queue=queue)
        self._maybe_log_stats()

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(cast(Request, request))

    def remove_request(self, request: Request) -> None:
        for queue in self._queues():
            matched_request = self._find_matching_request(queue, request)
            if matched_request is not None:
                queue.remove_request(matched_request)
                self._force_immediate_request_ids.discard(request.request_id)
                self._remove_counters[self._queue_name(queue)] += 1
                if queue is self._short_queue:
                    self._on_short_queue_changed()
                self._debug_state("remove", request=matched_request, queue=queue)
                self._maybe_log_stats()
                return
        raise ValueError("request not found in LAPS queue")

    def remove_requests(self, requests: Iterable[Request]) -> None:
        queue_to_requests: dict[int, list[Request]] = {}
        queue_map = {id(queue): queue for queue in self._queues()}
        removed_count = 0
        for request in requests:
            for queue in self._queues():
                matched_request = self._find_matching_request(queue, request)
                if matched_request is not None:
                    queue_to_requests.setdefault(id(queue), []).append(matched_request)
                    break
        for queue_id, matched_requests in queue_to_requests.items():
            removed_count += len(matched_requests)
            queue = queue_map[queue_id]
            self._remove_counters[self._queue_name(queue)] += len(matched_requests)
            queue.remove_requests(matched_requests)
            for request in matched_requests:
                self._force_immediate_request_ids.discard(request.request_id)
        self._on_short_queue_changed()
        if removed_count:
            self._debug_state("remove_batch", extra=f"count={removed_count}")
            self._maybe_log_stats()

    def __bool__(self) -> bool:
        return any(bool(queue) for queue in self._queues())

    def __len__(self) -> int:
        return (
            len(self._immediate_queue)
            + len(self._short_queue)
            + len(self._long_queue)
        )

    def __iter__(self) -> Iterator[Request]:
        yield from self._immediate_queue
        yield from self._short_queue
        yield from self._long_queue

    def __contains__(self, request: object) -> bool:
        return any(
            self._find_matching_request(queue, request) is not None
            for queue in self._queues()
        )


class LAPSSchedulerMixin:
    """Inject a LAPS-style waiting queue into vLLM's scheduler."""

    def _init_laps_waiting_queue(
        self,
        immediate_predicate: Callable[[Request], bool] | None = None,
    ) -> None:
        self.laps_long_prefill_cap = max(envs.VLLM_ASCEND_LAPS_LONG_PREFILL_CAP, 0)
        self.laps_short_reserved_ratio = min(
            max(envs.VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO, 0.0), 1.0
        )
        if self.policy != SchedulingPolicy.FCFS:
            logger.warning_once(
                "VLLM_ASCEND_LAPS_SCHEDULING currently supports only FCFS "
                "scheduler policy; keeping the default waiting queue."
            )
            return

        threshold = envs.VLLM_ASCEND_LAPS_THRESHOLD
        wait_window_ms = envs.VLLM_ASCEND_LAPS_WAIT_WINDOW_MS
        wait_max_batch = envs.VLLM_ASCEND_LAPS_WAIT_MAX_BATCH
        self.waiting = LAPSRequestQueue(
            policy=self.policy,
            threshold=threshold,
            wait_window_ms=wait_window_ms,
            wait_max_batch=wait_max_batch,
            immediate_predicate=immediate_predicate,
        )
        logger.info(
            "LAPS scheduling enabled on Ascend: threshold=%d, "
            "wait_window_ms=%.3f, wait_max_batch=%d, "
            "long_prefill_cap=%d, short_reserved_ratio=%.3f",
            threshold,
            wait_window_ms,
            wait_max_batch,
            envs.VLLM_ASCEND_LAPS_LONG_PREFILL_CAP,
            envs.VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO,
        )

    def _laps_waiting_queue(self) -> LAPSRequestQueue | None:
        if isinstance(self.waiting, LAPSRequestQueue):
            return self.waiting
        return None

    def _laps_threshold(self) -> int:
        laps_waiting = self._laps_waiting_queue()
        if laps_waiting is not None:
            return laps_waiting.threshold
        return envs.VLLM_ASCEND_LAPS_THRESHOLD

    def _is_prefill_request(
        self, request: Request, num_computed_tokens: int | None = None
    ) -> bool:
        computed_tokens = (
            request.num_computed_tokens
            if num_computed_tokens is None
            else num_computed_tokens
        )
        return computed_tokens < request.num_prompt_tokens

    def _is_short_prefill_request(
        self, request: Request, num_computed_tokens: int | None = None
    ) -> bool:
        return self._is_prefill_request(
            request, num_computed_tokens
        ) and request.num_prompt_tokens <= self._laps_threshold()

    def _is_long_prefill_request(
        self, request: Request, num_computed_tokens: int | None = None
    ) -> bool:
        return self._is_prefill_request(
            request, num_computed_tokens
        ) and request.num_prompt_tokens > self._laps_threshold()

    def _laps_short_reserved_tokens(self, token_budget: int) -> int:
        laps_waiting = self._laps_waiting_queue()
        if (
            laps_waiting is None
            or token_budget <= 0
            or getattr(self, "laps_short_reserved_ratio", 0.0) <= 0
            or not laps_waiting.has_short_requests()
        ):
            return 0
        return min(
            token_budget,
            int(self.max_num_scheduled_tokens * self.laps_short_reserved_ratio),
        )

    def _laps_long_budgeting_enabled(self) -> bool:
        return (
            getattr(self, "laps_long_prefill_cap", 0) > 0
            or getattr(self, "laps_short_reserved_ratio", 0.0) > 0
        )

    def _apply_long_prefill_cap(
        self,
        request: Request,
        num_new_tokens: int,
        num_computed_tokens: int | None = None,
    ) -> tuple[int, bool]:
        if (
            getattr(self, "laps_long_prefill_cap", 0) <= 0
            or not self._is_prefill_request(request, num_computed_tokens)
            or request.num_prompt_tokens <= self.laps_long_prefill_cap
        ):
            return num_new_tokens, False
        if num_new_tokens > self.laps_long_prefill_cap:
            return self.laps_long_prefill_cap, True
        return num_new_tokens, False

    def _apply_long_budget_limit(
        self,
        request: Request,
        num_new_tokens: int,
        long_budget_remaining: int,
        num_computed_tokens: int | None = None,
    ) -> int:
        if not self._is_long_prefill_request(request, num_computed_tokens):
            return num_new_tokens
        return min(num_new_tokens, max(long_budget_remaining, 0))

    def _record_laps_step_usage(
        self,
        request: Request,
        num_scheduled_tokens: int,
        *,
        short_actual_used_tokens: int,
        long_actual_used_tokens: int,
        num_computed_tokens: int | None = None,
    ) -> tuple[int, int]:
        if self._is_short_prefill_request(request, num_computed_tokens):
            short_actual_used_tokens += num_scheduled_tokens
        elif self._is_long_prefill_request(request, num_computed_tokens):
            long_actual_used_tokens += num_scheduled_tokens
        return short_actual_used_tokens, long_actual_used_tokens

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        waiting = getattr(self, "waiting", None)
        if isinstance(waiting, LAPSRequestQueue):
            queue = waiting.select_waiting_queue_for_scheduling()
            if queue is not None:
                return queue
            skipped_waiting = getattr(self, "skipped_waiting", None)
            return skipped_waiting or None
        return super()._select_waiting_queue_for_scheduling()

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        super()._preempt_request(request, timestamp)
        waiting = getattr(self, "waiting", None)
        if isinstance(waiting, LAPSRequestQueue):
            waiting.remove_request(request)
            waiting.prepend_request(request, force_immediate=True)


from vllm.v1.core.sched.scheduler import Scheduler as BaseScheduler


class LAPSScheduler(LAPSSchedulerMixin, BaseScheduler):
    """vLLM scheduler with the Ascend LAPS waiting queue installed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_laps_waiting_queue()

    def schedule(self) -> SchedulerOutput:
        laps_waiting = self._laps_waiting_queue()
        if not self._laps_long_budgeting_enabled():
            scheduler_output = super().schedule()
            if laps_waiting is not None:
                laps_waiting.record_schedule_step_stats(
                    long_capped_count=0,
                    short_reserved_tokens=0,
                    short_actual_used_tokens=0,
                    long_actual_used_tokens=0,
                )
            return scheduler_output

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0

        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        scheduled_timestamp = time.monotonic()

        short_reserved_tokens = self._laps_short_reserved_tokens(token_budget)
        long_budget_remaining = max(token_budget - short_reserved_tokens, 0)
        long_capped_count = 0
        short_actual_used_tokens = 0
        long_actual_used_tokens = 0

        self.kv_cache_manager.new_step_starts()

        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
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
            num_new_tokens, was_long_capped = self._apply_long_prefill_cap(
                request, num_new_tokens
            )
            if was_long_capped:
                long_capped_count += 1
            num_new_tokens = self._apply_long_budget_limit(
                request, num_new_tokens, long_budget_remaining
            )
            num_new_tokens = min(num_new_tokens, token_budget)

            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
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

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                req_index += 1
                continue

            with record_function_or_nullcontext("schedule: allocate_slots"):
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
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            released_tokens = num_scheduled_tokens.pop(preempted_req_id)
                            token_budget += released_tokens
                            if self._is_long_prefill_request(preempted_req):
                                long_budget_remaining += released_tokens
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        break

            if new_blocks is None:
                break

            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            if self._is_long_prefill_request(request):
                long_budget_remaining -= num_new_tokens
            (
                short_actual_used_tokens,
                long_actual_used_tokens,
            ) = self._record_laps_step_usage(
                request,
                num_new_tokens,
                short_actual_used_tokens=short_actual_used_tokens,
                long_actual_used_tokens=long_actual_used_tokens,
            )
            req_index += 1

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
                request.spec_token_ids = []

            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
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

        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                if request_queue is None:
                    break

                request = request_queue.peek_request()
                request_id = request.request_id

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
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                if request.num_computed_tokens == 0:
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens
                else:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget
                was_long_capped = False

                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    num_new_tokens, was_long_capped = self._apply_long_prefill_cap(
                        request,
                        num_new_tokens,
                        num_computed_tokens=num_computed_tokens,
                    )

                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and not self._laps_long_budgeting_enabled()
                        and num_new_tokens > token_budget
                    ):
                        break

                    num_new_tokens = self._apply_long_budget_limit(
                        request,
                        num_new_tokens,
                        long_budget_remaining,
                        num_computed_tokens=num_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        if self._is_long_prefill_request(
                            request, num_computed_tokens=num_computed_tokens
                        ):
                            break
                        num_new_tokens = min(request.num_tokens - num_computed_tokens, token_budget)
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
                            shift_computed_tokens=1 if self.use_eagle else 0,
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
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

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

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
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

                request = request_queue.pop_request()
                if load_kv_async:
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    request.num_computed_tokens = num_computed_tokens
                    continue

                self.running.append(request)
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
                if self._is_long_prefill_request(
                    request, num_computed_tokens=num_computed_tokens
                ):
                    long_budget_remaining -= num_new_tokens
                if was_long_capped:
                    long_capped_count += 1
                (
                    short_actual_used_tokens,
                    long_actual_used_tokens,
                ) = self._record_laps_step_usage(
                    request,
                    num_new_tokens,
                    short_actual_used_tokens=short_actual_used_tokens,
                    long_actual_used_tokens=long_actual_used_tokens,
                    num_computed_tokens=num_computed_tokens,
                )
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
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

            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
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
                    req, req_to_new_blocks[req.request_id].get_block_ids()
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

        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        if self.ec_connector is not None:
            ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)

        if laps_waiting is not None:
            laps_waiting.record_schedule_step_stats(
                long_capped_count=long_capped_count,
                short_reserved_tokens=short_reserved_tokens,
                short_actual_used_tokens=short_actual_used_tokens,
                long_actual_used_tokens=long_actual_used_tokens,
            )
            laps_waiting._maybe_log_stats()
        return scheduler_output
