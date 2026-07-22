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
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, cast

from vllm.logger import logger
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler


class ShortRequestFirstRequestQueue(RequestQueue):
    """Three-lane waiting queue with short-request priority and simple aging.

    The queue implements vLLM's standard ``RequestQueue`` contract directly.
    ``peek_request`` pins the selected lane so its corresponding
    ``pop_request`` removes the same request even when aging changes lane
    priority between those two scheduler calls.
    """

    STATS_LOG_INTERVAL_S = 5.0
    AGED_LONG_WARNING_STREAK = 3

    def __init__(
        self,
        policy: SchedulingPolicy,
        threshold: int,
        long_max_wait_ms: float,
        immediate_predicate: Callable[[Request], bool] | None = None,
    ) -> None:
        self.policy = policy
        self.threshold = threshold
        self.long_max_wait_ms = long_max_wait_ms
        self.immediate_predicate = immediate_predicate
        self._immediate_queue = create_request_queue(policy)
        self._short_queue = create_request_queue(policy)
        self._long_queue = create_request_queue(policy)
        self._long_enqueue_at: dict[str, float] = {}
        self._prepend_counters = {"immediate": 0, "short": 0, "long": 0}
        self._dispatch_counters = {"immediate": 0, "short": 0, "long": 0}
        self._queue_index: dict[str, RequestQueue] = {}
        self._pinned_queue: RequestQueue | None = None
        self._pinned_request_id: str | None = None
        self._aged_long_promotions = 0
        self._consecutive_aged_long_promotions = 0
        self._last_stats_log_at = time.monotonic()

    def _queues(self) -> tuple[RequestQueue, ...]:
        return (self._immediate_queue, self._short_queue, self._long_queue)

    @property
    def _debug_logging_enabled(self) -> bool:
        return logger.isEnabledFor(logging.DEBUG)

    def _queue_name(self, queue: RequestQueue) -> str:
        if queue is self._immediate_queue:
            return "immediate"
        if queue is self._short_queue:
            return "short"
        if queue is self._long_queue:
            return "long"
        return "unknown"

    def _maybe_log_stats(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_stats_log_at) < self.STATS_LOG_INTERVAL_S:
            return
        self._last_stats_log_at = now
        logger.info(
            "ShortRequestFirst stats: threshold=%d long_max_wait_ms=%.3f "
            "sizes=(immediate=%d short=%d long=%d) "
            "prepends=%s dispatches=%s "
            "aged_long_promotions=%d consecutive_aged_long_promotions=%d",
            self.threshold,
            self.long_max_wait_ms,
            len(self._immediate_queue),
            len(self._short_queue),
            len(self._long_queue),
            self._prepend_counters,
            self._dispatch_counters,
            self._aged_long_promotions,
            self._consecutive_aged_long_promotions,
        )

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
            "ShortRequestFirst queue %s: request_id=%s, prompt_tokens=%d, target_queue=%s, "
            "sizes=(immediate=%d, short=%d, long=%d)%s",
            event,
            request_id,
            prompt_tokens,
            queue_name,
            len(self._immediate_queue),
            len(self._short_queue),
            len(self._long_queue),
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

    def _classify_queue(self, request: Request) -> RequestQueue:
        if self.immediate_predicate is not None and self.immediate_predicate(request):
            return self._immediate_queue
        if request.num_prompt_tokens <= self.threshold:
            return self._short_queue
        return self._long_queue

    def _long_head_wait_ms(self) -> float | None:
        if self.long_max_wait_ms <= 0 or not self._long_queue:
            return None
        head = self._long_queue.peek_request()
        enqueued_at = self._long_enqueue_at.get(head.request_id)
        if enqueued_at is None:
            return None
        return (time.monotonic() - enqueued_at) * 1000.0

    def _reset_degradation_streak(self) -> None:
        self._consecutive_aged_long_promotions = 0

    def _reset_degradation_streak_if_short_queue_empty(self) -> None:
        if not self._short_queue:
            self._reset_degradation_streak()

    def _select_schedulable_queue(self) -> RequestQueue | None:
        if self._immediate_queue:
            return self._immediate_queue
        if self._long_queue:
            wait_ms = self._long_head_wait_ms()
            if wait_ms is not None and wait_ms >= self.long_max_wait_ms and self._short_queue:
                return self._long_queue
        if self._short_queue:
            return self._short_queue
        if self._long_queue:
            return self._long_queue
        return None

    @staticmethod
    def _request_id(request: Request | object) -> str | None:
        return getattr(request, "request_id", None)

    def add_request(self, request: Request) -> None:
        self._clear_pin()
        queue = self._classify_queue(request)
        queue.add_request(request)
        self._queue_index[request.request_id] = queue
        if queue is self._long_queue:
            self._long_enqueue_at.setdefault(request.request_id, time.monotonic())
        self._debug_state("enqueue", request=request, queue=queue)
        self._maybe_log_stats()

    def pop_request(self) -> Request:
        queue = self._get_pinned_queue() or self._select_schedulable_queue()
        if queue is None:
            raise IndexError("pop from empty ShortRequestFirst queue")
        return self._pop_request_from_queue(queue)

    def _maybe_warn_degradation(self) -> None:
        if self._consecutive_aged_long_promotions < self.AGED_LONG_WARNING_STREAK:
            return
        logger.warning_once(
            "ShortRequestFirst scheduling is repeatedly promoting long requests ahead of waiting short requests "
            "and may degrade toward long-request priority. Consider increasing the short-request threshold "
            "or disabling this waiting-queue policy."
        )

    def _pop_request_from_queue(self, queue: RequestQueue) -> Request:
        request = queue.pop_request()
        self._clear_pin()
        self._queue_index.pop(request.request_id, None)
        enqueued_at = self._long_enqueue_at.pop(request.request_id, None)
        self._dispatch_counters[self._queue_name(queue)] += 1
        event_name = "dispatch"
        if queue is self._short_queue:
            self._reset_degradation_streak()
        elif queue is self._long_queue:
            wait_ms = (time.monotonic() - enqueued_at) * 1000.0 if enqueued_at is not None else None
            aged = self.long_max_wait_ms > 0 and wait_ms is not None and wait_ms >= self.long_max_wait_ms
            jumped_shorts = len(self._short_queue) > 0
            if aged and jumped_shorts:
                self._aged_long_promotions += 1
                self._consecutive_aged_long_promotions += 1
                self._maybe_warn_degradation()
            elif not jumped_shorts:
                self._reset_degradation_streak()
        self._reset_degradation_streak_if_short_queue_empty()
        self._debug_state(event_name, request=request, queue=queue)
        self._maybe_log_stats()
        return request

    def peek_request(self) -> Request:
        queue = self._get_pinned_queue() or self._select_schedulable_queue()
        if queue is None:
            raise IndexError("peek from an empty ShortRequestFirst queue")
        request = queue.peek_request()
        self._pinned_queue = queue
        self._pinned_request_id = request.request_id
        return request

    def prepend_request(self, request: Request) -> None:
        self._clear_pin()
        queue = self._classify_queue(request)
        queue.prepend_request(request)
        self._queue_index[request.request_id] = queue
        if queue is self._long_queue:
            self._long_enqueue_at.setdefault(request.request_id, time.monotonic())
        self._prepend_counters[self._queue_name(queue)] += 1
        self._debug_state("prepend", request=request, queue=queue)
        self._maybe_log_stats()

    def prepend_requests(self, requests: RequestQueue) -> None:
        self._clear_pin()
        for request in requests:
            self.prepend_request(cast(Request, request))

    def remove_request(self, request: Request) -> None:
        self._clear_pin()
        queue = self._queue_index.get(request.request_id)
        if queue is None:
            raise ValueError("request not found in ShortRequestFirst queue")
        queue.remove_request(request)
        self._queue_index.pop(request.request_id, None)
        self._long_enqueue_at.pop(request.request_id, None)
        self._reset_degradation_streak_if_short_queue_empty()
        self._debug_state("remove", request=request, queue=queue)
        self._maybe_log_stats()

    def remove_requests(self, requests: Iterable[Request]) -> None:
        self._clear_pin()
        queue_to_requests: dict[int, list[Request]] = {}
        queue_map = {id(q): q for q in self._queues()}
        removed_count = 0
        for request in requests:
            queue = self._queue_index.get(request.request_id)
            if queue is None:
                continue
            queue_to_requests.setdefault(id(queue), []).append(request)
        for queue_id, matched_requests in queue_to_requests.items():
            removed_count += len(matched_requests)
            queue = queue_map[queue_id]
            queue.remove_requests(matched_requests)
            for matched in matched_requests:
                self._queue_index.pop(matched.request_id, None)
                self._long_enqueue_at.pop(matched.request_id, None)
        if removed_count:
            self._reset_degradation_streak_if_short_queue_empty()
            self._debug_state("remove_batch", extra=f"count={removed_count}")
            self._maybe_log_stats()

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return len(self._immediate_queue) + len(self._short_queue) + len(self._long_queue)

    def __iter__(self) -> Iterator[Request]:
        yield from self._immediate_queue
        yield from self._short_queue
        yield from self._long_queue

    def __contains__(self, request: object) -> bool:
        request_id = self._request_id(request)
        return request_id is not None and request_id in self._queue_index

    def _clear_pin(self) -> None:
        self._pinned_queue = None
        self._pinned_request_id = None

    def _get_pinned_queue(self) -> RequestQueue | None:
        if self._pinned_queue is None or self._pinned_request_id is None:
            return None
        try:
            request = self._pinned_queue.peek_request()
        except IndexError:
            self._clear_pin()
            return None
        if request.request_id != self._pinned_request_id:
            self._clear_pin()
            return None
        return self._pinned_queue


def is_recovery_request(request: Request) -> bool:
    """Return whether a request must resume ahead of fresh waiting work."""
    return (
        request.status == RequestStatus.PREEMPTED
        or getattr(request, "num_computed_tokens", 0) > 0
        or getattr(request, "num_output_tokens", 0) > 0
    )


def install_short_request_first_waiting_queue(
    scheduler: Scheduler,
    *,
    threshold: int,
    long_max_wait_ms: float,
) -> ShortRequestFirstRequestQueue:
    """Install the SRF queue on a newly constructed FCFS scheduler."""
    policy = scheduler.policy
    if policy != SchedulingPolicy.FCFS:
        raise ValueError(f"ShortRequestFirst requires FCFS scheduling policy, got {policy!s}")

    waiting = scheduler.waiting
    if isinstance(waiting, ShortRequestFirstRequestQueue):
        return waiting
    if waiting:
        raise RuntimeError("ShortRequestFirst waiting queue must be installed before request admission.")

    queue = ShortRequestFirstRequestQueue(
        policy=policy,
        threshold=threshold,
        long_max_wait_ms=long_max_wait_ms,
        immediate_predicate=is_recovery_request,
    )
    scheduler.waiting = queue
    logger.info(
        "ShortRequestFirst waiting queue installed: threshold=%d, long_max_wait_ms=%.3f",
        threshold,
        long_max_wait_ms,
    )
    return queue
