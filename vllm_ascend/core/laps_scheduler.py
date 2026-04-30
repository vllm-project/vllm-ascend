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
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.request import Request

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
        self._debug_logging_enabled = logger.isEnabledFor(logging.DEBUG)

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
            "enqueues=%s dispatches=%s removals=%s short_ready_reasons=%s",
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
        )

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

    def _classify_queue(self, request: Request) -> RequestQueue:
        if self.immediate_predicate is not None and self.immediate_predicate(
            request
        ):
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

    def add_request(self, request: Request) -> None:
        queue = self._classify_queue(request)
        queue.add_request(request)
        self._enqueue_counters[self._queue_name(queue)] += 1
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

    def prepend_request(self, request: Request) -> None:
        queue = self._classify_queue(request)
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
            if request in queue:
                queue.remove_request(request)
                self._remove_counters[self._queue_name(queue)] += 1
                if queue is self._short_queue:
                    self._on_short_queue_changed()
                self._debug_state("remove", request=request, queue=queue)
                self._maybe_log_stats()
                return
        raise ValueError("request not found in LAPS queue")

    def remove_requests(self, requests: Iterable[Request]) -> None:
        queue_to_requests: dict[int, list[Request]] = {}
        queue_map = {id(queue): queue for queue in self._queues()}
        removed_count = 0
        for request in requests:
            for queue in self._queues():
                if request in queue:
                    queue_to_requests.setdefault(id(queue), []).append(request)
                    break
        for queue_id, matched_requests in queue_to_requests.items():
            removed_count += len(matched_requests)
            queue = queue_map[queue_id]
            self._remove_counters[self._queue_name(queue)] += len(matched_requests)
            queue.remove_requests(matched_requests)
        self._on_short_queue_changed()
        if removed_count:
            self._debug_state("remove_batch", extra=f"count={removed_count}")
            self._maybe_log_stats()

    def __bool__(self) -> bool:
        return self._select_schedulable_queue() is not None

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
        return any(candidate is request for candidate in self)


class LAPSSchedulerMixin:
    """Inject a LAPS-style waiting queue into vLLM's scheduler."""

    def _init_laps_waiting_queue(
        self,
        immediate_predicate: Callable[[Request], bool] | None = None,
    ) -> None:
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
            "wait_window_ms=%.3f, wait_max_batch=%d",
            threshold,
            wait_window_ms,
            wait_max_batch,
        )


from vllm.v1.core.sched.scheduler import Scheduler as BaseScheduler


class LAPSScheduler(LAPSSchedulerMixin, BaseScheduler):
    """vLLM scheduler with the Ascend LAPS waiting queue installed."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_laps_waiting_queue()
