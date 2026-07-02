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
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from vllm.logger import logger
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.ascend_config import get_ascend_config


@runtime_checkable
class SteppedWaitingQueue(Protocol):
    """Extension contract a waiting queue must satisfy for LAPS-style scheduling.

    The scheduler ([`LapsSchedulerMixin`]) depends only on this surface, not on
    `LapsRequestQueue`'s concrete internals. It is a superset of the base
    `RequestQueue` ABC with the per-step / queue-aware hooks LAPS needs:

    - `begin_step`: refill per-step admission state once per `schedule()`.
    - `select_waiting_queue_for_scheduling`: pick the sub-queue to pop from next.
    - `has_schedulable_requests`: whether anything can be dispatched right now.
    - `pop_request_from_queue`: pop from a specific sub-queue with accounting
      (dispatch vs. skip-or-requeue, token charging).
    - `mark_force_immediate`: route a request to the immediate lane on re-add.
    - `owns_queue`: whether a given queue object is one of its sub-queues.
    """

    def begin_step(self, token_budget: int) -> None: ...

    def select_waiting_queue_for_scheduling(self) -> RequestQueue | None: ...

    def has_schedulable_requests(self) -> bool: ...

    def pop_request_from_queue(
        self,
        queue: RequestQueue,
        *,
        count_as_removal: bool = ...,
        skip_or_requeue_reason: str | None = ...,
        long_charge_tokens: int = ...,
    ) -> Request: ...

    def mark_force_immediate(self, request_id: str) -> None: ...

    def owns_queue(self, queue: RequestQueue | None) -> bool: ...


class LapsRequestQueue(RequestQueue):
    """Two-level waiting queue for short and long prefills."""

    _SKIP_OR_REQUEUE_REASONS = (
        "blocked_waiting_status",
        "max_loras",
        "remote_kv_not_ready",
    )

    def __init__(
        self,
        policy: SchedulingPolicy,
        threshold: int,
        long_max_wait_ms: float,
        long_token_reservation: float = 0.0,
        long_burst_steps: int = 4,
        immediate_predicate: Callable[[Request], bool] | None = None,
        stats_log_interval_s: float = 0.0,
    ) -> None:
        # Inputs are assumed pre-validated: the only production caller
        # (LapsSchedulerMixin) builds this from a LapsConfig, which is the single
        # validation boundary (it raises on out-of-range values). This
        # constructor therefore trusts its arguments rather than re-clamping them.
        self.policy = policy
        self.threshold = threshold
        # Burst allowance for the aged-long admission bucket, in scheduling
        # steps: the bucket caps at `reservation * token_budget *
        # long_burst_steps`, so an idle aged-long lane accumulates at most this
        # many steps' worth of reservation before a burst of admissions,
        # bounding short-request impact.
        self.long_burst_steps = long_burst_steps
        # Anti-starvation: a long request waiting longer than this many ms becomes
        # eligible for promotion ahead of short prefills, but only through the
        # token-reservation bucket (see begin_step), which rate-limits aged-long
        # admissions to the reservation fraction of token throughput. The bucket
        # is the single admission channel, so a backlog of aged longs can never
        # flip the queue into long-first and starve shorts. <= 0 disables aging
        # entirely; when > 0 it requires long_token_reservation > 0 (enforced by
        # LapsConfig), otherwise the bucket never refills and aging is inert.
        self.long_max_wait_ms = long_max_wait_ms
        # Average fraction of token throughput reserved for admitting aged-long
        # prefills ahead of waiting shorts (token-bucket rate; see begin_step).
        # This bucket is the only channel that promotes an aged long ahead of a
        # waiting short, so it must be > 0 whenever aging is enabled (enforced by
        # LapsConfig). Larger values drain aged-long requests sooner at the cost
        # of short-request latency. Expected range [0.0, 1.0].
        self.long_token_reservation = long_token_reservation
        self.immediate_predicate = immediate_predicate
        self._immediate_queue = create_request_queue(policy)
        self._short_queue = create_request_queue(policy)
        self._long_queue = create_request_queue(policy)
        # Tracks when each request entered the long queue, for aging.
        self._long_enqueue_at: dict[str, float] = {}
        # Aged-long requests admitted ahead of waiting shorts (soft phase only).
        self._long_starvation_promotions = 0
        # Token bucket throttling aged-long admissions (see begin_step). The
        # bucket refills by `long_token_reservation * token_budget` every
        # scheduling step and is charged a long's full remaining prefill each time
        # an aged long is admitted ahead of waiting shorts. Gating on `bucket > 0`
        # caps the average aged-long admission rate at the reservation fraction of
        # token throughput: a big long drives the bucket well below zero, and the
        # per-step refill repays that debt over the steps the long actually runs,
        # so shorts are preferred until it is repaid (a bounded burst, no cliff).
        self._long_bucket = 0.0
        self._long_bucket_capacity = 0.0
        self._long_refill_per_step = 0.0
        # Total tokens charged to the bucket so far (observability).
        self._long_tokens_charged = 0
        self._stats_log_interval_s = stats_log_interval_s
        self._last_stats_log_at = time.monotonic()
        self._prepend_counters = {"immediate": 0, "short": 0, "long": 0}
        self._dispatch_counters = {"immediate": 0, "short": 0, "long": 0}
        self._skip_or_requeue_counters = {
            reason: {"immediate": 0, "short": 0, "long": 0} for reason in self._SKIP_OR_REQUEUE_REASONS
        }
        self._force_immediate_request_ids: set[str] = set()
        self._queue_index: dict[str, RequestQueue] = {}

    def _queues(self) -> tuple[RequestQueue, ...]:
        return (self._immediate_queue, self._short_queue, self._long_queue)

    @property
    def _debug_logging_enabled(self) -> bool:
        # Checked live (not cached at __init__) so a runtime log-level change
        # takes effect. logger.isEnabledFor is cheap (cached effective level);
        # the guard exists only to skip building the debug-log arguments.
        return logger.isEnabledFor(logging.DEBUG)

    def owns_queue(self, queue: RequestQueue | None) -> bool:
        """Whether `queue` is one of this queue's internal sub-queues.

        Public accessor used by the scheduler to decide whether a queue returned
        by `select_waiting_queue_for_scheduling` should be popped through this
        wrapper's accounting (vs. an externally-owned queue such as
        `skipped_waiting`). Avoids reaching into `_queues()` from outside.
        """
        return any(queue is q for q in self._queues())

    def _queue_name(self, queue: RequestQueue) -> str:
        if queue is self._immediate_queue:
            return "immediate"
        if queue is self._short_queue:
            return "short"
        if queue is self._long_queue:
            return "long"
        return "unknown"

    def _maybe_log_stats(self, force: bool = False) -> None:
        if self._stats_log_interval_s <= 0:
            return
        now = time.monotonic()
        if not force and (now - self._last_stats_log_at) < self._stats_log_interval_s:
            return
        self._last_stats_log_at = now
        logger.info(
            "LAPS stats: threshold=%d long_max_wait_ms=%.3f "
            "long_token_reservation=%.3f "
            "sizes=(immediate=%d short=%d long=%d) "
            "prepends=%s dispatches=%s skip_or_requeues=%s "
            "long_starvation_promotions=%d "
            "long_tokens_charged=%d "
            "bucket=%.0f capacity=%.0f",
            self.threshold,
            self.long_max_wait_ms,
            self.long_token_reservation,
            len(self._immediate_queue),
            len(self._short_queue),
            len(self._long_queue),
            self._prepend_counters,
            self._dispatch_counters,
            self._skip_or_requeue_counters,
            self._long_starvation_promotions,
            self._long_tokens_charged,
            self._long_bucket,
            self._long_bucket_capacity,
        )

    def _increment_skip_or_requeue_counter(self, queue: RequestQueue, reason: str) -> None:
        if reason not in self._skip_or_requeue_counters:
            raise ValueError(f"Unknown skip_or_requeue reason: {reason}")
        self._skip_or_requeue_counters[reason][self._queue_name(queue)] += 1

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

    def has_short_requests(self) -> bool:
        return len(self._short_queue) > 0

    def has_long_requests(self) -> bool:
        return len(self._long_queue) > 0

    def has_immediate_requests(self) -> bool:
        return len(self._immediate_queue) > 0

    def _classify_queue(self, request: Request, *, force_immediate: bool = False) -> RequestQueue:
        if force_immediate or request.request_id in self._force_immediate_request_ids:
            return self._immediate_queue
        if self.immediate_predicate is not None and self.immediate_predicate(request):
            return self._immediate_queue
        if request.num_prompt_tokens <= self.threshold:
            return self._short_queue
        return self._long_queue

    def _long_head_wait_ms(self) -> float | None:
        """Milliseconds the oldest long request has waited, or None if N/A.

        Single source of truth for the long-queue head age (one peek +
        monotonic() + dict lookup), reused by the soft and hard aging checks so
        the hot path does not recompute it.
        """
        if self.long_max_wait_ms <= 0 or not self._long_queue:
            return None
        head = self._long_queue.peek_request()  # FCFS head = oldest long request
        enqueued_at = self._long_enqueue_at.get(head.request_id)
        if enqueued_at is None:
            return None
        return (time.monotonic() - enqueued_at) * 1000.0

    def begin_step(self, token_budget: int) -> None:
        """Refill the aged-long admission token bucket for this scheduling step.

        Called once at the start of each schedule(). The bucket refills by
        `long_token_reservation * token_budget` tokens per step (capped at
        `long_burst_steps` steps' worth), so over any window the aged-long lane
        is admitted at an average rate of the reservation fraction of token
        throughput. With reservation=0 the bucket stays empty, so an aged long is
        admitted only via the stall-avoidance path (no short waiting); LapsConfig
        therefore requires reservation > 0 whenever aging is enabled.
        """
        self._long_refill_per_step = self.long_token_reservation * token_budget
        self._long_bucket_capacity = self._long_refill_per_step * self.long_burst_steps
        self._long_bucket = min(
            self._long_bucket_capacity,
            self._long_bucket + self._long_refill_per_step,
        )
        if self._debug_logging_enabled:
            logger.debug(
                "LAPS begin_step: token_budget=%d long_token_reservation=%.3f "
                "refill_per_step=%.0f bucket=%.0f capacity=%.0f",
                token_budget,
                self.long_token_reservation,
                self._long_refill_per_step,
                self._long_bucket,
                self._long_bucket_capacity,
            )

    def _debug_select(self, event: str) -> None:
        """Lazy debug log for a selection decision (skips f-string when off)."""
        if self._debug_logging_enabled:
            self._debug_state(
                event,
                extra=f"bucket={self._long_bucket:.0f} capacity={self._long_bucket_capacity:.0f}",
            )

    def _select_schedulable_queue(self) -> RequestQueue | None:
        # Pure query (no side effects): called repeatedly per scheduling step.
        if self._immediate_queue:
            return self._immediate_queue
        if self._long_queue:
            # Read the long-queue head age once for the aging check.
            wait_ms = self._long_head_wait_ms()
            if wait_ms is not None and wait_ms >= self.long_max_wait_ms:
                # The token-reservation bucket is the single channel for admitting
                # aged longs ahead of waiting shorts, rate-limited to the
                # reservation fraction of token throughput. When the short queue
                # is non-empty, only promote long while the bucket has credit;
                # when it is empty, always allow long (stall avoidance). There is
                # deliberately no unconditional deadline bypass: under sustained
                # overload every long is perpetually past any absolute wall-clock
                # bound, so a bypass would degenerate into long-first scheduling
                # and reintroduce the head-of-line blocking LAPS exists to remove.
                if not self._short_queue:
                    # Stall avoidance: allow long when no short is waiting.
                    self._debug_select("aged-long_selected_stall_avoidance")
                    return self._long_queue
                if self._long_bucket > 0.0:
                    # Reservation credit available: promote aged-long ahead.
                    self._debug_select("aged-long_selected_with_budget")
                    return self._long_queue
                # Bucket exhausted and short queue non-empty: prefer short so the
                # aged-long admission rate stays bounded by the reservation (fall
                # through below).
                self._debug_select("aged-long_blocked_budget_exhausted")
        if self._short_queue:
            return self._short_queue
        if self._long_queue:
            return self._long_queue
        return None

    def has_schedulable_requests(self) -> bool:
        """Return whether a request can be dispatched right now."""
        return self._select_schedulable_queue() is not None

    def select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        return self._select_schedulable_queue()

    def mark_force_immediate(self, request_id: str) -> None:
        self._force_immediate_request_ids.add(request_id)

    @staticmethod
    def _request_id(request: Request | object) -> str | None:
        return getattr(request, "request_id", None)

    def add_request(self, request: Request) -> None:
        queue = self._classify_queue(request)
        queue.add_request(request)
        self._queue_index[request.request_id] = queue
        if queue is self._long_queue:
            self._long_enqueue_at.setdefault(request.request_id, time.monotonic())
        if queue is not self._immediate_queue:
            self._force_immediate_request_ids.discard(request.request_id)
        self._debug_state("enqueue", request=request, queue=queue)
        self._maybe_log_stats()

    def pop_request(self) -> Request:
        queue = self._select_schedulable_queue()
        if queue is None:
            raise IndexError("pop from empty LAPS queue")
        return self.pop_request_from_queue(queue)

    def pop_request_from_queue(
        self,
        queue: RequestQueue,
        *,
        count_as_removal: bool = False,
        skip_or_requeue_reason: str | None = None,
        long_charge_tokens: int = 0,
    ) -> Request:
        # `long_charge_tokens` is the cost charged to the aged-long token bucket
        # when this dispatch is an aged-long promotion ahead of waiting shorts.
        # The caller passes the long's *full remaining prefill* (num_tokens -
        # num_computed_tokens), not just this step's chunk: a long monopolizes
        # the per-step token budget for many steps, so charging only the first
        # chunk would let the per-step bucket refill outpace the charge and the
        # reservation would never actually throttle longs.
        request = queue.pop_request()
        self._queue_index.pop(request.request_id, None)
        enqueued_at = self._long_enqueue_at.pop(request.request_id, None)
        if count_as_removal:
            if skip_or_requeue_reason is not None:
                self._increment_skip_or_requeue_counter(queue, skip_or_requeue_reason)
                event_name = f"skip_or_requeue:{skip_or_requeue_reason}"
            else:
                event_name = "remove"
        else:
            self._dispatch_counters[self._queue_name(queue)] += 1
            event_name = "dispatch"
            # Count long requests dispatched after aging past their bound.
            wait_ms = (time.monotonic() - enqueued_at) * 1000.0 if enqueued_at is not None else None
            aged = (
                queue is self._long_queue
                and self.long_max_wait_ms > 0
                and wait_ms is not None
                and wait_ms >= self.long_max_wait_ms
            )
            if aged:
                # A stall-avoidance admission (no short waiting) did not jump the
                # queue, so it counts as neither a starvation promotion nor a
                # bucket charge; only count/charge when shorts were actually
                # waiting and got passed. Such a jump can only have been selected
                # via the soft phase (bucket had credit), so it is always charged
                # the long's full remaining prefill. The bucket may dip far below
                # zero (bounded by one long's prefill); the per-step refill repays
                # that debt over the many steps the long actually runs, so over
                # any window aged longs are admitted at the reservation fraction
                # of token throughput. While the debt is outstanding, shorts are
                # preferred (a negative bucket blocks further soft promotions),
                # except that stall avoidance still admits a long when no short is
                # waiting (that branch precedes the bucket check in selection).
                jumped_shorts = len(self._short_queue) > 0
                charged = False
                if jumped_shorts:
                    self._long_starvation_promotions += 1
                    if long_charge_tokens > 0:
                        charged = True
                        self._long_bucket -= long_charge_tokens
                        self._long_tokens_charged += long_charge_tokens
                if self._debug_logging_enabled:
                    self._debug_state(
                        "aged-long_dispatched",
                        request=request,
                        queue=queue,
                        extra=f"long_charge_tokens={long_charge_tokens} charged={charged} "
                        f"bucket={self._long_bucket:.0f} "
                        f"capacity={self._long_bucket_capacity:.0f}",
                    )
        self._force_immediate_request_ids.discard(request.request_id)
        self._debug_state(event_name, request=request, queue=queue)
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
        self._queue_index[request.request_id] = queue
        if queue is self._long_queue:
            self._long_enqueue_at.setdefault(request.request_id, time.monotonic())
        self._prepend_counters[self._queue_name(queue)] += 1
        self._debug_state("prepend", request=request, queue=queue)
        self._maybe_log_stats()

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(cast(Request, request))

    def remove_request(self, request: Request) -> None:
        queue = self._queue_index.get(request.request_id)
        if queue is None:
            raise ValueError("request not found in LAPS queue")
        # `_queue_index` tracks the canonical Request object the sub-queue holds
        # (the scheduler keeps one object per request_id), so hand it straight to
        # the sub-queue and let it match by identity, exactly as the base vLLM
        # queues do. Remove from the sub-queue *before* mutating our bookkeeping
        # so a missing request raises ValueError without corrupting `_queue_index`.
        queue.remove_request(request)
        self._queue_index.pop(request.request_id, None)
        self._long_enqueue_at.pop(request.request_id, None)
        self._force_immediate_request_ids.discard(request.request_id)
        self._debug_state("remove", request=request, queue=queue)
        self._maybe_log_stats()

    def remove_requests(self, requests: Iterable[Request]) -> None:
        queue_to_requests: dict[int, list[Request]] = {}
        queue_map = {id(q): q for q in self._queues()}
        removed_count = 0
        # Group the requests by sub-queue via `_queue_index` (O(1) each) and hand
        # the objects straight through. A previous per-request linear scan here
        # made this batch path O(k * queue_len); dropping it restores the base
        # queue's single-pass O(queue_len), which matters when the waiting queue
        # is long (i.e. exactly the overload case LAPS targets).
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
                self._force_immediate_request_ids.discard(matched.request_id)
        if removed_count:
            self._debug_state("remove_batch", extra=f"count={removed_count}")
            self._maybe_log_stats()

    def __bool__(self) -> bool:
        # Equivalent to `_select_schedulable_queue() is not None` (that query
        # returns None iff all three sub-queues are empty), but cheaper: this is
        # on the scheduler's per-step hot path (the `while self.waiting` guard),
        # so avoid the peek + monotonic() + starvation checks here. Use
        # `has_schedulable_requests()` when the full selection logic is needed.
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


if TYPE_CHECKING:
    # Compile-time assertion that LapsRequestQueue satisfies the extension
    # contract the scheduler depends on (see SteppedWaitingQueue).
    _laps_queue_conforms: type[SteppedWaitingQueue] = LapsRequestQueue


class LapsSchedulerMixin:
    """Inject a LAPS-style waiting queue into vLLM's scheduler."""

    if TYPE_CHECKING:
        # `policy` is provided by vLLM's Scheduler base at runtime; the mixin is
        # always combined with it (see RecomputeScheduler). Declared here only
        # for the type checker so self.policy resolves without faking
        # inheritance. The super() calls below reach Scheduler through the same
        # combination and are annotated with type: ignore[misc] accordingly.
        policy: SchedulingPolicy

    def _is_recovery_request(self, request: Request) -> bool:
        """Recovery-style requests are prioritized via the immediate queue."""
        return (
            request.status == RequestStatus.PREEMPTED
            or request.num_computed_tokens > 0
            or request.num_external_computed_tokens > 0
            or request.num_output_tokens > 0
        )

    def _init_laps_waiting_queue(
        self,
        immediate_predicate: Callable[[Request], bool] | None = None,
    ) -> None:
        if self.policy != SchedulingPolicy.FCFS:
            # The non-FCFS warning is emitted once at config time in
            # NPUPlatform.check_and_update_config; keep this path quiet (debug
            # only) so the same condition is not logged twice.
            logger.debug(
                "LAPS scheduling supports only FCFS policy (current=%s); keeping the default waiting queue.",
                self.policy,
            )
            return

        if immediate_predicate is None:
            immediate_predicate = self._is_recovery_request
        laps_config = get_ascend_config().laps_config
        self.waiting = LapsRequestQueue(
            policy=self.policy,
            threshold=laps_config.threshold,
            long_max_wait_ms=laps_config.long_max_wait_ms,
            long_token_reservation=laps_config.long_token_reservation,
            long_burst_steps=laps_config.long_burst_steps,
            immediate_predicate=immediate_predicate,
            stats_log_interval_s=laps_config.stats_log_interval_s,
        )
        logger.info(
            "LAPS scheduling enabled on Ascend: threshold=%d, long_max_wait_ms=%.3f, long_token_reservation=%.3f",
            laps_config.threshold,
            laps_config.long_max_wait_ms,
            laps_config.long_token_reservation,
        )

    def _laps_waiting_queue(self) -> LapsRequestQueue | None:
        if isinstance(self.waiting, LapsRequestQueue):
            return self.waiting
        return None

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        waiting = getattr(self, "waiting", None)
        if isinstance(waiting, LapsRequestQueue):
            skipped_waiting = getattr(self, "skipped_waiting", None)
            if self.policy == SchedulingPolicy.FCFS and skipped_waiting:
                return skipped_waiting
            queue = waiting.select_waiting_queue_for_scheduling()
            if queue is not None:
                return queue
            return skipped_waiting or None
        return super()._select_waiting_queue_for_scheduling()  # type: ignore[misc]

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        waiting = getattr(self, "waiting", None)
        if isinstance(waiting, LapsRequestQueue):
            waiting.mark_force_immediate(request.request_id)
        super()._preempt_request(request, timestamp)  # type: ignore[misc]
