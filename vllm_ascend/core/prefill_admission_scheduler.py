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

"""Token-level prefill admission throttling for low-bandwidth PP.

The controller does not introduce a prefill/decode scheduler phase. It only
decides whether prefill tokens are eligible in the current scheduler step and,
when PP bubbles are available, narrows that step's token budget. The upstream
vLLM scheduler still owns request ordering, KV allocation, preemption, and the
per-request ``num_scheduled_tokens`` result used by both model runner versions.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm.logger import logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.ascend_config import PrefillAdmissionConfig, init_ascend_config


@dataclass(frozen=True)
class PrefillAdmissionDecision:
    """Admission constraint to apply to one token-level scheduling step."""

    throttle_prefills: bool
    token_budget: int | None
    reason: str
    pending_prefill_ids: frozenset[str]


class PrefillAdmissionController:
    """Compute periodic, decode-aware prefill admission decisions."""

    def __init__(
        self,
        config: PrefillAdmissionConfig,
        pipeline_parallel_size: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.prefill_interval = config.prefill_interval
        self.decode_low_watermark = config.decode_low_watermark or pipeline_parallel_size
        self.max_prefill_wait_s = config.max_prefill_wait_ms / 1000.0
        self.prefill_tokens_per_pp_bubble = config.prefill_tokens_per_pp_bubble
        self._clock = clock
        self._deferred_since: dict[str, float] = {}

    @staticmethod
    def _decode_token_demand(request: Request) -> int:
        return max(
            0,
            request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens,
        )

    def decide(
        self,
        running_requests: Iterable[Request],
        pending_prefills: Iterable[Request],
        *,
        scheduler_step: int,
        max_token_budget: int,
    ) -> PrefillAdmissionDecision:
        """Return the admission constraint for the next upstream schedule call."""
        now = self._clock()
        pending_prefill_ids = frozenset(request.request_id for request in pending_prefills)
        self._deferred_since = {
            request_id: deferred_at
            for request_id, deferred_at in self._deferred_since.items()
            if request_id in pending_prefill_ids
        }
        if not pending_prefill_ids:
            return PrefillAdmissionDecision(False, None, "no_prefill", pending_prefill_ids)

        eligible_decodes = [
            request
            for request in running_requests
            if not request.is_prefill_chunk
            and self._decode_token_demand(request) > 0
            and scheduler_step >= request.next_decode_eligible_step
        ]
        decode_batch_size = len(eligible_decodes)
        cadence_release = scheduler_step % self.prefill_interval == 0
        bubble_release = decode_batch_size < self.decode_low_watermark
        starvation_release = any(
            now - deferred_at >= self.max_prefill_wait_s for deferred_at in self._deferred_since.values()
        )

        if not (cadence_release or bubble_release or starvation_release):
            for request_id in pending_prefill_ids:
                self._deferred_since.setdefault(request_id, now)
            return PrefillAdmissionDecision(True, None, "decode_priority", pending_prefill_ids)

        if starvation_release:
            reason = "max_wait"
        elif bubble_release:
            reason = "pp_bubble"
        else:
            reason = "periodic"

        # Running decode demand remains first in the token budget. Missing
        # decode slots approximate available PP bubbles; cadence/max-wait
        # releases reserve one small prefill quantum even at the watermark.
        bubble_capacity = max(self.decode_low_watermark - decode_batch_size, 1)
        prefill_token_budget = bubble_capacity * self.prefill_tokens_per_pp_bubble
        decode_token_budget = min(
            max_token_budget,
            sum(self._decode_token_demand(request) for request in eligible_decodes),
        )
        token_budget = min(max_token_budget, decode_token_budget + prefill_token_budget)
        return PrefillAdmissionDecision(False, token_budget, reason, pending_prefill_ids)

    def observe(self, decision: PrefillAdmissionDecision, scheduler_output: SchedulerOutput) -> None:
        """Reset aging for prefills that made token-level progress."""
        scheduled_request_ids = scheduler_output.num_scheduled_tokens.keys()
        for request_id in decision.pending_prefill_ids.intersection(scheduled_request_ids):
            self._deferred_since.pop(request_id, None)


class _PrefillAdmissionSchedulerMixin:
    """Apply controller decisions while delegating scheduling to vLLM."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        scheduler_extension_config = init_ascend_config(self.vllm_config).scheduler_config
        admission_config = scheduler_extension_config.prefill_admission_config
        self._prefill_admission_controller: PrefillAdmissionController | None = None

        # Keep ShortRequestFirst composable with the admission constraint. The
        # installer is idempotent when a parent scheduler already installed it.
        short_request_config = scheduler_extension_config.short_request_first_config
        if short_request_config.enabled:
            from vllm_ascend.core.short_request_first_scheduler import install_short_request_first_waiting_queue

            install_short_request_first_waiting_queue(
                self,
                threshold=short_request_config.threshold,
                long_max_wait_ms=short_request_config.long_max_wait_ms,
            )

        if not admission_config.enabled:
            return

        self._prefill_admission_controller = PrefillAdmissionController(
            admission_config,
            self.parallel_config.pipeline_parallel_size,
        )
        logger.info(
            "Prefill admission throttling enabled: prefill_interval=%d, "
            "decode_low_watermark=%d, max_prefill_wait_ms=%.3f, "
            "prefill_tokens_per_pp_bubble=%d",
            admission_config.prefill_interval,
            self._prefill_admission_controller.decode_low_watermark,
            admission_config.max_prefill_wait_ms,
            admission_config.prefill_tokens_per_pp_bubble,
        )

    @staticmethod
    def _waiting_request_needs_prefill(request: Request) -> bool:
        return request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED) and (
            request.num_computed_tokens < request.num_tokens - 1
        )

    def _pending_prefills(self) -> list[Request]:
        pending: dict[str, Request] = {
            request.request_id: request for request in self.running if request.is_prefill_chunk
        }
        for request_queue in (self.waiting, self.skipped_waiting):
            for request in request_queue:
                if self._waiting_request_needs_prefill(request):
                    pending.setdefault(request.request_id, request)
                    # Queue order determines admission order. Tracking the
                    # first eligible Prefill avoids an O(waiting) hot-path scan
                    # while still aging the next request that can make progress.
                    break
        return list(pending.values())

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        controller = self._prefill_admission_controller
        if controller is None:
            return super().schedule(throttle_prefills)

        decision = controller.decide(
            self.running,
            self._pending_prefills(),
            scheduler_step=self.current_step + 1,
            max_token_budget=self.max_num_scheduled_tokens,
        )
        original_token_budget = self.max_num_scheduled_tokens
        original_prefill_capacity_bound = self.prefill_capacity_bound
        if decision.token_budget is not None:
            self.max_num_scheduled_tokens = decision.token_budget
        if decision.throttle_prefills:
            # Upstream DP throttling may release prefills when the prior release
            # was capacity-bound. This feature has its own bubble/max-wait
            # release rules, so suppress only that override for this call.
            self.prefill_capacity_bound = False

        try:
            scheduler_output = super().schedule(throttle_prefills or decision.throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = original_token_budget
            if decision.throttle_prefills:
                self.prefill_capacity_bound = original_prefill_capacity_bound

        controller.observe(decision, scheduler_output)
        return scheduler_output


class PrefillAdmissionScheduler(_PrefillAdmissionSchedulerMixin, Scheduler):
    """Synchronous vLLM scheduler with prefill admission throttling."""


class PrefillAdmissionAsyncScheduler(_PrefillAdmissionSchedulerMixin, AsyncScheduler):
    """Asynchronous vLLM scheduler with prefill admission throttling."""
