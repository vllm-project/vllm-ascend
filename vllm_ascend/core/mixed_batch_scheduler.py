# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.ascend_config import init_ascend_config


class MixedBatchScheduler(Scheduler):
    """Limit decode rows in a potential mixed step, then give decodes a step."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._decode_cap = init_ascend_config(self.vllm_config).scheduler_config.max_num_decode_seqs_in_mixed_batch
        if self._decode_cap is not None and not hasattr(Scheduler, "_reserve_prefill_lookahead"):
            raise ValueError("Mixed-batch decode limiting requires vLLM's prefill lookahead scheduling hook.")
        self._remaining_mixed_decodes: int | None = None
        self._drain_decodes = False

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if self._decode_cap is None:
            return super().schedule(throttle_prefills=throttle_prefills)

        capacity_bound = self.prefill_capacity_bound
        has_decode = any(not req.is_prefill_chunk for req in self.running)
        drain = self._drain_decodes and has_decode
        self._drain_decodes = False
        defer_prefills = has_decode and (drain or (throttle_prefills and not capacity_bound))
        waiting_can_prefill = (
            bool(self.waiting) and len(self.running) + self.num_waiting_for_streaming_input < self.max_num_active_reqs
        )
        if (
            has_decode
            and not defer_prefills
            and self.pause_state == PauseState.UNPAUSED
            and (waiting_can_prefill or any(req.is_prefill_chunk for req in self.running))
        ):
            self._remaining_mixed_decodes = self._decode_cap

        if drain:
            # The upstream throttle normally releases when the previous prefill
            # was capacity-bound. Keep the one-step decode hold in that case and
            # preserve the saturation state for the next ordinary step.
            self.prefill_capacity_bound = False
        try:
            return super().schedule(throttle_prefills=throttle_prefills or drain)
        finally:
            self._remaining_mixed_decodes = None
            if drain:
                self.prefill_capacity_bound = capacity_bound

    def _reserve_prefill_lookahead(self, request: Request, num_computed_tokens: int, num_new_tokens: int) -> int:
        num_new_tokens = super()._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)
        # The paired Scheduler calls this after computing runnable tokens and
        # before allocating KV slots. A zero result takes its ordinary skip
        # path with the request still in the running list.
        remaining = self._remaining_mixed_decodes
        if (
            remaining is None
            or num_new_tokens == 0
            or request.status != RequestStatus.RUNNING
            or request.is_prefill_chunk
        ):
            return num_new_tokens
        if remaining == 0:
            # Arm the drain even if the prefill later fails admission.
            self._drain_decodes = True
            return 0
        self._remaining_mixed_decodes = remaining - 1
        return num_new_tokens


class AsyncMixedBatchScheduler(MixedBatchScheduler, AsyncScheduler):
    """Mixed-batch policy with the upstream asynchronous bookkeeping."""
