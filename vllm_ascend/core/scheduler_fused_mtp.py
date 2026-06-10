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
"""Scheduler that preserves Fused MTP decode graph without starving prefill.

The parent vLLM scheduler first schedules RUNNING requests, then uses the
remaining token budget for WAITING requests. This scheduler keeps that parent
behavior intact and only decides whether WAITING queues should be visible.

Policy:
* If RUNNING is a full pure-decode batch, hide WAITING so the fused MTP decode
  graph can run.
* Once enough decode slots are open for a budget-sized refill batch, schedule a
  prefill-only refill pass for those slots. This avoids mixing new prefill with
  the existing decode batch, which would break the fused MTP decode graph.
* If only a small number of slots are open, wait for a short adaptive window so
  nearby completions can be grouped into a larger prefill-only refill pass.
* While that refill batch is still prefilling, hide additional WAITING requests
  so refill does not cascade into many mixed steps.

This is deliberately based on request phase and scheduler capacity, not on a
fixed max_num_seqs watermark.
"""

import math
import os
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler

_ENV_MAX_DECODE_STEPS = "VLLM_ASCEND_FUSED_MTP_MAX_DECODE_STEPS"
_ENV_MAX_REFILL_SLOTS = "VLLM_ASCEND_FUSED_MTP_MAX_REFILL_SLOTS"


class _FusedMTPSchedulerMixin:
    """Decode-priority scheduler for Fused MTP Full Graph."""

    # Attributes provided by the concrete vLLM scheduler base classes.
    vllm_config: VllmConfig
    running: list[Any]
    waiting: RequestQueue
    skipped_waiting: RequestQueue
    scheduler_config: Any
    max_num_scheduled_tokens: int
    max_num_running_reqs: int
    policy: Any
    _pause_state: PauseState

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._refill_in_progress = False
        self._decode_steps_since_refill = 0
        self._max_decode_steps_before_refill_override = self._get_max_decode_steps_before_refill_override()
        self._max_refill_slots_override = self._get_non_negative_int_env(_ENV_MAX_REFILL_SLOTS)

        self._fused_mtp_enabled = self._detect_fused_mtp(self.vllm_config)

        logger.info(
            "[FusedMTPScheduler] Initialized. "
            "fused_mtp_enabled=%s, refill_policy=adaptive_group_refill, "
            "max_decode_steps_before_refill=%s, max_refill_slots=%s",
            self._fused_mtp_enabled,
            (
                self._max_decode_steps_before_refill_override
                if self._max_decode_steps_before_refill_override is not None
                else "adaptive"
            ),
            (self._max_refill_slots_override if self._max_refill_slots_override is not None else "adaptive"),
        )

    @staticmethod
    def _get_non_negative_int_env(
        env_name: str,
        default: int | None = None,
    ) -> int | None:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            return default
        try:
            return max(0, int(raw_value))
        except ValueError:
            logger.warning(
                "Invalid %s=%r. Falling back to %s.",
                env_name,
                raw_value,
                default if default is not None else "adaptive",
            )
            return default

    def _schedule_parent(self) -> SchedulerOutput:
        raise NotImplementedError

    @classmethod
    def _get_max_decode_steps_before_refill_override(cls) -> int | None:
        return cls._get_non_negative_int_env(_ENV_MAX_DECODE_STEPS)

    @staticmethod
    def _is_pd_disaggregated(vllm_config: VllmConfig) -> bool:
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        kv_role = getattr(kv_transfer_config, "kv_role", None)
        return kv_role in ("kv_producer", "kv_consumer")

    @staticmethod
    def _detect_fused_mtp(vllm_config: VllmConfig) -> bool:
        if _FusedMTPSchedulerMixin._is_pd_disaggregated(vllm_config):
            return False
        spec_config = vllm_config.speculative_config
        if spec_config is None:
            return False
        method = getattr(spec_config, "method", None)
        if method not in ("mtp", "deepseek_mtp"):
            return False
        env_val = os.environ.get("VLLM_ASCEND_ENABLE_FUSED_MTP_FULL_GRAPH", "0")
        if int(env_val):
            return True
        try:
            from vllm_ascend.ascend_config import get_ascend_config

            return bool(
                getattr(
                    get_ascend_config(),
                    "enable_fused_mtp_full_graph",
                    False,
                )
            )
        except Exception:
            return False

    @staticmethod
    def _effective_computed_tokens(request) -> int:
        return max(0, request.num_computed_tokens - request.num_output_placeholders)

    @classmethod
    def _is_decode_request(cls, request) -> bool:
        return cls._effective_computed_tokens(request) >= request.num_prompt_tokens

    def _is_pure_decode(self) -> bool:
        if not self.running:
            return False
        return all(self._is_decode_request(req) for req in self.running)

    def _has_running_prefill(self) -> bool:
        return any(not self._is_decode_request(req) for req in self.running)

    def _has_running_decode(self) -> bool:
        return any(self._is_decode_request(req) for req in self.running)

    def _estimate_prefill_tokens(self, request) -> int:
        num_new_tokens = max(0, request.num_tokens - request.num_computed_tokens)
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold
        return min(num_new_tokens, self.max_num_scheduled_tokens)

    def _estimate_decode_tokens(self, request) -> int:
        num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
        if num_new_tokens <= 0:
            return 0
        threshold = self.scheduler_config.long_prefill_token_threshold
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold
        return min(num_new_tokens, self.max_num_scheduled_tokens)

    def _peek_waiting_prefill_tokens(self) -> int:
        for request_queue in (self.waiting, self.skipped_waiting):
            for request in request_queue:
                num_new_tokens = self._estimate_prefill_tokens(request)
                if num_new_tokens > 0:
                    return num_new_tokens
        return 0

    def _get_refill_slots_target(self) -> int:
        if self.max_num_running_reqs <= 1:
            return 1

        prefill_tokens = self._peek_waiting_prefill_tokens()
        if prefill_tokens <= 0:
            return 1

        decode_token_budget = sum(
            self._estimate_decode_tokens(req) for req in self.running if self._is_decode_request(req)
        )
        prefill_budget = max(1, self.max_num_scheduled_tokens - decode_token_budget)
        refill_slots = prefill_budget // prefill_tokens
        if refill_slots <= 0:
            refill_slots = 1
        if self._max_refill_slots_override is not None:
            max_refill_slots = self._max_refill_slots_override
        else:
            max_refill_slots = self.max_num_running_reqs
        refill_slots = min(refill_slots, max_refill_slots)
        return max(1, min(refill_slots, self.max_num_running_reqs))

    def _get_decode_steps_before_refill_limit(
        self,
        refill_slots_target: int,
    ) -> int:
        if self._max_decode_steps_before_refill_override is not None:
            return self._max_decode_steps_before_refill_override

        return max(
            1,
            math.ceil(self.max_num_running_reqs / max(1, refill_slots_target)),
        )

    @staticmethod
    def _estimate_decode_steps_until_length_cap(request) -> int | None:
        max_tokens = getattr(request, "max_tokens", None)
        if max_tokens is None:
            return None

        estimates: list[int] = []
        num_output_tokens = getattr(request, "num_output_tokens", None)
        if num_output_tokens is not None:
            estimates.append(max(0, max_tokens - num_output_tokens))

        num_prompt_tokens = getattr(request, "num_prompt_tokens", None)
        num_computed_tokens = getattr(request, "num_computed_tokens", None)
        if num_prompt_tokens is not None and num_computed_tokens is not None:
            num_output_placeholders = getattr(request, "num_output_placeholders", 0)
            effective_computed_tokens = max(
                0,
                num_computed_tokens - num_output_placeholders,
            )
            estimates.append(
                max(
                    0,
                    num_prompt_tokens + max_tokens - effective_computed_tokens,
                )
            )

        if not estimates:
            return None
        return min(estimates)

    def _get_group_aware_decode_steps_before_refill_limit(
        self,
        refill_slots_target: int,
        open_slots: int,
    ) -> int:
        base_limit = self._get_decode_steps_before_refill_limit(refill_slots_target)
        needed_slots = refill_slots_target - open_slots
        if needed_slots <= 0:
            return base_limit

        steps_until_finished = sorted(
            steps
            for req in self.running
            if self._is_decode_request(req)
            for steps in [self._estimate_decode_steps_until_length_cap(req)]
            if steps is not None
        )
        if len(steps_until_finished) < needed_slots:
            return base_limit

        steps_to_group = steps_until_finished[needed_slots - 1]
        group_window = base_limit * max(1, refill_slots_target + needed_slots)
        if steps_to_group <= group_window:
            return max(base_limit, steps_to_group)
        return base_limit

    def _is_waiting_for_refill_group(
        self,
        pure_decode: bool,
        has_waiting: bool,
    ) -> bool:
        if not pure_decode or not has_waiting:
            return False
        if len(self.running) >= self.max_num_running_reqs:
            return False
        open_slots = self.max_num_running_reqs - len(self.running)
        return open_slots < self._get_refill_slots_target()

    def _update_fused_mtp_state_after_schedule(
        self,
        output: SchedulerOutput,
        pre_pure_decode: bool,
        pre_has_waiting: bool,
        pre_running: int,
        waiting_hidden: bool,
    ) -> None:
        if waiting_hidden:
            if self._is_waiting_for_refill_group(pre_pure_decode, pre_has_waiting):
                self._decode_steps_since_refill += 1
        elif pre_pure_decode and pre_has_waiting and pre_running < self.max_num_running_reqs:
            self._refill_in_progress = bool(output.scheduled_new_reqs)
            if output.scheduled_new_reqs:
                self._decode_steps_since_refill = 0

        if not self._has_running_prefill():
            self._refill_in_progress = False

    @staticmethod
    def _prepend_queue(dst: RequestQueue, src: RequestQueue) -> None:
        for req in reversed(list(src)):
            dst.prepend_request(req)

    def _schedule_with_waiting_hidden(self) -> SchedulerOutput:
        saved_waiting = self.waiting
        saved_skipped = self.skipped_waiting
        tmp_waiting = create_request_queue(self.policy)
        tmp_skipped = create_request_queue(self.policy)
        self.waiting = tmp_waiting
        self.skipped_waiting = tmp_skipped

        try:
            output = self._schedule_parent()
        finally:
            self._prepend_queue(saved_waiting, tmp_waiting)
            self._prepend_queue(saved_skipped, tmp_skipped)
            self.waiting = saved_waiting
            self.skipped_waiting = saved_skipped

        return output

    def _schedule_prefill_only_refill(self, open_slots: int) -> SchedulerOutput:
        saved_running = self.running
        saved_max_num_running_reqs = self.max_num_running_reqs
        self.running = []
        self.max_num_running_reqs = max(1, min(open_slots, saved_max_num_running_reqs))

        try:
            output = self._schedule_parent()
            refill_running = self.running
        except Exception:
            self.running = saved_running
            raise
        finally:
            self.max_num_running_reqs = saved_max_num_running_reqs

        self.running = saved_running + refill_running
        return output

    def _should_hide_waiting(
        self,
        pure_decode: bool,
        has_waiting: bool,
    ) -> bool:
        if not has_waiting or self._pause_state != PauseState.UNPAUSED:
            return False

        if self._refill_in_progress:
            if not self._has_running_prefill():
                self._refill_in_progress = False
            elif self._has_running_decode():
                return True
            else:
                self._refill_in_progress = False

        if not pure_decode:
            return False

        if len(self.running) == self.max_num_running_reqs:
            return True

        open_slots = self.max_num_running_reqs - len(self.running)
        refill_slots_target = self._get_refill_slots_target()
        if open_slots >= refill_slots_target:
            return False

        max_decode_steps = self._get_group_aware_decode_steps_before_refill_limit(
            refill_slots_target,
            open_slots,
        )
        if max_decode_steps <= 0:
            return True
        return self._decode_steps_since_refill < max_decode_steps

    def schedule(self) -> SchedulerOutput:
        if not self._fused_mtp_enabled:
            return self._schedule_parent()

        pure_decode = self._is_pure_decode()
        has_waiting = bool(self.waiting or self.skipped_waiting)
        if not has_waiting:
            self._decode_steps_since_refill = 0
        pre_running = len(self.running)
        should_hide = self._should_hide_waiting(pure_decode, has_waiting)

        if should_hide:
            output = self._schedule_with_waiting_hidden()
            self._update_fused_mtp_state_after_schedule(
                output,
                pure_decode,
                has_waiting,
                pre_running,
                waiting_hidden=True,
            )
            return output

        if pure_decode and has_waiting and pre_running < self.max_num_running_reqs:
            output = self._schedule_prefill_only_refill(self.max_num_running_reqs - pre_running)
        else:
            output = self._schedule_parent()
        self._update_fused_mtp_state_after_schedule(
            output,
            pure_decode,
            has_waiting,
            pre_running,
            waiting_hidden=False,
        )

        return output


class FusedMTPScheduler(_FusedMTPSchedulerMixin, Scheduler):
    """Synchronous fused MTP scheduler."""

    def _schedule_parent(self) -> SchedulerOutput:
        return Scheduler.schedule(self)


class FusedMTPAsyncScheduler(_FusedMTPSchedulerMixin, AsyncScheduler):
    """Async fused MTP scheduler preserving vLLM async speculative semantics."""

    def _schedule_parent(self) -> SchedulerOutput:
        return AsyncScheduler.schedule(self)
