# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime backport of vLLM PR #47728 for v0.25.1: free sliding-window blocks
on the processed-token basis under async scheduling. No-op on vLLM revisions
that already ship `Request.num_in_flight_tokens`."""
from functools import wraps

import vllm.v1.core.kv_cache_coordinator as _kvcc
import vllm.v1.core.sched.scheduler as _sched
from vllm.v1.request import Request

# request_id -> tokens scheduled but not yet settled (PR #47728 keeps this on
# Request.num_in_flight_tokens, which v0.25.1 lacks).
_inflight: dict[str, int] = {}

# max_concurrent_batches * max_num_batched_tokens, published at Scheduler init
# for patch_kv_cache_coordinator._select_kv_token_budget.
_inflight_token_budget: int | None = None

if not hasattr(Request, "num_in_flight_tokens"):
    _orig_remove_skipped_blocks = _kvcc.KVCacheCoordinator.remove_skipped_blocks
    _orig_scheduler_init = _sched.Scheduler.__init__
    _orig_update_after_schedule = _sched.Scheduler._update_after_schedule
    _orig_update_from_output = _sched.Scheduler.update_from_output

    @wraps(_orig_remove_skipped_blocks)
    def _remove_skipped_blocks(self, request_id, total_computed_tokens,
                               num_prompt_tokens=None):
        processed = max(0, total_computed_tokens - _inflight.get(request_id, 0))
        _orig_remove_skipped_blocks(self, request_id, processed, num_prompt_tokens)

    @wraps(_orig_scheduler_init)
    def _scheduler_init(self, *args, **kwargs):
        global _inflight_token_budget
        # vllm_config may arrive as a keyword arg (EngineCore passes it as one
        # and AsyncScheduler forwards via `super().__init__(*args, **kwargs)`)
        # or positionally; handle both. If neither, args[0] fast-fails.
        vllm_config = kwargs.get("vllm_config")
        _inflight_token_budget = (
            vllm_config.max_concurrent_batches
            * vllm_config.scheduler_config.max_num_batched_tokens)
        return _orig_scheduler_init(self, *args, **kwargs)

    @wraps(_orig_update_after_schedule)
    def _update_after_schedule(self, scheduler_output):
        ret = _orig_update_after_schedule(self, scheduler_output)
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            _inflight[rid] = _inflight.get(rid, 0) + n
        return ret

    @wraps(_orig_update_from_output)
    def _update_from_output(self, scheduler_output, model_runner_output):
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            v = _inflight.get(rid, 0) - n
            if v <= 0:
                _inflight.pop(rid, None)
            else:
                _inflight[rid] = v
        return _orig_update_from_output(self, scheduler_output, model_runner_output)

    _kvcc.KVCacheCoordinator.remove_skipped_blocks = _remove_skipped_blocks
    _sched.Scheduler.__init__ = _scheduler_init
    _sched.Scheduler._update_after_schedule = _update_after_schedule
    _sched.Scheduler.update_from_output = _update_from_output
