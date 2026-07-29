# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Backport of vLLM PR #47728 ("[Bugfix][V1] Free out-of-window blocks on the
processed-token basis under async scheduling") for vLLM v0.25.1.

v0.25.1's ``KVCacheManager.allocate_slots`` frees sliding-window (and
chunked-local) blocks on the *optimistic* ``num_computed_tokens``. Under
``--async-scheduling`` + speculative decoding that value leads the real
committed position by the in-flight step's tokens (the async look-ahead plus
``1 + num_speculative_tokens``; rejected draft tokens roll it back further).
Out-of-window blocks that are still inside the in-flight step's attention
window get freed, recycled by another request, and overwritten before that step
finishes reading them (a load-WAR). The in-flight step's SWA attention (target
verification + DSpark draft) then reads foreign KV, the draft proposal quality
collapses, and the spec-decode acceptance length drops (e.g. 4.3 -> 2.5).
"""

from functools import wraps

import vllm.v1.core.kv_cache_coordinator as _kvcc
import vllm.v1.core.sched.scheduler as _sched
from vllm.v1.request import Request

# Active only on vLLM revisions missing `Request.num_in_flight_tokens`, i.e.
# real v0.25.1 (PR #47728 is absent). On newer vLLM the whole module is a no-op.
_NEEDS_BACKPORT = not hasattr(Request, "num_in_flight_tokens")

_inflight: dict[str, int] = {}

_inflight_token_budget: int | None = None

_PATCH_FLAG = "_vllm_ascend_swa_inflight_patched"


def _inflight_for(request_id: str) -> int:
    # 0 is the correct value for an untracked request (no step in flight), not a
    # degraded fallback: it matches PR #47728's `Request.num_in_flight_tokens`
    # default and makes the free fall back to the committed basis for the first
    # schedule of a request, before any step has been optimistically counted.
    return _inflight.get(request_id, 0)


if _NEEDS_BACKPORT:
    _orig_remove_skipped_blocks = _kvcc.KVCacheCoordinator.remove_skipped_blocks
    _orig_scheduler_init = _sched.Scheduler.__init__
    _orig_update_after_schedule = _sched.Scheduler._update_after_schedule
    _orig_update_from_output = _sched.Scheduler.update_from_output

    @wraps(_orig_remove_skipped_blocks)
    def _remove_skipped_blocks(
        self,
        request_id: str,
        total_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        # Free on the processed-token basis: subtract tokens of steps whose
        # output is not yet settled, which `num_computed_tokens` still counts.
        processed = max(0, total_computed_tokens - _inflight_for(request_id))
        _orig_remove_skipped_blocks(self, request_id, processed, num_prompt_tokens)

    @wraps(_orig_scheduler_init)
    def _scheduler_init(self, *args, **kwargs):
        global _inflight_token_budget
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        if vllm_config is not None:
            _inflight_token_budget = (
                vllm_config.max_concurrent_batches * vllm_config.scheduler_config.max_num_batched_tokens
            )
        return _orig_scheduler_init(self, *args, **kwargs)

    @wraps(_orig_update_after_schedule)
    def _update_after_schedule(self, scheduler_output):
        # `allocate_slots` already ran earlier in `schedule()`; the count it
        # sees must reflect only the *previous* (still-settling) step, so the
        # increment happens after the upstream post-schedule update, matching
        # PR #47728 which bumps `num_in_flight_tokens` inside this method.
        ret = _orig_update_after_schedule(self, scheduler_output)
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            _inflight[rid] = _inflight.get(rid, 0) + n
        return ret

    @wraps(_orig_update_from_output)
    def _update_from_output(self, scheduler_output, model_runner_output):
        # Settle the step: drop its tokens before the upstream accounting rolls
        # `num_computed_tokens` back for rejected spec tokens (PR #47728
        # decrements at the top of this method).
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            v = _inflight.get(rid, 0) - n
            if v <= 0:
                _inflight.pop(rid, None)
            else:
                _inflight[rid] = v
        for rid in [r for r in _inflight if r not in self.requests]:
            _inflight.pop(rid, None)
        return _orig_update_from_output(self, scheduler_output, model_runner_output)

    if not getattr(_sched.Scheduler.update_from_output, _PATCH_FLAG, False):
        _kvcc.KVCacheCoordinator.remove_skipped_blocks = _remove_skipped_blocks
        _sched.Scheduler.__init__ = _scheduler_init
        _sched.Scheduler._update_after_schedule = _update_after_schedule
        _sched.Scheduler.update_from_output = _update_from_output
        # Mark the outermost wrapper so a re-import does not stack another layer
        # on top of `patch_pp_mtp`'s wrappers (which sit just below this one).
        _update_from_output._vllm_ascend_swa_inflight_patched = True  # type: ignore[attr-defined]
