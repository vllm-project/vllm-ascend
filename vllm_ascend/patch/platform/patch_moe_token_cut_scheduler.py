# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Handle explicit MoE token-cut deferrals in scheduler output accounting."""

from __future__ import annotations

from functools import wraps

from vllm_ascend.worker.moe_token_cut_handoff import get_deferred_request_ids
from vllm_ascend.worker.moe_token_cut_scheduler import (
    apply_scheduler_rollbacks,
    prepare_scheduler_rollbacks,
    restore_scheduler_rollbacks,
)

_PATCH_MARKER = "_vllm_ascend_moe_token_cut_patched"


def update_parked_request_ids(
    parked_request_ids: set[str],
    scheduled_request_ids: set[str],
    deferred_request_ids: set[str],
) -> set[str]:
    """Advance KV-pin ownership after one successful Scheduler update."""
    parked = set(parked_request_ids)
    parked.difference_update(scheduled_request_ids - deferred_request_ids)
    parked.update(deferred_request_ids)
    return parked


def _patch_scheduler_update_from_output() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    current_update_from_output = Scheduler.update_from_output
    if getattr(current_update_from_output, _PATCH_MARKER, False):
        return

    @wraps(current_update_from_output)
    def _patched_update_from_output(self, scheduler_output, model_runner_output):
        parked = set(getattr(self, "_moe_token_cut_parked_request_ids", ()))
        deferred = set(get_deferred_request_ids(model_runner_output))
        scheduled = set(scheduler_output.num_scheduled_tokens)
        rollbacks = prepare_scheduler_rollbacks(
            self,
            scheduler_output,
            model_runner_output,
        )
        if not rollbacks:
            result = current_update_from_output(
                self,
                scheduler_output,
                model_runner_output,
            )
            self._moe_token_cut_parked_request_ids = update_parked_request_ids(
                parked, scheduled, deferred
            )
            return result

        apply_scheduler_rollbacks(rollbacks)
        try:
            # Upstream still sees every scheduled row, so it retires
            # num_in_flight_tokens. Empty sampled-token rows emit no output.
            result = current_update_from_output(
                self,
                scheduler_output,
                model_runner_output,
            )
            self._moe_token_cut_parked_request_ids = update_parked_request_ids(
                parked, scheduled, deferred
            )
            return result
        except Exception:
            # Undo only the mutation owned by this wrapper. The caller retains
            # the original exception and can handle any upstream mutations.
            restore_scheduler_rollbacks(rollbacks)
            raise

    setattr(_patched_update_from_output, _PATCH_MARKER, True)
    Scheduler.update_from_output = _patched_update_from_output

    current_free_blocks = Scheduler._free_request_blocks

    @wraps(current_free_blocks)
    def _patched_free_request_blocks(self, request):
        parked = getattr(self, "_moe_token_cut_parked_request_ids", ())
        if request.request_id in parked and not request.is_finished():
            raise RuntimeError("cannot free KV blocks owned by a parked MoE token-cut request")
        result = current_free_blocks(self, request)
        if request.is_finished():
            getattr(self, "_moe_token_cut_parked_request_ids", set()).discard(request.request_id)
        return result

    Scheduler._free_request_blocks = _patched_free_request_blocks

    current_preempt = Scheduler._preempt_request

    @wraps(current_preempt)
    def _patched_preempt_request(self, request, *args, **kwargs):
        parked = getattr(self, "_moe_token_cut_parked_request_ids", ())
        if request.request_id in parked:
            raise RuntimeError(
                "cannot preempt a MoE token-cut request before its saved "
                "cross-layer state is restored"
            )
        return current_preempt(self, request, *args, **kwargs)

    Scheduler._preempt_request = _patched_preempt_request


_patch_scheduler_update_from_output()
