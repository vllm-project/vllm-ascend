# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Free consumed encoder cache entries when the scheduler advances its state.

vLLM PR #68799 moves the encoder-input free from ``update_from_output`` into
``_update_after_schedule``. The vLLM main branch contains that change, but the
v0.34.0 release tag paired with this branch does not. With async scheduling,
the old placement keeps encoder entries occupied until the previous step's
output is processed, so an encoder-cache-bound multimodal workload can stall
for one extra scheduler step and preempt requests that would otherwise fit.
"""

from functools import wraps

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.utils import vllm_version_is

_ENCODER_FREE_PATCH_MARKER = "_vllm_ascend_encoder_free_patch"


def _patch_update_after_schedule() -> None:
    original = Scheduler._update_after_schedule
    if getattr(original, _ENCODER_FREE_PATCH_MARKER, False):
        return

    @wraps(original)
    def update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        original(self, scheduler_output)
        # Token accounting is already advanced above. A freed entry stays in
        # the worker's cache until a later step evicts it, and the worker runs
        # steps in order, so freeing before the step is dispatched is safe.
        # This mirrors the placement of upstream vLLM PR #68799.
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is not None and getattr(request, "has_encoder_inputs", False):
                self._free_encoder_inputs(request)

    setattr(update_after_schedule, _ENCODER_FREE_PATCH_MARKER, True)
    Scheduler._update_after_schedule = update_after_schedule


# vLLM main already contains PR #68799; only the v0.34.0 release tag needs it.
if vllm_version_is("0.34.0"):
    _patch_update_after_schedule()
