# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Readiness publication for application-directed Mamba checkpoints.

vllm-project/vllm#55873/#55875 add application-directed Mamba prefix
checkpoints to the V1 scheduler: a control token in the prompt marks the
position where the GDN recurrent state is snapshotted into the prefix cache,
so fan-out requests sharing that prefix resume from the snapshot instead of
re-prefilling it.

The coordination state machine lives in the vLLM core scheduler (checkpoint
registration in MambaManager, the unready-hash gate in BlockPool, and the
waiting-loop deferral in Scheduler.schedule). This module keeps the readiness
publication correct on Ascend:

- ``update_from_output`` publishes a checkpoint entry once its committing
  forward completes.
- ``schedule`` publishes entries whose committing forward was dispatched in
  the previous pass one pass early: the device executes steps in submission
  order, so the snapshot is committed by the time this pass's forward runs —
  async scheduling otherwise delays consumer wake-up by a full pass.

Every hook is inert unless the running vLLM provides the checkpoint state
(``Request.mamba_checkpoint_position`` /
``KVCacheManager.mark_checkpoint_ready``), i.e. before
vllm-project/vllm#55873/#55875 land.
"""

import functools

from vllm.v1.core.sched.scheduler import Scheduler

_original_schedule = Scheduler.schedule
_original_update_from_output = Scheduler.update_from_output


def _checkpoint_position(request):
    return getattr(request, "mamba_checkpoint_position", None)


def _starts_from_output(output):
    """Map req_id -> first token position of this pass's scheduled chunk."""
    starts = {req_data.req_id: req_data.num_computed_tokens for req_data in output.scheduled_new_reqs}
    cached_reqs = output.scheduled_cached_reqs
    starts.update(
        dict(
            zip(
                cached_reqs.req_ids,
                cached_reqs.num_computed_tokens,
                strict=True,
            )
        )
    )
    return starts


def _checkpoint_machinery(scheduler):
    """Return (kv_cache_manager, dispatched) when the coordinator supports
    the feature, else (None, None)."""
    kvc = getattr(scheduler, "kv_cache_manager", None)
    if kvc is None or not hasattr(kvc, "mark_checkpoint_ready"):
        return None, None
    dispatched = getattr(scheduler, "_mamba_checkpoint_dispatched", None)
    if dispatched is None:
        dispatched = set()
        scheduler._mamba_checkpoint_dispatched = dispatched
    return kvc, dispatched


def _record_dispatched(self, output, dispatched):
    starts = _starts_from_output(output)
    for req_id, num_scheduled in output.num_scheduled_tokens.items():
        request = self.requests.get(req_id)
        cp = _checkpoint_position(request) if request is not None else None
        start = starts.get(req_id)
        if cp is None or start is None:
            continue
        # The chunk ends exactly at the checkpoint: its forward commits the
        # state snapshot; publish the cache entry once that forward ran.
        if start + num_scheduled == cp:
            dispatched.add(req_id)


@functools.wraps(_original_schedule)
def _checkpoint_aware_schedule(self, *args, **kwargs):
    kvc, dispatched = _checkpoint_machinery(self)
    if kvc is not None and dispatched:
        # Publish entries whose committing forward was dispatched in the
        # previous pass (see _record_dispatched). Consumers sharing the
        # checkpoint resume this pass instead of waiting for
        # update_from_output of the previous step, which async scheduling
        # runs only after this schedule pass.
        for req_id in list(dispatched):
            kvc.mark_checkpoint_ready(req_id)
        dispatched.clear()

    output = _original_schedule(self, *args, **kwargs)

    if kvc is not None and hasattr(kvc, "has_unready_checkpoint"):
        _record_dispatched(self, output, dispatched)
    return output


@functools.wraps(_original_update_from_output)
def _checkpoint_aware_update_from_output(self, scheduler_output, model_runner_output, *args, **kwargs):
    kvc, _ = _checkpoint_machinery(self)
    if kvc is None:
        return _original_update_from_output(self, scheduler_output, model_runner_output, *args, **kwargs)
    # Snapshot the checkpoint positions (and chunk starts) before the
    # original call: it removes finished requests from self.requests, and a
    # producer whose final chunk ends exactly at the checkpoint would
    # otherwise be missed — its entry never published and its consumers
    # left waiting indefinitely.
    positions = {req_id: _checkpoint_position(request) for req_id, request in self.requests.items()}
    starts = _starts_from_output(scheduler_output)
    result = _original_update_from_output(self, scheduler_output, model_runner_output, *args, **kwargs)
    for req_id, num_scheduled in scheduler_output.num_scheduled_tokens.items():
        cp = positions.get(req_id)
        start = starts.get(req_id)
        if cp is not None and start is not None and start + num_scheduled == cp:
            # Runs after the forward that committed the snapshot state.
            kvc.mark_checkpoint_ready(req_id)
    return result


Scheduler.schedule = _checkpoint_aware_schedule
Scheduler.update_from_output = _checkpoint_aware_update_from_output
