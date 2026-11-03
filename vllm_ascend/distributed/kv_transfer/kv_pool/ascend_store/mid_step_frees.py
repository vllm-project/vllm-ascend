# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge for block frees that RUNNING requests suffer mid-step.

The KV cache coordinator frees blocks of *running* requests inside
``schedule()`` (sliding-window advancement, R-SWA gap eviction, mamba state
retirement), and the freed blocks can be reallocated within the same
``schedule()`` call. ``SchedulerOutput`` only surfaces request terminal
states (finished/preempted), so a block-reuse fence built from the output
alone misses these frees: a queued store put may still be reading a block
the scheduler already handed to another request.

The patched coordinator (``patch_kv_cache_coordinator``) records such
request ids here while a collector is registered; the AscendStore scheduler
connector merges them into ``AscendConnectorMetadata.released_req_ids`` and
clears the set on every ``build_connector_meta``. One scheduler per process
keeps the module-level state safe.
"""

_mid_step_released_req_ids: set[str] = set()
_tracking = False


def activate_mid_step_release_tracking() -> None:
    """Let the coordinator patch start recording (idempotent)."""
    global _tracking
    _tracking = True


def is_tracking() -> bool:
    return _tracking


def record_mid_step_release(request_id: str) -> None:
    if _tracking:
        _mid_step_released_req_ids.add(request_id)


def take_mid_step_released_req_ids() -> set[str]:
    """Return and clear the ids recorded since the previous call."""
    global _mid_step_released_req_ids
    recorded = _mid_step_released_req_ids
    _mid_step_released_req_ids = set()
    return recorded
