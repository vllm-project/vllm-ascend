# mypy: ignore-errors
"""Balance scheduling EngineCore monkey patch.

This module keeps the conditional EngineCore activation and the
``BalanceDPEngineCoreProc`` hook. Scheduler implementations live in
:mod:`vllm_ascend.core.balance_scheduler`.
"""

from vllm.v1.engine.core import DPEngineCoreProc

class BalanceDPEngineCoreProc(DPEngineCoreProc):
    """Minimal DP engine core hook for balance scheduling.

    The only thing balance scheduling needs from the engine core is the DP
    process group, which the scheduler uses for its per-step all-gather. The
    group is created in ``_init_data_parallel`` (during ``__init__``, before
    the scheduler exists) and the scheduler is created in ``EngineCore.__init__``,
    so both are present by the time the busy loop runs.

    The per-step gather is hooked via ``_has_global_unfinished_reqs`` (called
    every iteration by upstream's ``run_busy_loop`` on every non-idle path),
    NOT from inside ``schedule()``: a rank that has drained its local requests
    never enters ``schedule()`` (it runs a dummy batch instead), so an
    ``all_gather`` living in ``schedule()`` would be skipped by that rank while
    busy ranks call it -- a collective mismatch that deadlocks.

    Why ``_has_global_unfinished_reqs`` and not ``_process_engine_step``:
    ``_has_global_unfinished_reqs`` is itself a cross-rank collective (an
    all-reduce every 32 steps internally) and is the only point in the busy
    loop that re-synchronizes ranks on wave/idle state. Hooking the gather
    immediately after ``super()._has_global_unfinished_reqs()`` keeps the
    every-step all-gather in the same lock-stepped region, so ranks enter the
    gather having just agreed on ``engines_running``. Hooking it earlier --
    inside ``_process_engine_step``, before that sync and before the idle
    ``continue`` gate -- decouples the gather from the synchronization: at
    wave boundaries one rank can reach the gather while another is still
    blocked in ``_process_input_queue`` or ``future.result()``, deadlocking
    the all-gather. The stuck EngineCore then can't drain its worker shm
    channel, the worker's ``sample_tokens`` response has nowhere to land, and
    after 60s the engine dies with ``RPC call to sample_tokens timed out``.
    Because ``_has_global_unfinished_reqs`` is only called on iterations that
    did NOT take the idle ``continue``, gather is skipped consistently by
    every rank when all are idle -- no rank does an extra gather. This matches
    the pre-refactor copied ``run_busy_loop``, where gather sat right after
    the all-reduce (after schedule + execute + update_from_output).
    """

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        result = super()._has_global_unfinished_reqs(local_unfinished)
        # Inject the DP group (idempotent) and refresh the cross-rank running
        # snapshot once per non-idle iteration. balance_gather is a no-op until
        # dp_group is injected, so this is safe on every path. MUST run
        # immediately after the sync collective so ranks enter the all-gather
        # already aligned -- see class docstring.
        self.scheduler.dp_group = self.dp_group
        self.scheduler.balance_gather()
        return result


# The patch for engine core has been moved to patch_engine_core.py
