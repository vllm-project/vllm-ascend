# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symbol-hook metrics for async scheduling.

``_update_seq_lens_cpu`` covers both async scheduling rows at once: the whole
call is the seq_lens update, and the blocking D2H wait happens inside it. The
wait is timed from the model runner itself, because no hook can separate it from
the rest of the call. Output placeholders still have no per-step hook
(see async_scheduling_metrics.yaml).

``d2h_sync_ms`` only emits samples when that call really waits on the D2H copy:
a speculator is active, or speculative PP keeps the CPU token counts in sync
and the skip fast path does not apply. Plain async scheduling without
speculative decoding never blocks there, so only ``seq_lens_update_ms`` is
reported for that configuration.
"""

from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_DETAIL,
    MetricSpec,
    MetricType,
    invoke_original,
    metric_enabled,
    record_observation,
)

_recorders: dict[str, Any] = {}
_KEY = "async_scheduling"

_SEQ_LENS_SPEC = MetricSpec(
    "async_scheduling:seq_lens_update_ms",
    MetricType.HISTOGRAM,
    (),
    level=METRIC_LEVEL_DETAIL,
)
_D2H_SYNC_SPEC = MetricSpec(
    "async_scheduling:d2h_sync_ms",
    MetricType.HISTOGRAM,
    (),
    level=METRIC_LEVEL_DETAIL,
)
_SPECS = (_SEQ_LENS_SPEC, _D2H_SYNC_SPEC)


def seq_lens_update_handler(original_func, runner, *args, **kwargs):
    """Wall clock of ``_update_seq_lens_cpu``, including the D2H synchronize."""
    # Level gate first: skip the timing reads on the per-step path when the
    # detail level is not enabled.
    if not metric_enabled(_SEQ_LENS_SPEC):
        return original_func(runner, *args, **kwargs)
    result, elapsed_ms = invoke_original(original_func, (runner, *args), kwargs)
    record_observation(
        _recorders,
        _KEY,
        _SPECS,
        _SEQ_LENS_SPEC,
        elapsed_ms,
        {},
        "async_scheduling",
    )
    return result


def d2h_sync_enabled() -> bool:
    """Whether the blocking D2H wait should be timed at the active level."""
    return metric_enabled(_D2H_SYNC_SPEC)


def record_d2h_sync_ms(elapsed_ms: float) -> None:
    """Report the blocking D2H wait from inside ``_update_seq_lens_cpu``."""
    record_observation(
        _recorders,
        _KEY,
        _SPECS,
        _D2H_SYNC_SPEC,
        elapsed_ms,
        {},
        "async_scheduling",
    )
