# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symbol-hook metrics for async scheduling.

``_update_seq_lens_cpu`` covers both async scheduling rows at once: the whole
call is the seq_lens update, and the blocking D2H wait happens inside it. The
wait is timed from the model runner itself, because no hook can separate it from
the rest of the call. Output placeholders still have no per-step hook
(see async_scheduling_metrics.yaml).
"""

from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_DETAIL,
    MetricSpec,
    MetricType,
    invoke_original,
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
