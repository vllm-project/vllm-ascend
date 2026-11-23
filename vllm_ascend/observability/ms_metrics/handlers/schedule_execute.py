# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symbol-hook metrics for the 调度与执行 directory.

Each handler wraps a function that already exists. Nothing in the scheduler
or model runner is edited. Per-step durations stay at detail so the default
basic level does not sample them.
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
_KEY = "schedule_execute"

_SAMPLE_SPEC = MetricSpec(
    "executor:sample_tokens:duration_ms",
    MetricType.HISTOGRAM,
    ("runner",),
    level=METRIC_LEVEL_DETAIL,
)
_SPECS = (_SAMPLE_SPEC,)


def _record(spec: MetricSpec, value: float, labels: dict[str, str]) -> None:
    record_observation(_recorders, _KEY, _SPECS, spec, value, labels, "schedule_execute")


def sample_tokens_handler(original_func, runner, *args, **kwargs):
    """Wall clock of ModelRunner V2 ``sample_tokens``, including grammar and rejection."""
    if not metric_enabled(_SAMPLE_SPEC):
        return original_func(runner, *args, **kwargs)
    result, elapsed_ms = invoke_original(original_func, (runner, *args), kwargs)
    _record(_SAMPLE_SPEC, elapsed_ms, {"runner": "v2"})
    return result
