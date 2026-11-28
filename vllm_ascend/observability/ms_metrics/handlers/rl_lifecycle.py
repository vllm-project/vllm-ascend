# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metrics for the RL sleep/wake lifecycle.

Durations wrap ``NPUWorker.sleep`` and ``NPUWorker.wake_up``. Both are symbol
hooks declared in ``rl_lifecycle_metrics.yaml``, so no business logic is
edited.
"""

from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_BASIC,
    MetricSpec,
    MetricType,
    invoke_original,
    metric_enabled,
    record_observation,
)

_recorders: dict[str, Any] = {}
_KEY = "rl_lifecycle"

_SLEEP_SPEC = MetricSpec(
    "lifecycle:sleep:duration_ms",
    MetricType.HISTOGRAM,
    ("level",),
    level=METRIC_LEVEL_BASIC,
)
_WAKE_SPEC = MetricSpec(
    "lifecycle:wake:duration_ms",
    MetricType.HISTOGRAM,
    ("tags",),
    level=METRIC_LEVEL_BASIC,
)
_SPECS = (_SLEEP_SPEC, _WAKE_SPEC)


def _record(spec: MetricSpec, value: float, labels: dict[str, str]) -> None:
    record_observation(_recorders, _KEY, _SPECS, spec, value, labels, "rl_lifecycle")


def _tag_label(tags) -> str:
    if not tags:
        return "all"
    return ",".join(str(tag) for tag in tags)


def sleep_handler(original_func, worker, level: int = 1, *args, **kwargs):
    """Wall clock of ``NPUWorker.sleep``. ``level`` is already an argument."""
    if not metric_enabled(_SLEEP_SPEC):
        return original_func(worker, level, *args, **kwargs)
    result, elapsed_ms = invoke_original(original_func, (worker, level, *args), kwargs)
    _record(_SLEEP_SPEC, elapsed_ms, {"level": str(level)})
    return result


def wake_handler(original_func, worker, tags: list[str] | None = None, *args, **kwargs):
    """Wall clock of ``NPUWorker.wake_up``.

    The method takes ``tags`` and has no ``sleep_opt`` argument, so the series
    is split on ``tags`` only.
    """
    if not metric_enabled(_WAKE_SPEC):
        return original_func(worker, tags, *args, **kwargs)
    result, elapsed_ms = invoke_original(original_func, (worker, tags, *args), kwargs)
    _record(_WAKE_SPEC, elapsed_ms, {"tags": _tag_label(tags)})
    return result
