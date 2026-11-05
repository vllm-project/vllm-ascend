# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metrics for the RL 休眠与恢复 directory.

Durations wrap ``NPUWorker.sleep``, ``wake_up``, and ``update_weights``.
``NPUWorker.sleep`` does not return the freed memory, so it reports that value
through :func:`record_sleep_freed_gb` instead of a symbol hook.
"""

from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_BASIC,
    MetricSpec,
    MetricType,
    invoke_original,
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
# Reported from ``NPUWorker.sleep`` because the freed memory is measured inside
# the call and never returned.
_SLEEP_FREED_SPEC = MetricSpec(
    "lifecycle:sleep:freed_gb",
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
_UPDATE_WEIGHTS_SPEC = MetricSpec(
    "lifecycle:update_weights:duration_ms",
    MetricType.HISTOGRAM,
    (),
    level=METRIC_LEVEL_BASIC,
)
_SPECS = (_SLEEP_SPEC, _SLEEP_FREED_SPEC, _WAKE_SPEC, _UPDATE_WEIGHTS_SPEC)


def _record(spec: MetricSpec, value: float, labels: dict[str, str]) -> None:
    record_observation(_recorders, _KEY, _SPECS, spec, value, labels, "rl_lifecycle")


def _tag_label(tags) -> str:
    if not tags:
        return "all"
    return ",".join(str(tag) for tag in tags)


def record_sleep_freed_gb(level: int, freed_gb: float) -> None:
    """Report the memory released by ``NPUWorker.sleep`` for ``level``.

    The freed memory is only measured inside ``NPUWorker.sleep`` and is not
    returned, so the worker reports it directly instead of a symbol hook.
    """
    _record(_SLEEP_FREED_SPEC, freed_gb, {"level": str(level)})


def sleep_handler(original_func, worker, level: int = 1, *args, **kwargs):
    """Wall clock of ``NPUWorker.sleep``. ``level`` is already an argument."""
    result, elapsed_ms = invoke_original(original_func, (worker, level, *args), kwargs)
    _record(_SLEEP_SPEC, elapsed_ms, {"level": str(level)})
    return result


def wake_handler(original_func, worker, tags: list[str] | None = None, *args, **kwargs):
    """Wall clock of ``NPUWorker.wake_up``.

    The method takes ``tags`` and has no ``sleep_opt`` argument, so the series
    is split on ``tags`` only.
    """
    result, elapsed_ms = invoke_original(original_func, (worker, tags, *args), kwargs)
    _record(_WAKE_SPEC, elapsed_ms, {"tags": _tag_label(tags)})
    return result


def update_weights_handler(original_func, worker, *args, **kwargs):
    """Wall clock of one ``NPUWorker.update_weights`` chunk."""
    result, elapsed_ms = invoke_original(original_func, (worker, *args), kwargs)
    _record(_UPDATE_WEIGHTS_SPEC, elapsed_ms, {})
    return result
