# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared building blocks for vLLM Ascend-owned metric handlers.

Each feature composes its handler from the helpers here instead of
re-implementing metric registration, recorder caching, and failure isolation.
This module is the only place that imports ``ms_service_metric`` directly, so
feature handler modules stay free of framework imports.

Recording is gated by the process-local level in :mod:`vllm_ascend.observability.ms_metrics.level`.
A spec is recorded when ``spec.level`` is less than or equal to that integer.
Runtime config sets it from ``metrics.level``.

Yaml entry convention for custom handlers
-----------------------------------------

``ms_service_metric`` treats a yaml entry that declares both ``handler`` and
``metrics`` as a factory: it calls ``handler(metrics_config, ...)`` expecting
the returned callable to be the hook. The handlers owned by this repository
have runtime signatures ``(original_func, obj, *args)`` and are NOT factories,
so their yaml entries must declare only ``symbol`` and ``handler`` — never a
``metrics`` block. A stray ``metrics`` block makes the framework attempt the
factory call, fail with ``TypeError``, and silently fall back to a pass-through
hook, so the metric is never recorded while "Hooked symbol" logs still claim
success. Metric declaration stays solely in code via :class:`MetricSpec`
(``label_names`` and ``level`` cannot be expressed in the yaml block anyway).
Entries that point at provider built-ins (e.g.
``ms_service_metric.provider_handlers:phase_all_handler``) do follow the
factory convention and keep their ``metrics`` blocks.
"""

import contextlib
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ms_service_metric.provider_api import MetricType, get_metric_recorder  # type: ignore[import-not-found]

from vllm_ascend.observability.ms_metrics.level import (
    METRIC_LEVEL_BASIC,
    METRIC_LEVEL_DETAIL,
    get_metric_level,
    set_metric_level,
)

__all__ = [
    "METRIC_LEVEL_BASIC",
    "METRIC_LEVEL_DETAIL",
    "MetricSpec",
    "MetricType",
    "get_metric_level",
    "invoke_original",
    "metric_enabled",
    "record_metric",
    "record_observation",
    "register_metrics",
    "safe_recording",
    "set_metric_level",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricSpec:
    """Declarative definition of a metric to register.

    ``level`` defaults to basic. Higher levels stay dark until the active
    level is raised.
    """

    name: str
    metric_type: Any
    label_names: tuple[str, ...]
    level: int = METRIC_LEVEL_BASIC


def metric_enabled(spec: MetricSpec) -> bool:
    """Return whether ``spec`` should be sampled at the active level."""
    return spec.level <= get_metric_level()


def register_metrics(
    metric_specs: Iterable[MetricSpec],
    recorders: dict[str, Any],
    recorder_key: str,
) -> Any:
    """Register enabled ``metric_specs`` once per recorder and active level.

    ``recorders`` is the feature-owned cache and ``recorder_key`` identifies
    the spec group within it. Re-registers when the recorder is replaced or
    the active level changes. Specs above the active level are left
    unregistered so high-cardinality series are not created until enabled.
    Already created series stay in the registry when the level drops.
    """
    metrics = get_metric_recorder()
    if metrics is None:
        return None
    active_level = get_metric_level()
    if recorders.get(recorder_key) == (metrics, active_level):
        return metrics
    for spec in metric_specs:
        if spec.level > active_level:
            continue
        metrics.get_or_create_metric(
            metric_name=spec.name,
            metric_type=spec.metric_type,
            label_names=list(spec.label_names),
        )
    recorders[recorder_key] = (metrics, active_level)
    return metrics


def invoke_original(original_func: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[Any, float]:
    """Call the wrapped function and return ``(result, elapsed_ms)``.

    The elapsed time is omitted when the call raises, so a failed inference
    step is not recorded as a successful sample.
    """
    started = time.perf_counter()
    result = original_func(*args, **kwargs)
    return result, (time.perf_counter() - started) * 1000.0


def record_observation(
    recorders: dict[str, Any],
    recorder_key: str,
    specs: Iterable[MetricSpec],
    spec: MetricSpec,
    value: float,
    labels: dict[str, str],
    feature: str,
) -> None:
    """Register ``specs`` and record one observation when the level allows it."""
    if not metric_enabled(spec):
        return
    with safe_recording(feature):
        metrics = register_metrics(specs, recorders, recorder_key)
        record_metric(metrics, spec, value, labels)


def record_metric(
    metrics: Any,
    spec: MetricSpec,
    value: float,
    labels: dict[str, str],
) -> None:
    """Record ``spec`` when the recorder exists and the active level allows it."""
    if metrics is None or spec.level > get_metric_level():
        return
    metrics.record_metric(spec.name, value=float(value), labels=labels)


@contextlib.contextmanager
def safe_recording(feature: str):
    """Unified error handling for recording"""
    try:
        yield
    except Exception:
        logger.warning(
            "Failed to record %s metrics; inference result is preserved",
            feature,
            exc_info=True,
        )
