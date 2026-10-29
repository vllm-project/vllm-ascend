# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared building blocks for vLLM Ascend-owned metric handlers.

Each feature composes its handler from the helpers here instead of
re-implementing metric registration, recorder caching, and failure isolation.
This module is the only place that imports ``ms_service_metric`` directly, so
feature handler modules stay free of framework imports.
"""

import contextlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ms_service_metric.provider_api import MetricType, get_metric_recorder  # type: ignore[import-not-found]

__all__ = ["MetricType", "MetricSpec", "register_metrics", "safe_recording"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricSpec:
    """Declarative definition of a metric to register."""

    name: str
    metric_type: Any
    label_names: tuple[str, ...]


def register_metrics(
    metric_specs: Iterable[MetricSpec],
    recorders: dict[str, Any],
    recorder_key: str,
) -> Any:
    """Register ``metric_specs`` (iterable of MetricSpec) once per recorder.

    ``recorders`` is the feature-owned cache and ``recorder_key`` identifies
    the spec group within it. Re-registers when the recorder is replaced.
    """
    metrics = get_metric_recorder()
    if metrics is None:
        return None
    if recorders.get(recorder_key) is metrics:
        return metrics
    for spec in metric_specs:
        metrics.get_or_create_metric(
            metric_name=spec.name,
            metric_type=spec.metric_type,
            label_names=list(spec.label_names),
        )
    recorders[recorder_key] = metrics
    return metrics


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
