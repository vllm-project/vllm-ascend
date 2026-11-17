# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime-guard anomaly alert metrics.

YAML for this symbol has no ``metrics:`` list (custom WRAP handler, not a
factory). See :mod:`vllm_ascend.observability.ms_metrics.handlers.common`.
"""

from __future__ import annotations

from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_BASIC,
    MetricSpec,
    MetricType,
    record_observation,
)

_recorders: dict[str, Any] = {}
_KEY = "runtime_guard_anomaly"

_ANOMALY_SPEC = MetricSpec(
    "runtime_guard:anomaly:count",
    MetricType.COUNTER,
    ("incident_type",),
    level=METRIC_LEVEL_BASIC,
)
_SPECS = (_ANOMALY_SPEC,)


def runtime_guard_handle_alert_handler(original_func, processor, alert, *args, **kwargs):
    """Count armed anomaly alerts by ``incident_type``.

    Hooks ``RuntimeGuardReportMixin._handle_alert``. Early no-op returns
    (``write_report=False`` and no ``action_override``) are not counted.
    Default level is ``basic`` so anomaly rates are visible without raising
    the process metric level.
    """
    write_report = kwargs.get("write_report", True)
    action_override = kwargs.get("action_override")
    result = original_func(processor, alert, *args, **kwargs)

    if not write_report and action_override is None:
        return result

    incident_type = str(getattr(alert, "incident_type", None) or "unknown")
    record_observation(
        _recorders,
        _KEY,
        _SPECS,
        _ANOMALY_SPEC,
        1.0,
        {"incident_type": incident_type},
        "runtime_guard_anomaly",
    )
    return result
