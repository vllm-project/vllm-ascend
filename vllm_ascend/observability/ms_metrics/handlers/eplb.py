# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM Ascend-owned metric handlers."""

import logging
from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_DETAIL,
    MetricSpec,
    MetricType,
    invoke_original,
    metric_enabled,
    record_metric,
    record_observation,
    register_metrics,
    safe_recording,
)

logger = logging.getLogger(__name__)

_eplb_metrics_recorders: dict[str, Any] = {}

_HOTNESS_SUMMARY_METRICS = (
    "eplb:expert_hotness:current_mean",
    "eplb:expert_hotness:current_max",
    "eplb:expert_hotness:update_mean",
    "eplb:expert_hotness:update_max",
)
_IMBALANCE_METRIC = "eplb:expert_hotness:imbalance"

_SUMMARY_SPECS = tuple(MetricSpec(name, MetricType.GAUGE, ("rank", "phase")) for name in _HOTNESS_SUMMARY_METRICS)
_IMBALANCE_SPEC = MetricSpec(_IMBALANCE_METRIC, MetricType.GAUGE, ("rank", "phase", "layer"))
_EPLB_METRIC_SPECS = (*_SUMMARY_SPECS, _IMBALANCE_SPEC)

_eplb_hotness_key = "eplb_hotness"


def _register_eplb_metrics():
    return register_metrics(
        _EPLB_METRIC_SPECS,
        _eplb_metrics_recorders,
        _eplb_hotness_key,
    )


def eplb_do_update_hotness_handler(original_func, worker, *args, **kwargs):
    """Record the aggregated expert hotness exposed by the EPLB rank-0 worker."""
    result = original_func(worker, *args, **kwargs)

    with safe_recording("expert_hotness"):
        rank_id = getattr(worker, "rank_id", -1)
        if rank_id != 0:
            return result

        hotness = getattr(worker, "latest_expert_hotness", None)
        if not hotness:
            logger.debug("Skip EPLB hotness metrics because latest_expert_hotness is missing")
            return result

        metrics = _register_eplb_metrics()
        if metrics is None:
            logger.debug("Skip EPLB hotness metrics because the metric recorder is unavailable")
            return result
        rank_labels = {"rank": str(rank_id), "phase": "all"}
        for spec in _SUMMARY_SPECS:
            if not metric_enabled(spec):
                continue
            value = hotness.get(spec.name.rpartition(":")[2])
            if value is not None:
                record_metric(metrics, spec, value, rank_labels)

        if metric_enabled(_IMBALANCE_SPEC):
            for phase, key in (
                ("current", "current_imbalance_list"),
                ("update", "update_imbalance_list"),
            ):
                imbalance_values: Any = hotness.get(key)
                if imbalance_values is None:
                    continue
                for layer_index, value in enumerate(imbalance_values):
                    record_metric(
                        metrics,
                        _IMBALANCE_SPEC,
                        value,
                        {
                            "rank": str(rank_id),
                            "phase": phase,
                            "layer": str(layer_index),
                        },
                    )
    return result


_TRANSFER_SPEC = MetricSpec(
    "eplb:expert_weight_transfer:duration",
    MetricType.HISTOGRAM,
    (),
    level=METRIC_LEVEL_DETAIL,
)


def eplb_expert_weight_transfer_handler(original_func, loader, *args, **kwargs):
    """Wall clock of ``D2DExpertWeightLoader.asyn_expert_weight_transfer``."""
    result, elapsed_ms = invoke_original(original_func, (loader, *args), kwargs)
    record_observation(
        _eplb_metrics_recorders,
        "eplb_weight_transfer",
        (_TRANSFER_SPEC,),
        _TRANSFER_SPEC,
        elapsed_ms,
        {},
        "eplb_weight_transfer",
    )
    return result
