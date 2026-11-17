# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layerwise AscendStore KV load timing.

The non-layerwise path reports one whole get through the connector
(``vllm:ascend_store_load_get_duration_seconds``). Layerwise loads bypass that
path: the receive thread dispatches one layer at a time and three
implementations coexist, so each mode hooks the receive-thread method that
handles exactly one layer.

The sample covers the layer's whole dispatch on the receive thread, including
the dependency waits that make the pipeline overlap (previous-save sync and
the attention start gate). It is therefore a per-layer load latency, not the
bare store call.
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
_KEY = "kvpool_layerwise_load"

_LOAD_SPEC = MetricSpec(
    "kvpool:layerwise_load_get_duration_ms",
    MetricType.HISTOGRAM,
    ("mode",),
    level=METRIC_LEVEL_DETAIL,
)
_SPECS = (_LOAD_SPEC,)


def _timed(original_func, obj, mode: str, args: tuple, kwargs: dict):
    if not metric_enabled(_LOAD_SPEC):
        return original_func(obj, *args, **kwargs)
    result, elapsed_ms = invoke_original(original_func, (obj, *args), kwargs)
    record_observation(_recorders, _KEY, _SPECS, _LOAD_SPEC, elapsed_ms, {"mode": mode}, "kvpool_layerwise_load")
    return result


def kvpool_layerwise_gva_load_handler(original_func, thread, *args, **kwargs):
    """Time one GVA-batched layer load on the layerwise receive thread."""
    return _timed(original_func, thread, "gva", args, kwargs)


def kvpool_layerwise_range_load_handler(original_func, thread, *args, **kwargs):
    """Time one key-major range layer load on the layerwise receive thread."""
    return _timed(original_func, thread, "range", args, kwargs)


def kvpool_layerwise_key_load_handler(original_func, thread, *args, **kwargs):
    """Time one block-key layer load on the key-based receive thread."""
    return _timed(original_func, thread, "key", args, kwargs)
