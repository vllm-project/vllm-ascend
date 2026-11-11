# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV Pool metrics reported by the Ascend store backends.

Layerwise loads bypass the connector-level load metric, so the three
receive-thread load paths are timed here. A sample covers one layer's whole
dispatch on the receive thread, including the pipeline dependency waits
(previous-save sync and the attention start gate), not the bare store call.
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

_KV_POOL_KEY = "kv_pool"

_PUT_FAILURE_SPEC = MetricSpec(
    "kvpool:put_failure_keys_total",
    MetricType.COUNTER,
    ("backend",),
)
_LOAD_ERROR_SPEC = MetricSpec(
    "kvpool:load_error_blocks_total",
    MetricType.COUNTER,
    ("mode",),
)
_LOAD_DURATION_SPEC = MetricSpec(
    "kvpool:layerwise_load_get_duration_ms",
    MetricType.HISTOGRAM,
    ("mode",),
    level=METRIC_LEVEL_DETAIL,
)
_SPECS = (_PUT_FAILURE_SPEC, _LOAD_ERROR_SPEC, _LOAD_DURATION_SPEC)

_recorders: dict[str, Any] = {}

# Stable label values; the class name alone would vary with the backend module.
_BACKEND_LABELS = {
    "MemcacheBackend": "memcache",
    "MooncakeBackend": "mooncake",
    "YuanrongBackend": "yuanrong",
}


def _backend_label(backend: Any) -> str:
    class_name = type(backend).__name__
    return _BACKEND_LABELS.get(class_name, class_name)


# The base pool worker method is shared by the standard and layerwise workers.
_WORKER_MODE_LABELS = {
    "KVPoolWorker": "standard",
    "LayerwiseKVPoolWorker": "layerwise",
}


def _worker_mode_label(worker: Any) -> str:
    class_name = type(worker).__name__
    return _WORKER_MODE_LABELS.get(class_name, class_name)


def _failed_key_count(result: Any, requested: int) -> int:
    """Return how many keys failed, given the backend return value.

    Backends return a per-key ``list[bool]`` on success and ``False`` when the
    whole batch raised, so both shapes are handled.
    """
    if isinstance(result, bool):
        return 0 if result else requested
    if isinstance(result, (list, tuple)):
        return sum(1 for stored in result if not stored)
    return 0


def kv_pool_put_failure_handler(original_func, backend, keys, *args, **kwargs):
    """Count keys whose KV Pool put failed from the backend return value."""
    result = original_func(backend, keys, *args, **kwargs)

    failed_keys = _failed_key_count(result, len(keys))
    if failed_keys:
        record_observation(
            _recorders,
            _KV_POOL_KEY,
            _SPECS,
            _PUT_FAILURE_SPEC,
            float(failed_keys),
            {"backend": _backend_label(backend)},
            "kv_pool",
        )
    return result


def kv_pool_load_error_handler(original_func, worker, *args, **kwargs):
    """Count blocks that failed to load from the KV Pool.

    ``get_block_ids_with_load_errors`` drains the pending set, so the handler
    calls it once and only inspects the returned block ids.
    """
    result = original_func(worker, *args, **kwargs)

    failed_blocks = len(result) if result else 0
    if failed_blocks:
        record_observation(
            _recorders,
            _KV_POOL_KEY,
            _SPECS,
            _LOAD_ERROR_SPEC,
            float(failed_blocks),
            {"mode": _worker_mode_label(worker)},
            "kv_pool",
        )
    return result


def _timed_layerwise_load(original_func, thread, mode: str, args: tuple, kwargs: dict):
    if not metric_enabled(_LOAD_DURATION_SPEC):
        return original_func(thread, *args, **kwargs)
    result, elapsed_ms = invoke_original(original_func, (thread, *args), kwargs)
    record_observation(_recorders, _KV_POOL_KEY, _SPECS, _LOAD_DURATION_SPEC, elapsed_ms, {"mode": mode}, "kv_pool")
    return result


def kv_pool_layerwise_gva_load_handler(original_func, thread, *args, **kwargs):
    """Time one GVA-batched layer load on the layerwise receive thread."""
    return _timed_layerwise_load(original_func, thread, "gva", args, kwargs)


def kv_pool_layerwise_range_load_handler(original_func, thread, *args, **kwargs):
    """Time one key-major range layer load on the layerwise receive thread."""
    return _timed_layerwise_load(original_func, thread, "range", args, kwargs)


def kv_pool_layerwise_key_load_handler(original_func, thread, *args, **kwargs):
    """Time one block-key layer load on the key-based receive thread."""
    return _timed_layerwise_load(original_func, thread, "key", args, kwargs)
