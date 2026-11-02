# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV Pool metrics reported by the Ascend store backends."""

from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    MetricSpec,
    MetricType,
    record_observation,
)

_KV_POOL_KEY = "kv_pool"

_PUT_FAILURE_SPEC = MetricSpec(
    "kvpool:put_failure_keys_total",
    MetricType.COUNTER,
    ("backend",),
)
_SPECS = (_PUT_FAILURE_SPEC,)

_recorders: dict[str, Any] = {}

# Stable label values; the class name alone would vary with the backend module.
_BACKEND_LABELS = {
    "MemcacheBackend": "memcache",
    "MooncakeBackend": "mooncake",
}


def _backend_label(backend: Any) -> str:
    class_name = type(backend).__name__
    return _BACKEND_LABELS.get(class_name, class_name)


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
