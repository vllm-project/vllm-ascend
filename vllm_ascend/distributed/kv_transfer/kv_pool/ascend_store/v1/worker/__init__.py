"""Worker-side ownership for AscendStore v1 execution.

Concrete routes are loaded lazily so Timeline can import Worker transfer
contracts without starting the Worker/Timeline import graph.
"""

from __future__ import annotations

import importlib
from types import MappingProxyType
from typing import Any

_WORKER_EXPORTS = MappingProxyType(
    {
        "AsynchronousBulkWorker": "bulk",
        "BulkWorker": "bulk",
        "GVALayerwiseWorker": "layerwise",
        "KVPoolWorker": "base",
        "KeyRangeLayerwiseWorker": "layerwise",
        "LayerwiseWorker": "layerwise",
        "SynchronousBulkWorker": "bulk",
    }
)

__all__ = tuple(_WORKER_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _WORKER_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
