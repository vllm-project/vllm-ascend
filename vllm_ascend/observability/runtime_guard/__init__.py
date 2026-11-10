"""Runtime anomaly detection and incident actions."""

from __future__ import annotations

from typing import Any

__all__ = ["RuntimeGuardProcessor"]


def __getattr__(name: str) -> Any:
    # Lazy: avoid import cycle when runtime_config._defaults loads detector schemas.
    if name == "RuntimeGuardProcessor":
        from vllm_ascend.observability.runtime_guard.processor import RuntimeGuardProcessor

        return RuntimeGuardProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
