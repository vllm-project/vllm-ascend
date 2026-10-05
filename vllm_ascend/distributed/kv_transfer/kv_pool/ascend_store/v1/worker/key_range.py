"""Layerwise Worker fixed to the Backend key-range data plane."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from ...attention_fence import reset_attention_compute_start_gate
from ..backend import KeyRangeBackend, LayerwiseAccessKind
from ..projection import KeyRangeLayerwiseProjection, KeyRangeLayerwiseProjectionBinder, LayerwiseProjection
from ..runtime.backend import KeyRangeBackendIO
from ..runtime.resources import KVPoolResources
from ..topology import KVPoolTopology
from .layerwise import LayerwiseWorker


class KeyRangeLayerwiseWorker(LayerwiseWorker):
    """Own native key-range sessions through one concrete adapter."""

    def __init__(
        self,
        topology: KVPoolTopology,
        projection_binder: KeyRangeLayerwiseProjectionBinder,
        resources: KVPoolResources,
        *,
        store_enabled: bool = True,
        layerwise_prefetch_layers: int = 2,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        if resources.backend_spec.layerwise_access is not LayerwiseAccessKind.KEY_RANGE:
            raise TypeError("KeyRangeLayerwiseWorker requires a KeyRange Backend")
        backend_io = KeyRangeBackendIO(cast(KeyRangeBackend, resources.backend), resources.backend_spec)
        backend_io.validate_support()
        self._key_range_backend_io = backend_io
        super().__init__(
            topology,
            projection_binder,
            resources,
            backend_io,
            store_enabled=store_enabled,
            layerwise_prefetch_layers=layerwise_prefetch_layers,
            start_gate_factory=start_gate_factory,
            source_ready_event_factory=source_ready_event_factory,
        )

    def _bind_backend_projection(self, projection: LayerwiseProjection) -> None:
        if not isinstance(projection, KeyRangeLayerwiseProjection):
            raise TypeError("KeyRange Backend requires a KeyRange Layerwise projection")
        self._key_range_backend_io.bind_projection(projection)


__all__ = ("KeyRangeLayerwiseWorker",)
