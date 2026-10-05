"""Layerwise Worker fixed to the GVA data plane."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from ...attention_fence import reset_attention_compute_start_gate
from ..backend import GVABackend, LayerwiseAccessKind
from ..projection import GVALayerwiseProjection, GVALayerwiseProjectionBinder, LayerwiseProjection
from ..runtime.backend import GVABackendIO
from ..runtime.resources import KVPoolResources
from ..topology import KVPoolTopology
from .layerwise import LayerwiseWorker


class GVALayerwiseWorker(LayerwiseWorker):
    """Own GVA leases, addresses, and publication through one concrete adapter."""

    def __init__(
        self,
        topology: KVPoolTopology,
        projection_binder: GVALayerwiseProjectionBinder,
        resources: KVPoolResources,
        *,
        store_enabled: bool = True,
        layerwise_prefetch_layers: int = 2,
        start_gate_factory: Callable[[], Any] = reset_attention_compute_start_gate,
        source_ready_event_factory: Callable[[], Any] | None = None,
    ) -> None:
        if resources.backend_spec.layerwise_access is not LayerwiseAccessKind.GVA:
            raise TypeError("GVALayerwiseWorker requires a GVA Backend")
        backend_io = GVABackendIO(cast(GVABackend, resources.backend), resources.backend_spec)
        self._gva_backend_io = backend_io
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
        if not isinstance(projection, GVALayerwiseProjection):
            raise TypeError("GVA Backend requires a GVA Layerwise projection")
        self._gva_backend_io.bind_projection(projection)


__all__ = ("GVALayerwiseWorker",)
