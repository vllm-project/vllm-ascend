"""Static inputs used to select one concrete AscendStore projection binder."""

from __future__ import annotations

from dataclasses import dataclass

from .topology import KVPoolTopology


@dataclass(frozen=True, slots=True)
class KVPoolRouteSpec:
    """Resolved configuration that selects one Worker execution route."""

    topology: KVPoolTopology
    backend_name: str
    max_model_len: int
    use_layerwise: bool = False
    use_eagle: bool = False
    retention_interval: int | None = None
