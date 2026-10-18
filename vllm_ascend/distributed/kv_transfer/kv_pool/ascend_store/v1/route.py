"""Static inputs used to assemble one concrete AscendStore Worker route."""

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
