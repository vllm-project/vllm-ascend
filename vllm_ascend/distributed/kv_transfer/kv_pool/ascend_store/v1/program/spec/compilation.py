"""Define the process-portable facts required to compile one KV Pool program."""

from __future__ import annotations

from dataclasses import dataclass

from .schedule import KVPoolSchedule
from .topology import KVPoolTopology


@dataclass(frozen=True, slots=True)
class KVPoolCompilationSpec:
    """Serializable startup facts consumed by the KV Pool program compiler."""

    topology: KVPoolTopology
    backend_name: str
    schedule: KVPoolSchedule
    max_model_len: int
    use_eagle: bool = False
    retention_interval: int | None = None
