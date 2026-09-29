"""Define the process-portable facts required to compile one KV Pool program."""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.kv_cache_interface import KVCacheSpec

from .schedule import KVPoolSchedule
from .topology import KVPoolTopology


@dataclass(frozen=True, slots=True)
class KVGroupReachabilitySpec:
    """Upstream cache semantics used to select reachable regions for one group."""

    group_id: int
    kv_cache_spec: KVCacheSpec
    is_eagle_group: bool = False


@dataclass(frozen=True, slots=True)
class KVPoolCompilationSpec:
    """Serializable startup facts consumed by the KV Pool program compiler."""

    topology: KVPoolTopology
    backend_name: str
    schedule: KVPoolSchedule
    max_model_len: int
    reachability_groups: tuple[KVGroupReachabilitySpec, ...]
    use_eagle: bool = False
    retention_interval: int | None = None
