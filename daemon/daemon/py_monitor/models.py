from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoolSpec:
    device_id: int
    granularity_bytes: int
    total_bytes: int
    cap_bytes: int


@dataclass(frozen=True)
class PoolTuneOverride:
    extend_threshold_bytes: int
    extend_step_bytes: int
    remove_threshold_bytes: int
    remove_step_bytes: int
