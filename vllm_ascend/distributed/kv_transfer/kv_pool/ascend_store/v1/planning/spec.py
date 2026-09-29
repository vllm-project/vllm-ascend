"""Static policies used by Scheduler-side transfer planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransferPlanningSpec:
    cache_transfer_granularity: int
    hash_block_size: int
    transfer_group_ids: tuple[int, ...]
    discard_partial_chunks: bool
