"""Terminal transfer results published by the KV Pool Worker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoadFailureLocation:
    """Public cache location whose Load success was not confirmed."""

    group_id: int
    block_id: int


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Terminal Load facts consumed through vLLM's split completion hooks."""

    completed_request_ids: frozenset[str]
    failed_request_ids: frozenset[str]
    failed_block_ids: frozenset[int]
    failed_locations: tuple[LoadFailureLocation, ...] = ()
