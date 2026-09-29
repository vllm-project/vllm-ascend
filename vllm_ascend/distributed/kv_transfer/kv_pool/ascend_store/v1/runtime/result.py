"""Results published by the KV Pool runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Terminal Load facts consumed through vLLM's split completion hooks."""

    completed_request_ids: frozenset[str]
    failed_request_ids: frozenset[str]
    failed_block_ids: frozenset[int]
