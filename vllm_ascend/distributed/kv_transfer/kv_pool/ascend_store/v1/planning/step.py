"""Immutable request facts consumed by Scheduler-side transfer planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from vllm.v1.core.kv_cache_utils import BlockHash

from ..protocol.transfer import StateCheckpointSource


class ScheduledRequestKind(str, Enum):
    """How this Scheduler step positions a request and its block table."""

    NEW = "new"
    RUNNING = "running"
    RESUMED = "resumed"


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """Current request coordinates required to plan one transfer step.

    ``block_ids_by_group`` is the complete table for a new or resumed request and
    the appended table suffix for a running request.
    """

    request_id: str
    kind: ScheduledRequestKind
    block_ids_by_group: tuple[tuple[int, ...], ...] | None
    block_hashes: tuple[BlockHash, ...]
    num_prompt_tokens: int
    current_token_count: int
    num_computed_tokens: int
    num_scheduled_tokens: int


@dataclass(frozen=True, slots=True)
class StateCheckpointHandoff:
    """Scheduler-owned boundary state offered for remote publication."""

    request_id: str
    block_hashes: tuple[BlockHash, ...]
    sources: tuple[StateCheckpointSource, ...]


@dataclass(frozen=True, slots=True)
class TransferPlanningStep:
    """Planner-owned Scheduler facts for one planning invocation."""

    scheduled_requests: tuple[ScheduledRequest, ...] = ()
    finished_request_ids: frozenset[str] = frozenset()
    preempted_request_ids: frozenset[str] = frozenset()
    checkpoint_handoffs: tuple[StateCheckpointHandoff, ...] = ()
