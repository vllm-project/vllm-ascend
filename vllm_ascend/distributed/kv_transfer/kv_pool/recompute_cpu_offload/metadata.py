# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for RecomputeCPUOffloadConnector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

INVALID_JOB_ID = -1


@dataclass
class RecomputeCPUOffloadMetadata(KVConnectorMetadata):
    """Recompute offload transfers passed from scheduler to worker."""

    # Whether any requests were preempted this step and need flush pending transfers.
    need_flush: bool = False

    # Store blocks of newly preempted requests before their GPU blocks can
    # be reused. The list may include a final partial block without a hash.
    preempt_store_event: int = INVALID_JOB_ID
    preempt_store_gpu_blocks: list[int] = field(default_factory=list)
    preempt_store_cpu_blocks: list[int] = field(default_factory=list)

    # Preemption load event. Used when a previously preempted request resumes.
    preempt_load_event: int = INVALID_JOB_ID
    preempt_load_gpu_blocks: list[int] = field(default_factory=list)
    preempt_load_cpu_blocks: list[int] = field(default_factory=list)
    preempt_load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class RecomputeCPUOffloadWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed store events.

    Each worker reports {event_idx: 1} for newly completed stores.
    ``aggregate()`` sums counts across workers within a step.
    The scheduler-side manager accumulates across steps and processes
    a store completion only when count reaches ``world_size``.
    """

    completed_store_events: dict[int, int]

    def aggregate(self, other: "KVConnectorWorkerMetadata") -> "KVConnectorWorkerMetadata":
        assert isinstance(other, RecomputeCPUOffloadWorkerMetadata)
        merged = dict(self.completed_store_events)
        for k, v in other.completed_store_events.items():
            merged[k] = merged.get(k, 0) + v
        return RecomputeCPUOffloadWorkerMetadata(completed_store_events=merged)
