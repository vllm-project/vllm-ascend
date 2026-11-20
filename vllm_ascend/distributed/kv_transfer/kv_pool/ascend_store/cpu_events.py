# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Internal physical-key evidence for blockwise CPU KV publications."""

from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic

from vllm.distributed.kv_events import BlockStored, KVCacheEvent

CPU_EVENT_RETENTION_SECONDS = 60.0
MAX_TRACKED_CPU_EVENTS = 4096


class CPUBlockStoreReport(KVCacheEvent):
    """Internal event transported by the existing worker event aggregator.

    Never forward this report to an external event publisher. Request identity
    allows retries to contribute without conflating independent requests.
    """

    event: BlockStored
    request_id: str
    required_keys: frozenset[str]
    completed_keys: frozenset[str]
    newly_stored: bool = True

    def __hash__(self) -> int:
        return hash((self.request_id, self.required_keys, self.completed_keys, self.newly_stored))


@dataclass
class _PendingBlock:
    created_at: float
    event: BlockStored | None
    completed_keys: set[str]
    newly_stored: bool = False


class CPUBlockEventTracker:
    """Scheduler-owned, bounded evidence and deduplication across batches.

    Evidence is isolated by request and exact physical key set.
    Expiry/capacity eviction drops evidence rather than publishing partial KV.
    Published entries retain only their identity until expiry, so duplicate
    reports from replicas do not publish again. No global lifetime state.
    """

    def __init__(self) -> None:
        self._blocks: OrderedDict[tuple[str, frozenset[str]], _PendingBlock] = OrderedDict()

    def expire(self) -> None:
        now = monotonic()
        while self._blocks:
            first = next(iter(self._blocks.values()))
            if now - first.created_at < CPU_EVENT_RETENTION_SECONDS:
                break
            self._blocks.popitem(last=False)

    def add(self, report: CPUBlockStoreReport) -> BlockStored | None:
        self.expire()
        if not report.required_keys or not report.completed_keys or not report.completed_keys <= report.required_keys:
            return None
        key = (report.request_id, report.required_keys)
        block = self._blocks.get(key)
        if block is None:
            if len(self._blocks) >= MAX_TRACKED_CPU_EVENTS:
                self._blocks.popitem(last=False)
            block = _PendingBlock(monotonic(), report.event, set())
            self._blocks[key] = block
        if block.event is None:
            return None
        block.completed_keys.update(report.completed_keys)
        block.newly_stored |= report.newly_stored
        if block.newly_stored and report.required_keys <= block.completed_keys:
            event = block.event
            block.event = None
            block.completed_keys.clear()
            return event
        return None

    def clear(self) -> None:
        self._blocks.clear()
