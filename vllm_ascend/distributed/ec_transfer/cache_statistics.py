# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Cumulative scheduler decisions, independent of cache metadata eviction."""

from dataclasses import dataclass
from threading import Lock
from uuid import uuid4


@dataclass(frozen=True)
class CacheStatistics:
    exact_hits: int = 0
    similarity_hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in self.as_tuple()):
            raise ValueError("Cache counters must be nonnegative integers")

    def as_tuple(self) -> tuple[int, int, int]:
        return self.exact_hits, self.similarity_hits, self.misses

    @property
    def total(self) -> int:
        return self.exact_hits + self.similarity_hits + self.misses

    @property
    def hit_rate(self) -> float:
        return (self.exact_hits + self.similarity_hits) / self.total if self.total else 0.0

    @property
    def similarity_hit_rate(self) -> float:
        return self.similarity_hits / self.total if self.total else 0.0

    @property
    def similarity_lookup_hit_rate(self) -> float:
        attempts = self.similarity_hits + self.misses
        return self.similarity_hits / attempts if attempts else 0.0


@dataclass(frozen=True)
class CacheStatisticsSnapshot:
    counts: CacheStatistics
    reporting_schedulers: int
    epoch: str


class CacheStatisticsAccumulator:
    """Retain per-instance cumulative reports for this index lifetime.

    Scheduler IDs must survive RPC retries but change on scheduler restart.
    Completed schedulers stay in the totals. The lock protects both reports
    and totals, so concurrent or replayed reports cannot double count.
    """

    def __init__(self) -> None:
        self._reports: dict[str, CacheStatistics] = {}
        self._totals = CacheStatistics()
        self._epoch = uuid4().hex
        self._lock = Lock()

    def report(self, scheduler_id: str, counts: CacheStatistics) -> CacheStatisticsSnapshot:
        if not isinstance(scheduler_id, str) or not scheduler_id:
            raise ValueError("Cache statistics require a nonempty scheduler instance ID")
        with self._lock:
            previous = self._reports.get(scheduler_id, CacheStatistics())
            # Taking the maximum also tolerates delayed older snapshots.
            updated = CacheStatistics(*(max(old, new) for old, new in zip(previous.as_tuple(), counts.as_tuple())))
            self._totals = CacheStatistics(
                *(
                    total + new - old
                    for total, new, old in zip(self._totals.as_tuple(), updated.as_tuple(), previous.as_tuple())
                )
            )
            self._reports[scheduler_id] = updated
            return CacheStatisticsSnapshot(self._totals, len(self._reports), self._epoch)
