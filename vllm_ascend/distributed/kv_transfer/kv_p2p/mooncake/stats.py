# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Transfer statistics and Prometheus metrics for Mooncake."""

import copy
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine


@dataclass
class MooncakeKVConnectorStats(KVConnectorStats):
    """D2D transfer statistics collected by a Mooncake worker."""

    def __post_init__(self) -> None:
        # Create the lock first so any defensive reset() (e.g. from
        # __setstate__) never observes a partially initialized object.
        # reset() itself must NOT acquire _lock (non-reentrant; see reset());
        # construction is single-threaded.
        self._lock = threading.Lock()
        if not self.data:
            self.reset()
        # The recording thread (pull execution) and the consuming thread
        # (worker loop via clone_and_reset) run concurrently; serialize
        # record/reset to avoid losing samples in the copy-then-reset window
        # (is_empty() outside the lock may miss the very last sample; that
        # sampling gap is tolerated for interval metrics).

    def reset(self) -> None:
        # Values must remain serializable because worker stats are aggregated
        # outside of the worker process.
        # Must NOT acquire self._lock: it is a non-reentrant Lock and all
        # callers either already hold it (record_*/clone_and_reset) or run
        # during single-threaded initialization (__post_init__ / __setstate__).
        self.data: dict[str, list[float | int]] = {
            "transfer_duration": [],
            "bytes_transferred": [],
            "num_failed_transfers": [],
        }

    def record_transfer(self, transfer_duration: float, bytes_transferred: int) -> None:
        """Record one completed decoder-side RDMA pull transfer (P -> D).

        ``transfer_duration`` is expressed in seconds.
        """
        with self._lock:
            self.data["transfer_duration"].append(transfer_duration)
            self.data["bytes_transferred"].append(bytes_transferred)

    def record_failed_transfer(self) -> None:
        """Record one failed decoder-side RDMA pull batch attempt."""
        with self._lock:
            self.data["num_failed_transfers"].append(1)

    def clone_and_reset(self) -> "MooncakeKVConnectorStats":
        """Return a snapshot of current samples and reset self.

        The snapshot shares the *old* data dict (reset() rebinds, never
        clears), so the caller may safely read it after reset without
        locking.
        """
        with self._lock:
            previous = copy.copy(self)
            self.reset()
            return previous

    def __getstate__(self) -> dict[str, Any]:
        # threading.Lock cannot be pickled/deep-copied. Worker outputs are
        # shipped to the engine core over a pickle-based channel
        # (shm_broadcast.MessageQueue), so the lock must be stripped whenever
        # the object leaves the worker process or is copied; __setstate__
        # rebuilds a fresh lock so copies never share one.
        # copy.copy/deepcopy both route through getstate/setstate (default
        # reduce protocol), so the lock is stripped on every copy path —
        # this is what makes clone_and_reset's shallow copy safe. Do NOT
        # add __copy__: it would bypass this and share the lock.
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Lock first, then any defensive reset — same order contract as
        # __post_init__ (reset must never observe a half-initialized self,
        # and must not acquire the lock itself; see reset()).
        self._lock = threading.Lock()
        self.__dict__.update(state)
        if not self.__dict__.get("data"):
            # Defensive: __getstate__ strips only the lock, so normal states
            # always carry data; tolerate future trimmed states anyway.
            self.reset()

    def is_empty(self) -> bool:
        # Lock-free on purpose (sampling trigger, see __post_init__). Under
        # concurrency the two reads may come from different reset generations
        # (the GIL keeps each reference read atomic, so this cannot crash —
        # but the fields may straddle a reset and disagree); this only
        # decides whether to clone, it is not exact accounting.
        return self.num_successful_transfers == 0 and not self.data["num_failed_transfers"]

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for key, values in other.data.items():
                accumulator = self.data[key]
                assert isinstance(accumulator, list)
                accumulator.extend(values)
        return self

    def reduce(self) -> dict[str, int | float]:
        if self.num_successful_transfers == 0:
            return {
                "Num successful transfers": 0,
                "Num failed transfers": len(self.data["num_failed_transfers"]),
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
            }

        durations = np.asarray(self.data["transfer_duration"])
        megabytes = np.asarray(self.data["bytes_transferred"]) / 2**20
        total_duration = durations.sum()
        throughput = megabytes.sum() / total_duration if total_duration > 0 else 0

        return {
            "Num successful transfers": self.num_successful_transfers,
            "Num failed transfers": len(self.data["num_failed_transfers"]),
            "Avg xfer time (ms)": round(durations.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(durations, 90).item() * 1e3, 3),
            "Avg MB per transfer": round(megabytes.mean(), 3),
            "Throughput (MB/s)": round(float(throughput), 3),
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class MooncakePromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for Mooncake D2D transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )

        transfer_duration = self._histogram_cls(
            name="vllm:mooncake_xfer_time_seconds",
            documentation=("Histogram of Mooncake KV cache D2D transfer duration."),
            buckets=[
                0.005,
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.2,
                0.3,
                0.5,
                0.75,
                1.0,
                5.0,
            ],
            labelnames=labelnames,
        )
        self.transfer_duration = create_metric_per_engine(transfer_duration, self.per_engine_labelvalues)

        bytes_transferred = self._histogram_cls(
            name="vllm:mooncake_bytes_transferred",
            documentation=("Histogram of bytes transferred per Mooncake KV cache D2D transfer."),
            buckets=[2 ** (10 + index) for index in range(1, 25, 2)],
            labelnames=labelnames,
        )
        self.bytes_transferred = create_metric_per_engine(bytes_transferred, self.per_engine_labelvalues)

        failed_transfers = self._counter_cls(
            name="vllm:mooncake_num_failed_transfers",
            documentation="Number of failed Mooncake KV cache D2D transfers.",
            labelnames=labelnames,
        )
        self.failed_transfers = create_metric_per_engine(failed_transfers, self.per_engine_labelvalues)

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        for duration in transfer_stats_data["transfer_duration"]:
            self.transfer_duration[engine_idx].observe(duration)
        for num_bytes in transfer_stats_data["bytes_transferred"]:
            self.bytes_transferred[engine_idx].observe(num_bytes)
        for failure in transfer_stats_data["num_failed_transfers"]:
            self.failed_transfers[engine_idx].inc(failure)
