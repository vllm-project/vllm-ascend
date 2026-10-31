# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Completed worker operations, collected through vLLM's connector metrics."""

from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorPromMetrics, KVConnectorStats
from vllm.v1.metrics.utils import create_metric_per_engine

COUNTERS = {
    "save_committed": "Committed and published pool PD worker saves.",
    "save_bytes": "Bytes committed by Prefill chunks, excluding borrowed objects.",
    "save_failed": "Failed pool PD worker saves, including failed publication.",
    "restore_completed": "Successful pool PD worker restores after native I/O and cleanup.",
    "restore_bytes": "Bytes in successful pool PD worker restores; excludes failed and cancelled attempts.",
    "restore_failed": "Failed pool PD worker restores, including incompatible layouts.",
    "restore_cancelled": "Cancelled pool PD worker restores after any active I/O drains.",
}


@dataclass
class PoolPDStats(KVConnectorStats):
    def reset(self) -> None:
        self.data = {}

    def is_empty(self) -> bool:
        return not any(self.data.values())

    def add(self, counter: str, value: int = 1) -> None:
        if counter not in COUNTERS or type(value) is not int or value < 0:
            raise ValueError("Invalid pool PD counter update")
        self.data[counter] = self.data.get(counter, 0) + value

    def aggregate(self, other: KVConnectorStats) -> "PoolPDStats":
        for key in COUNTERS:
            self.add(key, other.data.get(key, 0))
        return self

    def reduce(self) -> dict[str, int | float]:
        return {f"pool_pd_{key}": self.data.get(key, 0) for key in COUNTERS}


class PoolPDPromMetrics(KVConnectorPromMetrics):
    def __init__(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self.counters = {
            key: create_metric_per_engine(
                self._counter_cls(name=f"vllm:pool_pd_{key}_total", documentation=description, labelnames=labelnames),
                per_engine_labelvalues,
            )
            for key, description in COUNTERS.items()
        }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        for key, per_engine in self.counters.items():
            if engine_idx in per_engine:
                per_engine[engine_idx].inc(transfer_stats_data.get(key, 0))
