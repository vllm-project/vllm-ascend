# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass
from statistics import fmean
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine

LOAD_GET_HISTOGRAM_BUCKETS = (
    1e-3,
    5e-3,
    1e-2,
    5e-2,
    1e-1,
    2e-1,
    3e-1,
    4e-1,
    5e-1,
    7.5e-1,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
)


@dataclass
class AscendStoreKVConnectorStats(KVConnectorStats):
    """Serializable AscendStore connector metrics."""

    def reset(self) -> None:
        self.data.clear()

    def is_empty(self) -> bool:
        return not self.data

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if durations := other.data.get("load_get_duration_seconds"):
            self.data.setdefault("load_get_duration_seconds", []).extend(durations)
        if num_keys := other.data.get("load_get_keys"):
            self.data["load_get_keys"] = self.data.get("load_get_keys", 0) + num_keys
        return self

    def reduce(self) -> dict[str, int | float]:
        reduced: dict[str, int | float] = {}
        if durations := self.data.get("load_get_duration_seconds"):
            reduced["ascend_store_load_get_count"] = len(durations)
            reduced["ascend_store_load_get_avg_ms"] = round(fmean(durations) * 1e3, 3)
            reduced["ascend_store_load_get_keys"] = self.data.get("load_get_keys", 0)
        return reduced

    def record_operation(self, operation: str, duration_seconds: float, num_keys: int) -> None:
        if operation != "load_get":
            return
        self.data.setdefault("load_get_duration_seconds", []).append(duration_seconds)
        self.data["load_get_keys"] = self.data.get("load_get_keys", 0) + num_keys


class AscendStorePromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._load_get_duration = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:ascend_store_load_get_duration_seconds",
                documentation="Histogram of per-worker non-layerwise AscendStore Backend.get duration.",
                buckets=LOAD_GET_HISTOGRAM_BUCKETS,
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._load_get_keys = create_metric_per_engine(
            self._counter_cls(
                name="vllm:ascend_store_load_get_keys_total",
                documentation="Number of keys passed to completed AscendStore Backend.get calls.",
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        metric = self._load_get_duration.get(engine_idx)
        if metric is not None:
            for duration in transfer_stats_data.get("load_get_duration_seconds", ()):
                metric.observe(duration)
        metric = self._load_get_keys.get(engine_idx)
        if metric is not None:
            metric.inc(transfer_stats_data.get("load_get_keys", 0))
