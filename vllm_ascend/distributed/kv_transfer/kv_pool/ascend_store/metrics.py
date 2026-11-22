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
QUEUE_WAIT_HISTOGRAM_BUCKETS = (
    1e-3,
    5e-3,
    1e-2,
    5e-2,
    1e-1,
    2.5e-1,
    5e-1,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

ASCEND_STORE_OPERATIONS = ("load_get", "save_put")


@dataclass
class AscendStoreKVConnectorStats(KVConnectorStats):
    """Serializable AscendStore connector metrics."""

    def reset(self) -> None:
        self.data.clear()

    def is_empty(self) -> bool:
        return not self.data

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        for operation in ASCEND_STORE_OPERATIONS:
            if durations := other.data.get(f"{operation}_duration_seconds"):
                self.data.setdefault(f"{operation}_duration_seconds", []).extend(durations)
            for field in ("keys", "bytes"):
                if value := other.data.get(f"{operation}_{field}"):
                    self.data[f"{operation}_{field}"] = self.data.get(f"{operation}_{field}", 0) + value
            if queue_waits := other.data.get(f"{operation}_queue_wait_seconds"):
                self.data.setdefault(f"{operation}_queue_wait_seconds", []).extend(queue_waits)
        if "delayed_release_requests" in other.data:
            self.data["delayed_release_requests"] = other.data["delayed_release_requests"]
            self.data["delayed_release_blocks"] = other.data["delayed_release_blocks"]
        return self

    def reduce(self) -> dict[str, int | float]:
        reduced: dict[str, int | float] = {
            "ascend_store_delayed_release_requests": self.data.get("delayed_release_requests", 0),
            "ascend_store_delayed_release_blocks": self.data.get("delayed_release_blocks", 0),
        }
        for operation in ASCEND_STORE_OPERATIONS:
            prefix = f"ascend_store_{operation}"
            durations = self.data.get(f"{operation}_duration_seconds")
            if durations:
                reduced[f"{prefix}_count"] = len(durations)
                reduced[f"{prefix}_avg_ms"] = round(fmean(durations) * 1e3, 3)
                reduced[f"{prefix}_total_ms"] = round(sum(durations) * 1e3, 3)
                reduced[f"{prefix}_keys"] = self.data.get(f"{operation}_keys", 0)
                reduced[f"{prefix}_bytes"] = self.data.get(f"{operation}_bytes", 0)
            queue_waits = self.data.get(f"{operation}_queue_wait_seconds")
            if queue_waits:
                reduced[f"{prefix}_count_with_queue_wait"] = len(queue_waits)
                reduced[f"{prefix}_avg_queue_wait_ms"] = round(fmean(queue_waits) * 1e3, 3)
                reduced[f"{prefix}_max_queue_wait_ms"] = round(max(queue_waits) * 1e3, 3)
        return reduced

    def record_operation(
        self,
        operation: str,
        duration_seconds: float,
        num_keys: int,
        num_bytes: int = 0,
        queue_wait_seconds: float | None = None,
    ) -> None:
        if operation not in ASCEND_STORE_OPERATIONS:
            return
        self.data.setdefault(f"{operation}_duration_seconds", []).append(duration_seconds)
        self.data[f"{operation}_keys"] = self.data.get(f"{operation}_keys", 0) + num_keys
        self.data[f"{operation}_bytes"] = self.data.get(f"{operation}_bytes", 0) + num_bytes
        if queue_wait_seconds is not None:
            self.data.setdefault(f"{operation}_queue_wait_seconds", []).append(queue_wait_seconds)

    def set_delayed_release(self, num_requests: int, num_blocks: int) -> None:
        self.data["delayed_release_requests"] = num_requests
        self.data["delayed_release_blocks"] = num_blocks


class AscendStorePromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._delayed_release_requests = create_metric_per_engine(
            self._gauge_cls(
                name="vllm:ascend_store_delayed_release_requests",
                documentation=(
                    "Number of finished requests whose KV cache block release is "
                    "delayed by an asynchronous AscendStore save."
                ),
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._delayed_release_blocks = create_metric_per_engine(
            self._gauge_cls(
                name="vllm:ascend_store_delayed_release_blocks",
                documentation=(
                    "Number of KV cache block references retained for finished "
                    "requests while asynchronous AscendStore saves complete."
                ),
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
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
        self._load_get_bytes = create_metric_per_engine(
            self._counter_cls(
                name="vllm:ascend_store_load_get_bytes_total",
                documentation="Number of KV bytes requested by completed AscendStore Backend.get calls.",
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._load_get_queue_wait = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:ascend_store_load_get_queue_wait_seconds",
                documentation="Histogram of time a load waited between enqueue and Backend.get.",
                buckets=QUEUE_WAIT_HISTOGRAM_BUCKETS,
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._save_put_duration = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:ascend_store_save_put_duration_seconds",
                documentation="Histogram of per-worker non-layerwise AscendStore Backend.put duration.",
                buckets=LOAD_GET_HISTOGRAM_BUCKETS,
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._save_put_keys = create_metric_per_engine(
            self._counter_cls(
                name="vllm:ascend_store_save_put_keys_total",
                documentation="Number of keys passed to completed AscendStore Backend.put calls.",
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._save_put_bytes = create_metric_per_engine(
            self._counter_cls(
                name="vllm:ascend_store_save_put_bytes_total",
                documentation="Number of KV bytes written by completed AscendStore Backend.put calls.",
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._save_put_queue_wait = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:ascend_store_save_put_queue_wait_seconds",
                documentation="Histogram of time a save job waited before Backend.put.",
                buckets=QUEUE_WAIT_HISTOGRAM_BUCKETS,
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
        metric = self._load_get_bytes.get(engine_idx)
        if metric is not None:
            metric.inc(transfer_stats_data.get("load_get_bytes", 0))
        metric = self._load_get_queue_wait.get(engine_idx)
        if metric is not None:
            for queue_wait in transfer_stats_data.get("load_get_queue_wait_seconds", ()):
                metric.observe(queue_wait)
        metric = self._save_put_duration.get(engine_idx)
        if metric is not None:
            for duration in transfer_stats_data.get("save_put_duration_seconds", ()):
                metric.observe(duration)
        metric = self._save_put_keys.get(engine_idx)
        if metric is not None:
            metric.inc(transfer_stats_data.get("save_put_keys", 0))
        metric = self._save_put_bytes.get(engine_idx)
        if metric is not None:
            metric.inc(transfer_stats_data.get("save_put_bytes", 0))
        metric = self._save_put_queue_wait.get(engine_idx)
        if metric is not None:
            for queue_wait in transfer_stats_data.get("save_put_queue_wait_seconds", ()):
                metric.observe(queue_wait)
        metric = self._delayed_release_requests.get(engine_idx)
        if metric is not None:
            if "delayed_release_requests" in transfer_stats_data:
                metric.set(transfer_stats_data["delayed_release_requests"])
        metric = self._delayed_release_blocks.get(engine_idx)
        if metric is not None:
            if "delayed_release_blocks" in transfer_stats_data:
                metric.set(transfer_stats_data["delayed_release_blocks"])
