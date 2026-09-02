# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import dataclass
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine


@dataclass
class AscendStoreKVConnectorStats(KVConnectorStats):
    """Serializable AscendStore delayed-release observations."""

    def reset(self) -> None:
        self.data = {}

    def is_empty(self) -> bool:
        return not self.data

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if "delayed_release_requests" in other.data:
            self.data["delayed_release_requests"] = other.data["delayed_release_requests"]
        if started := other.data.get("delayed_release_started", 0):
            self.data["delayed_release_started"] = self.data.get("delayed_release_started", 0) + started
        return self

    def reduce(self) -> dict[str, int | float]:
        return {
            "delayed_release_requests": self.data.get("delayed_release_requests", 0),
            "delayed_release_started": self.data.get("delayed_release_started", 0),
        }

    def record_delayed_release_started(self) -> None:
        self.data["delayed_release_started"] = self.data.get("delayed_release_started", 0) + 1

    def set_delayed_release_requests(self, num_requests: int) -> None:
        self.data["delayed_release_requests"] = num_requests


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
                documentation=("Requests whose KV blocks are waiting for AscendStore save."),
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self._delayed_release_started = create_metric_per_engine(
            self._counter_cls(
                name="vllm:ascend_store_delayed_release_requests_total",
                documentation=("Requests that entered AscendStore delayed release."),
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        metric = self._delayed_release_requests.get(engine_idx)
        if metric is not None:
            if "delayed_release_requests" in transfer_stats_data:
                metric.set(transfer_stats_data["delayed_release_requests"])
        metric = self._delayed_release_started.get(engine_idx)
        if metric is not None:
            if started := transfer_stats_data.get("delayed_release_started", 0):
                metric.inc(started)
