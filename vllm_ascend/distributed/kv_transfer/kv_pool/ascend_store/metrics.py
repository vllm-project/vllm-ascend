# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Observability metrics for the AscendStore KV pool connector.

Two families of telemetry are collected:

- ``load``: per-request wall-clock duration of a KV pool load, together
  with the number of keys touched and keys that failed to load. One
  record per request, with the loading path (``sync`` / ``async`` /
  ``layerwise``) attached as a label. The measured span is the full load
  path on this rank: ``sync`` includes key preparation before
  ``m_store.get``; ``async`` additionally includes receive-thread queueing
  time; ``layerwise`` spans from task submission to the completion of the
  last layer transfer (the transfer thread's completion timestamp, not
  the compute-side wait return, so compute running past a finished load
  no longer stretches the sample; prefetch layers may still include
  waiting for the attention-start gate).
  Each record also carries ``not_overlapped_seconds``: the portion of the
  load the compute thread actually stalled on. ``layerwise`` sums the
  per-layer ``wait_for_layer_load`` stalls (wait entry -> event set);
  ``sync``/``async`` loads block the compute thread for their whole
  duration, so theirs defaults to ``duration_seconds``.
- ``delayed_release``: latest snapshot of the number of requests whose KV
  blocks are held in the delayed-release window on the scheduler side
  (i.e. ``len(pool_scheduler._delayed_free_req_ids)``).

``data`` only contains primitives so it can cross process boundaries
(worker -> scheduler -> engine core -> API server) via msgpack.
"""

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


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(0, min(len(sorted_values) - 1, int(percentile * len(sorted_values) - 1e-12)))
    return sorted_values[rank]


@dataclass
class AscendStoreKVConnectorStats(KVConnectorStats):
    """Serializable KV pool telemetry.

    Layout of ``data`` (all values are primitives):

    - ``load``: ``list`` of ``{duration_seconds: float, num_keys: int,
      num_failed_keys: int, path: str, not_overlapped_seconds: float}``,
      one entry per request whose KV cache was actually loaded from the
      pool (``num_keys > 0``).
    - ``delayed_release``: ``{num_requests: int}`` gauge snapshot recorded
      by the scheduler; the latest snapshot wins during aggregation.
    """

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data: dict[str, Any] = {}

    def is_empty(self) -> bool:
        return not self.data

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if other.is_empty():
            return self
        for key, value in other.data.items():
            if isinstance(value, list):
                self.data.setdefault(key, []).extend(value)
            else:
                # Gauge snapshot: the latest observation wins.
                self.data[key] = value
        return self

    def reduce(self) -> dict[str, int | float]:
        reduced: dict[str, int | float] = {}
        records = self.data.get("load")
        if records:
            durations = [float(record["duration_seconds"]) for record in records]
            reduced["load_count"] = len(records)
            reduced["load_avg_ms"] = round(fmean(durations) * 1e3, 3)
            reduced["load_p90_ms"] = round(_nearest_rank_percentile(durations, 0.9) * 1e3, 3)
            reduced["load_keys"] = sum(int(record["num_keys"]) for record in records)
            reduced["load_failed_keys"] = sum(int(record["num_failed_keys"]) for record in records)
            not_overlapped = [float(record["not_overlapped_seconds"]) for record in records]
            reduced["not_overlapped_avg_ms"] = round(fmean(not_overlapped) * 1e3, 3)
            reduced["not_overlapped_p90_ms"] = round(_nearest_rank_percentile(not_overlapped, 0.9) * 1e3, 3)
            ratios = [1.0 - no / d for no, d in zip(not_overlapped, durations) if d > 0]
            if ratios:
                reduced["overlap_ratio_avg"] = round(fmean(ratios), 4)
        delayed_release = self.data.get("delayed_release")
        if delayed_release is not None:
            reduced["delayed_release_requests"] = int(delayed_release["num_requests"])
        return reduced

    def record_load(
        self,
        duration_seconds: float,
        num_keys: int,
        *,
        num_failed_keys: int = 0,
        path: str = "sync",
        not_overlapped_seconds: float | None = None,
    ) -> None:
        # sync/async loads block the compute thread for the whole duration,
        # so their not-overlapped time defaults to the duration itself.
        if not_overlapped_seconds is None:
            not_overlapped_seconds = duration_seconds
        self.data.setdefault("load", []).append(
            {
                "duration_seconds": duration_seconds,
                "num_keys": num_keys,
                "num_failed_keys": num_failed_keys,
                "path": path,
                "not_overlapped_seconds": not_overlapped_seconds,
            }
        )

    def record_delayed_release(self, num_requests: int) -> None:
        self.data["delayed_release"] = {"num_requests": int(num_requests)}


class AscendStorePromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for the AscendStore KV pool.

    Metrics:

    - ``vllm:kv_pool_load_duration_seconds`` (Histogram, label ``path``):
      per-request KV pool load wall-clock duration. Measured spans differ
      per path: ``sync`` covers key preparation plus ``m_store.get``;
      ``async`` additionally includes queueing time in the receiving
      thread; ``layerwise`` covers from layer-task submission to the
      completion of the last layer transfer (transfer-thread completion
      time; compute overlap no longer stretches the sample).
    - ``vllm:kv_pool_load_keys_total`` (Counter, label ``path``): number of
      pool keys loaded. ``sync``/``async`` count this rank's key chunks
      (keys are circular-shifted across TP ranks, so sums across ranks
      equal the global key count); ``layerwise`` counts per-layer block
      transfers (approx. blocks x layers), which is not directly
      comparable with the other paths.
    - ``vllm:kv_pool_load_failed_keys_total`` (Counter, label ``path``):
      number of pool keys that failed to load. Only meaningful for
      ``sync``/``async``; on the ``layerwise`` path a transfer failure is
      fatal (the transfer thread raises), so this counter is expected to
      stay 0 there.
    - ``vllm:kv_pool_load_not_overlapped_seconds`` (Histogram, label
      ``path``): per-request portion of the pool load the compute thread
      actually stalled on. ``layerwise`` sums the per-layer
      ``wait_for_layer_load`` stalls; ``sync``/``async`` equal the full
      duration. Near-zero ``layerwise`` samples mean the transfers were
      effectively hidden behind compute.
    - ``vllm:kv_pool_load_overlap_ratio`` (Histogram, label ``path``):
      ``1 - not_overlapped / duration`` per request. 1.0 means the load
      was fully overlapped with compute; ``sync``/``async`` stay near 0
      by construction.
    - ``vllm:kv_pool_delayed_release_requests`` (Gauge): number of requests
      whose KV blocks are currently held in the delayed-release window.
      Latest-snapshot semantics: reflects the most recent scheduling step,
      not the peak within a scrape interval.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        metric_labelnames = labelnames + ["path"]
        self._metric_cache: dict[tuple[int, str], dict[str, PromMetric]] = {}

        self._histogram_load_duration = self._histogram_cls(
            name="vllm:kv_pool_load_duration_seconds",
            documentation="Histogram of per-request KV cache load duration from the KV pool.",
            buckets=[
                1e-3,
                5e-3,
                1e-2,
                2.5e-2,
                5e-2,
                7.5e-2,
                1e-1,
                2e-1,
                3e-1,
                5e-1,
                7.5e-1,
                1.0,
                2.5,
                5.0,
            ],
            labelnames=metric_labelnames,
        )
        self._counter_load_keys = self._counter_cls(
            name="vllm:kv_pool_load_keys_total",
            documentation="Number of KV pool keys loaded per request.",
            labelnames=metric_labelnames,
        )
        self._counter_load_failed_keys = self._counter_cls(
            name="vllm:kv_pool_load_failed_keys_total",
            documentation="Number of KV pool keys that failed to load.",
            labelnames=metric_labelnames,
        )
        self._histogram_load_not_overlapped = self._histogram_cls(
            name="vllm:kv_pool_load_not_overlapped_seconds",
            documentation="Histogram of per-request KV pool load time not overlapped with compute.",
            buckets=[
                1e-3,
                5e-3,
                1e-2,
                2.5e-2,
                5e-2,
                7.5e-2,
                1e-1,
                2e-1,
                5e-1,
                1.0,
                2.5,
                5.0,
            ],
            labelnames=metric_labelnames,
        )
        self._histogram_load_overlap_ratio = self._histogram_cls(
            name="vllm:kv_pool_load_overlap_ratio",
            documentation="Histogram of per-request KV pool load overlap ratio (1 - not_overlapped/duration).",
            buckets=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
            labelnames=metric_labelnames,
        )
        self._gauge_delayed_release = self._gauge_cls(
            name="vllm:kv_pool_delayed_release_requests",
            documentation="Number of requests whose KV blocks are currently held in the delayed-release window.",
            labelnames=labelnames,
        )
        self._delayed_release_per_engine = create_metric_per_engine(self._gauge_delayed_release, per_engine_labelvalues)

    def _get_load_metrics(self, engine_idx: int, path: str) -> dict[str, PromMetric]:
        cache_key = (engine_idx, path)
        if cache_key not in self._metric_cache:
            label_values = self.per_engine_labelvalues[engine_idx] + [path]
            self._metric_cache[cache_key] = {
                "duration": self._histogram_load_duration.labels(*label_values),
                "keys": self._counter_load_keys.labels(*label_values),
                "failed_keys": self._counter_load_failed_keys.labels(*label_values),
                "not_overlapped": self._histogram_load_not_overlapped.labels(*label_values),
                "overlap_ratio": self._histogram_load_overlap_ratio.labels(*label_values),
            }
        return self._metric_cache[cache_key]

    def observe(self, transfer_stats_data: dict[str, Any] | None, engine_idx: int = 0):
        if not transfer_stats_data:
            return
        for record in transfer_stats_data.get("load", []):
            if not isinstance(record, dict):
                # Malformed entry after deserialization; skip it rather than
                # crash the metrics pipeline (assert would vanish under -O).
                continue
            metrics = self._get_load_metrics(engine_idx, str(record["path"]))
            duration = float(record["duration_seconds"])
            not_overlapped = float(record.get("not_overlapped_seconds", duration))
            metrics["duration"].observe(duration)
            metrics["keys"].inc(int(record["num_keys"]))
            metrics["failed_keys"].inc(int(record["num_failed_keys"]))
            metrics["not_overlapped"].observe(not_overlapped)
            if duration > 0:
                metrics["overlap_ratio"].observe(1.0 - not_overlapped / duration)
        delayed_release = transfer_stats_data.get("delayed_release")
        if delayed_release is not None:
            gauge = self._delayed_release_per_engine.get(engine_idx)
            if gauge is not None:
                gauge.set(int(delayed_release["num_requests"]))
