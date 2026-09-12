# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, call

from prometheus_client import Counter, Gauge, Histogram

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import (
    MooncakeKVConnectorStats,
    MooncakePromMetrics,
)


def test_stats_record_reduce_and_reset() -> None:
    stats = MooncakeKVConnectorStats(data={})
    stats.record_transfer(0.01, 2**20)
    stats.record_transfer(0.03, 3 * 2**20)
    stats.record_failed_transfer()

    reduced = stats.reduce()

    assert reduced["Num successful transfers"] == 2
    assert reduced["Num failed transfers"] == 1
    assert reduced["Avg xfer time (ms)"] == 20.0
    assert reduced["Avg MB per transfer"] == 2.0
    assert reduced["Throughput (MB/s)"] == 100.0

    previous = stats.clone_and_reset()
    assert previous.num_successful_transfers == 2
    assert stats.is_empty()


def test_stats_aggregate_ignores_empty_and_merges_nonempty() -> None:
    stats = MooncakeKVConnectorStats(data={})
    empty = MooncakeKVConnectorStats(data={})
    other = MooncakeKVConnectorStats(data={})
    other.record_transfer(0.1, 1024)
    other.record_failed_transfer()

    assert stats.aggregate(empty) is stats
    stats.aggregate(other)

    assert stats.data["transfer_duration"] == [0.1]
    assert stats.data["bytes_transferred"] == [1024]
    assert stats.data["num_failed_transfers"] == [1]


def test_stats_aggregate_uses_latest_delayed_release_snapshot() -> None:
    stats = MooncakeKVConnectorStats(data={})
    entered = MooncakeKVConnectorStats(data={})
    entered.set_delayed_release(2, 7)
    released = MooncakeKVConnectorStats(data={})
    released.set_delayed_release(0, 0)

    stats.aggregate(entered).aggregate(released)

    assert stats.data["delayed_release_requests"] == 0
    assert stats.data["delayed_release_blocks"] == 0
    assert not stats.is_empty()


def test_prom_metrics_observes_delayed_release_snapshot() -> None:
    gauge_cls = MagicMock()
    metric_types = {Gauge: gauge_cls, Counter: MagicMock(), Histogram: MagicMock()}
    metrics = MooncakePromMetrics(MagicMock(), metric_types, ["model_name"], {0: ["test-model"]})
    stats = MooncakeKVConnectorStats()
    stats.set_delayed_release(2, 7)

    metrics.observe(stats.data)

    assert [metric.kwargs["name"] for metric in gauge_cls.call_args_list] == [
        "vllm:mooncake_pd_delayed_release_requests",
        "vllm:mooncake_pd_delayed_release_blocks",
    ]
    assert gauge_cls.return_value.labels.return_value.set.call_args_list == [call(2), call(7)]
