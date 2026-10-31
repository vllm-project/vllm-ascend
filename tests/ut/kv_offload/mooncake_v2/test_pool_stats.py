# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from functools import partial
from types import SimpleNamespace

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.layerwise_pool_connector import LayerwisePoolConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pool_stats import PoolPDStats


def test_worker_stats_roundtrip_aggregate_and_reset():
    stats = PoolPDStats()
    stats.add("restore_completed")
    stats.add("restore_bytes", 16)
    other = LayerwisePoolConnector.build_kv_connector_stats(json.loads(json.dumps(stats.data)))
    other.add("restore_failed")
    stats.aggregate(other)
    assert stats.reduce()["pool_pd_restore_bytes"] == 32
    assert stats.reduce()["pool_pd_restore_completed"] == 2
    assert stats.reduce()["pool_pd_restore_failed"] == 1
    stats.reset()
    assert stats.is_empty()
    assert other.data["restore_bytes"] == 16


def test_prometheus_counts_are_per_engine_and_have_no_request_labels():
    registry = CollectorRegistry()
    metrics = LayerwisePoolConnector.build_prom_metrics(
        SimpleNamespace(kv_transfer_config=None),
        {kind: partial(kind, registry=registry) for kind in (Counter, Gauge, Histogram)},
        ["engine"],
        {0: ["p"], 1: ["d"]},
    )
    metrics.observe({"save_committed": 1, "save_bytes": 32}, engine_idx=0)
    metrics.observe({"restore_completed": 1, "restore_bytes": 16}, engine_idx=1)
    metrics.observe({"restore_failed": 1}, engine_idx=1)
    text = generate_latest(registry).decode()
    assert 'vllm:pool_pd_save_bytes_total{engine="p"} 32.0' in text
    assert 'vllm:pool_pd_restore_bytes_total{engine="d"} 16.0' in text
    assert 'vllm:pool_pd_restore_failed_total{engine="d"} 1.0' in text
    assert 'vllm:pool_pd_restore_bytes_total{engine="p"} 0.0' in text
