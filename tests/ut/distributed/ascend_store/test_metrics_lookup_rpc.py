# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""The lookup RPC failure counts as an exported metric.

These drive real ``prometheus_client`` objects against an isolated registry, so
each assertion is what a scrape would actually see -- metric name, the ``failure``
label, the value -- rather than only that a fake recorded a call.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metrics import (
    LOOKUP_RPC_FAILURES_KEY,
    AscendStoreKVConnectorStats,
    AscendStorePromMetrics,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

_COUNTS = {"recv:timeout": 3, "server:backend_error": 1}


def _types_for(registry):
    """Bind all three metric classes to one registry.

    Leaving Counter/Histogram on the default registry would make the second test in
    this file raise DuplicateTimeseries -- the constructor is reached for every
    AscendStorePromMetrics built, not only for the gauge under test.
    """

    def _bind(cls):
        def _factory(*args, **kwargs):
            return cls(*args, registry=registry, **kwargs)

        return _factory

    return {Gauge: _bind(Gauge), Counter: _bind(Counter), Histogram: _bind(Histogram)}


def _samples(registry):
    for metric in registry.collect():
        if metric.name == "vllm:ascend_store_lookup_rpc_failures":
            return {sample.labels.get("failure"): sample.value for sample in metric.samples}
    raise AssertionError("the lookup RPC gauge was not registered")


def test_lookup_counts_reach_a_real_prometheus_gauge():
    registry = CollectorRegistry()
    prom = AscendStorePromMetrics(MagicMock(), _types_for(registry), ["model_name"], {0: ["proto-model"]})
    stats = AscendStoreKVConnectorStats(data={LOOKUP_RPC_FAILURES_KEY: dict(_COUNTS)})
    assert stats.is_empty() is False

    prom.observe(stats.data)

    assert _samples(registry) == {"recv:timeout": 3.0, "server:backend_error": 1.0}


def test_gauge_is_bound_to_the_engine_labels_and_the_failure_category():
    registry = CollectorRegistry()
    prom = AscendStorePromMetrics(MagicMock(), _types_for(registry), ["model_name"], {0: ["proto-model"]})
    prom.observe({LOOKUP_RPC_FAILURES_KEY: {"decode:decode_error": 7}})

    for metric in registry.collect():
        if metric.name == "vllm:ascend_store_lookup_rpc_failures":
            assert [tuple(sorted(sample.labels.items())) for sample in metric.samples] == [
                (("failure", "decode:decode_error"), ("model_name", "proto-model"))
            ]
            assert metric.samples[0].value == 7.0
            return
    raise AssertionError("the lookup RPC gauge was not registered")


def test_scheduler_counts_survive_aggregation_into_worker_stats():
    worker = AscendStoreKVConnectorStats()
    worker.record_operation("load_get", 0.01, 3)
    worker.aggregate(AscendStoreKVConnectorStats(data={LOOKUP_RPC_FAILURES_KEY: {"recv:timeout": 2}}))
    worker.aggregate(AscendStoreKVConnectorStats(data={LOOKUP_RPC_FAILURES_KEY: {"recv:timeout": 5}}))

    assert worker.data[LOOKUP_RPC_FAILURES_KEY] == {"recv:timeout": 7}
    assert worker.data["load_get_keys"] == 3


def test_get_stats_publishes_only_what_the_client_has_counted():
    quiet = SimpleNamespace(client=SimpleNamespace(snapshot=lambda: {}))
    assert KVPoolScheduler.get_stats(quiet) is None  # type: ignore[arg-type]
    assert KVPoolScheduler.get_stats(SimpleNamespace(client=None)) is None  # type: ignore[arg-type]

    failing = SimpleNamespace(client=SimpleNamespace(snapshot=lambda: dict(_COUNTS)))
    stats = KVPoolScheduler.get_stats(failing)  # type: ignore[arg-type]
    assert stats is not None
    assert stats.data == {LOOKUP_RPC_FAILURES_KEY: _COUNTS}
    assert stats.is_empty() is False


def test_published_stats_cannot_be_rewritten_by_later_failures():
    # A stats object can outlive the call that made it (queued for the next metrics
    # flush), so it must not alias the client's live map. Handing out copies is the
    # contract of snapshot(); this pins that get_stats() actually goes through it.
    live = dict(_COUNTS)
    client = SimpleNamespace(snapshot=lambda: dict(live), failure_counts=live)
    stats = KVPoolScheduler.get_stats(SimpleNamespace(client=client))  # type: ignore[arg-type]
    live["recv:timeout"] = 99

    assert stats is not None
    assert stats.data == {LOOKUP_RPC_FAILURES_KEY: _COUNTS}


def test_an_empty_payload_never_starts_a_scrape_series():
    registry = CollectorRegistry()
    prom = AscendStorePromMetrics(MagicMock(), _types_for(registry), ["model_name"], {0: ["proto-model"]})
    prom.observe({})

    assert _samples(registry) == {}
