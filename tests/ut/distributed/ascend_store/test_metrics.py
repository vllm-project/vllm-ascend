# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metrics import (
    AscendStoreKVConnectorStats,
    AscendStorePromMetrics,
)


class _MetricChild:
    def __init__(self):
        self.observations = []
        self.value = 0

    def observe(self, value):
        self.observations.append(value)

    def inc(self, value=1):
        self.value += value

    def set(self, value):
        self.value = value


class _Metric:
    def __init__(self, *args, **kwargs):
        self.child = _MetricChild()

    def labels(self, *args):
        return self.child


def test_stats_aggregate_delayed_release_values():
    first = AscendStoreKVConnectorStats()
    first.record_delayed_release_started()
    second = AscendStoreKVConnectorStats()
    second.record_delayed_release_started()
    second.set_delayed_release_requests(2)

    first.aggregate(second)

    assert first.data == {
        "delayed_release_requests": 2,
        "delayed_release_started": 2,
    }


def test_prom_metrics_observe_request_level_values():
    prom = AscendStorePromMetrics(
        MagicMock(),
        {"gauge": _Metric, "counter": _Metric, "histogram": _Metric},
        ["model_name"],
        {0: ["test-model"]},
    )

    prom.observe(
        {
            "delayed_release_requests": 2,
            "delayed_release_started": 3,
        }
    )

    assert prom._delayed_release_requests[0].value == 2
    assert prom._delayed_release_started[0].value == 3
