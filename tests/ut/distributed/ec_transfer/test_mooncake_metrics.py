# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Observation accounting; no NPU needed."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import torch

from tests.ut.distributed.ec_transfer.test_mooncake_memory import _make_pool
from tests.ut.distributed.ec_transfer.test_mooncake_store_client import _client, _get
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake import metrics as module


def test_cumulative_metrics_are_thread_safe_and_duration_counts_are_separate():
    metrics = module.ECMetrics("test")
    with ThreadPoolExecutor(4) as executor:
        list(executor.map(lambda _: metrics.add(objects=1, payload_bytes=8), range(100)))
    metrics.duration("lease", 0.2)
    metrics.duration("lease", 0.4)
    result = metrics.snapshot()
    assert result["objects"] == 100 and result["payload_bytes"] == 800
    assert result["lease_count"] == 2
    assert abs(result["lease_seconds"] - 0.6) < 1e-8
    assert result["lease_max_seconds"] == 0.4


def test_lease_duration_recorded_only_once_after_release():
    pool = _make_pool(256)
    with patch("time.monotonic", return_value=100):
        lease = pool.reserve_bytes([32], torch.device("cpu"))
    with patch("time.monotonic", return_value=102):
        pool.release(lease)
        pool.release(lease)
    result = pool.metrics.snapshot()
    assert result["get_lease_count"] == 1
    assert result["get_lease_seconds"] == 2


def test_pool_unavailable_is_distinct_from_successful_get():
    client, pool, store = _client(256)
    held = pool.reserve_bytes([256], torch.device("cpu"))
    loaded, _ = _get(client, ["key"])
    assert loaded == {}
    store.batch_get_into.assert_not_called()
    result = client.metrics.snapshot()
    assert result["get_pool_unavailable_objects"] == 1
    assert result["get_pool_unavailable_payload_bytes"] == 32
    pool.release(held)
    loaded, _ = _get(client, ["key"])
    assert "key" in loaded
    result = client.metrics.snapshot()
    assert result["get_candidate_objects"] == 2
    assert result["get_loaded_objects"] == 1
