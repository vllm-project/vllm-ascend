#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import time
import unittest
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metrics import (
    AscendStoreKVConnectorStats,
    AscendStorePromMetrics,
)


# ---------------------------------------------------------------------------
# Minimal fake prometheus_client metric primitives (duck-typed).
# ---------------------------------------------------------------------------
class _FakeMetricChild:
    def __init__(self):
        self.observations = []
        self.value = 0

    def observe(self, value):
        self.observations.append(value)

    def inc(self, amount=1):
        self.value += amount

    def set(self, value):
        self.value = value


class _FakeMetric:
    def __init__(self, name, documentation, labelnames=None, buckets=None, **kwargs):
        self.name = name
        self.documentation = documentation
        self.labelnames = list(labelnames or [])
        self.buckets = buckets
        self.children = {}

    def labels(self, *labelvalues):
        key = tuple(labelvalues)
        if key not in self.children:
            self.children[key] = _FakeMetricChild()
        return self.children[key]


class _FakeGauge(_FakeMetric):
    pass


class _FakeCounter(_FakeMetric):
    pass


class _FakeHistogram(_FakeMetric):
    pass


def _make_metric_types():
    """Build the metric_types dict for both mock and real vllm envs.

    The mock KVConnectorPromMetrics resolves classes by dict insertion
    order; the real one resolves them by the prometheus_client types as
    keys. Build both key styles so the dict works either way.
    """
    try:
        from prometheus_client import Counter, Gauge, Histogram

        return {
            Gauge: _FakeGauge,
            Counter: _FakeCounter,
            Histogram: _FakeHistogram,
        }
    except ImportError:
        return {
            "gauge": _FakeGauge,
            "counter": _FakeCounter,
            "histogram": _FakeHistogram,
        }


def _make_prom_metrics():
    vllm_config = MagicMock()
    return AscendStorePromMetrics(
        vllm_config,
        _make_metric_types(),
        ["model_name"],
        {0: ["test-model"]},
    )


class TestAscendStoreKVConnectorStats(unittest.TestCase):
    def test_starts_empty_and_resets(self):
        stats = AscendStoreKVConnectorStats()
        self.assertTrue(stats.is_empty())
        stats.record_load(0.1, 4)
        self.assertFalse(stats.is_empty())
        stats.reset()
        self.assertTrue(stats.is_empty())

    def test_record_load_appends_records(self):
        stats = AscendStoreKVConnectorStats()
        stats.record_load(0.5, 10, num_failed_keys=2, path="async")
        stats.record_load(0.25, 5, path="sync")
        records = stats.data["load"]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["path"], "async")
        self.assertEqual(records[0]["num_failed_keys"], 2)
        self.assertEqual(records[1]["num_keys"], 5)

    def test_record_delayed_release_overwrites(self):
        stats = AscendStoreKVConnectorStats()
        stats.record_delayed_release(3)
        stats.record_delayed_release(7)
        self.assertEqual(stats.data["delayed_release"]["num_requests"], 7)

    def test_aggregate_merges_lists_and_keeps_latest_gauge(self):
        first = AscendStoreKVConnectorStats()
        first.record_load(0.1, 2)
        first.record_delayed_release(1)
        second = AscendStoreKVConnectorStats()
        second.record_load(0.2, 3)
        second.record_delayed_release(5)

        first.aggregate(second)
        self.assertEqual(len(first.data["load"]), 2)
        self.assertEqual(first.data["delayed_release"]["num_requests"], 5)

    def test_aggregate_with_empty_other_is_noop(self):
        stats = AscendStoreKVConnectorStats()
        stats.record_load(0.1, 2)
        stats.aggregate(AscendStoreKVConnectorStats())
        self.assertEqual(len(stats.data["load"]), 1)

    def test_reduce_load_metrics(self):
        stats = AscendStoreKVConnectorStats()
        for duration in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            stats.record_load(duration, 2, num_failed_keys=1)
        reduced = stats.reduce()
        self.assertEqual(reduced["load_count"], 10)
        self.assertAlmostEqual(reduced["load_avg_ms"], 550.0)
        # Nearest-rank p90 of 10 samples: index int(0.9*10-eps)=8 -> 0.9s
        self.assertAlmostEqual(reduced["load_p90_ms"], 900.0)
        self.assertEqual(reduced["load_keys"], 20)
        self.assertEqual(reduced["load_failed_keys"], 10)

    def test_reduce_delayed_release_only(self):
        stats = AscendStoreKVConnectorStats()
        stats.record_delayed_release(4)
        reduced = stats.reduce()
        self.assertEqual(reduced["delayed_release_requests"], 4)
        self.assertNotIn("load_count", reduced)

    def test_reduce_empty(self):
        self.assertEqual(AscendStoreKVConnectorStats().reduce(), {})

    def test_rebuild_from_plain_data(self):
        stats = AscendStoreKVConnectorStats()
        stats.record_load(0.1, 2)
        stats.record_delayed_release(3)
        # Simulate the msgpack round-trip: only plain dicts/lists/numbers.
        rebuilt = AscendStoreKVConnectorStats(data=stats.data)
        self.assertEqual(rebuilt.reduce()["load_count"], 1)
        self.assertEqual(rebuilt.reduce()["delayed_release_requests"], 3)


class TestAscendStorePromMetrics(unittest.TestCase):
    def test_observe_records_load_metrics_with_path_label(self):
        prom = _make_prom_metrics()
        prom.observe(
            {
                "load": [
                    {
                        "duration_seconds": 0.05,
                        "num_keys": 8,
                        "num_failed_keys": 1,
                        "path": "sync",
                    }
                ]
            }
        )
        duration_child = prom._histogram_load_duration.children[("test-model", "sync")]
        self.assertEqual(duration_child.observations, [0.05])
        keys_child = prom._counter_load_keys.children[("test-model", "sync")]
        self.assertEqual(keys_child.value, 8)
        failed_child = prom._counter_load_failed_keys.children[("test-model", "sync")]
        self.assertEqual(failed_child.value, 1)

    def test_observe_separates_paths(self):
        prom = _make_prom_metrics()
        prom.observe(
            {
                "load": [
                    {"duration_seconds": 0.1, "num_keys": 1, "num_failed_keys": 0, "path": "sync"},
                    {"duration_seconds": 0.2, "num_keys": 2, "num_failed_keys": 0, "path": "layerwise"},
                ]
            }
        )
        self.assertEqual(len(prom._histogram_load_duration.children), 2)
        self.assertIn(("test-model", "sync"), prom._histogram_load_duration.children)
        self.assertIn(("test-model", "layerwise"), prom._histogram_load_duration.children)

    def test_observe_sets_delayed_release_gauge(self):
        prom = _make_prom_metrics()
        prom.observe({"delayed_release": {"num_requests": 6}})
        gauge_child = prom._gauge_delayed_release.children[("test-model",)]
        self.assertEqual(gauge_child.value, 6)

    def test_observe_empty_data_is_noop(self):
        prom = _make_prom_metrics()
        prom.observe(None)
        prom.observe({})
        self.assertEqual(prom._histogram_load_duration.children, {})

    def test_metric_registration_names(self):
        prom = _make_prom_metrics()
        self.assertEqual(prom._histogram_load_duration.name, "vllm:kv_pool_load_duration_seconds")
        self.assertEqual(prom._counter_load_keys.name, "vllm:kv_pool_load_keys_total")
        self.assertEqual(prom._counter_load_failed_keys.name, "vllm:kv_pool_load_failed_keys_total")
        self.assertEqual(prom._gauge_delayed_release.name, "vllm:kv_pool_delayed_release_requests")


class TestKVPoolWorkerLoadTiming(unittest.TestCase):
    """Verify the load-timing instrumentation on KVPoolWorker."""

    def _make_worker(self):
        import importlib

        module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker"
        # Patching requires the module (and thus its attributes) to exist.
        importlib.import_module(module)
        with (
            patch(f"{module}.get_tensor_model_parallel_rank", return_value=0),
            patch(f"{module}.get_tensor_model_parallel_world_size", return_value=1),
            patch(f"{module}.get_pcp_group") as pcp_group,
            patch(f"{module}.get_decode_context_model_parallel_world_size", return_value=1),
            patch(f"{module}.get_decode_context_model_parallel_rank", return_value=0),
            patch(f"{module}.importlib") as mock_importlib,
        ):
            pcp_group.return_value.world_size = 1
            mock_importlib.import_module.return_value = MagicMock()

            config = MagicMock()
            config.model_config.model = "org/llama-7b"
            config.model_config.use_mla = False
            config.model_config.hf_text_config = MagicMock(spec=[])
            config.model_config.get_num_layers.return_value = 2
            config.model_config.get_total_num_kv_heads.return_value = 1
            config.parallel_config.data_parallel_rank = 0
            config.parallel_config.rank = 0
            config.parallel_config.pipeline_parallel_size = 1
            config.kv_transfer_config.kv_role = "kv_producer"
            config.kv_transfer_config.kv_connector_extra_config = {"backend": "mooncake"}
            config.cache_config.block_size = 16
            config.kv_events_config = None

            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
                KVPoolWorker,
            )

            return KVPoolWorker(config, use_layerwise=False)

    def test_sync_load_timing_recorded(self):
        worker = self._make_worker()
        worker._record_load_started("req-1")
        time.sleep(0.01)
        worker._record_load_finished("req-1", 4, num_failed_keys=1, path="sync")

        stats = worker.get_stats()
        self.assertIsNotNone(stats)
        records = stats.data["load"]
        self.assertEqual(len(records), 1)
        self.assertGreaterEqual(records[0]["duration_seconds"], 0.01)
        self.assertEqual(records[0]["num_keys"], 4)
        self.assertEqual(records[0]["num_failed_keys"], 1)
        self.assertEqual(records[0]["path"], "sync")

    def test_get_stats_resets_between_calls(self):
        worker = self._make_worker()
        worker._record_load_started("req-1")
        worker._record_load_finished("req-1", 2)
        self.assertIsNotNone(worker.get_stats())
        self.assertIsNone(worker.get_stats())

    def test_unstarted_request_is_ignored(self):
        worker = self._make_worker()
        worker._record_load_finished("ghost", 3)
        self.assertIsNone(worker.get_stats())

    def test_zero_keys_request_is_ignored(self):
        worker = self._make_worker()
        worker._record_load_started("req-0")
        worker._record_load_finished("req-0", 0)
        self.assertIsNone(worker.get_stats())
        # Start time must not leak.
        self.assertNotIn("req-0", worker._load_start_times)

    def test_layerwise_timing_recorded(self):
        worker = self._make_worker()

        block_range = MagicMock()
        block_range.request.req_id = "req-lw"
        block_range.start_block = 0
        block_range.end_block = 3
        block_range.partial_block_index = None
        task = MagicMock()
        task.block_ranges = [block_range]
        worker.layer_load_tasks = [[task]]

        worker._record_layerwise_load_started()
        time.sleep(0.01)
        worker._record_layerwise_load_finished()

        stats = worker.get_stats()
        self.assertIsNotNone(stats)
        records = stats.data["load"]
        self.assertEqual(len(records), 1)
        self.assertGreaterEqual(records[0]["duration_seconds"], 0.01)
        self.assertEqual(records[0]["num_keys"], 3)
        self.assertEqual(records[0]["path"], "layerwise")
        # Both bookkeeping dicts are drained.
        self.assertEqual(worker._load_start_times, {})
        self.assertEqual(worker._layerwise_load_keys, {})

    def test_layerwise_partial_block_counted(self):
        """Verify partial_block_index adds 1 to the key count."""
        worker = self._make_worker()

        block_range = MagicMock()
        block_range.request.req_id = "req-pw"
        block_range.start_block = 0
        block_range.end_block = 2
        block_range.partial_block_index = 5  # extra partial block
        task = MagicMock()
        task.block_ranges = [block_range]
        worker.layer_load_tasks = [[task]]

        worker._record_layerwise_load_started()
        time.sleep(0.01)
        worker._record_layerwise_load_finished()

        stats = worker.get_stats()
        self.assertIsNotNone(stats)
        records = stats.data["load"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["num_keys"], 3)  # 2 full + 1 partial
        self.assertEqual(records[0]["path"], "layerwise")

    def test_layerwise_duration_uses_event_set_time(self):
        """Layerwise end time must be the transfer-thread event set time.

        The load threads record their completion timestamp when they set
        the per-layer event. The metric must use that timestamp (the max
        across layers) instead of the compute-side wait return time, so a
        long compute tail after the load already finished does not stretch
        the sample.
        """
        worker = self._make_worker()

        block_range = MagicMock()
        block_range.request.req_id = "req-lw"
        block_range.start_block = 0
        block_range.end_block = 3
        block_range.partial_block_index = None
        task = MagicMock()
        task.block_ranges = [block_range]
        worker.layer_load_tasks = [[task]]

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
            _TimedLayerLoadEvent,
        )

        events = [_TimedLayerLoadEvent() for _ in range(2)]
        worker.layer_load_finished_events = events

        worker._record_layerwise_load_started()
        # Layer 1's transfer finishes first ...
        time.sleep(0.02)
        events[1].set()
        # ... then layer 0's transfer finishes later (max wins) ...
        time.sleep(0.03)
        events[0].set()
        # ... and compute keeps running long after the loads finished.
        time.sleep(0.2)
        worker._record_layerwise_load_finished()

        stats = worker.get_stats()
        self.assertIsNotNone(stats)
        record = stats.data["load"][0]
        # Duration must reflect the latest event set time (~0.05s), not
        # the wait-return time (~0.25s) and not layer 1 alone (~0.02s).
        self.assertGreaterEqual(record["duration_seconds"], 0.05)
        self.assertLess(record["duration_seconds"], 0.15)
        self.assertEqual(record["num_keys"], 3)
        self.assertEqual(record["path"], "layerwise")


class TestKVPoolSchedulerDelayedRelease(unittest.TestCase):
    """Verify the delayed-release gauge snapshot on KVPoolScheduler."""

    def _make_scheduler(self):
        import importlib

        module = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler"
        # Patching requires the module (and thus its attributes) to exist.
        importlib.import_module(module)
        with (
            patch(f"{module}.LookupKeyClient"),
            patch(f"{module}.importlib") as mock_importlib,
        ):
            mock_importlib.import_module.return_value = MagicMock()

            config = MagicMock()
            config.kv_transfer_config.kv_role = "kv_producer"
            config.kv_transfer_config.kv_connector_extra_config = {}
            config.kv_transfer_config.get_from_extra_config.return_value = True
            config.parallel_config.data_parallel_rank = 0
            config.parallel_config.prefill_context_parallel_size = 1
            config.parallel_config.decode_context_parallel_size = 1
            config.parallel_config.tensor_parallel_size = 1
            config.parallel_config.pipeline_parallel_size = 1
            config.parallel_config.rank = 0
            config.parallel_config.world_size = 1
            config.cache_config.block_size = 16
            config.cache_config.hash_block_size = 16
            config.model_config.model = "org/llama-7b"
            config.model_config.use_mla = False
            config.model_config.hf_text_config = MagicMock(spec=[])
            config.model_config.get_total_num_kv_heads.return_value = 1
            config.model_config.get_num_layers.return_value = 2

            from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
                KVPoolScheduler,
            )

            return KVPoolScheduler(config, use_layerwise=False)

    def test_snapshot_records_delayed_free_window(self):
        scheduler = self._make_scheduler()
        scheduler._delayed_free_req_ids.update({"r1", "r2"})
        scheduler._kv_stats.record_delayed_release(len(scheduler._delayed_free_req_ids))

        stats = scheduler.get_stats()
        self.assertEqual(stats.data["delayed_release"]["num_requests"], 2)

    def test_get_stats_resets_between_calls(self):
        scheduler = self._make_scheduler()
        scheduler._kv_stats.record_delayed_release(1)
        first = scheduler.get_stats()
        self.assertEqual(first.data["delayed_release"]["num_requests"], 1)
        # After the reset a fresh snapshot is empty until recorded again.
        self.assertTrue(scheduler.get_stats().is_empty())


if __name__ == "__main__":
    unittest.main()
