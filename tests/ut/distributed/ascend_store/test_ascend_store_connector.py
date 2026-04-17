#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402

import unittest
from unittest.mock import MagicMock, patch

from vllm.distributed.kv_events import KVCacheEvent
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreConnector,
    AscendStoreKVEvents,
)


class TestAscendStoreKVEvents(unittest.TestCase):
    def _make_events(self, num_workers=1):
        return AscendStoreKVEvents(num_workers=num_workers)

    def test_add_and_get_events(self):
        ev = self._make_events()
        mock_events = [MagicMock(spec=KVCacheEvent), MagicMock(spec=KVCacheEvent)]
        ev.add_events(mock_events)
        ev._aggregator.get_all_events.return_value = mock_events
        result = ev.get_all_events()
        self.assertEqual(result, mock_events)

    def test_aggregate(self):
        ev = self._make_events()
        common = [MagicMock()]
        ev._aggregator.get_common_events.return_value = common
        result = ev.aggregate()
        self.assertIs(result, ev)
        ev._aggregator.clear_events.assert_called()
        ev._aggregator.add_events.assert_called_with(common)
        ev._aggregator.reset_workers.assert_called()

    def test_increment_workers(self):
        ev = self._make_events()
        ev.increment_workers(3)
        ev._aggregator.increment_workers.assert_called_with(3)

    def test_get_number_of_workers(self):
        ev = self._make_events()
        ev._aggregator.get_number_of_workers.return_value = 5
        self.assertEqual(ev.get_number_of_workers(), 5)

    def test_clear_events(self):
        ev = self._make_events()
        ev.clear_events()
        ev._aggregator.clear_events.assert_called()
        ev._aggregator.reset_workers.assert_called()

    def test_repr(self):
        ev = self._make_events()
        ev._aggregator.get_all_events.return_value = []
        s = repr(ev)
        self.assertIn("AscendStoreKVEvents", s)


class TestAscendStoreConnector(unittest.TestCase):
    def _make_vllm_config(self, kv_role="kv_producer", extra_config=None):
        config = MagicMock()
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector = "AscendStoreConnector"
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {}
        config.parallel_config.rank = 0
        return config

    def test_requires_piecewise_for_cudagraph_true(self):
        self.assertTrue(
            AscendStoreConnector.requires_piecewise_for_cudagraph({"use_layerwise": True})
        )

    def test_requires_piecewise_for_cudagraph_false(self):
        self.assertFalse(
            AscendStoreConnector.requires_piecewise_for_cudagraph({"use_layerwise": False})
        )

    def test_requires_piecewise_for_cudagraph_missing(self):
        self.assertFalse(
            AscendStoreConnector.requires_piecewise_for_cudagraph({})
        )

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_init_scheduler_role(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        role = MagicMock()
        role.__eq__ = lambda self, other: other == "SCHEDULER"
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        mock_scheduler_cls.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_init_worker_role(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        mock_worker_cls.assert_called_once()
        mock_lookup_cls.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_scheduler_methods_delegate(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        mock_sched = mock_scheduler_cls.return_value

        # get_num_new_matched_tokens
        mock_sched.get_num_new_matched_tokens.return_value = (10, False)
        result = connector.get_num_new_matched_tokens(MagicMock(), 5)
        self.assertEqual(result, (10, False))

        # update_state_after_alloc
        connector.update_state_after_alloc(MagicMock(), MagicMock(), 10)
        mock_sched.update_state_after_alloc.assert_called_once()

        # build_connector_meta
        connector.build_connector_meta(MagicMock())
        mock_sched.build_connector_meta.assert_called_once()

        # request_finished
        mock_sched.request_finished.return_value = (True, None)
        result = connector.request_finished(MagicMock(), [1, 2])
        self.assertEqual(result, (True, None))

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_update_connector_output_no_events(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        output = MagicMock()
        output.kv_cache_events = None
        connector.update_connector_output(output)
        self.assertIsNone(connector._kv_cache_events)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_update_connector_output_with_events(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        events = AscendStoreKVEvents(num_workers=1)
        mock_kv_events = [MagicMock()]
        events._aggregator.get_all_events.return_value = mock_kv_events
        events._aggregator.get_number_of_workers.return_value = 1

        output = MagicMock()
        output.kv_cache_events = events
        connector.update_connector_output(output)
        self.assertIsNotNone(connector._kv_cache_events)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_update_connector_output_accumulate(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        # First update
        events1 = AscendStoreKVEvents(num_workers=1)
        events1._aggregator.get_all_events.return_value = [MagicMock()]
        events1._aggregator.get_number_of_workers.return_value = 1
        output1 = MagicMock()
        output1.kv_cache_events = events1
        connector.update_connector_output(output1)

        # Second update
        events2 = AscendStoreKVEvents(num_workers=1)
        events2._aggregator.get_all_events.return_value = [MagicMock()]
        events2._aggregator.get_number_of_workers.return_value = 1
        output2 = MagicMock()
        output2.kv_cache_events = events2
        connector.update_connector_output(output2)
        self.assertIsNotNone(connector._kv_cache_events)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_take_events(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        # No events
        result = list(connector.take_events())
        self.assertEqual(result, [])

        # With events
        events = AscendStoreKVEvents(num_workers=1)
        mock_event = MagicMock()
        events._aggregator.get_common_events.return_value = [mock_event]
        events._aggregator.get_all_events.return_value = [mock_event]
        connector._kv_cache_events = events
        result = list(connector.take_events())
        self.assertEqual(len(result), 1)
        self.assertIsNone(connector._kv_cache_events)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_worker_methods(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        mock_worker = mock_worker_cls.return_value

        # register_kv_caches
        connector.register_kv_caches({"layer1": MagicMock()})
        mock_worker.register_kv_caches.assert_called_once()

        # start_load_kv
        connector._get_connector_metadata = MagicMock(return_value=MagicMock())
        connector.start_load_kv(MagicMock())
        mock_worker.start_load_kv.assert_called_once()

        # wait_for_save (non-consumer)
        connector.kv_role = "kv_producer"
        connector.use_layerwise = False
        connector.wait_for_save()
        mock_worker.wait_for_save.assert_called_once()

        # get_finished
        mock_worker.get_finished.return_value = ({"r1"}, {"r2"})
        done_s, done_r = connector.get_finished({"r1"})
        self.assertEqual(done_s, {"r1"})

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_wait_for_layer_load_not_layerwise(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config(extra_config={"use_layerwise": False})
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        # Should return immediately without calling worker
        connector.wait_for_layer_load("layer_0")

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_save_kv_layer_not_layerwise(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config(extra_config={"use_layerwise": False})
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        connector.save_kv_layer("layer_0", MagicMock(), MagicMock())
        # Should return immediately

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_save_kv_layer_consumer(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config(kv_role="kv_consumer", extra_config={"use_layerwise": True})
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        connector.save_kv_layer("layer_0", MagicMock(), MagicMock())
        # Consumer should not save

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_wait_for_save_consumer(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config(kv_role="kv_consumer")
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        connector.wait_for_save()
        mock_worker_cls.return_value.wait_for_save.assert_not_called()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_get_kv_connector_kv_cache_events_empty(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        mock_worker_cls.return_value.get_kv_events.return_value = []
        result = connector.get_kv_connector_kv_cache_events()
        self.assertIsNone(result)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_get_kv_connector_kv_cache_events_with_events(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        mock_worker_cls.return_value.get_kv_events.return_value = [MagicMock()]
        result = connector.get_kv_connector_kv_cache_events()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AscendStoreKVEvents)


if __name__ == "__main__":
    unittest.main()
