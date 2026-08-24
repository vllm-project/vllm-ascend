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

import types
import unittest
from unittest.mock import MagicMock, patch

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm.distributed.kv_events import KVCacheEvent
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreConnector,
    AscendStoreKVEvents,
)

# isort: on


def _mock_events(num_workers=1):
    events = AscendStoreKVEvents(num_workers=num_workers)
    events._aggregator = MagicMock()
    return events


class TestAscendStoreKVEvents(unittest.TestCase):
    def test_event_lifecycle(self):
        ev = _mock_events()
        mock_events = [MagicMock(spec=KVCacheEvent), MagicMock(spec=KVCacheEvent)]
        ev.add_events(mock_events)
        ev._aggregator.get_all_events.return_value = mock_events
        self.assertEqual(ev.get_all_events(), mock_events)
        self.assertIn("AscendStoreKVEvents", repr(ev))

        ev.clear_events()
        ev._aggregator.clear_events.assert_called_once()
        ev._aggregator.reset_workers.assert_called_once()

    def test_worker_aggregation(self):
        ev = _mock_events()
        ev.increment_workers(3)
        ev._aggregator.increment_workers.assert_called_once_with(3)
        ev._aggregator.get_number_of_workers.return_value = 5
        self.assertEqual(ev.get_number_of_workers(), 5)

        common = [MagicMock()]
        ev._aggregator.get_common_events.return_value = common
        self.assertIs(ev.aggregate(), ev)
        ev._aggregator.clear_events.assert_called_once()
        ev._aggregator.add_events.assert_called_once_with(common)
        ev._aggregator.reset_workers.assert_called_once()


class TestAscendStoreConnector(unittest.TestCase):
    def _make_vllm_config(self, kv_role="kv_producer", extra_config=None):
        config = MagicMock()
        config.kv_transfer_config.kv_role = kv_role
        config.kv_transfer_config.kv_connector = "AscendStoreConnector"
        config.kv_transfer_config.kv_connector_extra_config = extra_config or {}
        config.parallel_config.rank = 0
        return config

    def test_pp_handshake_metadata_is_ignored(self):
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        metadata = {
            (0, 0): MagicMock(),
            (1, 0): MagicMock(),
        }
        original_metadata = metadata.copy()

        result = connector.set_xfer_handshake_metadata_pp_aware(metadata)

        self.assertIsNone(result)
        self.assertEqual(metadata, original_metadata)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_init_scheduler_role(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        _connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=MagicMock(),
        )
        mock_scheduler_cls.assert_called_once()

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.LookupKeyServer")
    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolWorker")
    def test_init_worker_role(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        _connector = AscendStoreConnector(
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
            kv_cache_config=MagicMock(),
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
    def test_update_connector_output_accumulates_events(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=MagicMock(),
        )
        output = MagicMock()
        output.kv_cache_events = None
        connector.update_connector_output(output)
        self.assertIsNone(connector._kv_cache_events)

        for _ in range(2):
            events = _mock_events()
            events._aggregator.get_all_events.return_value = [MagicMock()]
            events._aggregator.get_number_of_workers.return_value = 1
            output.kv_cache_events = events
            connector.update_connector_output(output)
        self.assertIsNotNone(connector._kv_cache_events)

    @patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector.KVPoolScheduler")
    def test_take_events(self, mock_scheduler_cls):
        config = self._make_vllm_config()
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=MagicMock(),
        )
        # No events
        result = list(connector.take_events())
        self.assertEqual(result, [])

        # With events
        events = _mock_events(num_workers=1)
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
    def test_layerwise_methods_return_early(self, mock_worker_cls, mock_lookup_cls):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        cases = [
            ("wait_for_layer_load", "kv_both", False),
            ("save_kv_layer", "kv_both", False),
            ("save_kv_layer", "kv_consumer", True),
            ("wait_for_save", "kv_consumer", False),
        ]
        for method_name, kv_role, use_layerwise in cases:
            with self.subTest(method=method_name, kv_role=kv_role, use_layerwise=use_layerwise):
                worker = mock_worker_cls.return_value
                worker.reset_mock()
                config = self._make_vllm_config(
                    kv_role=kv_role,
                    extra_config={"use_layerwise": use_layerwise},
                )
                connector = AscendStoreConnector(
                    vllm_config=config,
                    role=KVConnectorRole.WORKER,
                    kv_cache_config=None,
                )
                if method_name == "wait_for_layer_load":
                    connector.wait_for_layer_load("layer_0")
                elif method_name == "save_kv_layer":
                    connector.save_kv_layer("layer_0", MagicMock(), MagicMock())
                else:
                    connector.wait_for_save()
                getattr(worker, method_name).assert_not_called()

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
    def test_save_kv_layer_consumer_with_put_enabled(self, mock_worker_cls, mock_lookup_cls):
        config = self._make_vllm_config(
            kv_role="kv_consumer",
            extra_config={
                "use_layerwise": True,
                "consumer_is_to_put": True,
            },
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        connector = AscendStoreConnector(
            vllm_config=config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=None,
        )
        connector._get_connector_metadata = MagicMock(return_value=MagicMock())

        connector.save_kv_layer("layer_0", MagicMock(), MagicMock())

        mock_worker_cls.return_value.save_kv_layer.assert_called_once()

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
        for events, expected_type in (([], type(None)), ([MagicMock()], AscendStoreKVEvents)):
            mock_worker_cls.return_value.get_kv_events.return_value = events
            self.assertIsInstance(connector.get_kv_connector_kv_cache_events(), expected_type)


class TestAscendStoreConnectorLayerwise(unittest.TestCase):
    """Test connector methods that are specific to layerwise mode."""

    connector_mod: types.ModuleType

    @classmethod
    def setUpClass(cls):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import ascend_store_connector

        cls.connector_mod = ascend_store_connector

    def test_requires_piecewise_for_cudagraph(self):
        cases = [
            ({"use_layerwise": True}, True),
            ({"use_layerwise": False}, False),
            ({}, False),
        ]
        for config, expected in cases:
            with self.subTest(config=config):
                self.assertEqual(
                    self.connector_mod.AscendStoreConnector.requires_piecewise_for_cudagraph(config),
                    expected,
                )

    def test_layerwise_worker_paths(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        with (
            patch.object(self.connector_mod, "KVPoolWorker") as mock_worker_cls,
            patch.object(self.connector_mod, "LookupKeyServer") as _mock_lookup_cls,
        ):
            config = MagicMock()
            config.kv_transfer_config.kv_role = "kv_producer"
            config.kv_transfer_config.kv_connector = "AscendStoreConnector"
            config.kv_transfer_config.kv_connector_extra_config = {"use_layerwise": True}
            config.parallel_config.rank = 0

            connector = self.connector_mod.AscendStoreConnector(
                vllm_config=config,
                role=KVConnectorRole.WORKER,
                kv_cache_config=None,
            )
            connector.wait_for_save()
            mock_worker_cls.return_value.wait_for_save.assert_not_called()
            connector._get_connector_metadata = MagicMock(return_value=MagicMock())
            connector.save_kv_layer("layer_0", MagicMock(), MagicMock())
            mock_worker_cls.return_value.save_kv_layer.assert_called_once()

            config.kv_transfer_config.kv_role = "kv_consumer"
            connector = self.connector_mod.AscendStoreConnector(
                vllm_config=config,
                role=KVConnectorRole.WORKER,
                kv_cache_config=None,
            )
            connector.wait_for_layer_load("layer_0")
            mock_worker_cls.return_value.wait_for_layer_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
