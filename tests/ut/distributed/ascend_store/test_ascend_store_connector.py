import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

# Stub external modules that may not be available
for mod_name in [
    "vllm.v1.attention.backend",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
        if mod_name == "vllm.v1.attention.backend":
            sys.modules[mod_name].AttentionMetadata = MagicMock()  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreKVEvents,
)


class TestAscendStoreKVEvents(unittest.TestCase):
    def test_creation(self):
        events = AscendStoreKVEvents(num_workers=2)
        self.assertEqual(events.get_number_of_workers(), 2)
        self.assertEqual(events.get_all_events(), [])

    def test_add_events(self):
        events = AscendStoreKVEvents(num_workers=1)
        mock_event = MagicMock()
        mock_event.block_hashes = [b"hash1"]
        events.add_events([mock_event])
        all_events = events.get_all_events()
        self.assertEqual(len(all_events), 1)

    def test_clear_events(self):
        events = AscendStoreKVEvents(num_workers=1)
        mock_event = MagicMock()
        mock_event.block_hashes = [b"hash1"]
        events.add_events([mock_event])
        events.clear_events()
        self.assertEqual(events.get_all_events(), [])

    def test_increment_workers(self):
        events = AscendStoreKVEvents(num_workers=1)
        events.increment_workers(2)
        self.assertEqual(events.get_number_of_workers(), 3)

    def test_aggregate(self):
        events = AscendStoreKVEvents(num_workers=1)
        mock_event = MagicMock()
        mock_event.block_hashes = [b"hash1"]
        events.add_events([mock_event])
        result = events.aggregate()
        self.assertIs(result, events)

    def test_repr(self):
        events = AscendStoreKVEvents(num_workers=1)
        r = repr(events)
        self.assertIn("AscendStoreKVEvents", r)


class TestAscendStoreConnectorRequiresPiecewise(unittest.TestCase):
    def test_requires_piecewise_true(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
            AscendStoreConnector,
        )

        result = AscendStoreConnector.requires_piecewise_for_cudagraph(
            {"use_layerwise": True}
        )
        self.assertTrue(result)

    def test_requires_piecewise_false(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
            AscendStoreConnector,
        )

        result = AscendStoreConnector.requires_piecewise_for_cudagraph({})
        self.assertFalse(result)

    def test_requires_piecewise_explicit_false(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
            AscendStoreConnector,
        )

        result = AscendStoreConnector.requires_piecewise_for_cudagraph(
            {"use_layerwise": False}
        )
        self.assertFalse(result)


class TestAscendStoreConnectorUpdateConnectorOutput(unittest.TestCase):
    def _make_connector_scheduler_side(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
            AscendStoreConnector,
        )

        connector = object.__new__(AscendStoreConnector)
        connector._kv_cache_events = None
        connector.sended_but_unfinished_reqs = set()
        return connector

    def test_update_connector_output_none_events(self):
        connector = self._make_connector_scheduler_side()
        output = MagicMock()
        output.kv_cache_events = None
        connector.update_connector_output(output)
        self.assertIsNone(connector._kv_cache_events)

    def test_update_connector_output_non_ascend_events(self):
        connector = self._make_connector_scheduler_side()
        output = MagicMock()
        output.kv_cache_events = "not_ascend_events"
        connector.update_connector_output(output)
        self.assertIsNone(connector._kv_cache_events)

    def test_update_connector_output_first_events(self):
        connector = self._make_connector_scheduler_side()
        kv_events = AscendStoreKVEvents(num_workers=1)
        output = MagicMock()
        output.kv_cache_events = kv_events
        connector.update_connector_output(output)
        self.assertIs(connector._kv_cache_events, kv_events)

    def test_update_connector_output_merge_events(self):
        connector = self._make_connector_scheduler_side()
        existing_events = AscendStoreKVEvents(num_workers=1)
        connector._kv_cache_events = existing_events

        new_events = AscendStoreKVEvents(num_workers=1)
        mock_event = MagicMock()
        mock_event.block_hashes = [b"h"]
        new_events.add_events([mock_event])

        output = MagicMock()
        output.kv_cache_events = new_events
        connector.update_connector_output(output)
        self.assertIs(connector._kv_cache_events, existing_events)

    def test_take_events_empty(self):
        connector = self._make_connector_scheduler_side()
        events = list(connector.take_events())
        self.assertEqual(events, [])

    def test_take_events_with_data(self):
        connector = self._make_connector_scheduler_side()
        kv_events = AscendStoreKVEvents(num_workers=1)
        mock_event = MagicMock()
        mock_event.block_hashes = [b"h1"]
        kv_events.add_events([mock_event])
        connector._kv_cache_events = kv_events
        events = list(connector.take_events())
        self.assertEqual(len(events), 1)
        # After take, should be None
        self.assertIsNone(connector._kv_cache_events)


class TestAscendStoreConnectorWorkerDelegation(unittest.TestCase):
    def _make_connector_worker_side(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
            AscendStoreConnector,
        )

        connector = object.__new__(AscendStoreConnector)
        connector.kv_role = "kv_producer"
        connector.use_layerwise = False
        connector.consumer_is_to_put = False
        connector.kv_caches = {}
        connector._kv_cache_events = None
        connector.sended_but_unfinished_reqs = set()
        connector.connector_worker = MagicMock()
        connector.connector_scheduler = None
        return connector

    def test_register_kv_caches(self):
        connector = self._make_connector_worker_side()
        kv = {"layer0": torch.zeros(1)}
        connector.register_kv_caches(kv)
        connector.connector_worker.register_kv_caches.assert_called_once_with(kv)

    def test_wait_for_layer_load_not_layerwise(self):
        connector = self._make_connector_worker_side()
        connector.use_layerwise = False
        connector.wait_for_layer_load("layer0")
        connector.connector_worker.wait_for_layer_load.assert_not_called()

    def test_wait_for_layer_load_layerwise(self):
        connector = self._make_connector_worker_side()
        connector.use_layerwise = True
        connector.wait_for_layer_load("layer0")
        connector.connector_worker.wait_for_layer_load.assert_called_once()

    def test_save_kv_layer_not_layerwise(self):
        connector = self._make_connector_worker_side()
        connector.use_layerwise = False
        connector.save_kv_layer("l0", torch.zeros(1), MagicMock())
        connector.connector_worker.save_kv_layer.assert_not_called()

    def test_save_kv_layer_consumer_role(self):
        connector = self._make_connector_worker_side()
        connector.use_layerwise = True
        connector.kv_role = "kv_consumer"
        connector.save_kv_layer("l0", torch.zeros(1), MagicMock())
        connector.connector_worker.save_kv_layer.assert_not_called()

    def test_save_kv_layer_layerwise_producer(self):
        connector = self._make_connector_worker_side()
        connector.use_layerwise = True
        connector.kv_role = "kv_producer"
        connector._get_connector_metadata = MagicMock(return_value=MagicMock())
        connector.save_kv_layer("l0", torch.zeros(1), MagicMock())
        connector.connector_worker.save_kv_layer.assert_called_once()

    def test_wait_for_save_consumer_no_put(self):
        connector = self._make_connector_worker_side()
        connector.kv_role = "kv_consumer"
        connector.consumer_is_to_put = False
        connector.wait_for_save()
        connector.connector_worker.wait_for_save.assert_not_called()

    def test_wait_for_save_layerwise(self):
        connector = self._make_connector_worker_side()
        connector.use_layerwise = True
        connector.wait_for_save()
        connector.connector_worker.wait_for_save.assert_not_called()

    def test_wait_for_save_normal(self):
        connector = self._make_connector_worker_side()
        connector._get_connector_metadata = MagicMock(return_value=MagicMock())
        connector.wait_for_save()
        connector.connector_worker.wait_for_save.assert_called_once()

    def test_get_finished(self):
        connector = self._make_connector_worker_side()
        connector._get_connector_metadata = MagicMock(return_value=MagicMock())
        connector.connector_worker.get_finished.return_value = ({"r1"}, {"r2"})
        done_send, done_recv = connector.get_finished({"r1"})
        self.assertEqual(done_send, {"r1"})
        self.assertEqual(done_recv, {"r2"})

    def test_get_kv_connector_kv_cache_events_empty(self):
        connector = self._make_connector_worker_side()
        connector.connector_worker.get_kv_events.return_value = []
        result = connector.get_kv_connector_kv_cache_events()
        self.assertIsNone(result)

    def test_get_kv_connector_kv_cache_events_with_events(self):
        connector = self._make_connector_worker_side()
        mock_event = MagicMock()
        mock_event.block_hashes = [b"h"]
        connector.connector_worker.get_kv_events.return_value = [mock_event]
        result = connector.get_kv_connector_kv_cache_events()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, AscendStoreKVEvents)


class TestAscendStoreConnectorSchedulerDelegation(unittest.TestCase):
    def _make_connector_scheduler_side(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
            AscendStoreConnector,
        )

        connector = object.__new__(AscendStoreConnector)
        connector.kv_role = "kv_producer"
        connector.use_layerwise = False
        connector.consumer_is_to_put = False
        connector.kv_caches = {}
        connector._kv_cache_events = None
        connector.sended_but_unfinished_reqs = set()
        connector.connector_scheduler = MagicMock()
        connector.connector_worker = None
        return connector

    def test_get_num_new_matched_tokens(self):
        connector = self._make_connector_scheduler_side()
        connector.connector_scheduler.get_num_new_matched_tokens.return_value = (10, True)
        result = connector.get_num_new_matched_tokens(MagicMock(), 5)
        self.assertEqual(result, (10, True))

    def test_update_state_after_alloc(self):
        connector = self._make_connector_scheduler_side()
        connector.update_state_after_alloc(MagicMock(), MagicMock(), 10)
        connector.connector_scheduler.update_state_after_alloc.assert_called_once()

    def test_build_connector_meta(self):
        connector = self._make_connector_scheduler_side()
        connector.build_connector_meta(MagicMock())
        connector.connector_scheduler.build_connector_meta.assert_called_once()

    def test_request_finished(self):
        connector = self._make_connector_scheduler_side()
        connector.connector_scheduler.request_finished.return_value = (True, None)
        result = connector.request_finished(MagicMock(), [1, 2])
        self.assertEqual(result, (True, None))


if __name__ == "__main__":
    unittest.main()
