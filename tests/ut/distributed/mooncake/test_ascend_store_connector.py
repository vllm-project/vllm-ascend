import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (  # noqa: E402
    AscendStoreConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (  # noqa: E402
    KVPoolWorker,
)


class TestAscendStoreConnectorDeferredFinalize(unittest.TestCase):

    def test_wait_for_save_replays_finished_requests(self):
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        connector.kv_role = "kv_producer"
        connector.consumer_is_to_put = False
        connector.use_layerwise = False
        connector.connector_worker = MagicMock()
        connector._finished_req_ids_waiting_for_save = {"req-1"}
        connector._late_finished_sending = set()

        metadata = SimpleNamespace()
        connector._get_connector_metadata = MagicMock(return_value=metadata)
        connector.connector_worker.register_finished_requests.return_value = {"req-1"}

        AscendStoreConnector.wait_for_save(connector)

        connector.connector_worker.wait_for_save.assert_called_once_with(metadata)
        connector.connector_worker.register_finished_requests.assert_called_once_with({"req-1"})
        self.assertEqual(connector._late_finished_sending, {"req-1"})
        self.assertEqual(connector._finished_req_ids_waiting_for_save, set())

    def test_get_finished_drains_late_finished_sending(self):
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        connector.connector_worker = MagicMock()
        connector._finished_req_ids_waiting_for_save = set()
        connector._late_finished_sending = {"req-late"}
        connector._get_connector_metadata = MagicMock(return_value=SimpleNamespace())
        connector.connector_worker.get_finished.return_value = ({"req-now"}, {"req-recv"})

        done_sending, done_recving = AscendStoreConnector.get_finished(connector, {"req-finished"})

        self.assertEqual(done_sending, {"req-late", "req-now"})
        self.assertEqual(done_recving, {"req-recv"})
        self.assertEqual(connector._finished_req_ids_waiting_for_save, {"req-finished"})
        self.assertEqual(connector._late_finished_sending, set())


class TestKVPoolWorkerFinishedReplay(unittest.TestCase):

    def test_register_finished_requests_tracks_deferred_sends(self):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.kv_send_thread = SimpleNamespace(
            stored_requests={"req-pending": 1, "req-ready": 0},
            delete_finished_stored_request=MagicMock(),
        )
        worker.finished_store_req = set()

        finished_now = KVPoolWorker.register_finished_requests(
            worker, {"req-pending", "req-ready", "req-missing"}
        )

        self.assertEqual(finished_now, {"req-ready"})
        self.assertEqual(worker.finished_store_req, {"req-pending"})
        worker.kv_send_thread.delete_finished_stored_request.assert_called_once_with("req-ready")

        worker.kv_send_thread.stored_requests["req-pending"] = 0
        finished_later = KVPoolWorker.get_and_clear_finished_requests(
            worker, set(), SimpleNamespace(preempted_req_ids=set())
        )

        self.assertEqual(finished_later, {"req-pending"})
        worker.kv_send_thread.delete_finished_stored_request.assert_called_with("req-pending")


if __name__ == "__main__":
    unittest.main()
