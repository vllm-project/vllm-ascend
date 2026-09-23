# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.worker import kv_connector_model_runner_mixin as runner_mixin

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreLayerSendingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class TestLayerwiseConnectorLifecycle(unittest.TestCase):
    def _make_connector(self):
        # Two target layers and one MTP layer. Keep the actual runner,
        # connector, worker and sender lifecycle; mock only hardware and I/O.
        worker = object.__new__(KVPoolWorker)
        worker.num_layers = 3
        worker.current_layer = 0
        worker.use_layerwise = True
        worker.pp_size = 1
        worker.prefetch_layer_map = {}
        worker.layer_save_tasks = [[] for _ in range(worker.num_layers)]
        worker.layer_load_tasks = [[] for _ in range(worker.num_layers)]
        worker.layer_save_finished_events = [threading.Event() for _ in range(worker.num_layers)]
        worker.sync_save_events = [SimpleNamespace(record=lambda: None) for _ in range(worker.num_layers)]

        sender = object.__new__(KVCacheStoreLayerSendingThread)
        sender.layer_save_finished_events = worker.layer_save_finished_events
        sender.request_queue = queue.Queue()
        sender.raise_if_failed = lambda: None
        sender.add_stored_request = lambda _: None

        def send(tasks):
            sender.request_queue.put(tasks)
            sender._handle_request(sender.request_queue.get())

        sender.add_request = send
        worker.kv_send_thread = sender

        def prepare(_requests):
            worker.layer_save_tasks = [
                [SimpleNamespace(layer_id=i, shared_block_data=None, block_ranges=[], use_key_major_ranges=False)]
                for i in range(worker.num_layers)
            ]

        worker.process_layer_data = prepare
        connector = object.__new__(AscendStoreConnector)
        connector.connector_worker = worker
        connector.use_layerwise = True
        connector.kv_role = "kv_producer"
        connector.consumer_is_to_put = False
        connector.get_finished = lambda _: (set(), set())
        connector.get_block_ids_with_load_errors = lambda: set()
        connector.get_kv_connector_stats = lambda: None
        connector.get_kv_connector_kv_cache_events = lambda: None
        connector.build_connector_worker_meta = lambda: None
        return connector, worker

    def test_deferred_load_start_preserves_target_progress_for_mtp(self):
        for has_sync_loads in (False, True):
            with self.subTest(has_sync_loads=has_sync_loads):
                connector, worker = self._make_connector()
                metadata = AscendConnectorMetadata(set(), set())
                metadata.requests = [SimpleNamespace(req_id="test", load_spec=None)]
                scheduler_output = SimpleNamespace(
                    kv_connector_metadata=metadata,
                    has_sync_kv_loads=has_sync_loads,
                    finished_req_ids=set(),
                )
                with (
                    patch.object(worker, "start_load_kv", side_effect=AssertionError("layerwise setup is not a load")),
                    patch.object(worker, "process_layer_data", wraps=worker.process_layer_data) as prepare,
                    patch.object(runner_mixin, "get_kv_transfer_group", return_value=connector),
                    patch.object(runner_mixin, "get_forward_context", return_value=SimpleNamespace(attn_metadata={})),
                ):
                    # A second step catches stale completion events too.
                    for step in range(2):
                        with runner_mixin.KVConnectorModelRunnerMixin._get_kv_connector_output(
                            scheduler_output, defer_finalize=True
                        ):
                            self.assertEqual(worker.current_layer, 0)
                            self.assertEqual(prepare.call_count, step + 1)
                            for layer_id in range(2):
                                connector.save_kv_layer(f"model.layers.{layer_id}", None, None)
                            self.assertEqual(worker.current_layer, 2)

                        # The deferred start_load_kv call must not rewind
                        # the counter or replace the target's pending tasks.
                        self.assertEqual(worker.current_layer, 2)
                        self.assertEqual(prepare.call_count, step + 1)
                        connector.save_kv_layer("model.layers.2", None, None)
                        self.assertEqual(worker.current_layer, 3)
                        self.assertTrue(all(not event.is_set() for event in worker.layer_save_finished_events))
                        connector.clear_connector_metadata()


if __name__ == "__main__":
    unittest.main()
