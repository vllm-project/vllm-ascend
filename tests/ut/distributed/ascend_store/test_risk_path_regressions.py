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
"""Failure-path regressions for the AscendStore KV pool.

These cases cover paths that can otherwise hang a transfer thread, report a
false lookup hit, or release attention after a failed layer load.
"""

import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from zmq.error import ZMQError

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    LookupKeyServer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVTransferThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LayerBatchReqMeta,
    LayerBlockRange,
    LayerLoadTask,
    LayerTransferTask,
    LoadSpec,
    ReqMeta,
    SharedBlockData,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    LookupKeyClient,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)

# isort: on

CONNECTOR_MODULE = sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector"]
KV_TRANSFER_MODULE = sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer"]
POOL_SCHEDULER_MODULE = sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler"]


def _make_transfer_thread() -> KVTransferThread:
    store = SimpleNamespace(
        set_device=MagicMock(),
        exists=MagicMock(return_value=[]),
        requires_exists_before_put=False,
    )
    token_database = SimpleNamespace(group_block_len={0: [1]})
    return KVTransferThread(
        m_store=store,
        token_database=token_database,
        block_size=16,
        tp_rank=0,
        ready_event=threading.Event(),
    )


class TestTransferThreadFailureContainment(unittest.TestCase):
    def test_ready_event_precedes_dispatch_and_dispatch_error_is_fatal(self):
        thread = _make_transfer_thread()
        observations = []

        def fail(request):
            observations.append((thread.ready_event.is_set(), request))
            raise OSError("store I/O failed")

        thread._handle_request = fail
        thread.add_request("first")

        thread.run()

        self.assertEqual(observations, [(True, "first")])
        self.assertIsInstance(thread._fatal_error, OSError)
        with self.assertRaisesRegex(RuntimeError, "asynchronous transfer"):
            thread.raise_if_failed()

    def test_cleanup_error_does_not_replace_original_dispatch_error(self):
        thread = _make_transfer_thread()
        thread._handle_request = MagicMock(side_effect=RuntimeError("dispatch failed"))
        thread._handle_request_exception = MagicMock(side_effect=RuntimeError("cleanup failed"))
        thread.add_request("first")

        with patch.object(KV_TRANSFER_MODULE, "logger") as logger:
            thread.run()

        self.assertIsInstance(thread._fatal_error, RuntimeError)
        self.assertEqual(str(thread._fatal_error), "dispatch failed")
        thread._handle_request_exception.assert_called_once_with("first")
        logger.exception.assert_called_once()
        logger.error.assert_called_once()


def _make_gva_worker() -> KVPoolWorker:
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.use_layerwise_transfer = True
    worker.kv_role = "kv_producer"
    worker.consumer_is_to_put = False
    worker.can_put = True
    worker._save_disabled = threading.Event()
    worker.layerwise_offload = True
    worker.num_kv_cache_groups = 1
    worker.cacheable_group_ids = [0]
    worker.grouped_block_size = [16]
    worker.hash_block_size = 16
    worker.group_uses_align_state = [False]
    worker.m_store = MagicMock()
    worker._is_layerwise_save_owner = MagicMock(return_value=True)
    worker._global_group_alloc_size = MagicMock(return_value=64)
    worker._pool_keys_missing = MagicMock(side_effect=lambda keys: set(keys))
    worker._make_layerwise_full_key = lambda group_id, block_hash: f"g{group_id}@{block_hash}"
    worker._make_layerwise_partial_key = lambda request, group_id, block_index, token_len: "partial"
    worker._layerwise_save_full_end = MagicMock(return_value=3)
    worker._compute_reachable_load_masks = MagicMock(return_value=None)
    worker._resolve_layerwise_load_partial = MagicMock(return_value=(None, None))
    worker._layerwise_load_snapshot_start = MagicMock(return_value=None)
    worker._record_failed_request = MagicMock()
    worker._failed_requests = set()
    worker._step_load_keys = set()
    return worker


def _make_request(block_hashes: list[str], block_ids: list[int], token_len: int) -> ReqMeta:
    block_ids_array = np.asarray(block_ids, dtype=np.int64)
    return ReqMeta(
        req_id="r1",
        token_len_chunk=token_len,
        save_end_token=token_len,
        target_token_len=token_len,
        block_ids=block_ids,
        block_ids_by_group=[block_ids],
        block_hashes=block_hashes,
        can_save=True,
        block_ids_np=block_ids_array,
        block_ids_by_group_np=[block_ids_array],
    )


class TestGVAPreparationBoundaries(unittest.TestCase):
    def test_save_preparation_handles_empty_block_hashes(self):
        worker = _make_gva_worker()
        request = _make_request([], [], 0)

        worker._alloc_gvas_for_save([request])

        worker.m_store.batch_alloc.assert_not_called()
        self.assertEqual(request.save_keys, [])
        self.assertEqual(request.block_gvas_by_group_np[0].tolist(), [])

    def test_save_preparation_handles_single_block(self):
        worker = _make_gva_worker()
        worker._layerwise_save_full_end.return_value = 1
        worker.m_store.batch_alloc.return_value = [101]
        request = _make_request(["h0"], [7], 16)

        worker._alloc_gvas_for_save([request])

        worker.m_store.batch_alloc.assert_called_once()
        self.assertEqual(request.save_keys, ["g0@h0"])
        self.assertEqual(request.block_gvas_by_group_np[0].tolist(), [101])

    def test_save_preparation_stops_at_full_block_table(self):
        worker = _make_gva_worker()
        worker._layerwise_save_full_end.return_value = 4
        worker.m_store.batch_alloc.return_value = [101, 102, 103]
        request = _make_request(["h0", "h1", "h2", "h3"], [7, 8, 9], 64)

        worker._alloc_gvas_for_save([request])

        allocated_keys = worker.m_store.batch_alloc.call_args.args[0]
        self.assertEqual(allocated_keys, ["g0@h0", "g0@h1", "g0@h2"])
        self.assertEqual(request.block_gvas_by_group_np[0].tolist(), [101, 102, 103])

    def test_partial_save_failure_invalidates_whole_step(self):
        worker = _make_gva_worker()
        worker._layerwise_save_full_end.return_value = 2
        worker.m_store.batch_alloc.return_value = [101, 0]
        first = _make_request(["h0", "h1"], [7, 8], 32)
        second = _make_request(["h2"], [9], 16)
        second.req_id = "r2"
        first.save_keys = ["stale"]
        second.save_keys = ["stale"]

        worker._alloc_gvas_for_save([first, second])

        self.assertTrue(worker._save_disabled.is_set())
        worker.m_store.batch_alloc.assert_called_once()
        self.assertFalse(first.can_save)
        self.assertFalse(second.can_save)
        self.assertEqual(first.save_keys, [])
        self.assertEqual(second.save_keys, [])
        self.assertEqual(worker._record_failed_request.call_count, 2)

    def test_malformed_load_metadata_rejects_request(self):
        worker = _make_gva_worker()
        worker.m_store.batch_get_key_info.return_value = []
        request = _make_request(["h0"], [7], 16)
        request.can_save = None
        request.load_spec = LoadSpec(
            vllm_cached_tokens=16,
            kvpool_cached_tokens=16,
            can_load=True,
        )

        worker._prepare_load_gvas([request])

        self.assertFalse(request.load_spec.can_load)
        worker._record_failed_request.assert_called_once_with(request)

    def test_prepare_load_with_empty_batch_is_noop(self):
        worker = _make_gva_worker()

        worker._prepare_load_gvas([])

        worker.m_store.batch_get_key_info.assert_not_called()
        worker._record_failed_request.assert_not_called()

    def test_partial_load_failure_only_disables_failed_request(self):
        worker = _make_gva_worker()
        worker._invalid_block_ids = set()
        worker._invalid_block_ids_lock = threading.Lock()
        worker._record_failed_request = KVPoolWorker._record_failed_request.__get__(worker, KVPoolWorker)
        valid_info = MagicMock()
        valid_info.size.return_value = 64
        valid_info.gva_list.return_value = [201]
        worker.m_store.batch_get_key_info.side_effect = [[valid_info], [valid_info], []]
        worker.m_store.batch_add_lease.return_value = [0]
        first = _make_request(["h0"], [7], 16)
        first.load_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=16, can_load=True)
        second = _make_request(["h1"], [8], 16)
        second.req_id = "r2"
        second.load_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=16, can_load=True)

        worker._prepare_load_gvas([first, second])

        self.assertTrue(first.can_save)
        self.assertTrue(first.load_spec.can_load)
        self.assertFalse(second.can_save)
        self.assertFalse(second.load_spec.can_load)
        self.assertIn(8, worker._invalid_block_ids)

    def test_multi_group_load_failure_records_request_without_false_blocks(self):
        worker = _make_gva_worker()
        worker._invalid_block_ids = set()
        worker._invalid_block_ids_lock = threading.Lock()
        worker._record_failed_request = KVPoolWorker._record_failed_request.__get__(worker, KVPoolWorker)
        worker.num_kv_cache_groups = 2
        worker.grouped_block_size = [16, 16]
        worker.cacheable_group_ids = [0, 1]
        worker.group_uses_align_state = [False, False]
        valid_info = MagicMock()
        valid_info.size.return_value = 64
        valid_info.gva_list.return_value = [201]
        worker.m_store.batch_get_key_info.side_effect = [[valid_info], [valid_info], []]
        worker.m_store.batch_add_lease.return_value = [0]
        request = ReqMeta(
            req_id="r1",
            token_len_chunk=16,
            block_ids=[7, 8],
            block_ids_by_group=[[7], [8]],
            block_hashes=["h0"],
            can_save=True,
            block_ids_np=np.asarray([7, 8], dtype=np.int64),
            block_ids_by_group_np=[np.asarray([7], dtype=np.int64), np.asarray([8], dtype=np.int64)],
            load_spec=LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=16, can_load=True),
        )

        worker._prepare_load_gvas([request])

        self.assertFalse(request.can_save)
        self.assertFalse(request.load_spec.can_load)
        self.assertEqual(worker._failed_requests, {"r1"})
        self.assertEqual(worker._invalid_block_ids, set())
        self.assertTrue(worker._save_disabled.is_set())


class TestLookupRPCFailures(unittest.TestCase):
    def _make_server(self, frames, decoder_side_effect=None, poll_misses=0):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}

        with (
            patch.object(CONNECTOR_MODULE.threading, "Thread") as thread_cls,
            patch.object(CONNECTOR_MODULE.zmq, "Poller") as poller_cls,
            patch.object(CONNECTOR_MODULE, "make_zmq_socket") as make_socket,
            patch.object(CONNECTOR_MODULE.zmq, "Context") as context_cls,
        ):
            socket = MagicMock()
            socket.recv_multipart.return_value = frames
            make_socket.return_value = socket
            poller = MagicMock()
            poller.poll.side_effect = [[] for _ in range(poll_misses)] + [[(socket, 1)]]
            poller_cls.return_value = poller
            thread_cls.return_value = MagicMock()

            server = LookupKeyServer(MagicMock(), config)
            if decoder_side_effect is not None:
                server.decoder = MagicMock()
                server.decoder.decode.side_effect = decoder_side_effect
            socket.send.side_effect = lambda _: setattr(server, "running", False)
            target = thread_cls.call_args.kwargs["target"]
            target()

        return socket, context_cls.return_value, poller

    def test_server_rejects_truncated_frame_list(self):
        socket, context, _ = self._make_server([b"short"])

        socket.send.assert_called_once_with((0).to_bytes(4, "big"))
        socket.close.assert_called_once_with(linger=0)
        context.term.assert_called_once_with()

    def test_server_decode_failure_returns_zero_hit_count(self):
        frames = [
            (64).to_bytes(4, "big"),
            b"groups",
            (0).to_bytes(4, "big"),
            b"hashes",
        ]
        socket, context, _ = self._make_server(frames, decoder_side_effect=ValueError("bad payload"))

        socket.send.assert_called_once_with((0).to_bytes(4, "big"))
        socket.close.assert_called_once_with(linger=0)
        context.term.assert_called_once_with()

    def test_server_no_ready_request_retries_then_serves(self):
        frames = [
            (64).to_bytes(4, "big"),
            b"groups",
            (0).to_bytes(4, "big"),
            b"hashes",
        ]

        socket, _, poller = self._make_server(frames, poll_misses=2)

        self.assertEqual(poller.poll.call_count, 3)
        socket.send.assert_called_once()

    def test_server_rejects_oversized_hash_payload(self):
        frames = [
            (64).to_bytes(4, "big"),
            b"groups",
            (0).to_bytes(4, "big"),
            b"hashes-0",
            b"hashes-1",
        ]
        socket, context, _ = self._make_server(
            frames,
            decoder_side_effect=[[0], ValueError("payload too large")],
        )

        socket.send.assert_called_once_with((0).to_bytes(4, "big"))
        socket.close.assert_called_once_with(linger=0)
        context.term.assert_called_once_with()

    @patch.object(POOL_SCHEDULER_MODULE, "MsgpackEncoder")
    @patch.object(POOL_SCHEDULER_MODULE.zmq, "Context")
    @patch.object(POOL_SCHEDULER_MODULE, "make_zmq_socket")
    def test_client_send_failure_closes_private_context(self, make_socket, context_cls, encoder_cls):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}
        socket = MagicMock()
        socket.send_multipart.side_effect = ZMQError("disconnected")
        make_socket.return_value = socket
        encoder_cls.return_value.encode.side_effect = [[b"groups"], [b"hashes"]]

        client = LookupKeyClient(config)
        self.assertEqual(client.lookup(64, [b"h0"]), 0)

        self.assertTrue(client.closed)
        socket.close.assert_called_once_with(linger=0)
        context_cls.return_value.term.assert_called_once_with()

    @patch.object(POOL_SCHEDULER_MODULE, "MsgpackEncoder")
    @patch.object(POOL_SCHEDULER_MODULE.zmq, "Context")
    @patch.object(POOL_SCHEDULER_MODULE, "make_zmq_socket")
    def test_client_receive_timeout_closes_private_context(self, make_socket, context_cls, encoder_cls):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}
        socket = MagicMock()
        socket.recv.side_effect = ZMQError("timeout")
        make_socket.return_value = socket
        encoder_cls.return_value.encode.side_effect = [[b"groups"], [b"hashes"]]

        client = LookupKeyClient(config)

        self.assertEqual(client.lookup(64, [b"h0"]), 0)
        self.assertTrue(client.closed)
        socket.close.assert_called_once_with(linger=0)
        context_cls.return_value.term.assert_called_once_with()

    @patch.object(POOL_SCHEDULER_MODULE, "MsgpackEncoder")
    @patch.object(POOL_SCHEDULER_MODULE.zmq, "Context")
    @patch.object(POOL_SCHEDULER_MODULE, "make_zmq_socket")
    def test_client_empty_reply_is_reported_as_miss(self, make_socket, context_cls, encoder_cls):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}
        socket = MagicMock()
        socket.recv.return_value = b""
        make_socket.return_value = socket
        encoder_cls.return_value.encode.side_effect = [[b"groups"], [b"hashes"]]

        client = LookupKeyClient(config)

        self.assertEqual(client.lookup(64, [b"h0"]), 0)
        self.assertFalse(client.closed)
        client.close()

    @patch.object(POOL_SCHEDULER_MODULE, "MsgpackEncoder")
    @patch.object(POOL_SCHEDULER_MODULE.zmq, "Context")
    @patch.object(POOL_SCHEDULER_MODULE, "make_zmq_socket")
    def test_client_lookup_after_close_is_local_miss(self, make_socket, context_cls, encoder_cls):
        config = MagicMock()
        config.parallel_config.data_parallel_rank = 0
        config.kv_transfer_config.kv_connector_extra_config = {}
        socket = MagicMock()
        make_socket.return_value = socket

        client = LookupKeyClient(config)
        client.close()

        self.assertEqual(client.lookup(64, [b"h0"]), 0)
        socket.send_multipart.assert_not_called()
        encoder_cls.return_value.encode.assert_not_called()


class TestLayerwiseAttentionGateFailure(unittest.TestCase):
    def _make_thread(self):
        store = MagicMock()
        token_database = SimpleNamespace(group_block_len={0: [1]})
        load_finished = [threading.Event()]
        save_finished = [threading.Event()]
        builder = MagicMock()
        builder.build_addrs.return_value = LayerBatchReqMeta(
            req_ids=["r1"],
            layer_id=0,
            is_last_chunks=[True],
            addr_array=np.asarray([10]),
            size_array=np.asarray([16]),
            gvas_array=np.asarray([100]),
        )
        thread = KVCacheStoreLayerRecvingThread(
            m_store=store,
            token_database=token_database,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            page_size_bytes=16,
            ready_event=threading.Event(),
            get_event=threading.Event(),
            layer_load_finished_events=load_finished,
            layer_save_finished_events=save_finished,
            sync_save_events=[MagicMock()],
            num_layers=1,
            group_builders=[builder],
        )
        thread._batch_copy_with_limits = MagicMock(return_value=0)
        return thread, load_finished

    def _make_load_task(self, gate):
        request = ReqMeta("r1", block_ids_by_group=[[7]], block_hashes=[b"h0"])
        task = LayerTransferTask(
            layer_id=0,
            block_ranges=[LayerBlockRange(request=request, start_block=0, end_block=1)],
            shared_block_data=SharedBlockData(
                block_ids_arr=np.asarray([7]),
                block_gvas_arr=np.asarray([100]),
                req_ids=["r1"],
                is_last_chunks=[True],
            ),
        )
        return LayerLoadTask(
            wait_for_save_layer=None,
            transfer_tasks=[task],
            layer_id=0,
            attention_start_gate=gate,
        )

    def test_shutdown_while_waiting_for_attention_does_not_copy_or_mark_request_complete(self):
        thread, load_finished = self._make_thread()
        gate = MagicMock()

        def abort_wait(timeout):
            thread._stopping.set()
            return False

        gate.wait.side_effect = abort_wait
        load_task = self._make_load_task(gate)
        thread.request_queue.put(load_task)

        thread._handle_request(load_task)

        self.assertTrue(thread.load_disabled.is_set())
        thread._batch_copy_with_limits.assert_not_called()
        self.assertIn(7, thread._invalid_block_ids)
        self.assertNotIn("r1", thread.finished_requests)
        self.assertTrue(load_finished[0].is_set())
        self.assertTrue(thread.get_event.is_set())
        self.assertEqual(thread.request_queue.unfinished_tasks, 0)

    def test_gate_timeout_retry_then_open_allows_load_to_finish(self):
        thread, load_finished = self._make_thread()
        gate = MagicMock()
        gate.wait.side_effect = [False, True]
        load_task = self._make_load_task(gate)
        thread.request_queue.put(load_task)

        thread._handle_request(load_task)

        self.assertEqual(gate.wait.call_count, 2)
        self.assertFalse(thread.load_disabled.is_set())
        thread._batch_copy_with_limits.assert_called_once()
        self.assertIn("r1", thread.finished_requests)
        self.assertTrue(load_finished[0].is_set())
        self.assertEqual(thread.request_queue.unfinished_tasks, 0)

    def test_load_failure_after_gate_open_is_not_marked_finished(self):
        thread, load_finished = self._make_thread()
        thread._batch_copy_with_limits.return_value = -1
        gate = MagicMock()
        gate.wait.return_value = True
        load_task = self._make_load_task(gate)
        thread.request_queue.put(load_task)

        thread._handle_request(load_task)

        gate.wait.assert_called_once()
        thread._batch_copy_with_limits.assert_called_once()
        self.assertTrue(thread.load_disabled.is_set())
        self.assertTrue(thread.save_disabled.is_set())
        self.assertNotIn("r1", thread.finished_requests)
        self.assertIn(7, thread._invalid_block_ids)
        self.assertTrue(load_finished[0].is_set())
        self.assertEqual(thread.request_queue.unfinished_tasks, 0)


if __name__ == "__main__":
    unittest.main()
