import os
import queue
import socket
import sys
import threading
import time
import types
import unittest
from collections import defaultdict, deque
from unittest.mock import MagicMock, patch

import msgspec
import zmq
from vllm.utils import make_zmq_path

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine

from vllm_ascend.distributed.mooncake_connector import (  # noqa: E402
    KVCacheRecvingThread, KVCacheSendingThread, KVCacheTaskTracker,
    KVConnectorRole, MooncakeAgentMetadata, MooncakeConnector,
    MooncakeConnectorMetadata, MooncakeConnectorScheduler,
    MooncakeConnectorWorker, ReqMeta, ensure_zmq_recv, ensure_zmq_send,
    group_concurrent_contiguous, string_to_int64_hash, zmq_ctx)

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"


class TestKVCacheTaskTrackerInit(unittest.TestCase):

    def test_init_basic_properties(self):
        tracker = KVCacheTaskTracker()
        self.assertIsInstance(tracker.done_task_lock, type(threading.Lock()))
        self.assertIsInstance(tracker.finished_requests, set)
        self.assertIsInstance(tracker.delayed_free_requests, deque)


class TestGetAndClearFinishedSingleRequests(unittest.TestCase):

    def setUp(self):
        self.tracker = KVCacheTaskTracker()
        self.tracker.finished_requests = set()
        self.tracker.done_task_lock = threading.Lock()

    def test_empty_requests(self):
        result = self.tracker.get_and_clear_finished_requests()
        self.assertEqual(result, set())
        self.assertEqual(len(self.tracker.finished_requests), 0)

    def test_single_request(self):
        self.tracker.finished_requests = {"req_123"}
        result = self.tracker.get_and_clear_finished_requests()
        self.assertEqual(result, {"req_123"})
        self.assertEqual(len(self.tracker.finished_requests), 0)

    def test_multiple_requests(self):
        self.tracker.finished_requests = {"req_1", "req_2", "req_3"}
        result = self.tracker.get_and_clear_finished_requests()
        self.assertSetEqual(result, {"req_1", "req_2", "req_3"})
        self.assertEqual(len(self.tracker.finished_requests), 0)

    @patch("vllm_ascend.distributed.mooncake_connector.logger")
    def test_concurrent_access(self, mock_logger):
        from concurrent.futures import ThreadPoolExecutor
        self.tracker.finished_requests = {"req_1", "req_2"}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(self.tracker.get_and_clear_finished_requests)
                for _ in range(3)
            ]
            results = [f.result() for f in futures]
        self.assertEqual(sum(1 for r in results if r), 1)
        self.assertEqual(len(self.tracker.finished_requests), 0)


class TestKVCacheSendingThreadInit(unittest.TestCase):

    def setUp(self):
        self.common_args = {
            'tp_rank': 1,
            'decode_tp_size': 4,
            'local_engine_id': 'engine_1',
            'side_channel_host': 'localhost',
            'side_channel_port': 5555,
            'metadata': MagicMock(),
            'ready_event': threading.Event()
        }
        self.threads = []

    def tearDown(self):
        for thread in self.threads:
            if hasattr(thread, 'task_tracker') and hasattr(
                    thread.task_tracker, 'socket'):
                thread.task_tracker.socket.close()
            if hasattr(thread, 'is_alive') and thread.is_alive():
                thread.join(timeout=0.1)

    def test_thread_daemon_property(self):
        thread = KVCacheSendingThread(**self.common_args)
        self.threads.append(thread)
        self.assertTrue(thread.daemon)

    def test_thread_name_format(self):
        thread = KVCacheSendingThread(**self.common_args)
        self.threads.append(thread)
        self.assertEqual(thread.name, "KVCacheSendingThread")

    def test_ready_event_reference(self):
        custom_event = threading.Event()
        args = self.common_args.copy()
        args['ready_event'] = custom_event
        thread = KVCacheSendingThread(**args)
        self.threads.append(thread)
        self.assertIs(thread.ready_event, custom_event)


class TestGetAndClearFinishedRequests(unittest.TestCase):

    def setUp(self):
        self.common_args = {
            'tp_rank': 1,
            'decode_tp_size': 4,
            'local_engine_id': 'engine_1',
            'side_channel_host': 'localhost',
            'side_channel_port': 5555,
            'metadata': {
                "test": "metadata"
            },
            'ready_event': threading.Event()
        }
        self.thread = KVCacheSendingThread(**self.common_args)

    @patch.object(KVCacheTaskTracker, 'get_and_clear_finished_requests')
    def test_get_and_clear_finished_requests(self, mock_get_clear):
        expected_requests = {'req1', 'req2'}
        mock_get_clear.return_value = expected_requests
        result = self.thread.get_and_clear_finished_requests()
        mock_get_clear.assert_called_once()
        self.assertEqual(result, expected_requests)


class TestKVCacheSendingThread(unittest.TestCase):

    def test_run_handles_get_meta_and_done_recv_msgs(self):
        ready_event = threading.Event()
        metadata = MooncakeAgentMetadata(
            engine_id="engine1",
            te_rpc_port=9090,
            kv_caches_base_addr=[12345678],
            num_blocks=2,
        )
        host = "127.0.0.1"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            free_port = s.getsockname()[1]

        thread = KVCacheSendingThread(
            tp_rank=0,
            decode_tp_size=1,
            local_engine_id="engine1",
            side_channel_host=host,
            side_channel_port=free_port,
            metadata=metadata,
            ready_event=ready_event,
        )
        thread.start()
        self.assertTrue(ready_event.wait(timeout=3),
                        "Server thread startup timeout")

        context = zmq.Context()  # type: ignore
        sock = context.socket(zmq.DEALER)  # type: ignore
        sock.connect(f"tcp://{host}:{free_port}")
        encoder = msgspec.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=MooncakeAgentMetadata)

        sock.send_multipart([b"", encoder.encode((GET_META_MSG, ))])
        frames = sock.recv_multipart()
        self.assertEqual(frames[0], b"")
        meta = decoder.decode(frames[1])
        self.assertEqual(meta.engine_id, "engine1")
        self.assertEqual(meta.kv_caches_base_addr, [12345678])
        self.assertEqual(meta.num_blocks, 2)

        req_id = "request_42"
        sock.send_multipart(
            [b"", encoder.encode((DONE_RECVING_MSG, req_id, 0))])
        frames = sock.recv_multipart()
        self.assertEqual(frames[0], b"")
        self.assertEqual(frames[1], b"ACK")
        self.assertIn(req_id, thread.task_tracker.finished_requests)

        sock.close()
        context.term()


class TestKVCacheRecvingThreadBasic(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event)

    def test_add_request(self):
        test_req = {
            "request_id": "req1",
            "local_block_ids": [1, 2],
            "remote_block_ids": [3, 4],
            "remote_engine_id": "remote_engine",
            "remote_host": "localhost",
            "remote_handshake_port": 6666,
        }
        self.thread.add_request(**test_req)
        queued = self.thread.request_queue.get_nowait()
        self.assertEqual(queued["request_id"], "req1")
        self.assertEqual(queued["remote_host"], "localhost")

    @patch.object(KVCacheTaskTracker, 'get_and_clear_finished_requests')
    def test_get_finished_requests(self, mock_tracker):
        mock_tracker.return_value = {"req1", "req2"}
        result = self.thread.get_and_clear_finished_requests()
        self.assertEqual(result, {"req1", "req2"})


class TestSocketManagement(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event)
        self.thread.remote_sockets = defaultdict(deque)
        self.thread.remote_poller = MagicMock()

    @patch('vllm_ascend.distributed.mooncake_connector.zmq.Context')
    @patch('vllm_ascend.distributed.mooncake_connector.make_zmq_socket')
    def test_get_remote_socket(self, mock_make_socket, mock_context):
        mock_sock = MagicMock()
        mock_make_socket.return_value = mock_sock
        test_host = "test_host"
        test_port = 12345

        sock = self.thread._get_remote_socket(test_host, test_port)

        self.assertEqual(sock, mock_sock)
        mock_make_socket.assert_called_once()
        args, kwargs = mock_make_socket.call_args
        self.assertEqual(kwargs.get('path'), 'tcp://test_host:12345')
        self.assertEqual(kwargs.get('socket_type'), zmq.REQ)  # type: ignore
        self.assertFalse(kwargs.get('bind', True))
        self.thread.remote_poller.register.assert_called_with(
            mock_sock, zmq.POLLIN)  # type: ignore

    def test_return_socket_to_pool(self):
        mock_sock = MagicMock()
        test_host = "test_host"
        test_port = 12345
        test_path = make_zmq_path("tcp", test_host, test_port)

        self.thread._return_remote_socket(mock_sock, test_host, test_port)

        self.assertEqual(len(self.thread.remote_sockets[test_path]), 1)
        self.assertEqual(self.thread.remote_sockets[test_path][0], mock_sock)
        self.thread.remote_poller.register.assert_not_called()


class TestCoreFunctionality(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.mock_queue = MagicMock()
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event)
        self.thread.request_queue = self.mock_queue
        self.test_req = {
            "request_id": "req1",
            "local_block_ids": [1, 2],
            "remote_block_ids": [3, 4],
            "remote_engine_id": "remote_engine",
            "remote_host": "localhost",
            "remote_handshake_port": 6666,
            "remote_transfer_port": 7777
        }
        self.thread.task_tracker = MagicMock()
        self.engine.batch_transfer_sync_read.return_value = 0
        self.thread.remote_te_port = {"remote_engine": {6666: 7777}}

    @patch.object(KVCacheRecvingThread, '_transfer_kv_cache')
    @patch.object(KVCacheRecvingThread, '_send_done_recv_signal')
    def test_handle_request(self, mock_send, mock_transfer):
        self.thread._handle_request(self.test_req)
        mock_transfer.assert_called_once_with(self.test_req)
        mock_send.assert_called_once_with("req1", "localhost", 6666)
        self.thread.task_tracker.update_done_task_count.assert_called_once_with(
            "req1")
        self.mock_queue.task_done.assert_called_once()

    @patch.object(KVCacheRecvingThread, '_get_remote_metadata')
    def test_transfer_kv_cache(self, mock_get_meta):
        self.thread.kv_caches_base_addr["remote_engine"] = {
            6666: [0x3000, 0x4000]
        }

        self.thread._transfer_kv_cache(self.test_req)

        self.engine.batch_transfer_sync_read.assert_called_once()
        call_args, call_kwargs = self.engine.batch_transfer_sync_read.call_args
        self.assertEqual(call_args[0], "localhost:7777")
        self.assertIsInstance(call_args[1], list)
        self.assertIsInstance(call_args[2], list)
        self.assertIsInstance(call_args[3], list)
        self.assertEqual(len(call_args[1]), len(call_args[2]))
        self.assertEqual(len(call_args[1]), len(call_args[3]))
        mock_get_meta.assert_not_called()

    def test_transfer_kv_cache_failure(self):
        self.engine.batch_transfer_sync_read.return_value = -1
        self.thread.kv_caches_base_addr["remote_engine"] = {
            6666: [0x3000, 0x4000]
        }

        with self.assertRaises(RuntimeError):
            self.thread._transfer_kv_cache(self.test_req)


class TestMetadataHandling(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event)
        self.test_metadata = MooncakeAgentMetadata(
            engine_id="remote_engine",
            te_rpc_port=9090,
            kv_caches_base_addr=[0x3000, 0x4000],
            num_blocks=2)

    @patch('vllm_ascend.distributed.mooncake_connector.ensure_zmq_send')
    @patch('vllm_ascend.distributed.mooncake_connector.ensure_zmq_recv')
    def test_get_remote_metadata_success(self, mock_recv, mock_send):
        mock_recv.return_value = msgspec.msgpack.encode(self.test_metadata)

        with patch.object(self.thread, '_get_remote_socket') as mock_get_socket, \
                patch.object(self.thread, '_return_remote_socket') as mock_return_socket:
            mock_socket = MagicMock()
            mock_get_socket.return_value = mock_socket

            self.thread._get_remote_metadata("host1", 5555)

            mock_get_socket.assert_called_once_with("host1", 5555)
            mock_return_socket.assert_called_once_with(mock_socket, "host1",
                                                       5555)
            mock_send.assert_called_once_with(
                mock_socket, self.thread.encoder.encode((GET_META_MSG, "")))
            mock_recv.assert_called_once_with(mock_socket,
                                              self.thread.remote_poller)
            self.assertEqual(
                self.thread.kv_caches_base_addr["remote_engine"][5555],
                [0x3000, 0x4000])

    @patch('vllm_ascend.distributed.mooncake_connector.ensure_zmq_send')
    @patch('vllm_ascend.distributed.mooncake_connector.ensure_zmq_recv',
           side_effect=Exception("Network error"))
    def test_get_remote_metadata_failure(self, mock_recv, mock_send):
        with patch.object(self.thread, '_get_remote_socket') as mock_get_socket, \
                patch.object(self.thread, '_return_remote_socket') as mock_return_socket:
            mock_socket = MagicMock()
            mock_get_socket.return_value = mock_socket

            with self.assertRaises(Exception) as context:
                self.thread._get_remote_metadata("host1", 5555)

            self.assertEqual(str(context.exception), "Network error")
            mock_return_socket.assert_called_once()


class TestMainThreadLoop(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.ready_event = threading.Event()
        self.thread = KVCacheRecvingThread(
            tp_rank=0,
            tp_size=4,
            engine=self.engine,
            local_engine_id="local_engine",
            local_handshake_port=5555,
            local_kv_caches_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            ready_event=self.ready_event)
        self.thread.request_queue = queue.Queue()

    @patch.object(KVCacheRecvingThread, '_handle_request')
    def test_run_loop_normal(self, mock_handle):
        test_request = {
            "request_id": "req1",
            "local_block_ids": [1, 2],
            "remote_block_ids": [3, 4],
            "remote_engine_id": "remote_engine",
            "remote_host": "localhost",
            "remote_handshake_port": 6666,
            "remote_transfer_port": 7777
        }

        self.thread.request_queue.put(test_request)
        self.thread.request_queue.put(None)

        self.thread.start()
        time.sleep(0.1)
        self.thread.join(timeout=1.0)

        self.assertTrue(self.thread.ready_event.is_set())
        mock_handle.assert_called_once_with(test_request)
        self.assertTrue(self.thread.request_queue.empty())


class MockVllmConfig:

    def __init__(self):
        self.model_config = MagicMock()
        self.parallel_config = MagicMock()
        self.cache_config = MagicMock()
        self.kv_transfer_config = MagicMock()
        self.model_config.use_mla = True
        self.parallel_config.tensor_parallel_size = 2
        self.parallel_config.data_parallel_rank_local = 0
        self.parallel_config.data_parallel_size_local = 1
        self.cache_config.block_size = 16
        self.kv_transfer_config.kv_port = 5000
        self.kv_transfer_config.kv_role = 'kv_producer'
        self.kv_transfer_config.get_from_extra_config = MagicMock()
        self.kv_transfer_config.get_from_extra_config.side_effect = lambda k, d: {
            "prefill": {
                "tp_size": 2,
                "dp_size": 1
            },
            "decode": {
                "tp_size": 2,
                "dp_size": 1
            }
        }.get(k, d)


class MockRequest:

    def __init__(self,
                 request_id,
                 prompt_token_ids=None,
                 kv_transfer_params=None,
                 status=None):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids or [1, 2, 3, 4]
        self.kv_transfer_params = kv_transfer_params or {}
        self.status = status or "running"
        self.output_token_ids = [101, 102]


class TestKVCacheTaskTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = KVCacheTaskTracker()

    def test_update_done_task_count(self):
        self.assertEqual(len(self.tracker.finished_requests), 0)
        self.assertEqual(len(self.tracker.delayed_free_requests), 0)

        current_time = time.time()
        self.tracker.add_delayed_request("req_1", current_time)
        result = self.tracker.delayed_free_requests
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("req_1", current_time))

        self.tracker.update_done_task_count("req_1")
        result_finished = self.tracker.finished_requests
        result_delayed = self.tracker.delayed_free_requests
        self.assertEqual(result_finished, {"req_1"})
        self.assertEqual(len(result_delayed), 0)

    def test_retrieve_expired_requests(self):
        current_time = time.time()
        self.tracker.add_delayed_request("req_1", current_time - 600)
        self.tracker.add_delayed_request("req_2", current_time)
        result = self.tracker._retrieve_expired_requests()
        self.assertEqual(result, {
            "req_1",
        })
        result_delay = self.tracker.delayed_free_requests
        self.assertEqual(len(result_delay), 1)
        self.assertEqual(result_delay[0], ("req_2", current_time))

    def test_duplicate_task_update(self):
        self.tracker.update_done_task_count("req1")
        self.tracker.update_done_task_count("req1")
        self.tracker.update_done_task_count("req1")

        finished = self.tracker.get_and_clear_finished_requests()
        self.assertEqual(finished, {"req1"})


class TestMooncakeConnectorMetadata(unittest.TestCase):

    def test_add_new_req(self):
        meta = MooncakeConnectorMetadata()
        self.assertEqual(len(meta.requests), 0)
        self.assertEqual(len(meta.requests_to_send), 0)

        meta.add_new_req(request_id="req1",
                         local_block_ids=[1, 2, 3],
                         kv_transfer_params={
                             "remote_block_ids": [4, 5, 6],
                             "remote_engine_id": "remote_engine",
                             "remote_host": "localhost",
                             "remote_port": 5000
                         })

        self.assertEqual(len(meta.requests), 1)
        req_meta = meta.requests["req1"]
        self.assertIsInstance(req_meta, ReqMeta)
        self.assertEqual(req_meta.local_block_ids, [1, 2, 3])
        self.assertEqual(req_meta.remote_block_ids, [4, 5, 6])
        self.assertEqual(req_meta.remote_engine_id, "remote_engine")
        self.assertEqual(req_meta.remote_host, "localhost")
        self.assertEqual(req_meta.remote_port, 5000)


class TestMooncakeConnectorSchedulerMatchedTokens(unittest.TestCase):

    def setUp(self):
        config = MockVllmConfig()
        self.scheduler = MooncakeConnectorScheduler(config, "test_engine")

    def test_get_num_new_matched_tokens(self):
        request = MockRequest("req1")
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(
            request, 0)
        self.assertEqual(tokens, 0)
        self.assertFalse(async_flag)

        request.kv_transfer_params = {"do_remote_prefill": True}
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(
            request, 0)
        self.assertEqual(tokens, 3)
        self.assertTrue(async_flag)

    def test_build_connector_meta(self):
        request = MockRequest("req1")
        blocks_mock = MagicMock()
        blocks_mock.get_unhashed_block_ids.return_value = [4, 5, 6]
        self.scheduler._reqs_need_recv["req1"] = (request, [4, 5, 6])
        request.kv_transfer_params = {
            "remote_block_ids": [1, 2, 3],
            "remote_engine_id": "remote",
            "remote_host": "localhost",
            "remote_port": 5000
        }

        meta = self.scheduler.build_connector_meta(MagicMock())
        self.assertIsInstance(meta, MooncakeConnectorMetadata)
        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests["req1"].local_block_ids, [4, 5, 6])
        self.assertEqual(meta.requests["req1"].remote_block_ids, [1, 2, 3])
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)

    def test_get_finished_count(self):
        count = self.scheduler.get_finished_count()
        self.assertEqual(count, 2)


class TestHelperFunctions(unittest.TestCase):

    def test_group_concurrent_contiguous(self):
        src: list[int] = [1, 2, 3, 5, 6]
        dst: list[int] = [10, 11, 12, 14, 15]

        src_groups, dst_groups = group_concurrent_contiguous(src, dst)

        self.assertEqual(len(src_groups), 2)
        self.assertEqual(src_groups[0], [1, 2, 3])
        self.assertEqual(src_groups[1], [5, 6])
        self.assertEqual(dst_groups[0], [10, 11, 12])
        self.assertEqual(dst_groups[1], [14, 15])

    def test_group_concurrent_contiguous_empty(self):
        src: list[int] = []
        dst: list[int] = []
        src_groups, dst_groups = group_concurrent_contiguous(src, dst)
        self.assertEqual(src_groups, [])
        self.assertEqual(dst_groups, [])

    def test_string_to_int64_hash(self):
        hash1 = string_to_int64_hash("test_string")
        hash2 = string_to_int64_hash("test_string")
        self.assertEqual(hash1, hash2)

        hash3 = string_to_int64_hash("different_string")
        self.assertNotEqual(hash1, hash3)


class TestMooncakeConnectorForScheduler(unittest.TestCase):

    def test_scheduler_role(self):
        config = MockVllmConfig()
        connector = MooncakeConnector(config, KVConnectorRole.SCHEDULER)
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)

    @patch.object(MooncakeConnectorScheduler, "get_num_new_matched_tokens")
    def test_scheduler_methods(self, mock_method):
        config = MockVllmConfig()
        connector = MooncakeConnector(config, KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        connector.get_num_new_matched_tokens(request, 0)
        mock_method.assert_called_once_with(request, 0)


class MockKVCacheBlocks:

    def get_unhashed_block_ids(self):
        return [4, 5, 6]


class MockSchedulerOutput:
    pass


class MockForwardContext:
    pass


class TestMooncakeConnector(unittest.TestCase):

    def setUp(self):
        self.config = MockVllmConfig()
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

    def test_scheduler_initialization(self):
        connector = MooncakeConnector(self.config, KVConnectorRole.SCHEDULER)
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)

    @patch.object(MooncakeConnectorScheduler, "get_num_new_matched_tokens")
    def test_get_num_new_matched_tokens(self, mock_method):
        connector = MooncakeConnector(self.config, KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        connector.get_num_new_matched_tokens(request, 0)
        mock_method.assert_called_once_with(request, 0)

    @patch.object(MooncakeConnectorScheduler, "update_state_after_alloc")
    def test_update_state_after_alloc(self, mock_method):
        connector = MooncakeConnector(self.config, KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        blocks = MockKVCacheBlocks()
        connector.update_state_after_alloc(request, blocks, 3)
        mock_method.assert_called_once_with(request, blocks, 3)

    @patch.object(MooncakeConnectorScheduler, "build_connector_meta")
    def test_build_connector_meta(self, mock_method):
        connector = MooncakeConnector(self.config, KVConnectorRole.SCHEDULER)
        scheduler_output = MockSchedulerOutput()
        connector.build_connector_meta(scheduler_output)
        mock_method.assert_called_once_with(scheduler_output)

    @patch.object(MooncakeConnectorScheduler, "request_finished")
    def test_request_finished(self, mock_method):
        connector = MooncakeConnector(self.config, KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        connector.request_finished(request, [1, 2, 3])
        mock_method.assert_called_once_with(request, [1, 2, 3])


class TestMooncakeConnectorScheduler(unittest.TestCase):

    def setUp(self):
        self.config = MockVllmConfig()
        self.scheduler = MooncakeConnectorScheduler(self.config, "test_engine")

    def test_get_num_new_matched_tokens_no_remote_prefill(self):
        request = MockRequest("req1")
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(
            request, 0)
        self.assertEqual(tokens, 0)
        self.assertFalse(async_flag)

    def test_get_num_new_matched_tokens_with_remote_prefill(self):
        request = MockRequest("req1",
                              kv_transfer_params={"do_remote_prefill": True})
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(
            request, 0)
        self.assertEqual(tokens, 3)
        self.assertTrue(async_flag)

    def test_update_state_after_alloc_no_remote_prefill(self):
        request = MockRequest("req1")
        blocks = MagicMock()
        self.scheduler.update_state_after_alloc(request, blocks, 0)
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)

    def test_update_state_after_alloc_with_remote_prefill(self):
        request = MockRequest("req1",
                              kv_transfer_params={
                                  "do_remote_prefill": True,
                                  "remote_block_ids": [1, 2, 3],
                                  "remote_engine_id": "remote",
                                  "remote_host": "localhost",
                                  "remote_port": 5000
                              })
        blocks = MockKVCacheBlocks()
        self.scheduler.update_state_after_alloc(request, blocks, 3)
        self.assertEqual(len(self.scheduler._reqs_need_recv), 1)
        self.assertEqual(self.scheduler._reqs_need_recv["req1"][0], request)
        self.assertEqual(self.scheduler._reqs_need_recv["req1"][1], [4, 5, 6])

    def test_request_finished_no_remote_decode(self):
        request = MockRequest("req1")
        delay_free, params = self.scheduler.request_finished(
            request, [1, 2, 3])
        self.assertFalse(delay_free)
        self.assertIsNone(params)


class TestUtils(unittest.TestCase):

    def test_string_to_int64_hash(self):
        h1 = string_to_int64_hash("hello")
        h2 = string_to_int64_hash("hello")
        h3 = string_to_int64_hash("world")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertIsInstance(h1, int)

    def test_group_concurrent_contiguous(self):
        src: list[int] = [1, 2, 3, 5, 6]
        dst: list[int] = [10, 11, 12, 20, 21]
        src_g, dst_g = group_concurrent_contiguous(src, dst)
        self.assertEqual(src_g, [[1, 2, 3], [5, 6]])
        self.assertEqual(dst_g, [[10, 11, 12], [20, 21]])

    def test_group_empty(self):
        src_g, dst_g = group_concurrent_contiguous([], [])
        self.assertEqual(src_g, [])
        self.assertEqual(dst_g, [])

    def test_zmq_ctx_invalid_type(self):
        with self.assertRaises(ValueError):
            with zmq_ctx("INVALID", "tcp://127.0.0.1:5555"):
                pass

    @patch("vllm_ascend.distributed.mooncake_connector.make_zmq_socket")
    def test_zmq_ctx_ok(self, mock_make_socket):
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        with zmq_ctx(zmq.REQ, "tcp://localhost:1234") as s:  # type: ignore
            self.assertEqual(s, mock_socket)

    @patch("vllm_ascend.distributed.mooncake_connector.logger")
    def test_ensure_zmq_send_success(self, mock_logger):
        mock_socket = MagicMock()
        ensure_zmq_send(mock_socket, b"hello")
        mock_socket.send.assert_called_once_with(b"hello")

    @patch("vllm_ascend.distributed.mooncake_connector.logger")
    def test_ensure_zmq_send_retry_and_fail(self, mock_logger):
        mock_socket = MagicMock()
        mock_socket.send.side_effect = zmq.ZMQError(  # type: ignore
            "send failed")
        with self.assertRaises(RuntimeError):
            ensure_zmq_send(mock_socket, b"hello", max_retries=2)
        self.assertEqual(mock_socket.send.call_count, 2)

    @patch("vllm_ascend.distributed.mooncake_connector.logger")
    def test_ensure_zmq_recv_success(self, mock_logger):
        mock_socket = MagicMock()
        mock_socket.recv.return_value = b"response"
        mock_poller = MagicMock()
        mock_poller.poll.return_value = [
            (mock_socket, zmq.POLLIN)  # type: ignore
        ]
        data = ensure_zmq_recv(mock_socket, mock_poller)
        self.assertEqual(data, b"response")

    @patch("vllm_ascend.distributed.mooncake_connector.logger")
    def test_ensure_zmq_recv_timeout_and_fail(self, mock_logger):
        mock_socket = MagicMock()
        mock_poller = MagicMock()
        mock_poller.poll.return_value = []
        with self.assertRaises(RuntimeError):
            ensure_zmq_recv(mock_socket,
                            mock_poller,
                            timeout=0.01,
                            max_retries=2)


class MockMooncakeAgentMetadata:

    def __init__(self, **kwargs):
        pass


class MockMooncakeConnectorMetadata:

    def __init__(self):
        self.requests = {}


class MockKVCacheSendingThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.daemon = True
        self._finished_requests = set()

    def get_and_clear_finished_requests(self):
        return self._finished_requests

    def start(self):
        pass


class MockKVCacheRecvingThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.daemon = True
        self._finished_requests = set()
        self.add_request = MagicMock()

    def get_and_clear_finished_requests(self):
        return self._finished_requests

    def start(self):
        pass


class MockTensor:

    def __init__(self, *args, **kwargs):
        self.size = MagicMock(return_value=(10, 16, 8, 16))
        self.element_size = MagicMock(return_value=4)
        self.shape = (10, 16, 8, 16)
        self.data_ptr = MagicMock(return_value=0x1000)


mock_envs_ascend = MagicMock()
mock_envs_ascend.MOONCAKE_CONNECTOR_PROTOCOL = "mock_protocol"

mock_logger = MagicMock()


class MockTransferEngine:

    def initialize(self, *args, **kwargs):
        return 0

    def register_memory(self, *args, **kwargs):
        return 1


class MockEnvsAscend:
    MOONCAKE_CONNECTOR_PROTOCOL = "mock_protocol"
    PHYSICAL_DEVICES = "10,11"


def mock_get_tensor_model_parallel_rank():
    return 0


def mock_get_tp_group():
    return MagicMock()


def mock_get_ip():
    return "127.0.0.1"


def mock_string_to_int64_hash(s):
    return hash(s)


class TestMooncakeConnectorWorker(unittest.TestCase):

    def setUp(self):
        self.envs_ascend_mock = MockEnvsAscend()
        self.mock_transfer_engine = MagicMock()
        self.mock_transfer_engine.get_rpc_port.return_value = 9090
        self.mock_transfer_engine.initialize.return_value = 0
        self.mock_transfer_engine.register_memory.return_value = 0

        self.patches = [
            patch('os.getenv', return_value="10,11"),
            patch('torch.Tensor.size', return_value=(10, 16, 8, 16)),
            patch('torch.Tensor.element_size', return_value=4),
            patch('torch.Tensor.data_ptr', return_value=0x1000),
            patch('math.prod', return_value=128),
            patch('random.Random'),
            patch(
                'vllm_ascend.distributed.mooncake_connector.get_tensor_model_parallel_rank',
                mock_get_tensor_model_parallel_rank),
            patch('vllm_ascend.distributed.mooncake_connector.get_tp_group',
                  mock_get_tp_group),
            patch('vllm_ascend.distributed.mooncake_connector.get_ip',
                  mock_get_ip),
            patch(
                'vllm_ascend.distributed.mooncake_connector.string_to_int64_hash',
                mock_string_to_int64_hash),
            patch('vllm_ascend.distributed.mooncake_connector.TransferEngine',
                  return_value=self.mock_transfer_engine),
            patch(
                'vllm_ascend.distributed.mooncake_connector.KVCacheSendingThread',
                MagicMock()),
            patch(
                'vllm_ascend.distributed.mooncake_connector.KVCacheRecvingThread',
                MagicMock()),
            patch('vllm_ascend.distributed.mooncake_connector.logger',
                  MagicMock()),
            patch('vllm_ascend.distributed.mooncake_connector.threading.Event',
                  MagicMock()),
            patch.dict('sys.modules',
                       {'vllm_ascend.envs': self.envs_ascend_mock}),
        ]

        for p in self.patches:
            p.start()  # type: ignore

        self.vllm_config = MockVllmConfig()
        self.engine_id = "test_engine"
        self.kv_caches = {"layer1": (MagicMock(), MagicMock())}

    def tearDown(self):
        for p in self.patches:
            p.stop()  # type: ignore

    def test_worker_use_ascend_direct(self):
        test_case = [True, False]

        for use_ascend_direct in test_case:
            with self.subTest(use_ascend_direct=use_ascend_direct):
                config = MagicMock()
                config.kv_transfer_config = MagicMock()
                config.kv_transfer_config.get_from_extra_config.side_effect = (
                    lambda k, d: {
                        "prefill": {
                            "tp_size": 2,
                            "dp_size": 1
                        },
                        "decode": {
                            "tp_size": 2,
                            "dp_size": 1
                        },
                        "use_ascend_direct": use_ascend_direct,
                    }.get(k, d))

                config.parallel_config = MagicMock()
                config.parallel_config.tensor_parallel_size = 2
                config.parallel_config.data_parallel_rank_local = 0
                config.parallel_config.data_parallel_size_local = 1
                config.kv_transfer_config.kv_port = 8000
                config.kv_transfer_config.kv_role = 'worker'

                with patch(
                        "vllm_ascend.distributed.mooncake_connector.get_tensor_model_parallel_rank",
                        return_value=0):
                    with patch(
                            "vllm_ascend.distributed.mooncake_connector.get_tp_group",
                            return_value=None):
                        with patch(
                                "vllm_ascend.distributed.mooncake_connector.get_ip",
                                return_value="127.0.0.1"):
                            worker = MooncakeConnectorWorker(
                                config, self.engine_id)
                            self.assertIsNotNone(worker)

    def test_register_kv_caches_producer(self):
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertEqual(len(worker.kv_caches), 1)
        self.assertIsNotNone(worker.kv_send_thread)
        self.assertIsNone(worker.kv_recv_thread)

    def test_register_kv_caches_consumer(self):
        self.vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertIsNone(worker.kv_send_thread)
        self.assertIsNotNone(worker.kv_recv_thread)

    def test_register_kv_caches_mla_case(self):
        mla_cache1 = MagicMock()
        mla_cache1.size.return_value = (10, 16, 1, 16)
        mla_cache2 = MagicMock()
        mla_cache2.size.return_value = (10, 16, 1, 8)
        mla_caches = {"layer1": (mla_cache1, mla_cache2)}

        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        worker.register_kv_caches(mla_caches)
        self.assertTrue(worker.use_mla)
        self.assertEqual(len(worker.block_len), 2)

    def test_device_id_selection_with_physical_devices(self):
        # Test with physical devices set
        worker = MooncakeConnectorWorker(self.vllm_config, self.engine_id)
        # Default tp_rank is 0, so device_id should be 10
        self.assertEqual(worker.device_id, 10)


if __name__ == '__main__':
    unittest.main()
