import os
import sys
import threading
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import zmq

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine

from vllm_ascend.distributed.mooncake_layerwise_connector import (  # noqa: E402
    DecodeMooncakeAgentMetadata, KVCacheRecvingLayerThread,
    KVCacheSendingLayerThread, KVCacheTaskTracker, KVConnectorRole,
    MooncakeLayerwiseConnector, MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorScheduler, MooncakeLayerwiseConnectorWorker,
    ReqMeta, SendingLayerThread, ensure_zmq_recv, ensure_zmq_send,
    group_concurrent_contiguous, string_to_int64_hash, zmq_ctx)

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"


class TestKVCacheTaskTrackerInit(unittest.TestCase):

    def test_init_basic_properties(self):
        tracker = KVCacheTaskTracker()
        self.assertIsInstance(tracker.done_task_lock, type(threading.Lock()))
        self.assertIsInstance(tracker.finished_requests, set)
        self.assertIsInstance(tracker.delayed_free_requests, dict)


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

    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.logger")
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


class TestKVCacheSendingLayerThreadBasic(unittest.TestCase):

    def setUp(self):
        self.p1 = patch(
            'vllm_ascend.distributed.mooncake_layerwise_connector.get_ascend_config',
            new=MagicMock(return_value=SimpleNamespace(
                pd_tp_ratio=1, num_head_replica=0, pd_head_ratio=1)))
        self.p2 = patch(
            'vllm_ascend.distributed.mooncake_layerwise_connector.get_current_vllm_config',
            new=MagicMock(return_value=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_model_len=128))))
        self.p1.start()
        self.addCleanup(self.p1.stop)
        self.p2.start()
        self.addCleanup(self.p2.stop)
        self.engine = MagicMock()
        self.engine.register_memory.return_value = 0
        self.ready_event = threading.Event()

        batch_size, seq_len, hidden_dim, num_heads = 8, 128, 512, 8
        head_dim = hidden_dim // num_heads
        self.first_kv_cache = torch.zeros(
            (batch_size, num_heads, seq_len, head_dim),
            dtype=torch.float32,
            device='cpu')

        self.thread = KVCacheSendingLayerThread(
            tp_rank=0,
            tp_size=4,
            decode_tp_size=2,
            local_engine_id="local_engine",
            side_channel_host="localhost",
            side_channel_port=5555,
            metadata=MagicMock(),
            ready_event=self.ready_event,
            total_layers=3,
            engine=self.engine,
            local_kv_base_addr=[0x1000, 0x2000],
            block_len=[1024, 2048],
            use_mla=True,
            first_kv_cache=self.first_kv_cache)

    def test_add_request(self):
        req_id = "req1"
        meta = DecodeMooncakeAgentMetadata(
            req_id=req_id,
            block_ids=[3, 4],
            host="localhost",
            port=6666,
            engine_id="remote_engine",
            te_rpc_port=6000,
            kv_caches_base_addr=[0x3000, 0x4000],
            num_blocks=8)
        with self.thread.lock:
            self.thread.ready_decode[req_id] = meta

        local_block_ids = [1, 2]
        key = torch.zeros((1, 1), dtype=torch.float32)
        value = torch.zeros((1, 1), dtype=torch.float32)

        self.thread.add_request(request_id=req_id,
                                local_block_ids=local_block_ids,
                                layer_index=5,
                                key=key,
                                value=value)

        queued = self.thread.send_layer_thread.send_queue.get_nowait()
        # queued: (metadata, request_id, local_block_ids, layer_index, key, value)
        self.assertEqual(queued[1], "req1")
        self.assertEqual(queued[0].host, "localhost")

    @patch.object(KVCacheTaskTracker, 'get_and_clear_finished_requests')
    def test_get_finished_requests(self, mock_tracker):
        mock_tracker.return_value = {"req1", "req2"}
        result = self.thread.get_and_clear_finished_requests()
        self.assertEqual(result, {"req1", "req2"})

    @patch.object(KVCacheTaskTracker, 'add_delayed_request')
    def test_add_delayed_request_passthrough(self, mock_add):
        mock_add.return_value = None
        ret = self.thread.add_delayed_request("req1", 123.456)
        mock_add.assert_called_once_with("req1", 123.456)
        self.assertIsNone(ret)

    def test_abort_requests_removes_pending(self):
        with self.thread.lock:
            self.thread.pending_decode["keep"] = [([9], 1)]
            self.thread.pending_decode["dropA"] = [([1], 0)]
            self.thread.pending_decode["dropB"] = [([2], 0)]

        self.thread._abort_requests({"dropA", "dropB"})

        with self.thread.lock:
            self.assertNotIn("dropA", self.thread.pending_decode)
            self.assertNotIn("dropB", self.thread.pending_decode)
            self.assertIn("keep", self.thread.pending_decode)

    @patch('vllm_ascend.distributed.mooncake_layerwise_connector.zmq.Context')
    @patch(
        'vllm_ascend.distributed.mooncake_layerwise_connector.make_zmq_socket')
    @patch(
        'vllm_ascend.distributed.mooncake_layerwise_connector.ensure_zmq_send')
    def test_post_transfer_sends_and_receives_ack(self, mock_send,
                                                  mock_make_socket,
                                                  mock_context):
        req_id = "req_ok"
        meta = DecodeMooncakeAgentMetadata(
            req_id=req_id,
            block_ids=[1],
            host="127.0.0.1",
            port=7777,
            engine_id="remote",
            te_rpc_port=6000,
            kv_caches_base_addr=[0x1],
            num_blocks=1,
        )
        with self.thread.lock:
            self.thread.ready_decode[req_id] = meta

        fake_sock = MagicMock()
        fake_sock.recv.return_value = b"ACK"
        mock_make_socket.return_value = fake_sock

        self.thread._post_transfer(req_id)

        self.assertTrue(mock_make_socket.called)
        _, kwargs = mock_make_socket.call_args
        self.assertEqual(kwargs.get('path'), 'tcp://127.0.0.1:7777')
        self.assertEqual(kwargs.get('socket_type'), zmq.REQ)  # type: ignore
        self.assertFalse(kwargs.get('bind', True))

        mock_send.assert_called_once()
        with self.thread.lock:
            self.assertNotIn(req_id, self.thread.ready_decode)

    @patch('vllm_ascend.distributed.mooncake_layerwise_connector.zmq.Context')
    @patch(
        'vllm_ascend.distributed.mooncake_layerwise_connector.make_zmq_socket')
    @patch(
        'vllm_ascend.distributed.mooncake_layerwise_connector.ensure_zmq_send')
    def test_post_transfer_bad_ack_raises_value_error(self, _mock_send,
                                                      mock_make_socket,
                                                      _mock_context):
        req_id = "req_bad"
        meta = DecodeMooncakeAgentMetadata(
            req_id=req_id,
            block_ids=[1],
            host="127.0.0.1",
            port=8888,
            engine_id="remote",
            te_rpc_port=6000,
            kv_caches_base_addr=[0x2],
            num_blocks=1,
        )
        with self.thread.lock:
            self.thread.ready_decode[req_id] = meta

        fake_sock = MagicMock()
        fake_sock.recv.return_value = b"NOT_ACK"
        mock_make_socket.return_value = fake_sock

        with self.assertRaises(ValueError):
            self.thread._post_transfer(req_id)


class TestSendingLayerThread(unittest.TestCase):

    def setUp(self):
        self.p1 = patch(
            'vllm_ascend.distributed.mooncake_layerwise_connector.get_ascend_config',
            new=MagicMock(return_value=SimpleNamespace(
                pd_tp_ratio=1, num_head_replica=0, pd_head_ratio=1)))
        self.p2 = patch(
            'vllm_ascend.distributed.mooncake_layerwise_connector.get_current_vllm_config',
            new=MagicMock(return_value=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_model_len=128))))
        self.p1.start()
        self.addCleanup(self.p1.stop)
        self.p2.start()
        self.addCleanup(self.p2.stop)
        self.task_tracker = MagicMock(KVCacheTaskTracker)
        self.engine = MagicMock()
        self.engine.register_memory.side_effect = lambda addr, size: 0
        batch_size = 8
        seq_len = 128
        hidden_dim = 512
        num_heads = 8
        head_dim = hidden_dim // num_heads  # 512 // 8 = 64
        self.first_kv_cache = torch.zeros(
            (batch_size, num_heads, seq_len, head_dim),
            dtype=torch.float32,
            device='cpu')
        self.thread = SendingLayerThread(
            task_tracker=self.task_tracker,
            total_layers=3,
            engine=self.engine,
            local_kv_base_addr=["0x1000", "0x2000"],
            block_len=[1024, 2048],
            use_mla=True,
            tp_rank=0,
            first_kv_cache=self.first_kv_cache)

    @patch.object(SendingLayerThread, "_transfer_kv_cache", autospec=True)
    def test_handle_request(self, mock_transfer):
        req_id = "req_1"
        req_meta = MagicMock(spec=DecodeMooncakeAgentMetadata)
        key = torch.zeros((1, 1), dtype=torch.float32)
        value = torch.zeros((1, 1), dtype=torch.float32)
        item = (req_meta, req_id, [10, 11], 0, key, value)
        with patch.object(self.thread.task_tracker, "update_done_task_count") as mock_update_done, \
            patch.object(self.thread.send_queue, "task_done", autospec=True) as mock_task_done:
            self.thread._handle_request(item)
        mock_transfer.assert_called_once_with(self.thread, req_meta, [10, 11],
                                              0, key, value)
        mock_update_done.assert_called_once_with(req_id)
        mock_task_done.assert_called_once()

    @patch('torch.npu.synchronize')
    @patch(
        'vllm_ascend.distributed.mooncake_layerwise_connector.group_concurrent_contiguous'
    )
    def test_transfer_kv_cache(self, mock_group, mock_sync):
        key = torch.zeros((1, 1), dtype=torch.float32)
        value = torch.zeros((1, 1), dtype=torch.float32)
        mock_sync.return_value = None
        self.thread.pd_tp_ratio = 1

        self.thread.local_kv_base_addr = [1000, 2000]

        meta = DecodeMooncakeAgentMetadata(
            req_id="req-ok",
            block_ids=[0],
            host="127.0.0.1",
            port=7777,
            engine_id="remote",
            te_rpc_port=6000,
            kv_caches_base_addr=[4000, 8000],
            num_blocks=256,
        )

        mock_group.return_value = (
            [[10, 11, 12], [20, 21]],  # grouped_remote_block_ids
            [[5, 6, 7], [8, 9]],  # grouped_local_block_ids
        )

        self.engine.batch_transfer_sync_write.return_value = 1

        self.thread._transfer_kv_cache(meta,
                                       local_block_ids=[123],
                                       layer_index=0,
                                       key=key,
                                       value=value)

        # k=0 (block_len=1024):
        #   grp1: src=1000+5*1024=6120,  dst=4000+10*1024=14240, len=3*1024=3072
        #   grp2: src=1000+8*1024=9192,  dst=4000+20*1024=24480, len=2*1024=2048
        # k=1 (block_len=2048):
        #   grp1: src=2000+5*2048=12240, dst=8000+10*2048=28480, len=3*2048=6144
        #   grp2: src=2000+8*2048=18384, dst=8000+20*2048=48960, len=2*2048=4096
        exp_session = "127.0.0.1:6000"
        exp_src = [6120, 9192, 12240, 18384]
        exp_dst = [14240, 24480, 28480, 48960]
        exp_len = [3072, 2048, 6144, 4096]

        self.engine.batch_transfer_sync_write.assert_called_once()
        args, _ = self.engine.batch_transfer_sync_write.call_args
        self.assertEqual(args[0], exp_session)
        self.assertEqual(args[1], exp_src)
        self.assertEqual(args[2], exp_dst)
        self.assertEqual(args[3], exp_len)


class TestKVCacheRecvingLayerThreadBasic(unittest.TestCase):

    def setUp(self):
        self.ready_event = threading.Event()
        self.thread = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=4,
            local_engine_id="local_engine",
            ready_event=self.ready_event,
        )

    def test_get_finished_requests(self):

        with self.thread.lock:
            self.thread.done_requests.update({"req1", "req2"})

        result = self.thread.get_and_clear_finished_requests()
        self.assertEqual(result, {"req1", "req2"})

        result2 = self.thread.get_and_clear_finished_requests()
        self.assertEqual(result2, set())


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
        self.assertIn("req_1", result)
        self.assertEqual(result["req_1"], current_time)

        with patch.object(self.tracker, "on_done") as mock_on_done:
            for _ in range(getattr(self.tracker, "target_count", 1)):
                self.tracker.update_done_task_count("req_1")
            mock_on_done.assert_called_once_with("req_1")

        self.assertEqual(self.tracker.finished_requests, {"req_1"})

        result_delayed = self.tracker.delayed_free_requests
        self.assertEqual(len(result_delayed), 1)
        self.assertIn("req_1", result_delayed)
        self.assertEqual(result_delayed["req_1"], current_time)

    def test_retrieve_expired_requests(self):
        current_time = time.time()
        self.tracker.add_delayed_request("req_1", current_time - 600)
        self.tracker.add_delayed_request("req_2", current_time)
        result = self.tracker._retrieve_expired_requests()
        self.assertEqual(result, {
            "req_1",
        })
        result_delay = self.tracker.delayed_free_requests  # dict
        self.assertEqual(len(result_delay), 1)

        self.assertIn("req_2", result_delay)
        self.assertEqual(result_delay["req_2"], current_time)

    def test_duplicate_task_update(self):
        self.tracker.update_done_task_count("req1")
        self.tracker.update_done_task_count("req1")
        self.tracker.update_done_task_count("req1")

        finished = self.tracker.get_and_clear_finished_requests()
        self.assertEqual(finished, {"req1"})


class TestMooncakeLayerwiseConnectorMetadata(unittest.TestCase):

    def test_add_new_req(self):
        meta = MooncakeLayerwiseConnectorMetadata()
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


class TestMooncakeLayerwiseConnectorSchedulerMatchedTokens(unittest.TestCase):

    def setUp(self):
        config = MockVllmConfig()
        self.scheduler = MooncakeLayerwiseConnectorScheduler(
            config, "test_engine")

    def test_get_num_new_matched_tokens(self):
        request = MockRequest("req1")
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(
            request, 0)
        self.assertEqual(tokens, 0)
        self.assertFalse(async_flag)

        request.kv_transfer_params = {"do_remote_prefill": True}
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(
            request, 0)
        self.assertEqual(tokens, 4)
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
        self.assertIsInstance(meta, MooncakeLayerwiseConnectorMetadata)
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


class TestMooncakeLayerwiseConnectorForScheduler(unittest.TestCase):

    def test_scheduler_role(self):
        config = MockVllmConfig()
        connector = MooncakeLayerwiseConnector(config,
                                               KVConnectorRole.SCHEDULER)
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)

    @patch.object(MooncakeLayerwiseConnectorScheduler,
                  "get_num_new_matched_tokens")
    def test_scheduler_methods(self, mock_method):
        config = MockVllmConfig()
        connector = MooncakeLayerwiseConnector(config,
                                               KVConnectorRole.SCHEDULER)
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


class TestMooncakeLayerwiseConnector(unittest.TestCase):

    def setUp(self):
        self.config = MockVllmConfig()
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

    def test_scheduler_initialization(self):
        connector = MooncakeLayerwiseConnector(self.config,
                                               KVConnectorRole.SCHEDULER)
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)

    @patch.object(MooncakeLayerwiseConnectorScheduler,
                  "get_num_new_matched_tokens")
    def test_get_num_new_matched_tokens(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config,
                                               KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        connector.get_num_new_matched_tokens(request, 0)
        mock_method.assert_called_once_with(request, 0)

    @patch.object(MooncakeLayerwiseConnectorScheduler,
                  "update_state_after_alloc")
    def test_update_state_after_alloc(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config,
                                               KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        blocks = MockKVCacheBlocks()
        connector.update_state_after_alloc(request, blocks, 3)
        mock_method.assert_called_once_with(request, blocks, 3)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "build_connector_meta")
    def test_build_connector_meta(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config,
                                               KVConnectorRole.SCHEDULER)
        scheduler_output = MockSchedulerOutput()
        connector.build_connector_meta(scheduler_output)
        mock_method.assert_called_once_with(scheduler_output)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "request_finished")
    def test_request_finished(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config,
                                               KVConnectorRole.SCHEDULER)
        request = MockRequest("req1")
        connector.request_finished(request, [1, 2, 3])
        mock_method.assert_called_once_with(request, [1, 2, 3])


class TestMooncakeLayerwiseConnectorScheduler(unittest.TestCase):

    def setUp(self):
        self.config = MockVllmConfig()
        self.scheduler = MooncakeLayerwiseConnectorScheduler(
            self.config, "test_engine")

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
        self.assertEqual(tokens, 4)
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

    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.make_zmq_socket")
    def test_zmq_ctx_ok(self, mock_make_socket):
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        with zmq_ctx(zmq.REQ, "tcp://localhost:1234") as s:  # type: ignore
            self.assertEqual(s, mock_socket)

    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_send_success(self, mock_logger):
        mock_socket = MagicMock()
        ensure_zmq_send(mock_socket, b"hello")
        mock_socket.send.assert_called_once_with(b"hello")

    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_send_retry_and_fail(self, mock_logger):
        mock_socket = MagicMock()
        mock_socket.send.side_effect = zmq.ZMQError(  # type: ignore
            "send failed")
        with self.assertRaises(RuntimeError):
            ensure_zmq_send(mock_socket, b"hello", max_retries=2)
        self.assertEqual(mock_socket.send.call_count, 2)

    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_recv_success(self, mock_logger):
        mock_socket = MagicMock()
        mock_socket.recv.return_value = b"response"
        mock_poller = MagicMock()
        mock_poller.poll.return_value = [
            (mock_socket, zmq.POLLIN)  # type: ignore
        ]
        data = ensure_zmq_recv(mock_socket, mock_poller)
        self.assertEqual(data, b"response")

    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.logger")
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


class MockMooncakeLayerwiseConnectorMetadata:

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


class TestMooncakeLayerwiseConnectorWorker(unittest.TestCase):

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
                'vllm_ascend.distributed.mooncake_layerwise_connector.get_tensor_model_parallel_rank',
                mock_get_tensor_model_parallel_rank),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.get_tp_group',
                mock_get_tp_group),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.get_ip',
                mock_get_ip),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.string_to_int64_hash',
                mock_string_to_int64_hash),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.TransferEngine',
                return_value=self.mock_transfer_engine),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.KVCacheSendingLayerThread',
                MagicMock()),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.KVCacheRecvingLayerThread',
                MagicMock()),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.logger',
                MagicMock()),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.threading.Event',
                MagicMock()),
            patch.dict('sys.modules',
                       {'vllm_ascend.envs': self.envs_ascend_mock}),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.get_ascend_config',
                return_value=SimpleNamespace(pd_tp_ratio=1,
                                             num_head_replica=0,
                                             pd_head_ratio=1),
            ),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.get_current_vllm_config',
                return_value=SimpleNamespace(scheduler_config=SimpleNamespace(
                    max_model_len=128)),
            )
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
                        "vllm_ascend.distributed.mooncake_layerwise_connector.get_tensor_model_parallel_rank",
                        return_value=0):
                    with patch(
                            "vllm_ascend.distributed.mooncake_layerwise_connector.get_tp_group",
                            return_value=None):
                        with patch(
                                "vllm_ascend.distributed.mooncake_layerwise_connector.get_ip",
                                return_value="127.0.0.1"):
                            worker = MooncakeLayerwiseConnectorWorker(
                                config, self.engine_id)
                            self.assertIsNotNone(worker)

    def test_register_kv_caches_producer(self):
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config,
                                                  self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertEqual(len(worker.kv_caches), 1)
        self.assertIsNotNone(worker.kv_send_layer_thread)
        self.assertIsNone(worker.kv_recv_layer_thread)

    def test_register_kv_caches_consumer(self):
        self.vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config,
                                                  self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertIsNone(worker.kv_send_layer_thread)
        self.assertIsNotNone(worker.kv_recv_layer_thread)

    def test_register_kv_caches_mla_case(self):
        mla_cache1 = MagicMock()
        mla_cache1.size.return_value = (10, 16, 1, 16)
        mla_cache2 = MagicMock()
        mla_cache2.size.return_value = (10, 16, 1, 8)
        mla_caches = {"layer1": (mla_cache1, mla_cache2)}

        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config,
                                                  self.engine_id)
        worker.register_kv_caches(mla_caches)
        self.assertTrue(worker.use_mla)
        self.assertEqual(len(worker.block_len), 2)

    def test_device_id_selection_with_physical_devices(self):
        # Test with physical devices set
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config,
                                                  self.engine_id)
        # Default tp_rank is 0, so device_id should be 10
        self.assertEqual(worker.device_id, 10)


if __name__ == '__main__':
    unittest.main()
