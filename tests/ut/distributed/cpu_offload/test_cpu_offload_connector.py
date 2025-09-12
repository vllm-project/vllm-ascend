import queue
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, call, patch

import torch
from vllm.attention import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata, KVConnectorRole)
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import Request

from vllm_ascend.distributed.cpu_offload_connector import (
    CPUOffloadingConnector, CPUOffloadingConnectorMetadata,
    CPUOffloadingConnectorScheduler, CPUOffloadingConnectorWorker, ReqMeta,
    get_kv_cache_spec)
from vllm_ascend.worker.metadata import MetadataServerProc


class TestReqMeta(unittest.TestCase):
    """ReqMeta unit test"""

    def test_initialization(self):
        """test ReqMeta init"""
        meta = ReqMeta(gpu_block_ids=[1, 2, 3],
                       cpu_block_ids=[4, 5],
                       num_scheduled_tokens=10,
                       num_computed_tokens=5,
                       num_gpu_computed_tokens=3,
                       num_cpu_computed_tokens=2)

        self.assertEqual(meta.gpu_block_ids, [1, 2, 3])
        self.assertEqual(meta.cpu_block_ids, [4, 5])
        self.assertEqual(meta.num_scheduled_tokens, 10)
        self.assertEqual(meta.num_computed_tokens, 5)
        self.assertEqual(meta.num_gpu_computed_tokens, 3)
        self.assertEqual(meta.num_cpu_computed_tokens, 2)

    def test_update_method(self):
        """test update method of ReqMeta"""
        meta1 = ReqMeta(gpu_block_ids=[1, 2],
                        cpu_block_ids=[3],
                        num_scheduled_tokens=5,
                        num_computed_tokens=3,
                        num_gpu_computed_tokens=2,
                        num_cpu_computed_tokens=1)

        meta2 = ReqMeta(gpu_block_ids=[4, 5],
                        cpu_block_ids=[6, 7],
                        num_scheduled_tokens=8,
                        num_computed_tokens=6,
                        num_gpu_computed_tokens=4,
                        num_cpu_computed_tokens=2)

        meta1.update(meta2)

        self.assertEqual(meta1.gpu_block_ids, [1, 2, 4, 5])
        self.assertEqual(meta1.cpu_block_ids, [3, 6, 7])

        self.assertEqual(meta1.num_scheduled_tokens, 8)
        self.assertEqual(meta1.num_computed_tokens, 6)
        self.assertEqual(meta1.num_gpu_computed_tokens, 4)
        self.assertEqual(meta1.num_cpu_computed_tokens, 2)


class TestCPUOffloadingConnectorMetadata(unittest.TestCase):
    """CPUOffloadingConnectorMetadata UT"""

    def setUp(self):
        self.req_meta_1 = ReqMeta(gpu_block_ids=[1, 2],
                                  cpu_block_ids=[3, 4],
                                  num_scheduled_tokens=10,
                                  num_computed_tokens=8,
                                  num_gpu_computed_tokens=5,
                                  num_cpu_computed_tokens=3)

        self.req_meta_2 = ReqMeta(gpu_block_ids=[5, 6],
                                  cpu_block_ids=[7, 8],
                                  num_scheduled_tokens=15,
                                  num_computed_tokens=12,
                                  num_gpu_computed_tokens=8,
                                  num_cpu_computed_tokens=4)

        self.metadata = CPUOffloadingConnectorMetadata(
            requests={
                "req_1": self.req_meta_1,
                "req_2": self.req_meta_2
            },
            finished_req_ids={"req_3", "req_4"})

    def test_initialization(self):
        self.assertEqual(len(self.metadata.requests), 2)
        self.assertIn("req_1", self.metadata.requests)
        self.assertIn("req_2", self.metadata.requests)
        self.assertEqual(self.metadata.requests["req_1"], self.req_meta_1)
        self.assertEqual(self.metadata.requests["req_2"], self.req_meta_2)

        self.assertEqual(len(self.metadata.finished_req_ids), 2)
        self.assertIn("req_3", self.metadata.finished_req_ids)
        self.assertIn("req_4", self.metadata.finished_req_ids)


class TestCPUOffloadingConnector(unittest.TestCase):

    def setUp(self):
        self.mock_vllm_config = MagicMock()

        self.patches = [
            patch('vllm.distributed.parallel_state._PP', MagicMock()),
            patch('vllm.distributed.parallel_state._TP', MagicMock()),
            patch('torch.npu.Stream', MagicMock()),
            patch('vllm_ascend.worker.metadata.MetadataServer.ZMQRPCClient',
                  MagicMock()),
            patch('vllm_ascend.worker.metadata.MetadataServer', MagicMock()),
            patch('vllm.utils.logger', MagicMock()),
            patch('vllm.distributed.parallel_state.get_pp_group', MagicMock()),
            patch('vllm.distributed.parallel_state.get_pp_group', MagicMock()),
        ]

        for p in self.patches:
            p.start()  # type: ignore

    def tearDown(self):
        for p in self.patches:
            p.stop()  # type: ignore

    def test_initialization(self):
        # case1
        self.mock_vllm_config.cache_config.enable_prefix_caching = False
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.SCHEDULER)
        self.assertIsNone(self.connector.connector_scheduler)
        self.assertIsNone(self.connector.connector_worker)

        #case2
        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.SCHEDULER)
        self.assertIsInstance(self.connector.connector_scheduler,
                              CPUOffloadingConnectorScheduler)
        self.assertIsNone(self.connector.connector_worker)

        #case3
        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.WORKER)
        self.assertIsNone(self.connector.connector_scheduler)
        self.assertIsInstance(self.connector.connector_worker,
                              CPUOffloadingConnectorWorker)

    def test_bind_connector_metadata(self):
        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        # case1 connector_worker is not None
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.WORKER)
        self.assertIsNotNone(self.connector.connector_worker)

        self.connector.connector_worker.bind_connector_metadata = MagicMock()
        mock_metadata = MagicMock(spec=CPUOffloadingConnectorMetadata)

        self.connector.bind_connector_metadata(mock_metadata)
        self.connector.connector_worker.bind_connector_metadata.assert_called_once_with(
            mock_metadata)

        #case2 connector_worker is None
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.SCHEDULER)
        self.assertIsNone(self.connector.connector_worker)
        mock_metadata = MagicMock(spec=CPUOffloadingConnectorMetadata)

        # run no exception
        self.connector.bind_connector_metadata(mock_metadata)

    def test_clear_connector_metadata(self):
        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        # case1 connector_worker is not None
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.WORKER)
        self.assertIsNotNone(self.connector.connector_worker)

        self.connector.connector_worker.clear_connector_metadata = MagicMock()
        self.connector.clear_connector_metadata()
        self.connector.connector_worker.clear_connector_metadata.assert_called_once(
        )

        self.connector.connector_worker = None
        with self.assertRaises(AssertionError):
            self.connector.clear_connector_metadata()

    def test_register_kv_caches(self):
        self.test_kv_caches = {
            "layer_0": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "layer_1": torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        }
        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        # case1 connector_worker is not None
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.WORKER)
        self.assertIsNotNone(self.connector.connector_worker)

        self.connector.connector_worker.register_kv_caches = MagicMock()
        self.connector.register_kv_caches(self.test_kv_caches)
        self.connector.connector_worker.register_kv_caches.assert_called_once_with(
            self.test_kv_caches)

        self.connector.connector_worker = None
        # run no exception
        self.connector.register_kv_caches(self.test_kv_caches)

    def test_start_load_kv(self):
        self.forward_context = MagicMock(spec=ForwardContext)
        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        # case1 connector_worker is not None
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.WORKER)
        self.assertIsNotNone(self.connector.connector_worker)

        self.connector.connector_worker.start_load_kv = MagicMock()
        test_kwargs = {"priority": 1, "timeout": 100}
        self.connector.start_load_kv(self.forward_context, **test_kwargs)
        self.connector.connector_worker.start_load_kv.assert_called_once_with()

        self.connector.connector_worker = None
        # run no exception
        self.connector.start_load_kv(self.forward_context, param="test")

    def test_wait_for_layer_load(self):
        self.test_layer_name = "layer_0"
        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        # case1 connector_worker is not None
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.WORKER)
        self.assertIsNotNone(self.connector.connector_worker)

        self.connector.connector_worker.wait_for_layer_load = MagicMock()
        self.connector.wait_for_layer_load(self.test_layer_name)
        self.connector.connector_worker.wait_for_layer_load.assert_called_once_with(
        )

        self.connector.connector_worker = None
        # run no exception
        self.connector.wait_for_layer_load(self.test_layer_name)

    def test_get_finished(self):
        self.test_finished_ids = {"req1", "req2", "req3"}
        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        # case1 connector_worker is not None
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.WORKER)
        self.assertIsNotNone(self.connector.connector_worker)

        mock_worker_result = {"req4", "req5"}
        self.connector.connector_worker.get_finished = MagicMock(
            return_value=mock_worker_result)
        result, none_result = self.connector.get_finished(
            self.test_finished_ids)

        self.connector.connector_worker.get_finished.assert_called_once_with()

        self.assertEqual(result, mock_worker_result)
        self.assertIsNone(none_result)

        # connector_worker is None
        self.connector.connector_worker = None
        with self.assertRaises(AssertionError):
            self.connector.get_finished(self.test_finished_ids)

    def test_scheduler_get_num_new_matched_tokens(self):
        self.test_request = MagicMock(spec=Request)
        self.test_request.request_id = "req_001"
        self.test_num_computed = 10

        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        # case1 connector_worker is not None
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.SCHEDULER)
        self.assertIsNotNone(self.connector.connector_scheduler)
        mock_scheduler_result = (5, True)

        self.connector.connector_scheduler.get_num_new_matched_tokens = MagicMock(
            return_value=mock_scheduler_result)

        result, load_async = self.connector.get_num_new_matched_tokens(
            self.test_request, self.test_num_computed)

        self.connector.connector_scheduler.get_num_new_matched_tokens.assert_called_once_with(
            self.test_request, self.test_num_computed)

        self.assertEqual(result, mock_scheduler_result[0])
        self.assertEqual(load_async, mock_scheduler_result[1])

        # connector_scheduler is None
        self.connector.connector_scheduler = None
        result, load_async = self.connector.get_num_new_matched_tokens(
            self.test_request, self.test_num_computed)
        self.assertEqual(result, 0)
        self.assertFalse(load_async)

    def test_scheduler_update_state_after_alloc(self):
        self.test_request = MagicMock(spec=Request)
        self.test_request.request_id = "req_100"
        self.test_blocks = MagicMock(spec=KVCacheBlocks)
        self.test_num_external = 5

        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        # case1 connector_worker is not None
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.SCHEDULER)
        self.assertIsNotNone(self.connector.connector_scheduler)

        self.connector.connector_scheduler.update_state_after_alloc = MagicMock(
        )

        self.connector.update_state_after_alloc(self.test_request,
                                                self.test_blocks,
                                                self.test_num_external)

        self.connector.connector_scheduler.update_state_after_alloc.assert_called_once_with(
            self.test_request)

        # connector_scheduler is None
        self.connector.connector_scheduler = None
        # run with no exception
        self.connector.update_state_after_alloc(self.test_request,
                                                self.test_blocks,
                                                self.test_num_external)

    def test_scheduler_build_connector_meta(self):
        self.test_scheduler_output = MagicMock(spec=SchedulerOutput)

        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        # case1 connector_worker is not None
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.SCHEDULER)
        self.assertIsNotNone(self.connector.connector_scheduler)

        mock_meta = MagicMock(spec=KVConnectorMetadata)
        self.connector.connector_scheduler.build_connector_meta = MagicMock(
            return_value=mock_meta)

        result = self.connector.build_connector_meta(
            self.test_scheduler_output)

        self.connector.connector_scheduler.build_connector_meta.assert_called_once_with(
            self.test_scheduler_output)

        self.assertEqual(result, mock_meta)

        # connector_scheduler is None
        self.connector.connector_scheduler = None
        # run with no exception
        result = self.connector.build_connector_meta(
            self.test_scheduler_output)
        self.assertIsInstance(result, KVConnectorMetadata)

    def test_scheduler_request_finished(self):
        self.test_request = MagicMock(spec=Request)
        self.test_request.request_id = "req_200"
        self.test_block_ids = [10, 20, 30]

        self.mock_vllm_config.cache_config.enable_prefix_caching = True
        # case1 connector_worker is not None
        self.connector = CPUOffloadingConnector(self.mock_vllm_config,
                                                KVConnectorRole.SCHEDULER)
        self.assertIsNotNone(self.connector.connector_scheduler)

        self.connector.connector_scheduler.request_finished = MagicMock()

        # 执行测试方法
        result, extra = self.connector.request_finished(
            self.test_request, self.test_block_ids)

        self.connector.connector_scheduler.request_finished.assert_called_once_with(
            self.test_request)

        self.assertTrue(result)
        self.assertIsNone(extra)

        # connector_scheduler is None
        self.connector.connector_scheduler = None
        result, extra = self.connector.request_finished(
            MagicMock(spec=Request), [40, 50])
        self.assertTrue(result)
        self.assertIsNone(extra)


class TestCPUOffloadingConnectorScheduler(unittest.TestCase):

    def setUp(self):
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.cache_config.block_size = 128
        self.mock_vllm_config.model_config.use_mla = True
        self.mock_vllm_config.kv_transfer_config = MagicMock()

        self.patches = [
            patch('vllm_ascend.worker.metadata.MetadataServer.ZMQRPCClient',
                  MagicMock()),
            patch('vllm.utils.logger', MagicMock()),
        ]

        for p in self.patches:
            p.start()  # type: ignore

        self.scheduler = CPUOffloadingConnectorScheduler(self.mock_vllm_config)
        self.scheduler.zmq_rpc_client = MagicMock()

    def tearDown(self):
        for p in self.patches:
            p.stop()  # type: ignore

    def test_initialization(self):
        self.assertEqual(self.scheduler.block_size, 128)

    def test_get_num_new_matched_tokens(self):
        self.scheduler.swap_in_threshold = 5
        # case1
        mock_request1 = MagicMock()
        mock_request1.request_id = "req_456"
        num_computed_tokens = 15

        # mock ZMQ RPC response
        self.scheduler.zmq_rpc_client.call.return_value = (18, False)

        result, load_async = self.scheduler.get_num_new_matched_tokens(
            mock_request1, num_computed_tokens)

        # the diff 3 is less than threshold 5
        self.assertEqual(result, 0)
        self.assertEqual(load_async, False)

        self.assertEqual(self.scheduler.num_gpu_computed_tokens["req_456"], 15)
        self.assertEqual(self.scheduler.num_cpu_computed_tokens["req_456"], 18)

        #case 2
        mock_request2 = MagicMock()
        mock_request2.request_id = "req_123"
        num_computed_tokens = 10

        self.scheduler.zmq_rpc_client.call.return_value = (20, True)

        result, load_async = self.scheduler.get_num_new_matched_tokens(
            mock_request2, num_computed_tokens)

        # the diff 20-10 is great than threshold 5
        self.assertEqual(result, 10)
        self.assertEqual(load_async, True)

        self.assertEqual(mock_request2.get_hash_new_full_blocks, None)

        self.scheduler.zmq_rpc_client.call.assert_has_calls([
            call("get_matched_num_and_touch", mock_request1),
            call("get_matched_num_and_touch", mock_request2)
        ])
        self.assertEqual(self.scheduler.num_gpu_computed_tokens["req_123"], 10)
        self.assertEqual(self.scheduler.num_cpu_computed_tokens["req_123"], 20)

    def test_update_state_after_alloc(self):
        req1 = MagicMock(request_id="req_003")
        req2 = MagicMock(request_id="req_004")
        req3 = MagicMock(request_id="req_005")

        self.scheduler.update_state_after_alloc(req1)
        self.scheduler.update_state_after_alloc(req2)
        self.scheduler.update_state_after_alloc(req3)

        self.assertTrue("req_003" in self.scheduler.allocated_req_ids)
        self.assertTrue("req_004" in self.scheduler.allocated_req_ids)
        self.assertTrue("req_005" in self.scheduler.allocated_req_ids)
        self.assertEqual(len(self.scheduler.allocated_req_ids), 3)

    def test_build_connector_meta(self):
        self.scheduler.num_gpu_computed_tokens = {
            "req1": 10,
            "req2": 20,
            "req3": 30
        }
        self.scheduler.num_cpu_computed_tokens = {
            "req1": 15,
            "req2": 25,
            "req3": 35
        }
        self.scheduler.allocated_req_ids = {"req1", "req2"}
        self.scheduler.finished_req_ids = ["req4"]

        self.scheduler.zmq_rpc_client = MagicMock()
        # mock ZMQ RPC response cpu_block_ids
        self.scheduler.zmq_rpc_client.call.return_value = {
            "req1": [100, 101],
            "req2": [200, 201],
            "req3": [300, 301]
        }

        mock_new_req = MagicMock(req_id="req1",
                                 num_computed_tokens=10,
                                 block_ids=[[5, 6]])
        mock_cached_reqs = MagicMock(req_ids=["req2"],
                                     num_computed_tokens=[20],
                                     new_block_ids=[[7, 8]])

        mock_scheduler_output = MagicMock(
            scheduled_new_reqs=[mock_new_req],
            scheduled_cached_reqs=mock_cached_reqs,
            num_scheduled_tokens={
                "req1": 5,
                "req2": 3
            })

        metadata = self.scheduler.build_connector_meta(mock_scheduler_output)

        call_args = self.scheduler.zmq_rpc_client.call.call_args[0]
        self.assertEqual(call_args[1], {"req1": 15, "req2": 23})

        self.assertEqual(len(metadata.requests), 2)
        self.assertIn("req1", metadata.requests)
        self.assertIn("req2", metadata.requests)

    def test_request_finished(self):
        self.scheduler.finished_req_ids = []
        req1 = MagicMock(request_id="req_101")
        req2 = MagicMock(request_id="req_102")
        req3 = MagicMock(request_id="req_103")

        self.scheduler.request_finished(req1)
        self.scheduler.request_finished(req2)
        self.scheduler.request_finished(req3)

        self.assertEqual(len(self.scheduler.finished_req_ids), 3)
        self.assertEqual(self.scheduler.finished_req_ids,
                         ["req_101", "req_102", "req_103"])

        self.assertEqual(self.scheduler.zmq_rpc_client.call.call_count, 3)
        calls = [
            call("record_request_cache_and_free_slots", req1),
            call("record_request_cache_and_free_slots", req2),
            call("record_request_cache_and_free_slots", req3)
        ]
        self.scheduler.zmq_rpc_client.call.assert_has_calls(calls,
                                                            any_order=False)


class TestCPUOffloadingConnectorWorker(unittest.TestCase):

    def setUp(self):
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.cache_config.block_size = 128
        self.mock_vllm_config.model_config.use_mla = True
        self.mock_vllm_config.parallel_config.data_parallel_rank = 0
        self.mock_vllm_config.kv_transfer_config = MagicMock()

        self.patches = [
            patch('vllm.distributed.parallel_state._PP', MagicMock()),
            patch('vllm.distributed.parallel_state._TP', MagicMock()),
            patch('torch.npu.Stream', MagicMock()),
            patch('vllm_ascend.worker.metadata.MetadataServer.ZMQRPCClient',
                  MagicMock()),
            patch('vllm_ascend.worker.metadata.MetadataServer', MagicMock()),
            patch('vllm.utils.logger', MagicMock()),
            patch('vllm.distributed.parallel_state.get_pp_group', MagicMock()),
            patch('vllm.distributed.parallel_state.get_pp_group', MagicMock()),
        ]

        for p in self.patches:
            p.start()  # type: ignore

        self.worker = CPUOffloadingConnectorWorker(self.mock_vllm_config)
        self.worker.pp_rank = 0
        self.worker.tp_rank = 0
        self.worker.zmq_rpc_client = MagicMock()

    def tearDown(self):
        for p in self.patches:
            p.stop()  # type: ignore

    def test_initialization(self):
        self.assertEqual(self.worker.block_size, 128)

    @patch('threading.Thread')
    @patch(
        'vllm_ascend.worker.metadata.MetadataServerProc.run_metadata_server')
    def test_init_metadata_server(self, mock_run_server, mock_thread_class):
        mock_thread = MagicMock()
        mock_thread_class.return_value = mock_thread

        self.worker.init_metadata_server(self.mock_vllm_config)

        mock_thread_class.assert_called_once_with(
            target=MetadataServerProc.run_metadata_server,
            args=(self.mock_vllm_config, ))
        self.assertTrue(mock_thread.daemon)

        mock_thread.start.assert_called_once()
        self.assertEqual(self.worker.metadata_thread, mock_thread)

    @patch('time.sleep')
    def test_wait_for_metadata_process_start(self, mock_sleep):
        self.worker._wait_for_metadata_process_start()
        mock_sleep.assert_called_once_with(5)

    def test_bind_connector_metadata(self):
        req_meta = MagicMock(spec=ReqMeta)
        req_meta.num_gpu_computed_tokens = 0
        req_meta.num_computed_tokens = 256
        req_meta.cpu_block_ids = [101, 102]
        req_meta.gpu_block_ids = [201, 202]

        connector_metadata = MagicMock(spec=CPUOffloadingConnectorMetadata)
        connector_metadata.requests = {"req_1": req_meta}
        connector_metadata.finished_req_ids = set()

        self.worker.bind_connector_metadata(connector_metadata)

        self.assertIn("req_1", self.worker.requests)
        self.assertEqual(self.worker.requests["req_1"], req_meta)

        self.assertEqual(len(self.worker.load_block_mapping), 2)
        self.assertIn((101, 201), self.worker.load_block_mapping)
        self.assertIn((102, 202), self.worker.load_block_mapping)

        self.assertTrue(self.worker.save_input_queue.empty())

    def test_clear_connector_metadata(self):
        self.worker.load_block_mapping = [(101, 201), (102, 202), (103, 203)]
        self.assertEqual(len(self.worker.load_block_mapping), 3)
        self.worker.clear_connector_metadata()
        self.assertEqual(len(self.worker.load_block_mapping), 0)

    def test_register_kv_caches_without_mla(self):
        self.mock_vllm_config.model_config.use_mla = False

        kv_caches = {
            "layer1": [torch.randn(1, 10, 512),
                       torch.randn(1, 10, 512)],
            "layer2": [torch.randn(1, 10, 512),
                       torch.randn(1, 10, 512)]
        }

        mock_rpc_result = {"cache1": MagicMock(), "cache2": MagicMock()}
        self.worker.zmq_rpc_client.call.return_value = mock_rpc_result

        with patch(
                'vllm_ascend.distributed.cpu_offload_connector.get_kv_cache_spec'
        ) as mock_get_spec:
            mock_get_spec.return_value = MagicMock()

            self.worker.register_kv_caches(kv_caches)

            self.assertEqual(self.worker.gpu_kv_caches, kv_caches)
            self.worker.zmq_rpc_client.call.assert_called_once_with(
                "init_cpu_kv_caches",
                0,  # pp_rank
                0,  # tp_rank
                mock_get_spec.return_value,
                None  # mla_config 为 None
            )

            self.assertEqual(self.worker.cpu_kv_caches,
                             list(mock_rpc_result.values()))

    def test_register_kv_caches_with_mla(self):
        self.mock_vllm_config.model_config.use_mla = True
        self.mock_vllm_config.model_config.hf_text_config.kv_lora_rank = 8
        self.mock_vllm_config.model_config.hf_text_config.qk_rope_head_dim = 128

        kv_caches = {
            "layer1": [torch.randn(1, 10, 512),
                       torch.randn(1, 10, 512)],
            "layer2": [torch.randn(1, 10, 512),
                       torch.randn(1, 10, 512)]
        }

        mock_rpc_result = {"cache1": MagicMock(), "cache2": MagicMock()}
        self.worker.zmq_rpc_client.call.return_value = mock_rpc_result

        with patch(
                'vllm_ascend.distributed.cpu_offload_connector.get_kv_cache_spec'
        ) as mock_get_spec:
            mock_get_spec.return_value = MagicMock()
            self.worker.register_kv_caches(kv_caches)

            self.assertEqual(self.worker.gpu_kv_caches, kv_caches)

            self.worker.zmq_rpc_client.call.assert_called_once()

            call_args = self.worker.zmq_rpc_client.call.call_args[0]

            self.assertEqual(call_args[0], "init_cpu_kv_caches")
            self.assertEqual(call_args[1], 0)  # pp_rank
            self.assertEqual(call_args[2], 0)  # tp_rank
            self.assertEqual(call_args[3], mock_get_spec.return_value)
            self.assertEqual(self.worker.cpu_kv_caches,
                             list(mock_rpc_result.values()))

    def test_start_load_kv(self):
        self.worker.load_kv_layer = MagicMock()
        self.worker.gpu_kv_caches = {"layer1": "data1", "layer2": "data2"}
        
        self.worker.start_load_kv()
        
        self.assertEqual(self.worker.current_layer, 0)
        self.assertTrue(hasattr(self.worker, 'gpu_kv_caches_load_iter'))
        self.worker.load_kv_layer.assert_called_once_with(0)
        iter_values = list(self.worker.gpu_kv_caches_load_iter)
        self.assertEqual(iter_values, ["data1", "data2"])

    @patch('torch.npu.stream')
    def test_load_kv_layer(self, mock_npu_stream):
        self.worker.gpu_kv_caches = {
            "layer0": [MagicMock(), MagicMock()],
            "layer1": [MagicMock(), MagicMock()],
            "layer2": [MagicMock(), MagicMock()]
        }
        # create GPU KV cache iterators
        self.mock_gpu_kv_cache_1 = [MagicMock(), MagicMock()]
        self.mock_gpu_kv_cache_2 = [MagicMock(), MagicMock()]
        self.worker.gpu_kv_caches_load_iter = iter(
            [self.mock_gpu_kv_cache_1, self.mock_gpu_kv_cache_2])

        # create mocked CPU KV cache
        self.worker.cpu_kv_caches = [
            [MagicMock(), MagicMock()],  # layer0
            [MagicMock(), MagicMock()],  # layer1
            [MagicMock(), MagicMock()]  # layer2
        ]
        self.worker.load_block_mapping = [
            (101, 201),  # (cpu_block_id, gpu_block_id)
            (102, 202)
        ]
        self.worker.load_stream = MagicMock()
        self.mock_stream_context = MagicMock()
        self.mock_stream_context.__enter__ = MagicMock(return_value=None)
        self.mock_stream_context.__exit__ = MagicMock(return_value=None)

        mock_npu_stream.return_value = self.mock_stream_context

        # call load_kv_layer
        self.worker.load_kv_layer(1)  # load layer 1

        # test NPU load_stream
        mock_npu_stream.assert_called_once_with(self.worker.load_stream)
        self.mock_stream_context.__enter__.assert_called_once()
        self.mock_stream_context.__exit__.assert_called_once()

        for cpu_block_id, gpu_block_id in self.worker.load_block_mapping:
            for gpu_layer_part, cpu_layer_part in zip(
                    self.mock_gpu_kv_cache_1, self.worker.cpu_kv_caches[1]):
                gpu_layer_part.__getitem__.assert_any_call(gpu_block_id)
                cpu_layer_part.__getitem__.assert_any_call(cpu_block_id)

                gpu_block = gpu_layer_part.__getitem__.return_value
                cpu_block = cpu_layer_part.__getitem__.return_value
                gpu_block.copy_.assert_any_call(cpu_block, non_blocking=True)

    def test_wait_for_layer_load(self):
        self.worker.load_stream = MagicMock()
        self.worker.current_layer = 0
        self.worker.load_kv_layer = MagicMock()
        self.worker.current_layer = 3

        self.worker.wait_for_layer_load()

        self.worker.load_stream.synchronize.assert_called_once()
        self.assertEqual(self.worker.current_layer, 4)
        self.worker.load_kv_layer.assert_called_once_with(4)

    def test_get_finished_tp1(self):
        self.worker.save_output_queue = MagicMock()
        self.worker.requests = {}
        self.worker.tp_world_size = 2
        self.worker.tp_rank = 0
        self.worker.tp_group = MagicMock()
        self.worker.done_sending_count = defaultdict(int)

        # case1 tp=1
        self.worker.tp_world_size = 1
        self.worker.save_output_queue.get_nowait.side_effect = [
            "req_1", "req_2", queue.Empty
        ]
        self.worker.requests = {"req_1": MagicMock(), "req_2": MagicMock()}
        result = self.worker.get_finished()

        self.assertEqual(result, {"req_1", "req_2"})
        self.assertEqual(len(self.worker.requests), 0)
        self.assertEqual(self.worker.save_output_queue.get_nowait.call_count,
                         3)

    def test_get_finished_tp3(self):
        self.worker.save_output_queue = MagicMock()
        self.worker.requests = {}
        self.worker.tp_world_size = 2
        self.worker.tp_rank = 0
        self.worker.tp_group = MagicMock()
        self.worker.done_sending_count = defaultdict(int)

        # case2 dp=3
        self.worker.tp_world_size = 3
        self.worker.tp_rank = 0
        self.worker.save_output_queue.get_nowait.side_effect = [
            "req_1", "req_2", queue.Empty
        ]
        self.worker.requests = {"req_1": MagicMock(), "req_2": MagicMock()}
        self.worker.tp_group.recv_object.side_effect = [["req_3"], ["req_4"]]

        with patch('threading.Thread') as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread
            result = self.worker.get_finished()
            # verify result
            self.assertEqual(result, set())
            self.assertEqual(len(self.worker.requests), 0)
            self.assertEqual(
                self.worker.save_output_queue.get_nowait.call_count, 3)
            self.assertEqual(self.worker.tp_group.recv_object.call_count, 2)
            self.worker.tp_group.recv_object.assert_any_call(src=1)
            self.worker.tp_group.recv_object.assert_any_call(src=2)

            self.assertEqual(self.worker.done_sending_count["req_1"], 1)
            self.assertEqual(self.worker.done_sending_count["req_2"], 1)
            self.assertEqual(self.worker.done_sending_count["req_3"], 1)
            self.assertEqual(self.worker.done_sending_count["req_4"], 1)

            mock_thread_class.assert_called_once_with(
                target=self.worker._sending_finished, args=(set(), ))
            mock_thread.daemon = True
            mock_thread.start.assert_called_once()

    def test_sending_finished(self):
        test_req_id = "req_12345"
        self.worker._sending_finished([test_req_id])

        self.worker.zmq_rpc_client.call.assert_called_once_with(
            "cache_and_free_slots", test_req_id)

    def test_save_listener_normal_operation(self):
        self.worker.save_input_queue = MagicMock()
        self.worker.save_output_queue = MagicMock()
        self.worker.save_stream = MagicMock()
        self.worker.block_size = 128
        self.worker.use_mla = False
        self.worker.tp_rank = 0
        self.worker.tp_world_size = 2

        self.worker.cpu_kv_caches = [[MagicMock(), MagicMock()],
                                     [MagicMock(), MagicMock()]]

        self.worker.gpu_kv_caches = {
            "layer0": [MagicMock(), MagicMock()],
            "layer1": [MagicMock(), MagicMock()]
        }
        req = MagicMock()
        req.num_cpu_computed_tokens = 0
        req.num_computed_tokens = 256
        req.num_scheduled_tokens = 128
        req.cpu_block_ids = [101, 102, 103]
        req.gpu_block_ids = [201, 202, 203]

        self.worker.save_input_queue.get.side_effect = [("req_1", req),
                                                        KeyboardInterrupt]
        mock_stream_context = MagicMock()
        mock_stream_context.__enter__ = MagicMock(return_value=None)
        mock_stream_context.__exit__ = MagicMock(return_value=None)

        with patch('torch.npu.stream') as mock_npu_stream:
            mock_npu_stream.return_value = mock_stream_context

            self.worker._save_listener()

            self.assertEqual(self.worker.save_input_queue.get.call_count, 2)
            mock_npu_stream.assert_called_once_with(self.worker.save_stream)
            self.worker.save_stream.synchronize.assert_called_once()
            self.worker.save_output_queue.put.assert_called_once_with("req_1")

            for cpu_kv_caches, gpu_kv_caches in zip(
                    self.worker.cpu_kv_caches,
                    self.worker.gpu_kv_caches.values()):
                for cpu_layer_part, gpu_layer_part in zip(
                        cpu_kv_caches, gpu_kv_caches):
                    self.assertEqual(cpu_layer_part.__getitem__.call_count, 3)
                    self.assertEqual(gpu_layer_part.__getitem__.call_count, 3)


class TestGetKvCacheSpec(unittest.TestCase):

    def setUp(self):
        self.vllm_config = MagicMock(spec=VllmConfig)
        self.vllm_config.cache_config = MagicMock(block_size=1024)
        self.vllm_config.model_config = MagicMock(use_mla=False)
        self.vllm_config.compilation_config = MagicMock()
        self.vllm_config.compilation_config.static_forward_context = {}

        self.base_attn_props = {
            "num_kv_heads": 8,
            "head_size": 64,
            "dtype": "float16",
        }

    def test_get_kv_cache_spec_multiple_layers_mixed_types(self):
        decoder_attn = MagicMock(spec=Attention)
        decoder_attn.attn_type = AttentionType.DECODER
        decoder_attn.__dict__.update(self.base_attn_props)

        encoder_attn = MagicMock(spec=Attention)
        encoder_attn.attn_type = AttentionType.ENCODER
        encoder_attn.__dict__.update(self.base_attn_props)

        moe_module = MagicMock(spec=FusedMoE)

        self.vllm_config.compilation_config.static_forward_context = {
            "layer_6": decoder_attn,
            "layer_7": encoder_attn,
            "layer_8": moe_module
        }

        result = get_kv_cache_spec(self.vllm_config)

        self.assertEqual(len(result), 1)
        self.assertIn("layer_6", result)
        self.assertIsInstance(result["layer_6"], FullAttentionSpec)

    def test_get_kv_cache_spec_mla_config(self):
        self.vllm_config.model_config.use_mla = True

        decoder_attn = MagicMock(spec=Attention)
        decoder_attn.attn_type = AttentionType.DECODER
        decoder_attn.__dict__.update(self.base_attn_props)

        self.vllm_config.compilation_config.static_forward_context = {
            "layer_9": decoder_attn
        }

        result = get_kv_cache_spec(self.vllm_config)

        self.assertTrue(result["layer_9"].use_mla)


if __name__ == '__main__':
    unittest.main()
