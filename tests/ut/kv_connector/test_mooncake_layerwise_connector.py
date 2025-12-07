import os
import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import zmq

# fake mooncake.engine.TransferEngine
fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine

from vllm_ascend.distributed.mooncake_layerwise_connector import (  # noqa: E402
    KVCacheRecvingLayerThread, KVCacheSendingLayerThread, KVConnectorRole,
    MooncakeAgentMetadata, MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorMetadata, MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker, ReqMeta, ensure_zmq_recv,
    ensure_zmq_send, group_concurrent_contiguous, string_to_int64_hash,
    zmq_ctx)

GET_META_MSG = b"get_meta_msg"
DONE_SENDING_MSG = b"done_sending_msg"


class TestKVCacheSendingLayerThread(unittest.TestCase):

    def setUp(self):
        self.engine = MagicMock()
        self.engine.register_memory.return_value = 0
        self.engine.batch_transfer_sync_write.return_value = 1
        self._patcher_cs = patch(
            'vllm_ascend.distributed.mooncake_layerwise_connector.torch_npu.npu.current_stream'
        )
        self.mock_current_stream = self._patcher_cs.start()
        self.addCleanup(self._patcher_cs.stop)
        fake_stream = MagicMock(name="FakeStream")
        fake_stream.synchronize = MagicMock()
        self.mock_current_stream.return_value = fake_stream

        self.first_kv_cache = torch.zeros((2, 2, 2, 8),
                                          dtype=torch.float32,
                                          device="cpu")

        self.ready_event = threading.Event()

        self.thread = KVCacheSendingLayerThread(
            engine=self.engine,
            total_layers=3,
            ready_event=self.ready_event,
            tp_rank=0,
            pd_head_ratio=1,
            num_head_replica=1,
            kv_cache_base_addr=[1000, 2000, 3000, 4000, 5000,
                                6000],  # 2 * total_layers
            use_mla=True,
            block_len=[1024, 2048],
            decode_tp_size=1,
            first_kv_cache=self.first_kv_cache,
            callback_func=MagicMock())

        self.req_meta_base = ReqMeta(
            local_block_ids=[5, 8],
            token_ids=[1, 2, 3],
            remote_block_ids=[10, 20],
            remote_engine_id="remote_engine",
            remote_host="127.0.0.1",
            remote_port=7777,
            remote_te_rpc_port=6000,
            remote_kv_caches_base_addr=[4000, 8000, 14000, 18000],
            metaserver="http://dummy")

    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.torch.Tensor.data_ptr",
        autospec=True,
        return_value=0x200000)
    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.align_memory",
           side_effect=lambda x, _align: x)
    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.torch.npu.synchronize"
    )
    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.group_concurrent_contiguous"
    )
    def test_transfer_pd_gt1_uses_buffers_and_calls_engine(
            self, mock_group, _mock_sync, _mock_align, _mock_dataptr):

        thread = KVCacheSendingLayerThread(
            engine=self.engine,
            total_layers=2,
            ready_event=self.ready_event,
            tp_rank=0,
            pd_head_ratio=2,
            num_head_replica=1,
            kv_cache_base_addr=[1111, 2222, 3333, 4444],
            use_mla=False,
            block_len=[64],
            decode_tp_size=1,
            first_kv_cache=self.first_kv_cache,
            callback_func=MagicMock())

        req_meta = self.req_meta_base
        req_meta.remote_kv_caches_base_addr = [4000, 8000]

        mock_group.return_value = ([[10, 11], [20, 21]], [])

        cap = self.first_kv_cache.numel() // self.first_kv_cache.shape[-1]
        dim = self.first_kv_cache.shape[-1]

        key = torch.zeros((cap, dim), dtype=torch.float32)
        value = torch.zeros((cap, dim), dtype=torch.float32)

        thread._transfer_kv_cache(req_id="req1",
                                  req_meta=req_meta,
                                  layer_index=0,
                                  key=key,
                                  value=value)

        self.engine.batch_transfer_sync_write.assert_called_once()
        session_id, src_list, dst_list, length_list = self.engine.batch_transfer_sync_write.call_args[
            0]
        self.assertEqual(session_id, "127.0.0.1:6000")

        self.assertEqual(len(src_list), 4)
        self.assertEqual(len(dst_list), 4)
        self.assertEqual(len(length_list), 4)

        for L in length_list:
            self.assertGreater(L, 0)
            self.assertEqual(L % 64, 0)

        remote_block_len = 64 * 2  # 128
        expected_offsets = [10 * remote_block_len, 20 * remote_block_len]
        self.assertEqual(dst_list[0] - 4000, expected_offsets[0])  # K, group1
        self.assertEqual(dst_list[1] - 4000, expected_offsets[1])  # K, group2
        self.assertEqual(dst_list[2] - 8000, expected_offsets[0])  # V, group1
        self.assertEqual(dst_list[3] - 8000, expected_offsets[1])  # V, group2)

    def test_transfer_skips_when_no_local_blocks(self):
        req_meta = self.req_meta_base
        req_meta.local_block_ids = []
        self.thread._transfer_kv_cache("req2", req_meta, 0, torch.zeros(
            (1, 8)), torch.zeros((1, 8)))
        self.engine.batch_transfer_sync_write.assert_not_called()

    def test_transfer_skips_when_tp_not_sender(self):

        thread = KVCacheSendingLayerThread(engine=self.engine,
                                           total_layers=2,
                                           ready_event=self.ready_event,
                                           tp_rank=1,
                                           pd_head_ratio=1,
                                           num_head_replica=2,
                                           kv_cache_base_addr=[1000, 2000],
                                           use_mla=False,
                                           block_len=[1024],
                                           decode_tp_size=1,
                                           first_kv_cache=self.first_kv_cache,
                                           callback_func=MagicMock())
        req_meta = self.req_meta_base
        thread._transfer_kv_cache("req3", req_meta, 0, torch.zeros((1, 8)),
                                  torch.zeros((1, 8)))
        self.engine.batch_transfer_sync_write.assert_not_called()

    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.group_concurrent_contiguous",
        side_effect=group_concurrent_contiguous)
    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.torch.npu.synchronize"
    )
    def test_callback_invoked_on_final_layer(self, _mock_sync, _mock_group):

        req_meta = self.req_meta_base
        req_meta.local_block_ids = [5, 6]
        req_meta.remote_block_ids = [10, 11]

        req_meta.remote_kv_caches_base_addr = [
            7000, 8000, 9000, 10000, 11000, 12000
        ]

        key = torch.zeros((1, 8), dtype=torch.float32)
        value = torch.zeros((1, 8), dtype=torch.float32)

        self.thread._transfer_kv_cache("req5",
                                       req_meta,
                                       layer_index=2,
                                       key=key,
                                       value=value)

        self.thread.callback_func.assert_called_once()


class TestKVCacheRecvingLayerThread(unittest.TestCase):

    def setUp(self):

        self.meta = MooncakeAgentMetadata(te_rpc_port=6000,
                                          kv_caches_base_addr=[0x1, 0x2])
        self.ready_event = threading.Event()

    def test_get_and_clear_finished_requests(self):
        th = KVCacheRecvingLayerThread(tp_rank=0,
                                       side_channel_port=5555,
                                       tp_size=2,
                                       pd_head_ratio=1,
                                       local_engine_id="engineA",
                                       metadata=self.meta,
                                       ready_event=self.ready_event)

        with th.lock:
            th.done_requests.update({"r1", "r2"})
        got = th.get_and_clear_finished_requests()
        self.assertEqual(got, {"r1", "r2"})

        got2 = th.get_and_clear_finished_requests()
        self.assertEqual(got2, set())

    def test_update_task_aggregates_by_pd_head_ratio(self):
        th = KVCacheRecvingLayerThread(tp_rank=0,
                                       side_channel_port=5555,
                                       tp_size=2,
                                       pd_head_ratio=2,
                                       local_engine_id="engineA",
                                       metadata=self.meta,
                                       ready_event=self.ready_event)

        with th.lock:
            th.task_tracker["reqX"] = 0

        th.update_task("reqX")
        with th.lock:
            self.assertIn("reqX", th.task_tracker)
            self.assertNotIn("reqX", th.done_requests)

        th.update_task("reqX")
        with th.lock:
            self.assertNotIn("reqX", th.task_tracker)
            self.assertIn("reqX", th.done_requests)

    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.logger")
    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.get_ip",
           return_value="127.0.0.1")
    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.make_zmq_socket")
    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.make_zmq_path",
        side_effect=lambda proto, host, port: f"{proto}://{host}:{port}")
    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.msgspec.msgpack.Decoder"
    )
    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.msgspec.msgpack.Encoder"
    )
    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.zmq_ctx")
    def test_run_loop_handles_meta_done_invalid_unexpected_and_ack(
            self, mock_zmq_ctx, mock_Encoder, mock_Decoder, _mock_make_path,
            _mock_make_sock, _mock_get_ip, mock_logger):

        enc_inst = MagicMock()
        enc_inst.encode.return_value = b"ENCODED_META"
        mock_Encoder.return_value = enc_inst

        dec_inst = MagicMock()
        dec_inst.decode.side_effect = [
            (GET_META_MSG, ),
            (DONE_SENDING_MSG, "reqA"),
            (b"weird_msg", ),
        ]
        mock_Decoder.return_value = dec_inst

        sock = MagicMock()

        sock.recv_multipart.side_effect = [
            [b"ID", b"SOME_PAYLOAD"],
            [b"ID", b"SOME_PAYLOAD2"],
            [b"ONLY_ID"],  # invalid
            [b"ID", b"SOME_PAYLOAD3"],
            SystemExit,
        ]

        cm = MagicMock()
        cm.__enter__.return_value = sock
        mock_zmq_ctx.return_value = cm

        ready_event = threading.Event()
        th = KVCacheRecvingLayerThread(tp_rank=1,
                                       side_channel_port=6000,
                                       tp_size=2,
                                       pd_head_ratio=1,
                                       local_engine_id="engineZ",
                                       metadata=self.meta,
                                       ready_event=ready_event)

        with th.lock:
            th.task_tracker["reqA"] = 0

        with self.assertRaises(SystemExit):
            th.run()

        self.assertTrue(ready_event.is_set())

        self.assertGreaterEqual(sock.send_multipart.call_count, 2)
        calls = [c.args for c in sock.send_multipart.call_args_list]

        meta_call = calls[0]
        self.assertEqual(meta_call[0][0], b"ID")
        self.assertEqual(meta_call[0][1], b"")
        self.assertEqual(meta_call[0][2], b"ENCODED_META")

        ack_call = calls[1]
        self.assertEqual(ack_call[0][0], b"ID")
        self.assertEqual(ack_call[0][1], b"")
        self.assertEqual(ack_call[0][2], b"ACK")

        self.assertTrue(mock_logger.error.called)

        finished = th.get_and_clear_finished_requests()
        self.assertIn("reqA", finished)

    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.logger")
    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.get_ip",
           return_value="127.0.0.1")
    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.msgspec.msgpack.Decoder"
    )
    @patch(
        "vllm_ascend.distributed.mooncake_layerwise_connector.msgspec.msgpack.Encoder"
    )
    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.zmq_ctx")
    def test_run_loop_pd_head_ratio_gt1_requires_multiple_done(
            self, mock_zmq_ctx, mock_Encoder, mock_Decoder, _mock_get_ip,
            _mock_logger):

        enc_inst = MagicMock()
        enc_inst.encode.return_value = b"ENC"
        mock_Encoder.return_value = enc_inst

        dec_inst = MagicMock()
        dec_inst.decode.side_effect = [
            (DONE_SENDING_MSG, "reqB"),
            (DONE_SENDING_MSG, "reqB"),
        ]
        mock_Decoder.return_value = dec_inst

        sock = MagicMock()
        sock.recv_multipart.side_effect = [
            [b"ID", b"PAY1"],
            [b"ID", b"PAY2"],
            SystemExit,
        ]
        cm = MagicMock()
        cm.__enter__.return_value = sock
        mock_zmq_ctx.return_value = cm

        th = KVCacheRecvingLayerThread(tp_rank=0,
                                       side_channel_port=5555,
                                       tp_size=2,
                                       pd_head_ratio=2,
                                       local_engine_id="engineY",
                                       metadata=self.meta,
                                       ready_event=self.ready_event)
        with th.lock:
            th.task_tracker["reqB"] = 0
        with self.assertRaises(SystemExit):
            th.run()

        finished = th.get_and_clear_finished_requests()
        self.assertIn("reqB", finished)


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
        self.parallel_config.data_parallel_size = 1
        self.parallel_config.data_parallel_rank = 0
        self.cache_config.block_size = 16

        self.kv_transfer_config.engine_id = "test_engine"
        self.kv_transfer_config.kv_port = 5000
        self.kv_transfer_config.is_kv_producer = True
        self.kv_transfer_config.is_kv_consumer = False
        self.kv_transfer_config.get_from_extra_config = MagicMock()
        self.kv_transfer_config.get_from_extra_config.side_effect = lambda k, d: {
            "prefill": {
                "tp_size": 2,
                "dp_size": 1
            },
            "decode": {
                "tp_size": 2,
                "dp_size": 1
            },
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

        self.all_token_ids = list(self.prompt_token_ids)


class TestMooncakeLayerwiseConnectorMetadata(unittest.TestCase):

    def test_add_new_req(self):
        meta = MooncakeLayerwiseConnectorMetadata()
        self.assertEqual(len(meta.requests), 0)

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

        self.scheduler._reqs_need_recv["req1"] = (request, [], [4, 5, 6])
        request.kv_transfer_params = {
            "remote_block_ids": [1, 2, 3],
            "remote_engine_id": "remote",
            "remote_host": "localhost",
            "remote_port": 5000,
        }

        meta = self.scheduler.build_connector_meta(MagicMock())
        self.assertIsInstance(meta, MooncakeLayerwiseConnectorMetadata)
        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests["req1"].local_block_ids, [4, 5, 6])
        self.assertEqual(meta.requests["req1"].remote_block_ids, [1, 2, 3])
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)


class _MockBlocks:

    def __init__(self, unhashed, block_ids_tuple=None):
        self._unhashed = list(unhashed)
        self._block_ids_tuple = block_ids_tuple if block_ids_tuple is not None else (
            [1, 2], )

    def get_unhashed_block_ids(self):
        return list(self._unhashed)

    def get_block_ids(self):

        return self._block_ids_tuple


class _MockSchedulerOutput:

    def __init__(self,
                 cached_req_ids=None,
                 cached_new_block_ids=None,
                 cached_num_computed=None,
                 new_reqs=None,
                 num_sched=None):
        self.scheduled_cached_reqs = SimpleNamespace(
            req_ids=cached_req_ids or [],
            new_block_ids=cached_new_block_ids or [],
            num_computed_tokens=cached_num_computed or [],
        )
        self.scheduled_new_reqs = new_reqs or []
        self.num_scheduled_tokens = num_sched or {}


class TestMooncakeLayerwiseConnectorScheduler_More(unittest.TestCase):

    def setUp(self):
        self.config = MockVllmConfig()
        self.scheduler = MooncakeLayerwiseConnectorScheduler(
            self.config, "test_engine")

    def test_get_num_new_matched_tokens_with_prefill_block_aligned(self):

        req = MockRequest("req_prefill",
                          prompt_token_ids=list(range(32)),
                          kv_transfer_params={"do_remote_prefill": True})
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(
            req, num_computed_tokens=16)
        self.assertEqual(tokens, 16)
        self.assertTrue(async_flag)

    def test_update_state_after_alloc_prefill_records_and_resets_flag(self):
        req = MockRequest("req_u1",
                          prompt_token_ids=list(range(24)),
                          kv_transfer_params={"do_remote_prefill": True})
        blocks = _MockBlocks(unhashed=[4, 5, 6])

        self.scheduler.update_state_after_alloc(req,
                                                blocks,
                                                num_external_tokens=8)
        self.assertIn("req_u1", self.scheduler._reqs_need_recv)
        record = self.scheduler._reqs_need_recv["req_u1"]
        self.assertIs(record[0], req)
        self.assertEqual(record[1], [])
        self.assertEqual(record[2], [4, 5, 6])
        self.assertFalse(req.kv_transfer_params.get("do_remote_prefill", True))

    def test_update_state_after_alloc_decode_records_send_layerwise(self):
        req = MockRequest("req_u2",
                          prompt_token_ids=list(range(10)),
                          kv_transfer_params={"do_remote_decode": True})
        blocks = _MockBlocks(unhashed=[], block_ids_tuple=([7, 8, 9], ))
        self.scheduler.update_state_after_alloc(req,
                                                blocks,
                                                num_external_tokens=0)
        self.assertIn("req_u2", self.scheduler._reqs_need_send_layerwise)
        total_tokens, local_block_ids, req_ref = self.scheduler._reqs_need_send_layerwise[
            "req_u2"]
        self.assertEqual(total_tokens, 10)
        self.assertEqual(local_block_ids, [7, 8, 9])
        self.assertIs(req_ref, req)

    def test_build_connector_meta_consumes_reqs_need_recv_and_clears(self):
        req = MockRequest("req_b1",
                          kv_transfer_params={
                              "remote_block_ids": [1, 2],
                              "remote_engine_id": "E",
                              "remote_host": "H",
                              "remote_port": 5555,
                              "remote_te_rpc_port": 6000,
                              "remote_kv_caches_base_addr": [10, 11],
                          })
        self.scheduler._reqs_need_recv["req_b1"] = (req, [], [100, 101])
        meta = self.scheduler.build_connector_meta(_MockSchedulerOutput())
        self.assertIsInstance(meta, MooncakeLayerwiseConnectorMetadata)
        self.assertIn("req_b1", meta.requests)
        self.assertEqual(meta.requests["req_b1"].local_block_ids, [100, 101])
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)

    def test_build_connector_meta_accumulates_cached_blocks(self):
        req = MockRequest("req_b2",
                          prompt_token_ids=list(range(8)),
                          kv_transfer_params={"do_remote_decode": True})

        self.scheduler._reqs_need_send_layerwise["req_b2"] = (8, [1, 2], req)

        out = _MockSchedulerOutput(
            cached_req_ids=["req_b2"],
            cached_new_block_ids=[([3, 4], )],
            cached_num_computed=[4],
            new_reqs=[],
            num_sched={},
        )
        meta = self.scheduler.build_connector_meta(out)
        self.assertEqual(len(meta.requests), 0)
        total, block_ids, _ = self.scheduler._reqs_need_send_layerwise[
            "req_b2"]
        self.assertEqual(total, 8)
        self.assertEqual(block_ids, [1, 2, 3, 4])

    def test_build_connector_meta_emits_when_tokens_reach_total(self):

        req = MockRequest("req_b3",
                          prompt_token_ids=list(range(12)),
                          kv_transfer_params={
                              "do_remote_decode": True,
                              "remote_block_ids": [9],
                              "remote_engine_id": "E",
                              "remote_host": "H",
                              "remote_port": 5555,
                              "remote_te_rpc_port": 6000,
                              "remote_kv_caches_base_addr": [10, 11],
                          })
        self.scheduler._reqs_need_send_layerwise["req_b3"] = (12, [100,
                                                                   101], req)

        out = _MockSchedulerOutput(
            cached_req_ids=["req_b3"],
            cached_new_block_ids=[([50], )],
            cached_num_computed=[8],
            new_reqs=[SimpleNamespace(req_id="other", num_computed_tokens=0)],
            num_sched={"req_b3": 4},
        )
        meta = self.scheduler.build_connector_meta(out)
        self.assertIn("req_b3", meta.requests)
        rmeta = meta.requests["req_b3"]

        self.assertEqual(rmeta.local_block_ids, [100, 101, 50])

        self.assertNotIn("req_b3", self.scheduler._reqs_need_send_layerwise)

    def test_request_finished_returns_false_none(self):
        ok, params = self.scheduler.request_finished(MockRequest("req_fin"),
                                                     [1, 2])
        self.assertFalse(ok)
        self.assertIsNone(params)


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
    def test_ensure_zmq_send_success(self, _):
        mock_socket = MagicMock()
        ensure_zmq_send(mock_socket, b"hello")
        mock_socket.send.assert_called_once_with(b"hello")

    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_send_retry_and_fail(self, _):
        mock_socket = MagicMock()
        mock_socket.send.side_effect = zmq.ZMQError(  # type: ignore
            "send failed")
        with self.assertRaises(RuntimeError):
            ensure_zmq_send(mock_socket, b"hello", max_retries=2)
        self.assertEqual(mock_socket.send.call_count, 2)

    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_recv_success(self, _):
        mock_socket = MagicMock()
        mock_socket.recv.return_value = b"response"
        mock_poller = MagicMock()
        mock_poller.poll.return_value = [
            (mock_socket, zmq.POLLIN)  # type: ignore
        ]
        data = ensure_zmq_recv(mock_socket, mock_poller)
        self.assertEqual(data, b"response")

    @patch("vllm_ascend.distributed.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_recv_timeout_and_fail(self, _):
        mock_socket = MagicMock()
        mock_poller = MagicMock()
        mock_poller.poll.return_value = []
        with self.assertRaises(RuntimeError):
            ensure_zmq_recv(mock_socket,
                            mock_poller,
                            timeout=0.01,
                            max_retries=2)


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


class TestMooncakeLayerwiseConnectorWorker(unittest.TestCase):

    def setUp(self):
        self.mock_transfer_engine = MagicMock()
        self.mock_transfer_engine.get_rpc_port.return_value = 9090
        self.mock_transfer_engine.initialize.return_value = 0
        self.mock_transfer_engine.register_memory.return_value = 0

        self.patches = [
            patch('torch.Tensor.size', return_value=(10, 16, 8, 16)),
            patch('torch.Tensor.element_size', return_value=4),
            patch('torch.Tensor.data_ptr', return_value=0x1000),
            patch('math.prod', return_value=128),
            patch('random.Random'),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.get_tensor_model_parallel_rank',
                return_value=0),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.get_tp_group',
                return_value=None),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.get_ip',
                return_value="127.0.0.1"),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.string_to_int64_hash',
                side_effect=lambda s: hash(s)),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.global_te.get_transfer_engine',
                return_value=self.mock_transfer_engine),
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.global_te.register_buffer',
                return_value=None),
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
            patch(
                'vllm_ascend.distributed.mooncake_layerwise_connector.get_ascend_config',
                return_value=SimpleNamespace(pd_tp_ratio=1,
                                             num_head_replica=1,
                                             pd_head_ratio=1)),
        ]

        for p in self.patches:
            p.start()  # type: ignore

        self.vllm_config = MockVllmConfig()
        self.engine_id = "test_engine"
        self.kv_caches = {"layer1": (MagicMock(), MagicMock())}

    def tearDown(self):
        for p in self.patches:
            p.stop()  # type: ignore

    def test_register_kv_caches_producer(self):

        self.vllm_config.kv_transfer_config.is_kv_producer = True
        self.vllm_config.kv_transfer_config.is_kv_consumer = False
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config,
                                                  self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertEqual(len(worker.kv_caches), 1)
        self.assertIsNotNone(worker.kv_send_layer_thread)
        self.assertIsNone(worker.kv_recv_layer_thread)

    def test_register_kv_caches_consumer(self):

        self.vllm_config.kv_transfer_config.is_kv_producer = False
        self.vllm_config.kv_transfer_config.is_kv_consumer = True
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
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config,
                                                  self.engine_id)
        self.assertIsNotNone(worker.engine)


if __name__ == '__main__':
    unittest.main()
