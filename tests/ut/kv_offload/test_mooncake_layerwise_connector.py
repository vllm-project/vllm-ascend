import contextlib
import importlib.util
import os
import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import zmq

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine
fake_torch_npu = types.ModuleType("torch_npu")
fake_torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
fake_torch_npu.npu = MagicMock()  # type: ignore[attr-defined]
fake_torch_npu.npu.current_device = MagicMock(return_value=0)  # type: ignore[attr-defined]
fake_torch_npu.npu.Stream = MagicMock  # type: ignore[attr-defined]
fake_torch_npu.npu_fusion_attention = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("torch_npu", fake_torch_npu)
torch.npu = fake_torch_npu.npu  # type: ignore[attr-defined]
fake_uvloop = types.ModuleType("uvloop")
fake_uvloop.__spec__ = importlib.util.spec_from_loader("uvloop", loader=None)
sys.modules.setdefault("uvloop", fake_uvloop)

# Clean up stale mock modules installed by other test files
# (e.g., ascend_store/_mock_deps.py) that replace real kv_transfer
# subpackages with MagicMock/fake modules, breaking our imports.
# We save the removed modules so we can restore them after our imports
# complete, so other test files (ascend_store) still see their mocks.
_kv_xfer = "vllm_ascend.distributed.kv_transfer"
_vllm_kv_xfer = "vllm.distributed.kv_transfer"
_saved_modules: dict[str, types.ModuleType] = {}
_to_remove = []
for k in list(sys.modules):
    if k.startswith(_kv_xfer):
        suffix = k[len(_kv_xfer) :]
        if suffix == "" or suffix.startswith(".utils") or suffix.startswith(".kv_p2p"):
            _to_remove.append(k)
    elif k.startswith(_vllm_kv_xfer):
        _to_remove.append(k)
for _m in _to_remove:
    _saved_modules[_m] = sys.modules.pop(_m)

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (  # noqa: E402
    KVCacheRecvingLayerThread,
    KVCacheSendingLayerThread,
    KVConnectorRole,
    LayerMetadata,
    MooncakeAgentMetadata,
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
    ReqMeta,
    SendReqInfo,
    SendTask,
    ensure_zmq_recv,
    ensure_zmq_send,
    group_concurrent_contiguous,
    string_to_int64_hash,
    zmq_ctx,
)

# Restore the mocked modules so other test files still work correctly.
# For keys that our real import loaded, overwrite with the saved mock.
for _k, _v in _saved_modules.items():
    sys.modules[_k] = _v

GET_META_MSG = b"get_meta_msg"
DONE_SENDING_MSG = b"done_sending_msg"


def _make_layer_metadata(**overrides):
    defaults = dict(
        tensor_group_idx=[0],
        kv_caches_base_addr=[1000, 2000],
        block_len=[1024],
        block_size_scale=[1],
    )
    defaults.update(overrides)
    return LayerMetadata(**defaults)


def _make_mock_kv_cache_config(block_size=16):
    kv_cache_spec = MagicMock()
    kv_cache_spec.block_size = block_size
    group_spec = MagicMock()
    group_spec.kv_cache_spec = kv_cache_spec
    group_spec.layer_names = ["layer0"]
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = [group_spec]
    return kv_cache_config


class TestKVCacheSendingLayerThread(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.engine.register_memory.return_value = 0
        self.engine.batch_transfer_sync_write.return_value = 1
        fake_stream = MagicMock(name="FakeStream")
        fake_stream.synchronize = MagicMock()

        self.first_kv_cache = torch.zeros((2, 2, 2, 8), dtype=torch.float32, device="cpu")

        self.ready_event = threading.Event()

        self.fake_k_buffer = MagicMock()
        self.fake_v_buffer = MagicMock()
        fake_resharding_stream = MagicMock()

        self.layer_metadata = {
            "layer0": _make_layer_metadata(
                tensor_group_idx=[0],
                kv_caches_base_addr=[1000, 2000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
            "layer1": _make_layer_metadata(
                tensor_group_idx=[0],
                kv_caches_base_addr=[3000, 4000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
            "layer2": _make_layer_metadata(
                tensor_group_idx=[0],
                kv_caches_base_addr=[5000, 6000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
        }

        self.vllm_config = MagicMock()
        self.vllm_config.cache_config.mamba_cache_mode = None
        self.vllm_config.speculative_config = None

        self.kv_cache_config = _make_mock_kv_cache_config()
        self.kv_cache_specs = [MagicMock(block_size=16)]

        self.key = torch.zeros((4, 8), dtype=torch.float32)
        self.value = torch.zeros((4, 8), dtype=torch.float32)
        self.thread = KVCacheSendingLayerThread(
            engine=self.engine,
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
            kv_cache_specs=self.kv_cache_specs,
            attn_resharding_group_idx=set(),
            total_layers=3,
            ready_event=self.ready_event,
            tp_size=1,
            tp_rank=0,
            pd_head_ratio=1,
            num_head_replica=1,
            layer_metadata=self.layer_metadata,
            use_mla=True,
            use_attn_mamba_hybrid=False,
            k_buffer=self.fake_k_buffer,
            v_buffer=self.fake_v_buffer,
            enable_kv_quant=False,
            enable_c8_quant=False,
            resharding_stream=fake_resharding_stream,
            callback_func=MagicMock(),
        )

        self.req_meta_base = ReqMeta(
            local_block_ids=[[5, 8]],
            token_ids=[1, 2, 3],
            remote_block_ids=[[10, 20]],
            remote_block_size=[[16]],
            remote_engine_id="remote_engine",
            remote_host="127.0.0.1",
            remote_port=7777,
            remote_te_rpc_port=6000,
            remote_layer_metadata={
                "layer0": _make_layer_metadata(
                    kv_caches_base_addr=[4000, 8000],
                    block_len=[64, 64],
                    block_size_scale=[1, 1],
                ),
            },
            metaserver="http://dummy",
            remote_tp_size=8,
            remote_pcp_size=1,
            remote_dcp_size=1,
            chunk_finish=False,
        )

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.npu_stream_switch",
        side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.torch.Tensor.data_ptr",
        autospec=True,
        return_value=0x200000,
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.align_memory",
        side_effect=lambda x, _align: x,
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.torch.npu.synchronize")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.group_concurrent_contiguous")
    def test_transfer_pd_gt1_uses_buffers_and_calls_engine(
        self, mock_group, _mock_sync, _mock_align, _mock_dataptr, mock_stream_switch
    ):
        fake_resharding_stream = MagicMock()

        layer_metadata = {
            "layer0": _make_layer_metadata(
                tensor_group_idx=[0],
                kv_caches_base_addr=[1111, 2222],
                block_len=[64, 64],
                block_size_scale=[1, 1],
            ),
        }

        vllm_config = MagicMock()
        vllm_config.cache_config.mamba_cache_mode = None
        vllm_config.speculative_config = None

        kv_cache_config = _make_mock_kv_cache_config()
        kv_cache_specs = [MagicMock(block_size=16)]

        thread = KVCacheSendingLayerThread(
            engine=self.engine,
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            kv_cache_specs=kv_cache_specs,
            attn_resharding_group_idx=set(),
            total_layers=2,
            ready_event=self.ready_event,
            tp_size=1,
            tp_rank=0,
            pd_head_ratio=2,
            num_head_replica=1,
            layer_metadata=layer_metadata,
            use_mla=False,
            use_attn_mamba_hybrid=False,
            k_buffer=self.fake_k_buffer,
            v_buffer=self.fake_v_buffer,
            enable_kv_quant=False,
            enable_c8_quant=False,
            resharding_stream=fake_resharding_stream,
            callback_func=MagicMock(),
        )

        req_meta = self.req_meta_base
        req_meta.remote_block_ids = [[10, 20]]
        req_meta.remote_layer_metadata = {
            "layer0": _make_layer_metadata(
                kv_caches_base_addr=[4000, 8000],
                block_len=[64, 64],
                block_size_scale=[1, 1],
            ),
        }

        mock_group.return_value = ([[10, 11], [20, 21]], [])
        key = torch.zeros((1, 8), dtype=torch.float32)
        value = torch.zeros((1, 8), dtype=torch.float32)

        send_task = SendTask(
            send_request={"req1": req_meta},
            wait_event=MagicMock(),
            k_cache=key,
            v_cache=value,
            layer_idx=0,
            layer_name="layer0",
            group_rearrange_block_ids=[[5, 8]],
        )

        thread._transfer_kv_cache(send_task)

        self.engine.batch_transfer_sync_write.assert_called_once()
        session_id, src_list, dst_list, length_list = self.engine.batch_transfer_sync_write.call_args[0]
        self.assertEqual(session_id, "127.0.0.1:6000")

        self.assertEqual(len(src_list), 4)
        self.assertEqual(len(dst_list), 4)
        self.assertEqual(len(length_list), 4)

        for L in length_list:
            self.assertGreater(L, 0)
            self.assertEqual(L % 64, 0)

        remote_block_len = 64
        expected_offsets = [10 * remote_block_len, 20 * remote_block_len]
        self.assertEqual(dst_list[0] - 4000, expected_offsets[0])
        self.assertEqual(dst_list[1] - 4000, expected_offsets[1])
        self.assertEqual(dst_list[2] - 8000, expected_offsets[0])
        self.assertEqual(dst_list[3] - 8000, expected_offsets[1])

    def test_transfer_skips_when_no_local_blocks(self):
        req_meta = self.req_meta_base
        req_meta.local_block_ids = [[]]
        send_task = SendTask(
            send_request={"req2": req_meta},
            wait_event=MagicMock(),
            k_cache=torch.zeros((1, 8)),
            v_cache=torch.zeros((1, 8)),
            layer_idx=0,
            layer_name="layer0",
            group_rearrange_block_ids=[[]],
        )
        self.thread._transfer_kv_cache(send_task)
        self.engine.batch_transfer_sync_write.assert_not_called()

    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.group_concurrent_contiguous",
        side_effect=group_concurrent_contiguous,
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.torch.npu.synchronize")
    def test_callback_invoked_on_final_layer(self, _mock_sync, _mock_group):
        req_meta = self.req_meta_base
        req_meta.chunk_finish = True
        req_meta.local_block_ids = [[5, 6]]
        req_meta.remote_block_ids = [[10, 11]]
        req_meta.remote_layer_metadata = {
            "layer0": _make_layer_metadata(
                kv_caches_base_addr=[7000, 8000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
            "layer1": _make_layer_metadata(
                kv_caches_base_addr=[9000, 10000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
            "layer2": _make_layer_metadata(
                kv_caches_base_addr=[11000, 12000],
                block_len=[1024, 2048],
                block_size_scale=[1, 1],
            ),
        }

        key = torch.zeros((1, 8), dtype=torch.float32)
        value = torch.zeros((1, 8), dtype=torch.float32)

        send_task = SendTask(
            send_request={"req5": req_meta},
            wait_event=MagicMock(),
            k_cache=key,
            v_cache=value,
            layer_idx=2,
            layer_name="layer2",
            group_rearrange_block_ids=[[]],
        )
        self.thread._transfer_kv_cache(send_task)

        self.thread.callback_func.assert_called_once()


class TestKVCacheRecvingLayerThread(unittest.TestCase):
    def setUp(self):
        self.meta = MooncakeAgentMetadata(
            te_rpc_port=6000,
            layer_metadata={"layer0": _make_layer_metadata()},
        )
        self.ready_event = threading.Event()

    def test_get_and_clear_done_requests(self):
        th = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=2,
            pd_head_ratio=1,
            local_engine_id="engineA",
            metadata=self.meta,
            ready_event=self.ready_event,
        )

        with th.lock:
            th.done_requests.update({"r1", "r2"})
        got = th.get_and_clear_done_requests()
        self.assertEqual(got, {"r1", "r2"})

        got2 = th.get_and_clear_done_requests()
        self.assertEqual(got2, set())

    def test_get_and_clear_failed_requests(self):
        th = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=2,
            pd_head_ratio=1,
            local_engine_id="engineA",
            metadata=self.meta,
            ready_event=self.ready_event,
        )

        with th.lock:
            th.failed_requests.update({"r1", "r2"})
        got = th.get_and_clear_failed_requests()
        self.assertEqual(got, {"r1", "r2"})

        got2 = th.get_and_clear_failed_requests()
        self.assertEqual(got2, set())

    def test_update_failed_task_aggregates_by_pd_head_ratio(self):
        th = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=2,
            pd_head_ratio=2,
            local_engine_id="engineA",
            metadata=self.meta,
            ready_event=self.ready_event,
        )

        with th.lock:
            th.task_tracker["reqX"] = set()
            th.request_map = MagicMock()

        th.update_failed_task("reqX")
        with th.lock:
            self.assertNotIn("reqX", th.task_tracker)
            self.assertIn("reqX", th.failed_requests)

    def test_update_done_task_aggregates_by_pd_head_ratio(self):
        th = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=2,
            pd_head_ratio=2,
            local_engine_id="engineA",
            metadata=self.meta,
            ready_event=self.ready_event,
        )

        with th.lock:
            th.task_tracker["reqX"] = set()

        th.update_done_task("reqX", 2, "path1")
        with th.lock:
            self.assertIn("reqX", th.task_tracker)
            self.assertNotIn("reqX", th.done_requests)

        th.update_done_task("reqX", 2, "path2")
        with th.lock:
            self.assertNotIn("reqX", th.task_tracker)
            self.assertIn("reqX", th.done_requests)

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_ip", return_value="127.0.0.1")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.make_zmq_socket")
    @patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.make_zmq_path",
        side_effect=lambda proto, host, port: f"{proto}://{host}:{port}",
    )
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.msgspec.msgpack.Decoder")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.msgspec.msgpack.Encoder")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.zmq_ctx")
    def test_run_loop_handles_meta_done_invalid_unexpected_and_ack(
        self, mock_zmq_ctx, mock_Encoder, mock_Decoder, _mock_make_path, _mock_make_sock, _mock_get_ip, mock_logger
    ):
        enc_inst = MagicMock()
        enc_inst.encode.return_value = b"ENCODED_META"
        mock_Encoder.return_value = enc_inst

        dec_inst = MagicMock()
        dec_inst.decode.side_effect = [
            (GET_META_MSG,),
            (DONE_SENDING_MSG, "reqA", 1, "path1"),
            (b"weird_msg",),
        ]
        mock_Decoder.return_value = dec_inst

        sock = MagicMock()

        sock.recv_multipart.side_effect = [
            [b"ID", b"SOME_PAYLOAD"],
            [b"ID", b"SOME_PAYLOAD2"],
            [b"ONLY_ID"],
            [b"ID", b"SOME_PAYLOAD3"],
            SystemExit,
        ]

        cm = MagicMock()
        cm.__enter__.return_value = sock
        mock_zmq_ctx.return_value = cm

        ready_event = threading.Event()
        th = KVCacheRecvingLayerThread(
            tp_rank=1,
            side_channel_port=6000,
            tp_size=2,
            pd_head_ratio=1,
            local_engine_id="engineZ",
            metadata=self.meta,
            ready_event=ready_event,
        )

        with th.lock:
            th.task_tracker["reqA"] = set()

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

        finished = th.get_and_clear_done_requests()
        self.assertIn("reqA", finished)

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_ip", return_value="127.0.0.1")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.msgspec.msgpack.Decoder")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.msgspec.msgpack.Encoder")
    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.zmq_ctx")
    def test_run_loop_pd_head_ratio_gt1_requires_multiple_done(
        self, mock_zmq_ctx, mock_Encoder, mock_Decoder, _mock_get_ip, _mock_logger
    ):
        enc_inst = MagicMock()
        enc_inst.encode.return_value = b"ENC"
        mock_Encoder.return_value = enc_inst

        dec_inst = MagicMock()
        dec_inst.decode.side_effect = [
            (DONE_SENDING_MSG, "reqB", 2, "path1"),
            (DONE_SENDING_MSG, "reqB", 2, "path2"),
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

        th = KVCacheRecvingLayerThread(
            tp_rank=0,
            side_channel_port=5555,
            tp_size=2,
            pd_head_ratio=2,
            local_engine_id="engineY",
            metadata=self.meta,
            ready_event=self.ready_event,
        )
        with th.lock:
            th.task_tracker["reqB"] = set()
        with self.assertRaises(SystemExit):
            th.run()
        finished = th.get_and_clear_done_requests()
        self.assertIn("reqB", finished)


class MockVllmConfig:
    def __init__(self):
        self.model_config = MagicMock()
        self.parallel_config = MagicMock()
        self.cache_config = MagicMock()
        self.kv_transfer_config = MagicMock()
        self.speculative_config = None
        self.quant_config = None
        self.model_config.use_mla = True
        self.parallel_config.tensor_parallel_size = 2
        self.parallel_config.data_parallel_rank_local = 0
        self.parallel_config.data_parallel_size_local = 1
        self.parallel_config.data_parallel_size = 1
        self.parallel_config.data_parallel_rank = 0
        self.parallel_config.prefill_context_parallel_size = 1
        self.parallel_config.decode_context_parallel_size = 1
        self.cache_config.block_size = 16
        self.cache_config.mamba_cache_mode = None
        self.model_config.hf_config.num_key_value_heads = 1
        self.model_config.get_num_layers = MagicMock(return_value=1)
        self.model_config.get_total_num_kv_heads = MagicMock(return_value=1)
        self.model_config.hf_text_config = MagicMock()
        self.model_config.hf_text_config.model_type = "default"

        self.kv_transfer_config.engine_id = "test_engine"
        self.kv_transfer_config.kv_port = 5000
        self.kv_transfer_config.is_kv_producer = True
        self.kv_transfer_config.is_kv_consumer = False
        self.kv_transfer_config.get_from_extra_config = MagicMock()
        self.kv_transfer_config.get_from_extra_config.side_effect = lambda k, d: {
            "prefill": {"tp_size": 2, "dp_size": 1},
            "decode": {"tp_size": 2, "dp_size": 1},
        }.get(k, d)


class MockKVCacheConfig:
    def __init__(self, block_size=16):
        kv_cache_spec = MagicMock()
        kv_cache_spec.block_size = block_size
        group_spec = MagicMock()
        group_spec.kv_cache_spec = kv_cache_spec
        group_spec.layer_names = ["encoder.layer.0"]
        self.kv_cache_groups = [group_spec]
        self.kv_cache_tensors = []
        self.num_blocks = 10


class MockRequest:
    def __init__(self, request_id, prompt_token_ids=None, kv_transfer_params=None, status=None):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids or [1, 2, 3, 4]
        self.prompt_embeds = None
        self.kv_transfer_params = kv_transfer_params or {}
        self.status = status or "running"
        self.output_token_ids = [101, 102]
        self.num_computed_tokens = 0
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self.max_tokens = 16

        self.all_token_ids = list(self.prompt_token_ids)
        self._all_token_ids = list(self.prompt_token_ids)


class TestMooncakeLayerwiseConnectorMetadata(unittest.TestCase):
    def test_add_new_req(self):
        meta = MooncakeLayerwiseConnectorMetadata()
        self.assertEqual(len(meta.requests), 0)

        meta.add_new_req(
            request_id="req1",
            local_block_ids=[[1, 2, 3]],
            kv_transfer_params={
                "remote_block_ids": [[4, 5, 6]],
                "remote_block_size": [[16]],
                "remote_engine_id": "remote_engine",
                "remote_host": "localhost",
                "remote_port": 5000,
            },
        )

        self.assertEqual(len(meta.requests), 1)
        req_meta = meta.requests["req1"]
        self.assertIsInstance(req_meta, ReqMeta)
        self.assertEqual(req_meta.local_block_ids, [[1, 2, 3]])
        self.assertEqual(req_meta.remote_block_ids, [[4, 5, 6]])
        self.assertEqual(req_meta.remote_engine_id, "remote_engine")
        self.assertEqual(req_meta.remote_host, "localhost")
        self.assertEqual(req_meta.remote_port, 5000)


class TestMooncakeLayerwiseConnectorSchedulerMatchedTokens(unittest.TestCase):
    def setUp(self):
        config = MockVllmConfig()
        kv_cache_config = MockKVCacheConfig()
        self.scheduler = MooncakeLayerwiseConnectorScheduler(config, kv_cache_config, "test_engine")

    def test_get_num_new_matched_tokens(self):
        request = MockRequest("req1")
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(tokens, 0)
        self.assertFalse(async_flag)

        request.kv_transfer_params = {"do_remote_prefill": True}
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)
        self.assertEqual(tokens, 4)
        self.assertTrue(async_flag)

    def test_get_num_new_matched_tokens_hybrid_excludes_last_token(self):
        self.scheduler.need_truncate = True
        request = MockRequest("req1", prompt_token_ids=list(range(17)), kv_transfer_params={"do_remote_prefill": True})

        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)

        self.assertEqual(tokens, 16)
        self.assertTrue(async_flag)

    def test_get_num_new_matched_tokens_hybrid_truncates_prefill_request(self):
        self.scheduler.need_truncate = True
        request = MockRequest("req1", prompt_token_ids=list(range(4)), kv_transfer_params={"do_remote_decode": True})

        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(request, 0)

        self.assertEqual(tokens, 0)
        self.assertFalse(async_flag)
        self.assertEqual(request.prompt_token_ids, [0, 1, 2])
        self.assertEqual(request._all_token_ids, [0, 1, 2])
        self.assertEqual(request.num_prompt_tokens, 3)
        self.assertEqual(request.max_tokens, 1)
        self.assertTrue(request.kv_transfer_params["_p_side_truncated"])

    def test_build_connector_meta(self):
        self.scheduler.vllm_config.kv_transfer_config.is_kv_consumer = True
        request = MockRequest("req1")

        self.scheduler._reqs_need_recv["req1"] = (request, [], [[4, 5, 6]])
        request.kv_transfer_params = {
            "remote_block_ids": [[1, 2, 3]],
            "remote_block_size": [[16]],
            "remote_engine_id": "remote",
            "remote_host": "localhost",
            "remote_port": 5000,
        }

        meta = self.scheduler.build_connector_meta(MagicMock())
        self.assertIsInstance(meta, MooncakeLayerwiseConnectorMetadata)
        self.assertEqual(len(meta.requests), 1)
        self.assertEqual(meta.requests["req1"].local_block_ids, [[4, 5, 6]])
        self.assertEqual(meta.requests["req1"].remote_block_ids, [[1, 2, 3]])
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)

    def test_update_state_after_alloc_hybrid_trims_remote_block_with_only_last_token(self):
        self.scheduler.need_truncate = True
        request = MockRequest(
            "req1",
            prompt_token_ids=list(range(17)),
            kv_transfer_params={"do_remote_prefill": True, "metaserver": "http://meta"},
        )
        blocks = _MockBlocks(unhashed=[], block_ids_tuple=([4, 5],))
        self.scheduler.executor.submit = MagicMock()

        self.scheduler.update_state_after_alloc(request, blocks, num_external_tokens=16)

        _, kwargs = self.scheduler.executor.submit.call_args
        self.assertEqual(kwargs["message"]["remote_block_ids"], ([4],))


class _MockBlocks:
    def __init__(self, unhashed, block_ids_tuple=None):
        self._unhashed = list(unhashed)
        self._block_ids_tuple = block_ids_tuple if block_ids_tuple is not None else ([1, 2],)

    def get_unhashed_block_ids(self):
        return list(self._unhashed)

    def get_block_ids(self):
        return self._block_ids_tuple


class _MockSchedulerOutput:
    def __init__(
        self,
        cached_req_ids=None,
        cached_new_block_ids=None,
        cached_num_computed=None,
        new_reqs=None,
        num_sched=None,
        scheduled_spec_decode_tokens=None,
    ):
        self.scheduled_cached_reqs = SimpleNamespace(
            req_ids=cached_req_ids or [],
            new_block_ids=cached_new_block_ids or [],
            num_computed_tokens=cached_num_computed or [],
        )
        self.scheduled_spec_decode_tokens = scheduled_spec_decode_tokens or {}
        self.scheduled_new_reqs = new_reqs or []
        self.num_scheduled_tokens = num_sched or {}


class TestMooncakeLayerwiseConnectorScheduler_More(unittest.TestCase):
    def setUp(self):
        self.config = MockVllmConfig()
        self.kv_cache_config = MockKVCacheConfig()
        self.scheduler = MooncakeLayerwiseConnectorScheduler(self.config, self.kv_cache_config, "test_engine")

    def test_get_num_new_matched_tokens_with_prefill_block_aligned(self):
        req = MockRequest(
            "req_prefill", prompt_token_ids=list(range(32)), kv_transfer_params={"do_remote_prefill": True}
        )
        tokens, async_flag = self.scheduler.get_num_new_matched_tokens(req, num_computed_tokens=16)
        self.assertEqual(tokens, 16)
        self.assertTrue(async_flag)

    def test_update_state_after_alloc_prefill_records_and_resets_flag(self):
        req = MockRequest("req_u1", prompt_token_ids=list(range(24)), kv_transfer_params={"do_remote_prefill": True})
        req.num_computed_tokens = 0
        blocks = _MockBlocks(unhashed=[4, 5, 6], block_ids_tuple=([[4, 5, 6]],))

        self.scheduler.update_state_after_alloc(req, blocks, num_external_tokens=8)
        self.assertIn("req_u1", self.scheduler._reqs_need_recv)
        record = self.scheduler._reqs_need_recv["req_u1"]
        self.assertIs(record[0], req)
        self.assertEqual(record[1], [])
        self.assertEqual(record[2], ([[4, 5, 6]],))
        self.assertFalse(req.kv_transfer_params.get("do_remote_prefill", True))

    def test_update_state_after_alloc_decode_records_send_layerwise(self):
        req = MockRequest(
            "req_u2",
            prompt_token_ids=list(range(10)),
            kv_transfer_params={"do_remote_decode": True, "remote_block_ids": [], "remote_cached_tokens": 0},
        )
        blocks = _MockBlocks(unhashed=[], block_ids_tuple=([[7, 8, 9]],))
        self.scheduler.update_state_after_alloc(req, blocks, num_external_tokens=0)
        self.assertIn("req_u2", self.scheduler._reqs_need_send_layerwise)
        info = self.scheduler._reqs_need_send_layerwise["req_u2"]
        self.assertEqual(info.local_block_ids, [[[7, 8, 9]]])
        self.assertIs(info.request, req)

    def test_build_connector_meta_consumes_reqs_need_recv_and_clears(self):
        self.scheduler.vllm_config.kv_transfer_config.is_kv_consumer = True
        req = MockRequest(
            "req_b1",
            kv_transfer_params={
                "remote_block_ids": [[1, 2]],
                "remote_block_size": [[16]],
                "remote_engine_id": "E",
                "remote_host": "H",
                "remote_port": 5555,
                "remote_te_rpc_port": 6000,
                "remote_layer_metadata": {"layer0": _make_layer_metadata()},
            },
        )
        self.scheduler._reqs_need_recv["req_b1"] = (req, [], [[100, 101]])
        meta = self.scheduler.build_connector_meta(_MockSchedulerOutput())
        self.assertIsInstance(meta, MooncakeLayerwiseConnectorMetadata)
        self.assertIn("req_b1", meta.requests)
        self.assertEqual(meta.requests["req_b1"].local_block_ids, [[100, 101]])
        self.assertEqual(len(self.scheduler._reqs_need_recv), 0)

    def test_build_connector_meta_accumulates_cached_blocks(self):
        req_meta = MagicMock(spec=SendReqInfo)
        req_meta.local_block_ids = [[1, 2, 3]]
        req_meta.local_transferred_tokens = 50
        req_meta.local_computed_tokens = 75
        req_meta.request = MagicMock()
        req_meta.extend_local_block_ids = MagicMock()
        req_meta.update_computed_tokens = MagicMock()
        req_meta.update_transferred_tokens = MagicMock()
        req_meta.unpack = MagicMock(
            return_value=(
                req_meta.local_block_ids,
                req_meta.local_transferred_tokens,
                req_meta.local_computed_tokens,
                req_meta.request,
            )
        )

        self.scheduler._reqs_need_send_layerwise["req_b2"] = req_meta

        out = _MockSchedulerOutput(
            cached_req_ids=["req_b2"],
            cached_new_block_ids=[([[3, 4]],)],
            cached_num_computed=[4],
            new_reqs=[],
            num_sched={},
        )
        meta = self.scheduler.build_connector_meta(out)
        self.assertEqual(len(meta.requests), 0)

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.group_concurrent_contiguous")
    def test_build_connector_meta_emits_when_tokens_reach_total(self, mock_group_concurrent_contiguous):
        send_req_info = MagicMock(spec=SendReqInfo)
        send_req_info.local_block_ids = [[1, 2, 3]]
        send_req_info.local_transferred_tokens = 50
        send_req_info.local_computed_tokens = 75
        send_req_info.request = MagicMock()
        send_req_info.request.kv_transfer_params = {
            "remote_block_ids": [[4, 5]],
            "remote_block_size": [[16]],
            "remote_cached_tokens": 100,
        }
        send_req_info.request.all_token_ids = list(range(80))
        send_req_info.extend_local_block_ids = MagicMock()
        send_req_info.update_computed_tokens = MagicMock()
        send_req_info.update_transferred_tokens = MagicMock()
        send_req_info.unpack = MagicMock(
            return_value=(
                send_req_info.local_block_ids,
                send_req_info.local_transferred_tokens,
                send_req_info.local_computed_tokens,
                send_req_info.request,
            )
        )

        self.scheduler._reqs_need_send_layerwise["req_b3"] = send_req_info
        out = _MockSchedulerOutput(
            cached_req_ids=["req_b3"],
            cached_new_block_ids=[([[50]],)],
            cached_num_computed=[8],
            new_reqs=[MagicMock(req_id="other", num_computed_tokens=0)],
            num_sched={"req_b3": 4},
        )
        meta = self.scheduler.build_connector_meta(out)
        send_req_info.extend_local_block_ids.assert_called_once_with(([[50]],))
        self.assertIn("req_b3", meta.requests)

    def test_request_finished_returns_false_none(self):
        ok, params = self.scheduler.request_finished(MockRequest("req_fin"), [1, 2])
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
        with self.assertRaises(ValueError), zmq_ctx("INVALID", "tcp://127.0.0.1:5555"):
            pass

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.make_zmq_socket")
    def test_zmq_ctx_ok(self, mock_make_socket):
        mock_socket = MagicMock()
        mock_make_socket.return_value = mock_socket
        with zmq_ctx(zmq.REQ, "tcp://localhost:1234") as s:  # type: ignore
            self.assertEqual(s, mock_socket)

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_send_success(self, _):
        mock_socket = MagicMock()
        path = "127.0.0.1:12345"
        ensure_zmq_send(mock_socket, b"hello", path)
        mock_socket.send.assert_called_once_with(b"hello")

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_send_retry_and_fail(self, _):
        mock_socket = MagicMock()
        path = "127.0.0.1:12345"
        mock_socket.send.side_effect = zmq.ZMQError(  # type: ignore
            "send failed"
        )
        with self.assertRaises(RuntimeError):
            ensure_zmq_send(mock_socket, b"hello", path, max_retries=2)
        self.assertEqual(mock_socket.send.call_count, 2)

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_recv_success(self, _):
        mock_socket = MagicMock()
        mock_socket.recv.return_value = b"response"
        mock_poller = MagicMock()
        mock_poller.poll.return_value = [
            (mock_socket, zmq.POLLIN)  # type: ignore
        ]
        path = "127.0.0.1:12345"
        data = ensure_zmq_recv(mock_socket, mock_poller, path)
        self.assertEqual(data, b"response")

    @patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger")
    def test_ensure_zmq_recv_timeout_and_fail(self, _):
        mock_socket = MagicMock()
        mock_poller = MagicMock()
        mock_poller.poll.return_value = []
        path = "127.0.0.1:12345"
        with self.assertRaises(RuntimeError):
            ensure_zmq_recv(mock_socket, mock_poller, path, timeout=0.01, max_retries=2)


class TestMooncakeLayerwiseConnectorForScheduler(unittest.TestCase):
    def _make_config(self):
        config = MockVllmConfig()
        kv_cache_config = MockKVCacheConfig()
        return config, kv_cache_config

    def test_scheduler_role(self):
        config, kv_cache_config = self._make_config()
        connector = MooncakeLayerwiseConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "get_num_new_matched_tokens")
    def test_scheduler_methods(self, mock_method):
        config, kv_cache_config = self._make_config()
        connector = MooncakeLayerwiseConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
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
        self.kv_cache_config = MockKVCacheConfig()
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

    def test_scheduler_initialization(self):
        connector = MooncakeLayerwiseConnector(self.config, KVConnectorRole.SCHEDULER, self.kv_cache_config)
        self.assertIsNotNone(connector.connector_scheduler)
        self.assertIsNone(connector.connector_worker)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "get_num_new_matched_tokens")
    def test_get_num_new_matched_tokens(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config, KVConnectorRole.SCHEDULER, self.kv_cache_config)
        request = MockRequest("req1")
        connector.get_num_new_matched_tokens(request, 0)
        mock_method.assert_called_once_with(request, 0)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "update_state_after_alloc")
    def test_update_state_after_alloc(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config, KVConnectorRole.SCHEDULER, self.kv_cache_config)
        request = MockRequest("req1")
        blocks = MockKVCacheBlocks()
        connector.update_state_after_alloc(request, blocks, 3)
        mock_method.assert_called_once_with(request, blocks, 3)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "build_connector_meta")
    def test_build_connector_meta(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config, KVConnectorRole.SCHEDULER, self.kv_cache_config)
        scheduler_output = MockSchedulerOutput()
        connector.build_connector_meta(scheduler_output)
        mock_method.assert_called_once_with(scheduler_output)

    @patch.object(MooncakeLayerwiseConnectorScheduler, "request_finished")
    def test_request_finished(self, mock_method):
        connector = MooncakeLayerwiseConnector(self.config, KVConnectorRole.SCHEDULER, self.kv_cache_config)
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
            patch("torch.Tensor.size", return_value=(10, 16, 8, 16)),
            patch("torch.Tensor.element_size", return_value=4),
            patch("torch.Tensor.data_ptr", return_value=0x1000),
            patch("math.prod", return_value=128),
            patch("random.Random"),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_tp_group",
                return_value=None,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_ip",
                return_value="127.0.0.1",
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.string_to_int64_hash",
                side_effect=lambda s: hash(s),
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.global_te.get_transfer_engine",
                return_value=self.mock_transfer_engine,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.global_te.register_buffer",
                return_value=None,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.KVCacheSendingLayerThread",
                MagicMock(),
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.KVCacheRecvingLayerThread",
                MagicMock(),
            ),
            patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.logger", MagicMock()),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.threading.Event", MagicMock()
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_ascend_config",
                return_value=SimpleNamespace(pd_tp_ratio=1, num_head_replica=1, pd_head_ratio=1),
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_pcp_group",
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector.get_decode_context_model_parallel_rank",
                return_value=0,
            ),
        ]

        for p in self.patches:
            p.start()  # type: ignore

        self.vllm_config = MockVllmConfig()
        self.engine_id = "test_engine"
        mock_k = MagicMock()
        mock_k.shape = (10, 16, 8, 16)
        mock_k.data_ptr.return_value = 0x1000
        mock_k.element_size.return_value = 4
        mock_v = MagicMock()
        mock_v.shape = (10, 16, 8, 16)
        mock_v.data_ptr.return_value = 0x2000
        mock_v.element_size.return_value = 4
        self.kv_caches = {"encoder.layer.0": (mock_k, mock_v)}
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.vllm_config.parallel_config.decode_context_parallel_size = 1
        self.vllm_config.parallel_config.data_parallel_rank = 0
        self.vllm_config.kv_transfer_config.kv_port = 1234

        self.kv_cache_config = MockKVCacheConfig()

    def tearDown(self):
        for p in self.patches:
            p.stop()  # type: ignore

    def test_register_kv_caches_producer(self):
        self.vllm_config.kv_transfer_config.is_kv_producer = True
        self.vllm_config.kv_transfer_config.is_kv_consumer = False
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config, self.kv_cache_config, self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertEqual(len(worker.layer_metadata), 1)
        self.assertIsNotNone(worker.kv_send_layer_thread)
        self.assertIsNone(worker.kv_recv_layer_thread)

    def test_register_kv_caches_consumer(self):
        self.vllm_config.kv_transfer_config.is_kv_producer = False
        self.vllm_config.kv_transfer_config.is_kv_consumer = True
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config, self.kv_cache_config, self.engine_id)
        worker.register_kv_caches(self.kv_caches)
        self.assertEqual(len(worker.layer_metadata), 1)
        self.assertIsNone(worker.kv_send_layer_thread)
        self.assertIsNotNone(worker.kv_recv_layer_thread)

    def test_register_kv_caches_mla_case(self):
        mla_cache1 = MagicMock()
        mla_cache1.size.return_value = (10, 16, 1, 16)
        mla_cache1.shape = (10, 16, 1, 16)
        mla_cache1.data_ptr.return_value = 0x1000
        mla_cache1.element_size.return_value = 4
        mla_cache2 = MagicMock()
        mla_cache2.size.return_value = (10, 16, 1, 8)
        mla_cache2.shape = (10, 16, 1, 8)
        mla_cache2.data_ptr.return_value = 0x2000
        mla_cache2.element_size.return_value = 4
        mla_caches = {"encoder.layer.0": (mla_cache1, mla_cache2)}
        worker = MooncakeLayerwiseConnectorWorker(self.vllm_config, self.kv_cache_config, self.engine_id)
        worker.register_kv_caches(mla_caches)
        self.assertTrue(worker.use_mla)
        self.assertEqual(len(worker.layer_metadata["encoder.layer.0"].block_len), 2)
