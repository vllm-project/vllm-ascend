import sys
import threading
import time
import types
import unittest
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

import torch

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine

from vllm.v1.request import RequestStatus  # noqa: E402

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (  # noqa: E402
    MAX_REQUESTS_PER_PEER_HANDLER,
    KVCacheRecvingThread,
    KVCacheTaskTracker,
    MooncakeConnector,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeKVConnectorStats,
)


class MockRequest:
    def __init__(
        self,
        request_id,
        prompt_token_ids,
        kv_transfer_params,
        status,
        num_prompt_tokens=None,
    ):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        if num_prompt_tokens is None:
            num_prompt_tokens = len(prompt_token_ids) if prompt_token_ids is not None else 0
        self.num_prompt_tokens = num_prompt_tokens
        self.kv_transfer_params = kv_transfer_params
        self.status = status
        self.output_token_ids = [101]


class TestHybridKVCacheRecvingThreadDispatch(unittest.TestCase):
    def _make_thread(self):
        thread = object.__new__(KVCacheRecvingThread)
        thread.executor = ThreadPoolExecutor(max_workers=2)
        thread.peer_request_queues = defaultdict(deque)
        thread.active_peer_request_handlers = set()
        thread.peer_request_queues_lock = threading.Lock()
        thread.request_task_counts = defaultdict(int)
        thread.finished_request_markers = set()
        thread.request_task_counts_lock = threading.Lock()
        return thread

    def test_executor_workers_bind_kv_cache_device_before_handling_requests(self):
        expected_device = torch.device("npu:5")
        kv_cache = MagicMock(device=expected_device)
        model_config = types.SimpleNamespace(
            is_deepseek_mla=False,
            hf_config=types.SimpleNamespace(compress_ratios=[1]),
            hf_text_config=types.SimpleNamespace(num_hidden_layers=1),
        )
        vllm_config = types.SimpleNamespace(
            model_config=model_config,
            cache_config=types.SimpleNamespace(block_size=16),
        )
        kv_cache_config = types.SimpleNamespace(kv_cache_groups=[])
        worker_events: defaultdict[int, list[tuple[str, int | str]]] = defaultdict(list)
        events_lock = threading.Lock()
        both_workers_started = threading.Event()
        release_workers = threading.Event()

        def record_set_device(device):
            device_index = device if isinstance(device, int) else torch.device(device).index
            with events_lock:
                worker_events[threading.get_ident()].append(("set_device", device_index))

        with (
            patch("torch.npu.set_device", side_effect=record_set_device),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.is_vl_model",
                return_value=False,
            ),
        ):
            thread = KVCacheRecvingThread(
                tp_rank=1,
                tp_size=2,
                _prefill_pp_size=1,
                engine=MagicMock(),
                local_engine_id="local_engine",
                local_handshake_port=5555,
                side_channel_port=30000,
                local_kv_caches_base_addr=[0x1000],
                block_len_per_addr=[1024],
                block_stride_per_addr=[1024],
                addr_group_idx=[0],
                mamba_ssm_size=(0, 0),
                use_hybrid=False,
                has_mamba=False,
                hma_group_size=1,
                ready_event=threading.Event(),
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                kv_caches={"layer.0": (kv_cache, kv_cache)},
            )

            def handle_request(req_meta: dict[str, Any]):
                with events_lock:
                    worker_events[threading.get_ident()].append(("handle", req_meta["request_id"]))
                    handled_worker_count = sum(
                        any(event == "handle" for event, _ in events) for events in worker_events.values()
                    )
                    if handled_worker_count == 2:
                        both_workers_started.set()
                release_workers.wait()

            thread._handle_request = handle_request  # type: ignore[method-assign]
            try:
                for index in range(2):
                    thread._submit_request(
                        {
                            "request_id": f"req-{index}",
                            "remote_host": f"host-{index}",
                            "remote_handshake_port": 6000 + index,
                            "all_task_done": True,
                        }
                    )
                self.assertTrue(both_workers_started.wait(timeout=5.0), "executor did not start two workers")
            finally:
                release_workers.set()
                thread.executor.shutdown(wait=True, cancel_futures=True)

        handled_worker_events = [events for events in worker_events.values() if any(e == "handle" for e, _ in events)]
        self.assertEqual(len(handled_worker_events), 2)
        for events in handled_worker_events:
            self.assertEqual(events[0], ("set_device", expected_device.index))
            self.assertEqual(events[1][0], "handle")

    def test_submit_request_serializes_same_peer_fifo(self):
        thread = self._make_thread()
        release_first_request = threading.Event()
        first_request_started = threading.Event()
        other_peer_started = threading.Event()
        handled_requests: list[str] = []
        active_by_peer: defaultdict[tuple[str, int], int] = defaultdict(int)
        max_active_by_peer: defaultdict[tuple[str, int], int] = defaultdict(int)
        state_lock = threading.Lock()

        def handle_request(req_meta: dict[str, Any]):
            peer_key = (req_meta["remote_host"], req_meta["remote_handshake_port"])
            with state_lock:
                active_by_peer[peer_key] += 1
                max_active_by_peer[peer_key] = max(max_active_by_peer[peer_key], active_by_peer[peer_key])
                handled_requests.append(req_meta["request_id"])

            if req_meta["request_id"] == "same-peer-1":
                first_request_started.set()
                self.assertTrue(release_first_request.wait(timeout=2.0))
            elif req_meta["request_id"] == "other-peer-1":
                other_peer_started.set()

            time.sleep(0.01)
            with state_lock:
                active_by_peer[peer_key] -= 1

        thread._handle_request = handle_request  # type: ignore[method-assign]
        same_peer_1 = {
            "request_id": "same-peer-1",
            "remote_host": "host-a",
            "remote_handshake_port": 6000,
            "all_task_done": False,
        }
        same_peer_2 = {
            "request_id": "same-peer-2",
            "remote_host": "host-a",
            "remote_handshake_port": 6000,
            "all_task_done": True,
        }
        other_peer = {
            "request_id": "other-peer-1",
            "remote_host": "host-b",
            "remote_handshake_port": 6001,
            "all_task_done": True,
        }

        try:
            thread._submit_request(same_peer_1)
            self.assertTrue(first_request_started.wait(timeout=1.0))
            thread._submit_request(same_peer_2)
            thread._submit_request(other_peer)

            self.assertTrue(other_peer_started.wait(timeout=1.0))
            time.sleep(0.05)
            self.assertNotIn("same-peer-2", handled_requests)
        finally:
            release_first_request.set()
            thread.executor.shutdown(wait=True, cancel_futures=True)

        self.assertLess(handled_requests.index("same-peer-1"), handled_requests.index("same-peer-2"))
        self.assertEqual(max_active_by_peer[("host-a", 6000)], 1)
        self.assertEqual(max_active_by_peer[("host-b", 6001)], 1)

    def test_peer_handler_yields_after_batch_limit(self):
        thread = self._make_thread()
        peer_key = ("host-a", 6000)
        requests = [
            {
                "request_id": f"req-{idx}",
                "remote_host": peer_key[0],
                "remote_handshake_port": peer_key[1],
            }
            for idx in range(MAX_REQUESTS_PER_PEER_HANDLER + 1)
        ]
        handled_requests: list[str] = []
        thread.peer_request_queues[peer_key].extend(requests)
        thread.active_peer_request_handlers.add(peer_key)
        thread.executor = MagicMock()

        def handle_request(req_meta: dict[str, Any]):
            handled_requests.append(req_meta["request_id"])

        thread._handle_request = handle_request  # type: ignore[method-assign]

        thread._handle_peer_requests(peer_key)

        self.assertEqual(handled_requests, [f"req-{idx}" for idx in range(MAX_REQUESTS_PER_PEER_HANDLER)])
        self.assertEqual(
            [req["request_id"] for req in thread.peer_request_queues[peer_key]],
            [f"req-{MAX_REQUESTS_PER_PEER_HANDLER}"],
        )
        self.assertIn(peer_key, thread.active_peer_request_handlers)
        thread.executor.submit.assert_called_once_with(thread._handle_peer_requests, peer_key)


class TestMooncakeHybridConnectorScheduler(unittest.TestCase):
    def _make_scheduler(self):
        scheduler = object.__new__(MooncakeConnectorScheduler)
        scheduler.use_hybrid = True
        scheduler.use_compress = True
        scheduler.num_swa_blocks = [0, 2]
        scheduler.group_block_size = [128, 128]
        scheduler.group_compress_ratio = [4, 1]
        scheduler._reqs_need_send = {}
        scheduler.block_size = 128
        scheduler.engine_id = "engine"
        scheduler.side_channel_host = "127.0.0.1"
        scheduler.side_channel_port = 12345
        scheduler.tp_size = 1
        scheduler.multi_nodes_meta_mapping = {}
        return scheduler

    def test_compute_transfer_block_ids_trims_swa_groups(self):
        scheduler = self._make_scheduler()
        block_ids = (list(range(10)), [100, 101, 102, 103])

        transfer_block_ids = scheduler._compute_transfer_block_ids(block_ids, prompt_len=129)

        self.assertEqual(transfer_block_ids, ([0], [100, 101]))

    def test_request_finished_trims_before_swa_clip(self):
        scheduler = self._make_scheduler()
        request = MockRequest(
            "req1",
            prompt_token_ids=list(range(129)),
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        )
        block_ids = (list(range(10)), [100, 101, 102, 103])

        delay_free, params = scheduler.request_finished_all_groups(request, block_ids)

        self.assertTrue(delay_free)
        self.assertIsNotNone(params)
        self.assertEqual(params["remote_block_ids"], ([0], [100, 101]))
        self.assertEqual(params["num_prompt_blocks"], 2)
        self.assertIn("req1", scheduler._reqs_need_send)

    def test_request_finished_uses_num_prompt_tokens(self):
        scheduler = self._make_scheduler()
        request = MockRequest(
            "req1",
            prompt_token_ids=None,
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
            num_prompt_tokens=129,
        )
        block_ids = (list(range(10)), [100, 101, 102, 103])

        delay_free, params = scheduler.request_finished_all_groups(request, block_ids)

        self.assertTrue(delay_free)
        self.assertIsNotNone(params)
        self.assertEqual(params["remote_block_ids"], ([0], [100, 101]))
        self.assertEqual(params["num_prompt_blocks"], 2)


class TestMooncakeHybridConnectorStats(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.engine.batch_transfer_sync_read.return_value = 0
        self.req_meta = {
            "request_id": "req1",
            "remote_request_id": "req1",
            "local_block_ids": [[1, 2]],
            "remote_block_ids": [[3, 4]],
            "remote_engine_id": "remote_engine",
            "remote_host": "localhost",
            "remote_handshake_port": 6666,
            "remote_port_send_num": {},
            "offset": 0,
            "tp_num_need_pulls": 1,
            "all_task_done": True,
        }

    def _make_recv_thread(self, use_hybrid=False, xfer_stats=None):
        model_config = types.SimpleNamespace(
            is_deepseek_mla=False,
            hf_config=types.SimpleNamespace(compress_ratios=[1]),
            hf_text_config=types.SimpleNamespace(num_hidden_layers=1),
        )
        vllm_config = types.SimpleNamespace(
            model_config=model_config,
            cache_config=types.SimpleNamespace(block_size=16),
            speculative_config=None,
        )
        kv_cache = MagicMock(device=torch.device("npu:0"))
        with (
            patch("torch.npu.set_device"),
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.is_vl_model",
                return_value=False,
            ),
        ):
            thread = KVCacheRecvingThread(
                tp_rank=0,
                tp_size=1,
                _prefill_pp_size=1,
                engine=self.engine,
                local_engine_id="local_engine",
                local_handshake_port=5555,
                side_channel_port=30000,
                local_kv_caches_base_addr=[0x1000],
                block_len_per_addr=[1024],
                block_stride_per_addr=[1024],
                addr_group_idx=[],
                mamba_ssm_size=(0, 0),
                use_hybrid=use_hybrid,
                has_mamba=False,
                hma_group_size=1,
                ready_event=threading.Event(),
                vllm_config=vllm_config,
                kv_cache_config=types.SimpleNamespace(kv_cache_groups=[]),
                kv_caches={"layer.0": (kv_cache,)},
                xfer_stats=xfer_stats,
            )
        thread.kv_caches_base_addr["remote_engine"] = {6666: [0x3000]}
        thread.remote_te_port["remote_engine"] = {6666: 7777}
        thread.kv_cache_specs = [MagicMock()]
        return thread

    def test_transfer_success_records_stats(self):
        thread = self._make_recv_thread()

        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.get_ascend_config"
        ) as mock_config:
            mock_config.return_value.enable_kv_nz = False
            thread._transfer_kv_cache(self.req_meta)

        call_args, _ = self.engine.batch_transfer_sync_read.call_args
        stats_data = thread.xfer_stats.data
        self.assertEqual(len(stats_data["transfer_duration"]), 1)
        self.assertGreaterEqual(stats_data["transfer_duration"][0], 0)
        self.assertEqual(stats_data["bytes_transferred"], [sum(call_args[3])])
        self.assertEqual(stats_data["num_descriptors"], [len(call_args[1])])
        self.assertEqual(stats_data["num_failed_transfers"], [])

    def test_transfer_engine_failure_records_failed_transfer(self):
        self.engine.batch_transfer_sync_read.return_value = -1
        thread = self._make_recv_thread()

        with self.assertRaises(RuntimeError):
            thread._transfer_kv_cache(self.req_meta)

        stats_data = thread.xfer_stats.data
        self.assertEqual(stats_data["num_failed_transfers"], [1])
        self.assertEqual(stats_data["transfer_duration"], [])

    def test_hybrid_group_transfer_records_stats(self):
        thread = self._make_recv_thread(use_hybrid=True)

        thread._transfer_kv_cache_all_groups(self.req_meta)

        call_args, _ = self.engine.batch_transfer_sync_read.call_args
        stats_data = thread.xfer_stats.data
        self.assertEqual(stats_data["bytes_transferred"], [sum(call_args[3])])
        self.assertEqual(stats_data["num_descriptors"], [len(call_args[1])])

    @patch.object(KVCacheRecvingThread, "_send_done_signal_to_free_remote_port")
    @patch.object(KVCacheRecvingThread, "_send_done_recv_signal")
    @patch.object(KVCacheRecvingThread, "_transfer_kv_cache")
    def test_handle_request_exception_records_failed_recv(self, mock_transfer, mock_send, mock_free):
        mock_transfer.side_effect = RuntimeError("boom")
        thread = self._make_recv_thread()
        thread.request_queue = MagicMock()
        thread.task_tracker = MagicMock()

        thread._handle_request(self.req_meta)

        self.assertEqual(thread.xfer_stats.data["num_failed_recvs"], [1])

    def test_recv_thread_shares_worker_stats(self):
        shared = MooncakeKVConnectorStats()
        thread = self._make_recv_thread(xfer_stats=shared)
        self.assertIs(thread.xfer_stats, shared)
        self.assertIs(thread.task_tracker.xfer_stats, shared)

    def test_tracker_records_expired_request(self):
        from vllm import envs as vllm_envs

        stats = MooncakeKVConnectorStats()
        tracker = KVCacheTaskTracker(xfer_stats=stats)
        tracker.add_req_to_process("req1")
        tracker.add_delayed_request("req1", time.time() - vllm_envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT - 1)

        result = tracker.get_and_clear_finished_requests()

        self.assertEqual(result, {"req1"})
        self.assertEqual(stats.data["num_kv_expired_reqs"], [1])

    def test_tracker_without_stats_still_expires(self):
        from vllm import envs as vllm_envs

        tracker = KVCacheTaskTracker()
        tracker.add_req_to_process("req1")
        tracker.add_delayed_request("req1", time.time() - vllm_envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT - 1)
        self.assertEqual(tracker.get_and_clear_finished_requests(), {"req1"})

    def test_worker_get_kv_connector_stats_drains(self):
        worker = object.__new__(MooncakeConnectorWorker)
        worker.xfer_stats = MooncakeKVConnectorStats()

        self.assertIsNone(worker.get_kv_connector_stats())

        worker.xfer_stats.record_transfer(duration_s=0.5, total_bytes=2**20, num_descs=4)
        worker.xfer_stats.record_failed_recv()
        snapshot = worker.get_kv_connector_stats()

        assert snapshot is not None
        self.assertEqual(snapshot.data["transfer_duration"], [0.5])
        self.assertEqual(snapshot.data["num_failed_recvs"], [1])
        reduced = snapshot.reduce()
        self.assertEqual(reduced["Num successful transfers"], 1)
        self.assertEqual(reduced["Num failed recvs"], 1)
        # A second call in the same interval has nothing new to report.
        self.assertIsNone(worker.get_kv_connector_stats())

    def test_connector_stats_methods(self):
        connector = object.__new__(MooncakeConnector)
        connector.connector_worker = None
        self.assertIsNone(connector.get_kv_connector_stats())

        worker = MagicMock()
        sentinel = MooncakeKVConnectorStats()
        worker.get_kv_connector_stats.return_value = sentinel
        connector.connector_worker = worker
        self.assertIs(connector.get_kv_connector_stats(), sentinel)

        built = MooncakeConnector.build_kv_connector_stats()
        self.assertIsInstance(built, MooncakeKVConnectorStats)
        self.assertTrue(built.is_empty())
        rebuilt = MooncakeConnector.build_kv_connector_stats({"transfer_duration": [0.1]})
        assert rebuilt is not None
        self.assertEqual(rebuilt.data["transfer_duration"], [0.1])
