import sys
import threading
import time
import types
import unittest
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata as importlib_metadata
from pathlib import Path
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
    MooncakeAgentMetadata,
    MooncakeConnector,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (  # noqa: E402
    GlobalTE,
    validate_mooncake_runtime_installation,
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
        thread.failed_recv_requests = set()
        thread.invalid_block_ids = set()
        thread.failed_recv_requests_lock = threading.Lock()
        return thread

    def _make_transfer_thread(self):
        thread = self._make_thread()
        thread.use_hybrid = True
        thread.task_tracker = MagicMock()
        thread.request_queue = MagicMock()
        thread.proc_not_transfer_request = {}
        thread.proc_not_transfer_request_lock = threading.Lock()
        thread._send_done_signal_to_free_remote_port = MagicMock()
        thread._send_done_recv_signal = MagicMock()
        thread._transfer_kv_cache = MagicMock()
        thread._transfer_kv_cache_all_groups = MagicMock()
        return thread

    @staticmethod
    def _request_meta(local_block_ids, *, all_task_done):
        return {
            "request_id": "req-1",
            "remote_request_id": "remote-req-1",
            "local_block_ids": local_block_ids,
            "remote_host": "remote-host",
            "remote_handshake_port": 6000,
            "remote_port_send_num": {},
            "all_task_done": all_task_done,
        }

    def test_transfer_failure_reports_invalid_blocks(self):
        thread = self._make_transfer_thread()
        thread._transfer_kv_cache_all_groups.side_effect = RuntimeError("transfer failed")
        req_meta = self._request_meta(([10, 11], [20]), all_task_done=True)

        thread._handle_request(req_meta)

        thread.task_tracker.update_done_task_count.assert_called_once_with("req-1")
        self.assertEqual(thread.get_and_clear_invalid_block_ids(), {10, 11})
        self.assertEqual(thread.get_and_clear_invalid_block_ids(), set())
        self.assertFalse(thread._is_failed_recv_request("req-1"))
        thread.request_queue.task_done.assert_called_once_with()
        thread._send_done_signal_to_free_remote_port.assert_called_once_with("remote-req-1", "remote-host", {})
        thread._send_done_recv_signal.assert_called_once_with("remote-req-1", "remote-host", 6000, {})

    def test_later_transfer_is_skipped_after_request_failure(self):
        thread = self._make_transfer_thread()
        thread._transfer_kv_cache_all_groups.side_effect = RuntimeError("transfer failed")
        first_task = self._request_meta(([10], [20]), all_task_done=False)
        final_task = self._request_meta(([11], [21]), all_task_done=True)

        thread._handle_request(first_task)
        thread._handle_request(final_task)

        thread._transfer_kv_cache_all_groups.assert_called_once_with(first_task)
        thread.task_tracker.update_done_task_count.assert_called_once_with("req-1")
        self.assertEqual(thread.get_and_clear_invalid_block_ids(), {10, 11})
        self.assertFalse(thread._is_failed_recv_request("req-1"))

    def test_remote_transfer_endpoint_uses_advertised_host(self):
        metadata = MooncakeAgentMetadata(
            engine_id="remote-engine",
            te_rpc_port=12345,
            block_size=128,
            kv_caches_base_addr=[0x1000],
            num_blocks=8,
            block_lens=[1024],
            ssm_sizes=(0, 0),
            local_ip="192.0.2.10",
        )

        endpoint = KVCacheRecvingThread._resolve_remote_transfer_endpoint(metadata, "fallback-host")

        self.assertEqual(endpoint, ("192.0.2.10", 12345))

    def test_remote_transfer_endpoint_rejects_zero_port(self):
        metadata = MooncakeAgentMetadata(
            engine_id="remote-engine",
            te_rpc_port=0,
            block_size=128,
            kv_caches_base_addr=[0x1000],
            num_blocks=8,
            block_lens=[1024],
            ssm_sizes=(0, 0),
            local_ip="192.0.2.10",
        )

        with self.assertRaisesRegex(RuntimeError, "invalid transfer port: 0"):
            KVCacheRecvingThread._resolve_remote_transfer_endpoint(metadata, "fallback-host")

    def test_remote_runtime_mismatch_is_rejected_before_transfer(self):
        thread = object.__new__(KVCacheRecvingThread)
        thread.mooncake_runtime_id = "mooncake-transfer-engine-npu==0.3.11.post1;engine_sha256=decode"
        metadata = MooncakeAgentMetadata(
            engine_id="remote-engine",
            te_rpc_port=12345,
            block_size=128,
            kv_caches_base_addr=[0x1000],
            num_blocks=8,
            block_lens=[1024],
            ssm_sizes=(0, 0),
            runtime_id="mooncake-transfer-engine-npu==0.3.11.post1;engine_sha256=prefill",
        )

        with self.assertRaisesRegex(RuntimeError, "runtime mismatch between P/D workers"):
            thread._validate_remote_runtime(metadata)

    def test_missing_remote_runtime_identity_is_backward_compatible(self):
        thread = object.__new__(KVCacheRecvingThread)
        thread.mooncake_runtime_id = "mooncake-transfer-engine-npu==0.3.11.post1;engine_sha256=decode"
        metadata = MooncakeAgentMetadata(
            engine_id="remote-engine",
            te_rpc_port=12345,
            block_size=128,
            kv_caches_base_addr=[0x1000],
            num_blocks=8,
            block_lens=[1024],
            ssm_sizes=(0, 0),
        )

        thread._validate_remote_runtime(metadata)

    def test_runtime_installation_accepts_distribution_owned_engine(self):
        install_root = Path("/opt/mooncake-test")
        distribution = MagicMock(
            version="0.3.11.post1",
            files=[Path("mooncake/engine.so")],
        )
        distribution.locate_file.side_effect = lambda path: install_root / path

        def get_distribution(name):
            if name == "mooncake-transfer-engine-npu":
                return distribution
            raise importlib_metadata.PackageNotFoundError(name)

        with (
            patch.object(fake_engine, "__file__", str(install_root / "mooncake/engine.so"), create=True),
            patch(
                "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine.importlib_metadata.distribution",
                side_effect=get_distribution,
            ),
        ):
            validate_mooncake_runtime_installation()

    def test_runtime_installation_rejects_shadowing_engine(self):
        install_root = Path("/opt/mooncake-test")
        distribution = MagicMock(
            version="0.3.11.post1",
            files=[Path("mooncake/engine.so")],
        )
        distribution.locate_file.side_effect = lambda path: install_root / path

        def get_distribution(name):
            if name == "mooncake-transfer-engine-npu":
                return distribution
            raise importlib_metadata.PackageNotFoundError(name)

        stale_engine = install_root / "mooncake/engine.cpython-312-aarch64-linux-gnu.so"
        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine.importlib_metadata.distribution",
                side_effect=get_distribution,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine.importlib_util.find_spec",
                return_value=types.SimpleNamespace(origin=str(stale_engine)),
            ),
            self.assertRaisesRegex(RuntimeError, "stale engine.cpython-\\*\\.so"),
        ):
            validate_mooncake_runtime_installation()

    def test_runtime_installation_rejects_multiple_distributions(self):
        distributions = {
            "mooncake-transfer-engine-npu": MagicMock(version="0.3.11.post1"),
            "mooncake-transfer-engine": MagicMock(version="0.3.11.post1"),
        }

        def get_distribution(name):
            if name in distributions:
                return distributions[name]
            raise importlib_metadata.PackageNotFoundError(name)

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine.importlib_metadata.distribution",
                side_effect=get_distribution,
            ),
            self.assertRaisesRegex(RuntimeError, "Multiple Mooncake distributions"),
        ):
            validate_mooncake_runtime_installation()

    def test_runtime_installation_rejects_shadowing_package(self):
        install_root = Path("/opt/mooncake-test")
        distribution = MagicMock(
            version="0.3.11.post1",
            files=[Path("mooncake/engine.so")],
        )
        distribution.locate_file.side_effect = lambda path: install_root / path

        def get_distribution(name):
            if name == "mooncake-transfer-engine-npu":
                return distribution
            raise importlib_metadata.PackageNotFoundError(name)

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine.importlib_metadata.distribution",
                side_effect=get_distribution,
            ),
            patch(
                "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine.importlib_util.find_spec",
                return_value=None,
            ),
            self.assertRaisesRegex(RuntimeError, "engine module cannot be resolved"),
        ):
            validate_mooncake_runtime_installation()

    def test_transfer_engine_rejects_invalid_installation_before_import(self):
        transfer_engine = GlobalTE()

        with (
            patch.object(fake_engine, "TransferEngine") as transfer_engine_factory,
            patch(
                "vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine.validate_mooncake_runtime_installation",
                side_effect=RuntimeError("invalid Mooncake installation"),
            ),
            self.assertRaisesRegex(RuntimeError, "invalid Mooncake installation"),
        ):
            transfer_engine.get_transfer_engine("127.0.0.1:12345", None)

        transfer_engine_factory.assert_not_called()

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


class TestMooncakeHybridConnectorFailureReporting(unittest.TestCase):
    def test_connector_forwards_invalid_blocks(self):
        connector = object.__new__(MooncakeConnector)
        connector.connector_worker = MagicMock()
        connector.connector_worker.get_block_ids_with_load_errors.return_value = {3, 7}

        self.assertEqual(connector.get_block_ids_with_load_errors(), {3, 7})
        connector.connector_worker.get_block_ids_with_load_errors.assert_called_once_with()

    def test_worker_returns_consumer_invalid_blocks(self):
        worker = object.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_consumer"
        worker.kv_recv_thread = MagicMock()
        worker.kv_recv_thread.get_and_clear_invalid_block_ids.return_value = {5, 9}

        self.assertEqual(worker.get_block_ids_with_load_errors(), {5, 9})

    def test_worker_producer_has_no_invalid_blocks(self):
        worker = object.__new__(MooncakeConnectorWorker)
        worker.kv_role = "kv_producer"
        worker.kv_recv_thread = None

        self.assertEqual(worker.get_block_ids_with_load_errors(), set())
