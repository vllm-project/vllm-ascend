import queue
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

from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (  # noqa: E402
    MAX_REQUESTS_PER_PEER_HANDLER,
    KVCacheRecvingThread,
    KVCacheSendingThread,
    KVCacheTaskTracker,
    MooncakeAgentMetadata,
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    _get_tensor_transfer_span_bytes,
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

    def test_group_transfer_and_completion(self):
        thread = self._make_thread()
        self.addCleanup(thread.executor.shutdown, wait=True)
        thread.use_hybrid = True
        thread.tp_rank = 0
        thread._prefill_pp_size = 1
        thread.hma_group_size = 2
        thread.kv_cache_specs = [MagicMock(), MagicMock(spec=MambaSpec)]
        thread.task_tracker = KVCacheTaskTracker()
        thread.request_queue = queue.Queue()
        thread.proc_not_transfer_request = {}
        thread.proc_not_transfer_request_lock = threading.Lock()
        thread.side_channel_port = thread.local_handshake_port = 32000
        thread.local_engine_id = "decode"
        thread.remote_metadata_lock = threading.Lock()
        thread.kv_caches_base_addr = {
            "decode": {32000: [0x1000, 0x2000, 0x3000]},
            "prefill": {31002: [0x4000, 0x5000, 0x6000]},
        }
        thread.remote_te_port = {"prefill": {31002: 7777}}
        # Two buffers share the first group's blocks; the state group has its own stride.
        thread.addr_group_idx = [[0], [0], [1]]
        thread.block_len_per_addr = [16, 8, 32]
        thread.block_stride_per_addr = [32, 16, 64]
        thread.engine = MagicMock()
        thread._send_done_recv_signal = MagicMock()

        for outcome in ("success", "empty", "failure"):
            with self.subTest(outcome=outcome):
                request_id = f"decode-{outcome}"
                remote_request_id = f"prefill-{outcome}"
                thread.engine.reset_mock()
                thread.engine.batch_transfer_sync_read.return_value = -1 if outcome == "failure" else 0
                thread._send_done_recv_signal.reset_mock()
                thread.task_tracker.add_req_to_process(request_id)
                thread.add_request(
                    request_id=request_id,
                    remote_request_id=remote_request_id,
                    local_block_ids=([], []) if outcome == "empty" else ([2, 3], [4]),
                    remote_block_ids=([1, 2], [3]),
                    remote_engine_id="prefill",
                    remote_host="192.0.2.1",
                    remote_handshake_port=31002,
                    offset=0,
                    tp_num_need_pulls=1,
                    all_task_done=True,
                )
                req_meta = thread.request_queue.get_nowait()
                thread._mark_request_task_submitted(req_meta)
                with patch(
                    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.logger.exception"
                ) as log_exception:
                    thread._handle_request(req_meta)
                self.assertEqual(log_exception.call_count, int(outcome == "failure"))
                if outcome == "empty":
                    thread.engine.batch_transfer_sync_read.assert_not_called()
                else:
                    thread.engine.batch_transfer_sync_read.assert_called_once_with(
                        "192.0.2.1:7777",
                        [0x1040, 0x2020, 0x3100],
                        [0x4020, 0x5010, 0x60C0],
                        [32, 16, 32],
                    )
                thread._send_done_recv_signal.assert_called_once_with(remote_request_id, "192.0.2.1", 31002, {})
                self.assertEqual(thread.get_and_clear_finished_requests(), {request_id})
                self.assertEqual(thread.get_and_clear_finished_requests(), set())
                self.assertFalse(thread.task_tracker.reqs_to_process)
                self.assertFalse(thread.request_task_counts)
                self.assertFalse(thread.finished_request_markers)
                self.assertFalse(thread.proc_not_transfer_request)
                self.assertEqual(thread.request_queue.unfinished_tasks, 0)

    def test_executor_workers_bind_kv_cache_device_before_handling_requests(self):
        expected_device_index = 5
        kv_cache = MagicMock(device=expected_device_index)
        model_config = types.SimpleNamespace(
            is_deepseek_mla=False,
            hf_config=types.SimpleNamespace(compress_ratios=[1]),
            hf_text_config=types.SimpleNamespace(
                num_hidden_layers=1,
                head_dim=64,
                num_key_value_heads=8,
            ),
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
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.torch.npu.set_device",
                side_effect=record_set_device,
            ),
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
            self.assertEqual(events[0], ("set_device", expected_device_index))
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


class TestMooncakeHybridConnectorWorker(unittest.TestCase):
    def setUp(self):
        for patcher in (
            patch.dict("os.environ"),
            patch.multiple(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector",
                init_ascend_config=MagicMock(),
                get_ascend_config=MagicMock(),
                get_transfer_timeout_value=MagicMock(return_value=30),
                get_ip=MagicMock(return_value="127.0.0.1"),
                get_tp_group=MagicMock(),
                get_pp_group=MagicMock(return_value=types.SimpleNamespace(rank_in_group=0)),
                global_te=MagicMock(),
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _make_worker(self, prefill_tp_size, decode_tp_size, tp_rank, pcp_size, pcp_rank, dp_rank, use_mamba, role):
        tp_size = prefill_tp_size if role == "kv_producer" else decode_tp_size
        dp_size = 2 if role == "kv_producer" else 1
        extra_config = {
            "prefill": {"tp_size": prefill_tp_size, "dp_size": 2},
            "decode": {"tp_size": decode_tp_size, "dp_size": 1},
        }
        config = types.SimpleNamespace(
            parallel_config=types.SimpleNamespace(
                tensor_parallel_size=tp_size,
                prefill_context_parallel_size=pcp_size,
                decode_context_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_rank=dp_rank,
                data_parallel_size=dp_size,
                data_parallel_rank_local=dp_rank,
                data_parallel_size_local=dp_size,
            ),
            kv_transfer_config=types.SimpleNamespace(
                kv_role=role,
                kv_port=31000 if role == "kv_producer" else 32000,
                get_from_extra_config=extra_config.get,
            ),
            model_config=types.SimpleNamespace(
                is_deepseek_mla=not use_mamba,
                hf_config=types.SimpleNamespace(**({} if use_mamba else {"compress_ratios": [1, 4]})),
                hf_text_config=types.SimpleNamespace(num_key_value_heads=8),
            ),
            cache_config=types.SimpleNamespace(block_size=128),
            scheduler_config=types.SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        )
        state_spec = (
            MagicMock(spec=MambaSpec, block_size=128, shapes=((2, 2), (2, 2)), dtypes=(torch.float32, torch.float32))
            if use_mamba
            else types.SimpleNamespace(block_size=128)
        )
        cache_config = types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(kv_cache_spec=types.SimpleNamespace(block_size=512), layer_names=["layer.0"]),
                types.SimpleNamespace(kv_cache_spec=state_spec, layer_names=["layer.1"]),
            ]
        )
        host = f"192.0.2.{pcp_rank * tp_size + tp_rank + 1}"
        engine_id = f"{role}-{dp_rank}-{pcp_rank}-{tp_rank}"
        with patch.multiple(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector",
            get_tensor_model_parallel_rank=MagicMock(return_value=tp_rank),
            get_tensor_model_parallel_world_size=MagicMock(return_value=tp_size),
            get_pcp_group=MagicMock(return_value=types.SimpleNamespace(rank_in_group=pcp_rank, world_size=pcp_size)),
            get_ip=MagicMock(return_value=host),
        ):
            worker = MooncakeConnectorWorker(config, engine_id, cache_config)
            worker.use_sparse = False
            if role == "kv_producer":
                metadata = MooncakeAgentMetadata(engine_id, 7777, 128, [], 0, [], (0, 0), local_ip=host)
                worker.kv_send_thread = KVCacheSendingThread(
                    config,
                    tp_rank,
                    prefill_tp_size,
                    engine_id,
                    host,
                    worker.side_channel_port,
                    metadata,
                    threading.Event(),
                    {},
                    pcp_rank,
                )
            else:
                worker.kv_recv_thread = MagicMock()
        return worker

    def test_start_load_kv_replica_routing_and_completion(self):
        cases = [
            # PCP, P-TP, D-TP, P-DP rank, Mamba receive branch.
            (1, 2, 2, 0, False),
            (2, 2, 2, 0, False),
            (1, 4, 2, 1, False),
            (4, 4, 2, 1, False),
            (1, 2, 2, 1, True),
            (2, 2, 2, 1, True),
        ]
        for pcp_size, prefill_tp_size, decode_tp_size, dp_rank, use_mamba in cases:
            with self.subTest(pcp_size=pcp_size, prefill_tp_size=prefill_tp_size, mamba=use_mamba):
                senders: dict[int, MooncakeConnectorWorker] = {}
                handshake_metadata = {}
                for pcp_rank in range(pcp_size):
                    for tp_rank in range(prefill_tp_size):
                        worker = self._make_worker(
                            prefill_tp_size,
                            decode_tp_size,
                            tp_rank,
                            pcp_size,
                            pcp_rank,
                            dp_rank,
                            use_mamba,
                            "kv_producer",
                        )
                        self.assertNotIn(worker.handshake_port, senders)
                        senders[worker.handshake_port] = worker
                        key = (0, tp_rank) if pcp_size == 1 else (0, pcp_rank, tp_rank)
                        handshake_metadata[key] = worker.kv_send_thread.metadata
                        with (
                            patch(
                                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.zmq_ctx"
                            ) as ctx,
                            patch.object(worker.kv_send_thread, "run_busy_loop") as run_busy_loop,
                        ):
                            worker.kv_send_thread.run()
                            self.assertEqual(
                                ctx.call_args.args[1], f"tcp://{worker.side_channel_host}:{worker.handshake_port}"
                            )
                            run_busy_loop.assert_called_once()

                scheduler = MooncakeConnectorScheduler(worker.vllm_config, "prefill", worker.kv_cache_config)
                scheduler.set_xfer_handshake_metadata_from_workers(handshake_metadata)
                base_port = 31000 + dp_rank * prefill_tp_size * pcp_size
                self.assertEqual(scheduler.side_channel_port, base_port)
                self.assertEqual(set(senders), set(range(base_port, base_port + prefill_tp_size * pcp_size)))
                decoders = [
                    self._make_worker(prefill_tp_size, decode_tp_size, rank, 1, 0, 0, use_mamba, "kv_consumer")
                    for rank in range(decode_tp_size)
                ]

                # Reuse the workers across requests, with different P and D request IDs.
                for request_id in ("req-0", "req-1", "req-2", "req-3"):
                    with self.subTest(request_id=request_id):
                        request = MockRequest(
                            request_id,
                            list(range(513)),
                            {"do_remote_decode": True},
                            RequestStatus.FINISHED_LENGTH_CAPPED,
                        )
                        delay_free, params = scheduler.request_finished_all_groups(request, ([10, 11, 12], [30, 31]))
                        self.assertTrue(delay_free)
                        self.assertEqual(params["remote_block_ids"], ([10, 11], [30, 31]))
                        if pcp_size == 1:
                            # Old request metadata defaults to one replica.
                            params.pop("remote_pcp_size")
                        metadata = MooncakeConnectorMetadata()
                        metadata.add_new_req(f"decode-{request_id}", ([20, 21], [40, 41]), 513, params)
                        metadata.reqs_in_batch = {f"decode-{request_id}"}
                        source_ports = set()
                        for decoder in decoders:
                            decoder.kv_recv_thread.reset_mock()
                            decoder.start_load_kv(metadata)
                            decoder.kv_recv_thread.add_request.assert_called_once()
                            pull = decoder.kv_recv_thread.add_request.call_args.kwargs
                            port = pull["remote_handshake_port"]
                            self.assertIn(port, senders)
                            source_ports.add(port)
                            self.assertEqual(pull["remote_host"], senders[port].side_channel_host)
                            self.assertEqual(pull["remote_engine_id"], senders[port].engine_id)
                            self.assertEqual(pull["request_id"], f"decode-{request_id}")
                            self.assertEqual(pull["remote_request_id"], request_id)
                            self.assertEqual(pull["local_block_ids"], ([20, 21], [40, 41]))
                            self.assertEqual(pull["remote_block_ids"], ([10, 11], [30, 31]))
                            self.assertEqual(
                                (pull["offset"], pull["tp_num_need_pulls"], pull["all_task_done"]), (0, 1, True)
                            )
                            self.assertIsNone(pull.get("remote_port_send_num"))
                            if prefill_tp_size == decode_tp_size:
                                self.assertEqual((port - base_port) % prefill_tp_size, decoder.tp_rank)
                        self.assertEqual(len(source_ports), decode_tp_size)
                        self.assertEqual(len({(port - base_port) // prefill_tp_size for port in source_ports}), 1)

                        scheduler._reqs_in_batch.add(request_id)
                        send_metadata = scheduler.build_connector_meta(MagicMock())
                        for port, sender in senders.items():
                            sender.start_load_kv(send_metadata)
                            tracker = sender.kv_send_thread.task_tracker
                            if port in source_ports:
                                self.assertEqual(tracker.get_and_clear_finished_requests(), set())
                                self.assertEqual(set(tracker.delayed_free_requests), {request_id})
                                # Only actual sources wait for D's DONE message.
                                tracker.update_done_task_count(request_id)
                            self.assertEqual(tracker.get_and_clear_finished_requests(), {request_id})
                            self.assertEqual(tracker.get_and_clear_finished_requests(), set())
                            self.assertFalse(tracker.delayed_free_requests)
                            self.assertFalse(tracker.reqs_to_process)


class TestMooncakeHybridConnectorRegistration(unittest.TestCase):
    def test_transfer_span_includes_inter_block_padding(self):
        backing = torch.empty(18, dtype=torch.uint8)
        tensor = torch.as_strided(backing, size=(3, 4), stride=(6, 1))

        self.assertEqual(tensor.numel() * tensor.element_size(), 12)
        self.assertEqual(_get_tensor_transfer_span_bytes(tensor), 18)

    def test_hybrid_registration_uses_actual_merged_tensor_ranges(self):
        alignment = 2 * 1024 * 1024
        backing_size = 4 * alignment
        layer_names = [
            "model.layers.0.self_attn",
            "model.layers.1.self_attn",
        ]
        raw_tensor = torch.empty(backing_size + alignment, dtype=torch.uint8)
        aligned_offset = (-raw_tensor.data_ptr()) % alignment
        backing = raw_tensor[aligned_offset : aligned_offset + backing_size]
        kv_caches = {
            layer_names[0]: backing[: 2 * alignment],
            layer_names[1]: backing[alignment : 3 * alignment],
        }

        worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
        worker.vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                is_deepseek_mla=False,
                hf_text_config=types.SimpleNamespace(),
            )
        )
        worker.kv_cache_config = types.SimpleNamespace(
            num_blocks=1,
            kv_cache_groups=[types.SimpleNamespace(layer_names=layer_names)],
            kv_cache_tensors=[
                types.SimpleNamespace(
                    size=backing_size,
                    layers=[layer_name],
                    shared_by=[layer_name],
                )
                for layer_name in layer_names
            ],
        )
        worker.use_hybrid = True
        worker.use_mamba = False
        worker.use_compress = True

        class RegistrationCaptured(Exception):
            pass

        with (
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector.global_te.register_buffer",
                side_effect=RegistrationCaptured,
            ) as register_buffer,
            self.assertRaises(RegistrationCaptured),
        ):
            worker.register_kv_caches(kv_caches)

        register_buffer.assert_called_once_with(
            [backing.data_ptr()],
            [3 * alignment],
        )


class TestMooncakeHybridConnectorScheduler(unittest.TestCase):
    def _make_scheduler(self):
        scheduler = object.__new__(MooncakeConnectorScheduler)
        scheduler.use_hybrid = True
        scheduler.use_compress = True
        scheduler.num_swa_blocks = [0, 2]
        # C4 exposes a 512-token logical block backed by one 128-token page.
        scheduler.group_block_size = [512, 128]
        scheduler._reqs_need_send = {}
        scheduler.block_size = 128
        scheduler.engine_id = "engine"
        scheduler.side_channel_host = "127.0.0.1"
        scheduler.side_channel_port = 12345
        scheduler.tp_size = 1
        scheduler.pcp_size = 1
        scheduler.multi_nodes_meta_mapping = {}
        return scheduler

    def test_compute_transfer_block_ids_trims_swa_groups(self):
        scheduler = self._make_scheduler()
        block_ids = (list(range(10)), [100, 101, 102, 103])

        transfer_block_ids = scheduler._compute_transfer_block_ids(block_ids, prompt_len=129)

        self.assertEqual(transfer_block_ids, ([0], [100, 101]))

    def test_request_finished_preserves_group_layout_with_pcp(self):
        scheduler = self._make_scheduler()
        scheduler.group_block_size = [512, 16384, 128]
        scheduler.num_swa_blocks = [0, 0, 2]
        request = MockRequest(
            "req-compressed",
            prompt_token_ids=list(range(513)),
            kv_transfer_params={"do_remote_decode": True},
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
        )

        for pcp_size in (1, 2, 4):
            with self.subTest(pcp_size=pcp_size):
                scheduler.pcp_size = pcp_size
                delay_free, params = scheduler.request_finished_all_groups(
                    request,
                    ([10, 11, 12], [20, 21], [30, 31, 32, 33, 34, 35]),
                )
                self.assertTrue(delay_free)
                self.assertIsNotNone(params)
                assert params is not None
                self.assertEqual(params["remote_block_ids"], ([10, 11], [20], [33, 34]))
                self.assertEqual(params["remote_pcp_size"], pcp_size)
                self.assertEqual(params["remote_ptp_size"], scheduler.tp_size)
                self.assertIn(request.request_id, scheduler._reqs_need_send)
                # This unused compatibility field remains in the legacy physical-block unit.
                self.assertEqual(params["num_prompt_blocks"], 5)

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
