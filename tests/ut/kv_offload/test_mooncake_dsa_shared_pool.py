# SPDX-License-Identifier: Apache-2.0
import queue
import threading
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_connector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import (
    KVCacheRecvingThread,
    KVCacheTaskTracker,
    MooncakeConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_metadata import (
    DsaLocalResultKind,
    DsaStepRequest,
    RemoteEndpoint,
    RemoteSource,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_transfer import DsaCacheLayout

MAIN = "model.layers.0.self_attn.attn"
INDEXER = "model.layers.0.indexer.attn.index_cache"


def make_worker(rank=0, ptp=8, dcp=8, dtp=4):
    worker = object.__new__(MooncakeConnectorWorker)
    worker.tp_rank, worker.tp_size = rank, dtp
    worker._prefill_tp_size, worker._decode_tp_size = ptp, dtp
    worker._prefill_pp_size = 1
    worker.use_mla = worker.use_sparse = True
    worker.pcp_size = worker.dcp_size = 1
    worker.num_key_value_heads = 1
    worker.tp_num_need_pulls = 1
    worker.side_channel_port = 6000
    worker.handshake_port = 6000 + rank
    worker.engine_id = "decode"
    worker._dsa_decode = True
    worker._dsa_active_commands = {}
    worker._dsa_cancel_events = {}
    worker._dsa_results = queue.SimpleQueue()
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(is_deepseek_mla=True),
        parallel_config=SimpleNamespace(tensor_parallel_size=dtp),
        kv_transfer_config=SimpleNamespace(kv_port=5000),
    )
    thread = object.__new__(KVCacheRecvingThread)
    worker.kv_recv_thread = thread
    thread.tp_rank = rank
    thread.vllm_config = worker.vllm_config
    thread._prefill_pp_size = 1
    thread.pp_layer_indices = {0: (0, 1)}
    thread.num_draft_layers = 0
    thread.index_cache_plane_base = 1
    thread._dsa_transformer_layers = {MAIN: 0, INDEXER: 0}
    thread.request_queue = queue.Queue()
    thread.request_task_counts = defaultdict(int)
    thread.finished_request_markers = set()
    thread.request_task_counts_lock = threading.Lock()
    thread.failed_recv_requests = set()
    thread.dsa_failure_phases = {}
    thread.failed_recv_requests_lock = threading.Lock()
    thread.invalid_block_ids = set()
    thread.remote_metadata_lock = threading.Lock()
    groups = {0: ({"layer_names": [MAIN]}, [0]), 1: ({"layer_names": [INDEXER]}, [1])}
    thread.kv_group2layeridx = groups
    thread.remote_metadata_hosts = {"prefill": {5000 + p: "127.0.0.1" for p in range(ptp)}}
    thread.remote_kv_group2layeridx = {"prefill": {5000 + p: groups for p in range(ptp)}}
    thread.kv_caches_base_addr = {"prefill": {5000 + p: [[10000, 20000], [30000]] for p in range(ptp)}}
    thread.remote_block_stride_per_addr = {"prefill": {5000 + p: [[8, 8], [8]] for p in range(ptp)}}
    thread.remote_block_size_scale = {"prefill": {5000 + p: [[1, 1], [dcp]] for p in range(ptp)}}
    thread.remote_block_len_per_addr = {"prefill": {5000 + p: [[8, 8], [8]] for p in range(ptp)}}
    thread.remote_cache_dtypes = {"prefill": {5000 + p: [["bf16", "bf16"], ["bf16"]] for p in range(ptp)}}
    thread.remote_num_blocks = {"prefill": {5000 + p: 128 for p in range(ptp)}}
    thread.remote_te_port = {"prefill": {5000 + p: 7000 + p for p in range(ptp)}}
    thread._dsa_main_local_layout = [DsaCacheLayout(MAIN, i, 1000 + i * 1000, 8, 8, 1, 4, "bf16") for i in range(2)]
    thread._dsa_indexer_local_layout = [DsaCacheLayout(INDEXER, 0, 3000, 8, 8, 1, 4, "bf16")]
    thread.engine = SimpleNamespace(batch_transfer_sync_read=MagicMock(return_value=0))
    thread._send_done_recv_signal = MagicMock()
    endpoints = tuple(RemoteEndpoint("127.0.0.1", 5000 + p, "prefill") for p in range(ptp))
    command = DsaStepRequest(
        "request",
        RemoteSource(
            "remote",
            endpoints,
            (9, 7),
            (9, 7),
            remote_ptp_size=ptp,
            remote_dcp_size=dcp,
            remote_block_size=4,
            num_external_tokens=ptp * 4,
        ),
        tuple(range(20, 20 + ptp)),
        tuple(range(40, 40 + ptp)),
        invalid_block_ids=(81, 83),
    )
    return worker, thread, command


def submit(worker, command):
    worker._dispatch_dsa_commands((command,))
    tasks = []
    while not worker.kv_recv_thread.request_queue.empty():
        task = worker.kv_recv_thread.request_queue.get_nowait()
        worker.kv_recv_thread._mark_request_task_submitted(task)
        tasks.append(task)
    return tasks


@pytest.mark.parametrize("ptp,dcp,dtp", [(8, 8, 4), (16, 16, 8), (4, 1, 4), (8, 1, 4)])
def test_real_endpoint_handlers_cover_main_once_and_full_indexer_per_rank(ptp, dcp, dtp):
    writes = []
    done_counts = defaultdict(int)
    expected_counts = {}
    for rank in range(dtp):
        worker, thread, command = make_worker(rank, ptp, dcp, dtp)
        if dcp == 1:
            command = replace(
                command,
                source=replace(command.source, main_block_ids=tuple(range(ptp)), indexer_block_ids=tuple(range(ptp))),
            )
        tasks = submit(worker, command)
        for task in reversed(tasks):
            thread._handle_dsa_request(task)
        (result,) = worker.build_connector_worker_meta().results
        assert result.kind is DsaLocalResultKind.RECEIVE_COMPLETE
        assert worker.build_connector_worker_meta() is None
        indexer_bytes = 0
        for call in thread.engine.batch_transfer_sync_read.call_args_list:
            _, dsts, _, sizes = call.args
            for dst, size in zip(dsts, sizes):
                if dst >= 3000:
                    indexer_bytes += size
                else:
                    writes.extend(range(dst, dst + size))
        assert indexer_bytes == ptp * 8
        for call in thread._send_done_recv_signal.call_args_list:
            _, _, port, counts = call.args
            if counts[port]["num"]:
                done_counts[port] += 1
                expected_counts[port] = counts[port]["num"]
    assert len(writes) == len(set(writes)) == ptp * 8 * 2
    assert dict(done_counts) == expected_counts


def test_failure_and_last_task_first_wait_for_all_pending_and_report_group_zero_ids():
    worker, thread, command = make_worker()
    tasks = submit(worker, command)
    thread.engine.batch_transfer_sync_read.return_value = -1
    thread._handle_dsa_request(tasks[-1])
    assert worker.build_connector_worker_meta() is None
    for task in tasks[:-1]:
        thread._handle_dsa_request(task)
    (result,) = worker.build_connector_worker_meta().results
    assert result.kind is DsaLocalResultKind.TRANSFER_FAILED
    assert thread.get_and_clear_invalid_block_ids() == {81, 83}
    assert thread._send_done_recv_signal.call_count == len(tasks)


def test_cancel_skips_reads_but_preserves_all_notifications_and_terminal_result():
    worker, thread, command = make_worker()
    tasks = submit(worker, command)
    tasks[0]["cancelled"].set()
    for task in tasks:
        thread._handle_dsa_request(task)
    thread.engine.batch_transfer_sync_read.assert_not_called()
    assert thread._send_done_recv_signal.call_count == len(tasks)
    assert len(worker.build_connector_worker_meta().results) == 1


def test_notification_only_does_not_report_a_new_receive():
    worker, thread, command = make_worker()
    command = replace(command, notify_only=True, main_host_block_ids=(), indexer_hbm_block_ids=())
    tasks = submit(worker, command)
    for task in tasks:
        thread._handle_dsa_request(task)
    thread.engine.batch_transfer_sync_read.assert_not_called()
    assert thread._send_done_recv_signal.call_count == len(tasks)
    assert worker.build_connector_worker_meta() is None


def test_submission_gap_does_not_finish_before_last_marker():
    worker, thread, command = make_worker()
    worker._dispatch_dsa_commands((command,))
    while not thread.request_queue.empty():
        task = thread.request_queue.get_nowait()
        thread._mark_request_task_submitted(task)
        thread._handle_dsa_request(task)
        if not task["all_task_done"]:
            assert worker.build_connector_worker_meta() is None
    assert worker.build_connector_worker_meta() is not None


def test_duplicate_and_late_done_do_not_release_early_or_recreate_state():
    tracker = KVCacheTaskTracker()
    tracker.add_req_to_process("r")
    tracker.record_participant_done("r", "engine-A:port:0", 2)
    tracker.record_participant_done("r", "engine-A:port:0", 2)
    assert tracker.get_and_clear_finished_requests() == set()
    tracker.record_participant_done("r", "engine-B:port:0", 2)
    assert tracker.get_and_clear_finished_requests() == {"r"}
    tracker.record_participant_done("r", "engine-A:port:0", 2)
    assert not tracker.received_participants
    assert tracker.get_and_clear_finished_requests() == set()


@pytest.mark.parametrize("owner", [True, False])
def test_all_ranks_register_and_use_local_main_views(owner, monkeypatch):
    worker = object.__new__(MooncakeConnectorWorker)
    worker.num_blocks = 4
    worker.engine = MagicMock()
    worker._get_layer_spec = lambda name: SimpleNamespace(block_size=4)
    host_k = torch.empty((4, 4, 1), dtype=torch.bfloat16)
    host_v = torch.empty_like(host_k)
    pool = SimpleNamespace(is_owner=owner, register_local_writer=MagicMock())
    manager = SimpleNamespace(
        get_mooncake_host_pool=lambda: pool, get_local_host_kv_views=lambda name: (host_k, host_v)
    )
    monkeypatch.setattr(mooncake_connector, "get_sparse_kv_offload_manager", lambda: manager)
    indexer = torch.empty_like(host_k)
    indexer_layout, main_layout = worker._build_dsa_local_layouts(
        {MAIN: (None,) * 6, INDEXER: (indexer,)}, {MAIN: 0, INDEXER: 1}
    )
    assert main_layout[0].base == host_k.data_ptr()
    assert main_layout[1].base == host_v.data_ptr()
    assert indexer_layout[0].base == indexer.data_ptr()
    pool.register_local_writer.assert_called_once_with(worker.engine)


def test_cancel_inflight_read_waits_for_sync_return_before_terminal_callback():
    worker, thread, command = make_worker()
    tasks = submit(worker, command)
    entered, release = threading.Event(), threading.Event()

    def read(*args):
        entered.set()
        assert release.wait(5)
        return 0

    thread.engine.batch_transfer_sync_read.side_effect = read
    running = threading.Thread(target=thread._handle_dsa_request, args=(tasks[0],))
    running.start()
    assert entered.wait(5)
    tasks[0]["cancelled"].set()
    for task in tasks[1:]:
        thread._handle_dsa_request(task)
    assert worker.build_connector_worker_meta() is None
    assert "request" in worker._dsa_active_commands
    release.set()
    running.join(5)
    assert not running.is_alive()
    assert worker.build_connector_worker_meta() is not None
    thread.engine.batch_transfer_sync_read.assert_called_once()


def test_missing_remote_component_reports_failure_without_any_read():
    worker, thread, command = make_worker()
    tasks = submit(worker, command)
    thread.remote_cache_dtypes["prefill"][5000] = [["bf16"], ["bf16"]]
    for task in tasks:
        thread._handle_dsa_request(task)
    (result,) = worker.build_connector_worker_meta().results
    assert result.kind is DsaLocalResultKind.TRANSFER_FAILED
    thread.engine.batch_transfer_sync_read.assert_not_called()
    assert thread.get_and_clear_invalid_block_ids() == {81, 83}


def test_expiry_removes_participant_state(monkeypatch):
    tracker = KVCacheTaskTracker()
    tracker.add_req_to_process("r")
    tracker.add_delayed_request("r", 0)
    tracker.record_participant_done("r", "engine:port:0", 2)
    monkeypatch.setattr(mooncake_connector.time, "time", lambda: 10**12)
    assert tracker.get_and_clear_finished_requests() == {"r"}
    assert not tracker.received_participants
    tracker.record_participant_done("r", "engine:port:1", 2)
    assert tracker.get_and_clear_finished_requests() == set()


def test_changed_expected_count_is_rejected():
    tracker = KVCacheTaskTracker()
    tracker.add_req_to_process("r")
    tracker.record_participant_done("r", "a", 2)
    with pytest.raises(ValueError, match="count changed"):
        tracker.record_participant_done("r", "b", 1)
    assert tracker.get_and_clear_finished_requests() == set()


def test_cross_host_endpoint_translation_preserves_source_counts():
    worker, _, command = make_worker()
    endpoints = tuple(RemoteEndpoint(f"192.0.2.{p + 1}", 8000 + p, f"engine{p}") for p in range(8))
    command = replace(command, source=replace(command.source, endpoints_by_prefill_rank=endpoints))
    tasks = submit(worker, command)
    assert [t["dsa_remote_endpoint"] for t in tasks] == list(endpoints)
    assert [t["expected"] for t in tasks] == [4] * 8
    assert [t["cp_rank"] for t in tasks] == list(range(8))


def test_prefill_pp_keeps_indexer_task_in_every_stage():
    worker, _, command = make_worker(ptp=4, dcp=1, dtp=4)
    worker._prefill_pp_size = 2
    endpoints = tuple(RemoteEndpoint("127.0.0.1", 5000 + p, "prefill") for p in range(8))
    command = replace(command, source=replace(command.source, remote_pp_size=2, endpoints_by_prefill_rank=endpoints))
    plan = worker._plan_dsa_endpoints(command)
    readers = [task for task in plan if task[4] > 0]
    assert [(rank, pp, indexer) for rank, _, pp, indexer, _, _ in readers] == [(0, 0, True), (4, 1, True)]


@pytest.mark.parametrize("dsa", [False, True])
def test_dcp_pp_missing_source_stage_is_reported_before_transfer(dsa):
    worker, thread, command = make_worker(ptp=8, dcp=8, dtp=4)
    worker._prefill_pp_size = 2
    endpoints = tuple(RemoteEndpoint("127.0.0.1", 5000 + p, "prefill") for p in range(16))
    command = replace(command, source=replace(command.source, remote_pp_size=2, endpoints_by_prefill_rank=endpoints))
    with pytest.raises(AssertionError, match="Mooncake KV source coverage incomplete") as error:
        if dsa:
            worker._dispatch_dsa_commands((command,))
        else:
            worker._is_hma_required = False
            worker.pcp_rank = worker.dcp_rank = 0
            worker.block_size = 4
            worker.block_size_scale = [[1]]
            worker.kv_group2layeridx = {0: ({"kv_cache_spec_type": "FullAttentionSpec"}, [0])}
            worker.local_remote_block_port_mapping = {}
            worker.remote_port_send_num = {}
            meta = SimpleNamespace(
                remote_pcp_size=1,
                remote_dcp_size=8,
                remote_ptp_size=8,
                remote_port=5000,
                remote_host="localhost",
                remote_engine_id="prefill",
                remote_request_id="remote",
                remote_multi_nodes_meta_mapping={},
                remote_block_size=4,
                num_external_tokens=32,
                num_computed_tokens=0,
                num_prompt_blocks=8,
                local_block_ids=(list(range(8)),),
                remote_block_ids=([9],),
            )
            worker._get_kv_split_metadata("remote", meta)
    message = str(error.value)
    assert "request=remote" in message
    assert f"missing_(pp_rank,cp_rank)={[(1, rank) for rank in range(8)]}" in message
    assert "selected_ports=[5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007]" in message
    assert thread.request_queue.empty()
    thread.engine.batch_transfer_sync_read.assert_not_called()


def test_dsa_main_and_indexer_have_distinct_metadata_indices():
    worker = object.__new__(MooncakeConnectorWorker)
    worker._dsa_pd_offload = True
    worker.total_layers = 4
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="deepseek_v3"))
    )
    names = [MAIN, "model.layers.0.self_attn.indexer.k_cache", "model.mtp.layers.0.attn"]
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=names)])
    worker._get_layer_spec = lambda name: SimpleNamespace(num_kv_heads=1)
    worker._get_spec_total_num_kv_heads = lambda spec, index: 1
    groups = worker._build_kv_group2layeridx()
    mapping = mooncake_connector.build_layer_name_to_metadata_idx(groups)
    assert len(set(mapping.values())) == len(names)
    assert set(mapping) == set(names)


def test_remote_group_and_layer_order_does_not_change_component_addresses():
    worker, thread, command = make_worker()
    tasks = submit(worker, command)
    port = tasks[0]["remote_handshake_port"]
    # Remote metadata order differs from Decode, as do its transfer group IDs.
    thread.remote_kv_group2layeridx["prefill"][port] = {
        7: ({"layer_names": [INDEXER]}, [0]),
        9: ({"layer_names": [MAIN]}, [1]),
    }
    for table in (
        thread.kv_caches_base_addr,
        thread.remote_block_stride_per_addr,
        thread.remote_block_size_scale,
        thread.remote_block_len_per_addr,
        thread.remote_cache_dtypes,
    ):
        table["prefill"][port] = list(reversed(table["prefill"][port]))
    thread._execute_dsa_receive(tasks[0])
    assert thread.engine.batch_transfer_sync_read.call_count == 2
    indexer_call, main_call = thread.engine.batch_transfer_sync_read.call_args_list
    assert all(src >= 30000 for src in indexer_call.args[2])
    assert all(10000 <= src < 30000 for src in main_call.args[2])
