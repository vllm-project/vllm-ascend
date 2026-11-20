# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import json
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.dsv41_state import (
    STATE_ACQUIRE_MSG,
    STATE_DONE_MSG,
    STATE_META_MSG,
    CachePlane,
    StateSnapshot,
    build_copy_plan,
    validate_pd_config,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import GET_META_MSG
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.dsv41 import (
    transfer_addresses,
    zero_state_destination,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import (
    MooncakePullConnectorScheduler,
    MooncakeSchedulerRecvingThread,
    MooncakeSchedulerSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullConnectorWorker,
    MooncakePullRecvingThread,
    UnfencedStateTransferError,
)

LAYERS = ["m.long_kv_cache", "m.indexer.k_cache", "m.swa_cache", "m.compressor.state_cache"]


def worker_metadata(history_capacity=8, state_capacity=5, rank=0):
    planes = [
        CachePlane(LAYERS[0], 0, "history", 0, history_capacity, 64, 4, 256, 2, "bfloat16", (64, 2)),
        CachePlane(LAYERS[1], 0, "history", 0, history_capacity, 64, 2, 256, 2, "int8", (64, 2)),
        CachePlane(LAYERS[1], 1, "history", 0, history_capacity, 64, 2, 256, 2, "float16", (64, 1)),
        CachePlane(LAYERS[2], 0, "window", 1, state_capacity, 128, 4, 512, 1, "bfloat16", (128, 2)),
        CachePlane(LAYERS[3], 0, "compressor", 2, state_capacity, 32, 8, 256, 1, "float32", (32, 2)),
    ]
    return MooncakeTransferMetadata(
        engine_id="producer",
        te_rpc_port=10000 + rank,
        block_size=128,
        cp_kv_cache_interleave_size=1,
        num_blocks=history_capacity,
        layer_names=LAYERS,
        layer_block_sizes=[128, 128, 128, 32],
        group_indices=[0, 0, 1, 2],
        kv_caches_base_addr=[[100000 + rank * 1000], [200000, 200128], [300000], [400000]],
        block_strides=[[256], [256, 256], [512], [256]],
        block_lens=[[256], [128, 128], [512], [256]],
        block_shapes=[[(64, 2)], [(64, 2), (64, 1)], [(128, 2)], [(32, 2)]],
        block_size_scales=[[1], [1, 1], [1], [1]],
        indexer_replication_sizes=[0] * 4,
        dcp_sharded=[False] * 4,
        local_ip="127.0.0.1",
        state_layout={
            "version": 1,
            "model_id": "weights",
            "tp_size": 4,
            "fingerprint": "format",
            "planes": [asdict(plane) for plane in planes],
        },
    )


def sender(*, address_delta=0):
    metadata = {}
    for rank in range(4):
        worker = worker_metadata(rank=rank)
        metadata[rank] = replace(
            worker,
            kv_caches_base_addr=[[addr + address_delta for addr in row] for row in worker.kv_caches_base_addr],
        )
    return MooncakeSchedulerSendingThread(
        "127.0.0.1",
        10020,
        "producer",
        metadata,
        4,
        1,
        1,
        1,
        False,
        threading.Event(),
        kv_lease_duration=10,
    )


def snapshot(boundary=256, generation="generation", request_id="prefill"):
    return StateSnapshot.create(generation, request_id, "weights", boundary, boundary + 1, 4)


def decode_handshake(lease_sender):
    return msgspec.msgpack.decode(lease_sender.encoded_metadata, type=MooncakeTransferMetadataGroups)


@pytest.mark.parametrize(
    ("boundary", "history_rows", "window_rows", "compressor_rows"),
    [
        (1, 0, 1, 1),
        (31, 15, 31, 31),
        (32, 16, 32, 32),
        (127, 63, 127, 32),
        (128, 64, 128, 32),
        (129, 64, 128, 32),
        (257, 128, 128, 32),
    ],
)
def test_encoded_copy_ranges_at_boundaries(boundary, history_rows, window_rows, compressor_rows):
    snap = snapshot(boundary)
    restored = StateSnapshot.from_wire(json.loads(json.dumps(snap.to_wire())))
    assert restored == snap
    remote = decode_handshake(sender()).metadata_by_pp_rank[0]
    local = worker_metadata(11, 7)
    local_ids = ([8, 10, 5], [6], [4])
    remote_ids = ([1, 2, 3], [2], [3])
    ranges = build_copy_plan(restored, local.state_layout, remote.state_layout, local_ids, remote_ids)
    counts = {(name, index): 0 for name in LAYERS for index in (0, 1)}
    for item in ranges:
        counts[item.name, item.cache_index] += item.nbytes
    assert [counts[name, 0] for name in LAYERS] == [
        history_rows * 4,
        history_rows * 2,
        window_rows * 4,
        compressor_rows * 8,
    ]
    assert counts[LAYERS[1], 1] == history_rows * 2
    sources, destinations, sizes = transfer_addresses(restored, local, remote, 2, local_ids, remote_ids)
    assert sum(sizes) == sum(counts.values())
    if history_rows:
        assert sources[0] == 102000 + 256
        assert destinations[0] == 100000 + 8 * 256
    with pytest.raises(ValueError, match="invalid|Invalid"):
        build_copy_plan(restored, local.state_layout, remote.state_layout, local_ids, ([1, 2, 3], [0], [3]))


def test_generation_lease_survives_partial_tp_until_real_aggregate_completion():
    p = sender()
    snap = snapshot()
    ids = ([1, 2], [3], [4])
    p.add_state_lease(snap, time.monotonic(), ids)
    acquire = (STATE_ACQUIRE_MSG, snap.transfer_id, "d/request", 2, snap.to_wire(), ids, decode_handshake(p).engine_id)
    assert p._state_control(acquire) == b"ACK"
    assert p._state_control(acquire) == b"ACK"
    p._handle_finished_request(snap.source_request_id)
    p._handle_heartbeat([snap.source_request_id])
    with patch("time.monotonic", return_value=p.state_leases[snap.transfer_id].expires_at + 1):
        assert not p.get_and_clear_finished_requests()
    assert p._state_control((STATE_DONE_MSG, snap.transfer_id, "another-d")) == b"REJECT"
    assert not p.get_and_clear_finished_requests()
    assert p._state_control((STATE_DONE_MSG, snap.transfer_id, "d/request")) == b"ACK"
    assert p.get_and_clear_finished_requests() == {snap.source_request_id}
    next_snap = snapshot(generation="new-generation")
    p.add_state_lease(next_snap, time.monotonic(), ids)
    assert p._state_control((STATE_DONE_MSG, snap.transfer_id, "d/request")) == b"ACK"
    assert not p.get_and_clear_finished_requests()
    assert p._state_control(acquire) == b"REJECT"
    with patch("time.monotonic", return_value=p.state_leases[next_snap.transfer_id].expires_at + 1):
        assert p.get_and_clear_finished_requests() == {next_snap.source_request_id}


def test_terminal_reject_and_transient_completion_retry():
    d = MooncakeSchedulerRecvingThread(threading.Event())
    message = ("127.0.0.1", 10020, "p", "generation", "d/request")
    socket = Mock()
    with (
        patch.object(d, "_get_remote_socket", return_value=socket),
        patch.object(d, "_return_remote_socket") as returned,
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler.ensure_zmq_send") as send,
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler.ensure_zmq_recv",
            side_effect=[TimeoutError(), b"REJECT"],
        ),
    ):
        d._notify_completion(message)
    assert send.call_count == 2
    returned.assert_called_once()


def test_restart_with_same_configured_engine_id_fetches_fresh_registration():
    old_p, new_p = sender(), sender(address_delta=1_000_000)
    old_id, new_id = decode_handshake(old_p).engine_id, decode_handshake(new_p).engine_id
    d = object.__new__(MooncakePullRecvingThread)
    d.local_metadata = worker_metadata()
    d.remote_metadata = {}
    d.encoder = msgspec.msgpack.Encoder()
    d.decoder = msgspec.msgpack.Decoder(MooncakeTransferMetadataGroups)
    with (
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.zmq_ctx"),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.ensure_zmq_send"),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.ensure_zmq_recv",
            side_effect=[old_p.encoded_metadata, new_p.encoded_metadata],
        ) as recv,
    ):
        old = d._get_remote_metadata(old_id, old_p.host, old_p.port)
        new = d._get_remote_metadata(new_id, new_p.host, new_p.port)
        assert d._get_remote_metadata(new_id, new_p.host, new_p.port) is new
    assert recv.call_count == 2
    assert old_id != new_id
    for metadata, expected in ((old, 102000), (new, 1102000)):
        pp = metadata.metadata_by_pp_rank[0]
        rank = pp.metadata_by_pcp_rank[0].metadata_by_tp_rank[2]
        assert rank.kv_caches_base_addr[pp.layer_names.index(LAYERS[0])] == [expected]

    # A fresh lease cannot authorize READ with a stale address advertisement.
    snap = snapshot()
    ids = ([1, 2], [3], [4])
    new_p.add_state_lease(snap, time.monotonic(), ids)
    acquire = (STATE_ACQUIRE_MSG, snap.transfer_id, "d/request", 2, snap.to_wire(), ids)
    assert new_p._state_control((*acquire, old_id)) == b"REJECT"
    assert not new_p.state_leases[snap.transfer_id].acquired_ranks
    assert new_p._state_control((*acquire, new_id)) == b"ACK"


def test_copy_failure_quarantines_instance_without_completion():
    remote = decode_handshake(sender())
    local = worker_metadata(11, 7)
    req = ReqMeta(
        local_block_ids=([8, 10], [6], [4]),
        local_full_block_ids=([8, 10], [6], [4]),
        local_num_prompt_tokens=257,
        num_external_tokens=256,
        num_computed_tokens=0,
        remote_block_ids=([1, 2], [2], [3]),
        remote_host="127.0.0.1",
        remote_port=10020,
        remote_engine_id=remote.engine_id,
        remote_request_id="prefill",
        remote_num_prompt_tokens=256,
        state_snapshot=snapshot().to_wire(),
        state_consumer="d/request",
    )
    thread = object.__new__(MooncakePullRecvingThread)
    thread.tp_size = 4
    thread.tp_rank = 2
    thread.local_metadata = local
    thread.state_caches = {}
    thread._shutdown_lock = threading.Lock()
    thread._unfenced_state_requests = {}
    thread.encoder = msgspec.msgpack.Encoder()
    thread.engine = Mock()
    with (
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.zmq_ctx"),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.ensure_zmq_send") as send,
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker.ensure_zmq_recv", return_value=b"ACK"),
    ):
        thread.engine.batch_transfer_sync_read.return_value = 0
        assert thread._handle_state_requests(remote, req.remote_host, req.remote_port, {"d": req}) == set()
        assert msgspec.msgpack.decode(send.call_args.args[1])[-1] == remote.engine_id
        _, destinations, sources, sizes = thread.engine.batch_transfer_sync_read.call_args.args
        assert destinations[0] == 100000 + 8 * 256
        assert sources[0] == 102000 + 256
        assert sum(sizes) == 128 * 8 + 128 * 4 + 32 * 8
        thread.engine.batch_transfer_sync_read.return_value = -1
        assert thread._handle_state_requests(remote, req.remote_host, req.remote_port, {"d": req}) == {"d"}
    with pytest.raises(UnfencedStateTransferError, match="terminal DMA fence"):
        thread.get_and_clear_finished_requests()


def test_destination_zeroes_only_its_request_slots():
    layout = worker_metadata().state_layout
    rings = {
        LAYERS[2]: torch.ones((5, 128, 2)),
        LAYERS[3]: torch.ones((5, 32, 2)),
    }
    zero_state_destination(rings, ([1, 2], [2], [3]), layout)
    assert not rings[LAYERS[2]][2].any() and rings[LAYERS[2]][3].all()
    assert not rings[LAYERS[3]][3].any() and rings[LAYERS[3]][2].all()


def state_decoder():
    decoder = object.__new__(MooncakePullConnectorScheduler)
    decoder.state_mode, decoder.need_truncate, decoder.state_model_id, decoder.tp_size = True, True, "weights", 4
    decoder.engine_id = "decoder"
    decoder._reqs_need_recv, decoder._reqs_recv_info = {}, {}
    decoder._failed_state_receives = set()
    decoder._state_receive_completions = OrderedDict()
    decoder._active_state_receives = set()
    decoder._reqs_in_batch = set()
    decoder._sending_thread = decoder._heartbeat_thread = None
    decoder._recving_thread = Mock()
    decoder._log_mamba_request_allocation = Mock()
    return decoder


def test_scheduler_handoff_boundary_and_aggregate_completion():
    p = object.__new__(MooncakePullConnectorScheduler)
    p.state_mode = True
    p.need_truncate = True
    p.state_model_id = "weights"
    p.tp_size = 4
    p.engine_id = "producer"
    p.side_channel_host = "127.0.0.1"
    p.side_channel_port = 10020
    p._kv_lease_duration = 10
    p._heartbeat_thread = None
    p._sending_thread = sender()
    p._reqs_recv_info = {}
    p._reqs_need_send = {}
    p.group_unique_specs = []
    p._get_transfer_block_ids = Mock(return_value=([1, 2], [2], [3]))
    request = Request("prefill", list(range(257)), SamplingParams(max_tokens=4), pooling_params=None)
    request.kv_transfer_params = {"do_remote_decode": True}
    p.on_new_request(request)
    assert request.num_prompt_tokens == 256 and len(request.all_token_ids) == 256
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    request.num_computed_tokens = 256
    request.append_output_token_ids(99)
    retain, wire = p.request_finished(request, ([1, 2], [2], [3]))
    assert retain and wire["remote_num_prompt_tokens"] == 256
    assert wire["remote_engine_id"] == decode_handshake(p._sending_thread).engine_id

    d = state_decoder()
    receiver = Request("decode", list(range(257)), SamplingParams(max_tokens=4), pooling_params=None)
    receiver.kv_transfer_params = wire
    assert d.get_num_new_matched_tokens(receiver, 0) == (256, True)
    blocks = Mock()
    blocks.get_unhashed_block_ids_all_groups.return_value = ([5, 7], [1], [4])
    blocks.get_block_ids.return_value = ([5, 7], [1], [4])
    d.update_state_after_alloc(receiver, blocks, 256)
    metadata = d.build_connector_meta(None)
    assert isinstance(metadata, MooncakeConnectorMetadata)
    assert metadata.requests["decode"].state_snapshot["boundary"] == 256
    assert not d.request_finished(receiver, blocks.get_block_ids())[0]
    d._recving_thread.add_request.assert_not_called()
    d.update_connector_output(KVConnectorOutput(finished_recving={"decode"}))
    d._recving_thread.add_request.assert_called_once_with(
        "127.0.0.1", 10020, "prefill", wire["dsv41_state_snapshot"]["transfer_id"], "decoder/decode"
    )


@pytest.mark.parametrize("invalid", ["missing_snapshot", "checkpoint", "boundary", "replay", "missing_route"])
def test_rejected_handoff_reports_request_failure_without_read_or_done(invalid):
    d = state_decoder()
    source = snapshot()
    params = {
        "do_remote_prefill": True,
        "remote_engine_id": "producer",
        "remote_request_id": source.source_request_id,
        "remote_host": "127.0.0.1",
        "remote_port": 10020,
        "remote_num_prompt_tokens": source.boundary,
        "remote_block_ids": ([1], [2], [3]),
        "dsv41_state_snapshot": source.to_wire(),
    }
    if invalid == "missing_snapshot":
        del params["dsv41_state_snapshot"]
    elif invalid == "checkpoint":
        params["dsv41_state_snapshot"]["model_id"] = "other-weights"
    elif invalid == "boundary":
        params["remote_num_prompt_tokens"] += 1
    elif invalid == "replay":
        d._active_state_receives.add(("127.0.0.1", 10020, source.transfer_id))
    else:
        del params["remote_host"]

    request = Request("rejected", list(range(source.original_prompt_tokens)), SamplingParams(max_tokens=1), None)
    request.kv_transfer_params = params
    assert d.get_num_new_matched_tokens(request, 0) == (source.boundary, True)
    blocks = Mock()
    d.update_state_after_alloc(request, blocks, source.boundary)
    blocks.get_block_ids.assert_not_called()
    meta = d.build_connector_meta(None)
    assert not meta.requests and meta.pull_failed_requests == {"rejected"}
    assert not d.build_connector_meta(None).pull_failed_requests

    worker = object.__new__(MooncakePullConnectorWorker)
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    receiver = worker._recving_thread = object.__new__(MooncakePullRecvingThread)
    receiver.finished_requests = queue.SimpleQueue()
    receiver.raise_if_unfenced_state_dma = Mock()
    receiver.add_requests = Mock()
    worker.start_load_kv(meta)
    results = worker.get_transfer_results()
    assert results.finished_recving == results.failed_recving == {"rejected"}
    receiver.add_requests.assert_not_called()
    assert d.request_finished(request, ([], [], [])) == (False, None)
    d.update_connector_output(
        KVConnectorOutput(finished_recving=results.finished_recving, failed_recving=results.failed_recving)
    )
    d._recving_thread.add_request.assert_not_called()
    assert bool(d._active_state_receives) == (invalid == "replay")


def test_one_token_prompt_has_no_transferable_prefix():
    p = state_decoder()
    request = Request("one-token", [7], SamplingParams(max_tokens=4), None)
    request.kv_transfer_params = {"do_remote_decode": True}
    p.on_new_request(request)
    assert request.num_prompt_tokens == request.max_tokens == 1
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    retain, params = p.request_finished(request, ([1], [2], [3]))
    assert not retain
    assert params == {"do_remote_prefill": False, "do_remote_decode": False}

    d = state_decoder()
    request.kv_transfer_params = params
    assert d.get_num_new_matched_tokens(request, 0) == (0, False)


@pytest.mark.parametrize("generation_status", ["unused", "active", "completed"])
def test_explicit_cache_bypass_returns_only_an_unstarted_source(generation_status):
    d = state_decoder()
    snap = snapshot()
    key = ("127.0.0.1", 10020, snap.transfer_id)
    params = {
        "do_remote_prefill": True,
        "remote_engine_id": "producer",
        "remote_host": key[0],
        "remote_port": key[1],
        "remote_request_id": snap.source_request_id,
        "remote_num_prompt_tokens": snap.boundary,
        "remote_block_ids": ([1, 2], [3], [4]),
        "dsv41_state_snapshot": snap.to_wire(),
    }
    if generation_status == "active":
        d._active_state_receives.add(key)
    elif generation_status == "completed":
        d._state_receive_completions[key] = None
    request = Request(
        "score", list(range(snap.original_prompt_tokens)), SamplingParams(max_tokens=1, prompt_logprobs=1), None
    )
    assert request.skip_reading_external_kv
    request.kv_transfer_params = params
    assert d.get_num_new_matched_tokens(request, 0) == (snap.boundary, True)
    d.update_state_after_alloc(request, Mock(), 0)  # Core overrides the match for a bypass.
    assert not d.build_connector_meta(None).requests
    assert d.request_finished(request, ([], [], [])) == (False, None)
    if generation_status == "unused":
        d._recving_thread.add_request.assert_called_once_with(*key[:2], snap.source_request_id, key[2], "decoder/score")
    else:
        d._recving_thread.add_request.assert_not_called()


def test_transfer_config_and_peer_layout_validation():
    extra = {"state_model_id": "weights"}
    cfg = SimpleNamespace(
        additional_config={},
        kv_transfer_config=SimpleNamespace(
            kv_connector="MooncakePullConnector",
            kv_role="kv_producer",
            kv_load_failure_policy="fail",
            get_from_extra_config=extra.get,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        speculative_config=None,
        model_config=SimpleNamespace(enable_sleep_mode=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    assert validate_pd_config(cfg) == "weights"
    cfg.kv_transfer_config.kv_load_failure_policy = "recompute"
    with pytest.raises(ValueError, match="kv_load_failure_policy"):
        validate_pd_config(cfg)
    incoming = {rank: worker_metadata(rank=rank) for rank in range(4)}
    incoming[3] = replace(incoming[3], state_layout=None)
    with pytest.raises(ValueError, match="state_layout"):
        MooncakeSchedulerSendingThread("127.0.0.1", 10020, "producer", incoming, 4, 1, 1, 1, False, threading.Event())


def test_destination_event_completes_before_publish():
    worker = object.__new__(MooncakePullConnectorWorker)
    worker.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
    worker._shutdown = False
    receiver = worker._recving_thread = Mock()
    receiver.local_metadata = worker_metadata()
    rings = receiver.state_caches = {
        LAYERS[2]: torch.ones((5, 128, 2)),
        LAYERS[3]: torch.ones((5, 32, 2)),
    }
    meta = SimpleNamespace(
        state_snapshot=snapshot().to_wire(), local_full_block_ids=([1, 2], [2], [3]), remote_engine_id="p"
    )
    order = []
    event = Mock()

    def record(_stream):
        assert not rings[LAYERS[2]][2].any()
        assert not rings[LAYERS[3]][3].any()
        order.append("record")

    event.record.side_effect = record
    event.synchronize.side_effect = lambda: order.append("synchronized")
    receiver.add_requests.side_effect = lambda *args, **kwargs: order.append("published")
    metadata = MooncakeConnectorMetadata()
    metadata.requests["d"] = meta
    with patch.object(torch, "npu", SimpleNamespace(Event=lambda: event, current_stream=lambda: None), create=True):
        worker.start_load_kv(metadata)
    assert order == ["record", "synchronized", "published"]
    receiver.add_requests.assert_called_once_with("p", {"d": meta}, state_destination_ready=True)


def test_state_peer_never_publishes_addresses_to_legacy_requests():
    p = sender()
    socket = Mock()

    class EndOfTest(BaseException):
        pass

    socket.recv_multipart.side_effect = [
        [b"d", b"", msgspec.msgpack.encode((GET_META_MSG,))],
        [b"d", b"", msgspec.msgpack.encode((STATE_META_MSG,))],
        EndOfTest(),
    ]
    with pytest.raises(EndOfTest):
        p._run_busy_loop(socket)
    assert socket.send_multipart.call_args_list[0].args == ((b"d", b"", b""),)
    assert socket.send_multipart.call_args_list[1].args == ((b"d", b"", p.encoded_metadata),)
