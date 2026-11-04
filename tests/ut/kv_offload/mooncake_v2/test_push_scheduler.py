# SPDX-License-Identifier: Apache-2.0

import json
import socket
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import httpx
import msgspec
import pytest
import zmq
from vllm.v1.request import RequestStatus

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import bootstrap_server, push_scheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_scheduler import GET_META_MSG
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import (
    ACK_MSG,
    DONE_RECVING_MSG,
    MooncakeSchedulerSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_scheduler import MooncakePushConnectorScheduler


def workers():
    return {
        (pp, pcp, tp): MooncakeTransferMetadata(
            engine_id="d",
            te_rpc_port=9000 + pcp * 2 + tp,
            block_size=16,
            num_blocks=8,
            layer_names=[f"layer.{pp}"],
            layer_block_sizes=[16],
            group_indices=[0],
            kv_caches_base_addr=[[4096 + pp * 1024 + pcp * 256 + tp * 128]],
            block_strides=[[128]],
            block_lens=[[128]],
            block_shapes=[[(16, 8)]],
            block_size_scales=[[1]],
            local_ip=f"10.0.0.{pp + 1}",
            handshake_port=8000 + pcp * 2 + tp,
        )
        for pp in range(2)
        for pcp in range(2)
        for tp in range(2)
    }


@pytest.fixture
def scheduler():
    obj = MooncakePushConnectorScheduler.__new__(MooncakePushConnectorScheduler)
    obj.kv_transfer_config = SimpleNamespace(is_kv_producer=False)
    obj.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    obj.engine_id = "d"
    obj.pp_size = obj.pcp_size = obj.tp_size = 2
    obj.dcp_size = 4
    obj.side_channel_host = "127.0.0.1"
    obj.side_channel_port = 0
    obj._metadata_thread = None
    obj.bootstrap_server = None
    obj._registered_worker_metadata = None
    obj._reqs_need_send_layerwise = {}
    obj._reqs_sent_to_worker = set()
    obj._reqs_need_send = {}
    obj._reqs_finished_sending = set()
    obj._reqs_pending_send = {}
    obj._reqs_aborted = {}
    obj.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=2))
    yield obj
    obj.shutdown()


def query(thread, message):
    with zmq.Context.instance().socket(zmq.REQ) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.setsockopt(zmq.SNDTIMEO, 2000)
        sock.connect(thread.address)
        sock.send(msgspec.msgpack.encode(message))
        return sock.recv()


def test_d_serves_all_ranks_and_snapshot_is_stable(scheduler):
    metadata = workers()
    scheduler.set_xfer_handshake_metadata_from_workers(metadata)
    thread = scheduler._metadata_thread
    response = query(thread, (GET_META_MSG, "d"))
    groups = msgspec.msgpack.decode(response, type=MooncakeTransferMetadataGroups)
    assert groups.engine_id == "d"
    assert (groups.pp_size, groups.pcp_size, groups.tp_size, groups.dcp_size) == (2, 2, 2, 4)
    for (pp, pcp, tp), worker in metadata.items():
        actual = groups.metadata_by_pp_rank[pp].metadata_by_pcp_rank[pcp].metadata_by_tp_rank[tp]
        assert actual.kv_caches_base_addr == worker.kv_caches_base_addr
        assert actual.local_ip == worker.local_ip
    scheduler.set_xfer_handshake_metadata_from_workers(metadata)
    assert scheduler._metadata_thread is thread
    metadata[(0, 0, 0)].kv_caches_base_addr[0][0] += 128
    assert query(thread, (GET_META_MSG, "d")) == response
    with pytest.raises(RuntimeError, match="Replacing"):
        scheduler.set_xfer_handshake_metadata_from_workers(metadata)
    scheduler.shutdown()
    assert not thread.is_alive()
    scheduler.shutdown()


def test_invalid_queries_do_not_stop_listener(scheduler):
    scheduler.set_xfer_handshake_metadata_from_workers(workers())
    for message in ((GET_META_MSG, "stale-engine"), (b"done_recving_msg", "request"), ()):
        assert "error" in msgspec.msgpack.decode(query(scheduler._metadata_thread, message))
    assert msgspec.msgpack.decode(query(scheduler._metadata_thread, (GET_META_MSG, "d")))["engine_id"] == "d"


def test_missing_or_foreign_metadata_does_not_start_thread(scheduler):
    metadata = workers()
    with pytest.raises(ValueError, match="incomplete"):
        scheduler.set_xfer_handshake_metadata_from_workers({(0, 0, 0): metadata[(0, 0, 0)]})
    metadata[(0, 0, 0)] = replace(metadata[(0, 0, 0)], engine_id="other")
    with pytest.raises(ValueError, match="engine ID mismatch"):
        scheduler.set_xfer_handshake_metadata_from_workers(metadata)
    assert scheduler._metadata_thread is None


def test_p_does_not_start_metadata_service(scheduler, monkeypatch):
    scheduler.kv_transfer_config.is_kv_producer = True
    registered = []
    monkeypatch.setattr(push_scheduler, "get_mooncake_bootstrap_addr", lambda config: ("bootstrap", 8998))
    monkeypatch.setattr(push_scheduler, "register_workers_with_bootstrap", lambda **kwargs: registered.append(kwargs))
    scheduler.set_xfer_handshake_metadata_from_workers(workers())
    assert len(registered) == 1
    assert scheduler._metadata_thread is None


def test_bind_failure_is_reported_and_thread_is_cleaned_up(scheduler):
    with zmq.Context.instance().socket(zmq.REP) as occupied:
        occupied.bind("tcp://127.0.0.1:*")
        scheduler.side_channel_port = int(occupied.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(":", 1)[1])
        with pytest.raises(RuntimeError, match="metadata thread failed"):
            scheduler.set_xfer_handshake_metadata_from_workers(workers())
        assert scheduler._metadata_thread is None


def test_shared_service_preserves_pull_query_and_completion_protocol():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    ready = threading.Event()
    thread = MooncakeSchedulerSendingThread("127.0.0.1", port, "d", workers(), 2, 2, 2, 4, False, ready)
    thread.start()
    try:
        assert ready.wait(5)
        assert thread.error is None
        # Pull clients send the original one-element request.
        groups = msgspec.msgpack.decode(query(thread, (GET_META_MSG,)), type=MooncakeTransferMetadataGroups)
        assert groups.engine_id == "d"
        thread.add_delayed_request("request", time.time())
        assert query(thread, (DONE_RECVING_MSG, "request")) == ACK_MSG
        assert "request" not in thread.delayed_free_requests
        assert thread.finished_requests.get_nowait() == "request"
        assert query(thread, (DONE_RECVING_MSG, "request")) == ACK_MSG
        assert thread.finished_requests.empty()
    finally:
        thread.shutdown()
    assert not thread.is_alive()


@pytest.fixture
def producer_scheduler(scheduler):
    scheduler.kv_transfer_config.is_kv_producer = True
    return scheduler


def prefill_request(request_id="p-request"):
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=48,
        kv_transfer_params={"do_remote_decode": True, "transfer_id": "xfer-123"},
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )


def step(request, blocks=None, computed=0, scheduled=16, *, resumed=False):
    cached = SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set())
    new = []
    if computed == 0:
        new.append(SimpleNamespace(req_id=request.request_id, block_ids=blocks, num_computed_tokens=0))
    else:
        cached = SimpleNamespace(
            req_ids=[request.request_id],
            new_block_ids=[blocks],
            num_computed_tokens=[computed],
            resumed_req_ids={request.request_id} if resumed else set(),
        )
    return SimpleNamespace(
        scheduled_new_reqs=new,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={request.request_id: scheduled},
        preempted_req_ids=set(),
    )


def allocate(scheduler, request, blocks=([1], [11])):
    scheduler.update_state_after_alloc(request, SimpleNamespace(get_block_ids=lambda: blocks), 0)


def test_producer_snapshots_new_and_cached_prefill_chunks(producer_scheduler):
    scheduler = producer_scheduler
    request = prefill_request()
    blocks = ([1], [11])
    allocate(scheduler, request, blocks)
    first = scheduler.build_connector_meta(step(request, blocks))
    assert first.push_send_requests[request.request_id].local_block_ids == ((1,), (11,))
    second = scheduler.build_connector_meta(step(request, ([2], [12]), computed=16))
    chunk = second.push_send_requests[request.request_id]
    assert chunk.local_block_ids == ((1, 2), (11, 12))
    assert (chunk.num_computed_tokens, chunk.num_scheduled_tokens, chunk.num_prompt_tokens) == (16, 16, 48)
    assert chunk.transfer_id == "xfer-123"
    assert not second.push_requests
    third = scheduler.build_connector_meta(step(request, ([3], [13]), computed=32, scheduled=20))
    assert third.push_send_requests[request.request_id].num_scheduled_tokens == 16
    assert first.push_send_requests[request.request_id].local_block_ids == ((1,), (11,))
    assert not scheduler.build_connector_meta(step(request, None, computed=48, scheduled=1)).push_send_requests


def test_prefix_hit_publishes_cached_prefix_and_new_token_range(producer_scheduler):
    request = prefill_request()
    allocate(producer_scheduler, request, ([1, 2, 3], [11, 12, 13]))
    output = step(request, ([1, 2, 3], [11, 12, 13]))
    output.scheduled_new_reqs[0].num_computed_tokens = 32
    chunk = producer_scheduler.build_connector_meta(output).push_send_requests[request.request_id]
    assert (chunk.num_computed_tokens, chunk.num_scheduled_tokens) == (32, 16)
    assert chunk.local_block_ids == ((1, 2, 3), (11, 12, 13))


@pytest.mark.parametrize("aborted", [False, True])
def test_producer_retains_blocks_until_worker_finishes(producer_scheduler, aborted):
    scheduler = producer_scheduler
    request = prefill_request()
    allocate(scheduler, request)
    scheduler.build_connector_meta(step(request, ([1], [11])))
    if aborted:
        request.status = RequestStatus.FINISHED_ABORTED
    assert scheduler.request_finished(request, ([1], [11])) == (True, None)
    terminal_step = step(request, None, computed=16, scheduled=0)
    meta = scheduler.build_connector_meta(terminal_step)
    assert set(meta.requests_to_send) == {request.request_id}
    assert meta.push_aborted_requests == ({request.request_id: "xfer-123"} if aborted else {})
    assert not meta.push_send_requests
    assert not scheduler.build_connector_meta(terminal_step).requests_to_send
    output = SimpleNamespace(finished_sending={request.request_id}, finished_recving=None)
    scheduler.update_connector_output(output)
    assert output.finished_sending == {request.request_id}
    assert not scheduler._reqs_need_send
    assert not scheduler._reqs_need_send_layerwise
    # A duplicate completion cannot cause core to free an already-freed request.
    scheduler.update_connector_output(output)
    assert not output.finished_sending


def test_early_completion_does_not_free_running_request(producer_scheduler):
    scheduler = producer_scheduler
    request = prefill_request()
    allocate(scheduler, request)
    scheduler.build_connector_meta(step(request, ([1], [11])))
    output = SimpleNamespace(finished_sending={request.request_id}, finished_recving=None)
    scheduler.update_connector_output(output)
    assert not output.finished_sending
    assert request.request_id in scheduler._reqs_finished_sending
    assert scheduler.request_finished(request, ([1], [11])) == (False, None)
    assert not scheduler._reqs_need_send_layerwise
    assert not scheduler._reqs_finished_sending


def test_abort_before_dispatch_does_not_pin_blocks(producer_scheduler):
    request = prefill_request()
    allocate(producer_scheduler, request)
    request.status = RequestStatus.FINISHED_ABORTED
    assert producer_scheduler.request_finished(request, ([1], [11])) == (False, None)
    assert not producer_scheduler._reqs_need_send_layerwise
    meta = producer_scheduler.build_connector_meta(step(request, None, computed=16, scheduled=0))
    assert meta.push_aborted_requests == {request.request_id: "xfer-123"}


def test_missing_transfer_id_is_rejected_before_dispatch(producer_scheduler):
    request = prefill_request()
    del request.kv_transfer_params["transfer_id"]
    with pytest.raises(ValueError, match="transfer_id"):
        allocate(producer_scheduler, request)
    assert not producer_scheduler._reqs_need_send_layerwise


def test_preemption_of_published_source_is_rejected(producer_scheduler):
    request = prefill_request()
    allocate(producer_scheduler, request)
    producer_scheduler.build_connector_meta(step(request, ([1], [11])))
    output = step(request, None, computed=16)
    output.preempted_req_ids = {request.request_id}
    with pytest.raises(RuntimeError, match="Preempting"):
        producer_scheduler.build_connector_meta(output)
    with pytest.raises(RuntimeError, match="Preempting"):
        allocate(producer_scheduler, request)


def test_register_all_workers_from_scheduler_once(producer_scheduler, monkeypatch):
    scheduler = producer_scheduler
    calls = []
    factory = httpx.Client

    def handle(request):
        calls.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"status": "ok"})

    monkeypatch.setattr(push_scheduler, "get_mooncake_bootstrap_addr", lambda config: ("bootstrap", 8998))
    monkeypatch.setattr(
        bootstrap_server.httpx, "Client", lambda **kwargs: factory(transport=httpx.MockTransport(handle))
    )
    scheduler.set_xfer_handshake_metadata_from_workers(workers())
    scheduler.set_xfer_handshake_metadata_from_workers(workers())
    assert len(calls) == 8
    assert {url for url, _ in calls} == {"http://bootstrap:8998/register"}
    for _, payload in calls:
        key = (payload["pp_rank"], payload["pcp_rank"], payload["tp_rank"])
        worker = workers()[key]
        assert payload["addr"] == f"tcp://{worker.local_ip}:{worker.handshake_port}"
        assert payload["dp_rank"] == 2
        assert (payload["pp_size"], payload["pcp_size"], payload["tp_size"], payload["dcp_size"]) == (2, 2, 2, 4)
    changed = workers()
    changed[(0, 0, 0)] = replace(changed[(0, 0, 0)], handshake_port=9999)
    with pytest.raises(RuntimeError, match="Replacing"):
        scheduler.set_xfer_handshake_metadata_from_workers(changed)
    assert len(calls) == 8
    assert scheduler._metadata_thread is None


def test_registration_error_does_not_retry_or_mark_complete(producer_scheduler, monkeypatch):
    factory = httpx.Client
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(500)

    monkeypatch.setattr(push_scheduler, "get_mooncake_bootstrap_addr", lambda config: ("bootstrap", 8998))
    monkeypatch.setattr(
        bootstrap_server.httpx, "Client", lambda **kwargs: factory(transport=httpx.MockTransport(handle))
    )
    with pytest.raises(httpx.HTTPStatusError):
        producer_scheduler.set_xfer_handshake_metadata_from_workers(workers())
    assert len(calls) == 1
    assert producer_scheduler._registered_worker_metadata is None


def test_validate_every_worker_before_registration(producer_scheduler, monkeypatch):
    def unexpected_client(**kwargs):
        pytest.fail("Must validate the full handshake before any HTTP registration")

    monkeypatch.setattr(push_scheduler, "get_mooncake_bootstrap_addr", lambda config: ("bootstrap", 8998))
    monkeypatch.setattr(bootstrap_server.httpx, "Client", unexpected_client)
    invalid = workers()
    invalid[(1, 1, 1)] = replace(invalid[(1, 1, 1)], handshake_port=0)
    with pytest.raises(ValueError, match="listener address"):
        producer_scheduler.set_xfer_handshake_metadata_from_workers(invalid)


def test_abort_before_allocation_notifies_worker_by_transfer_id(producer_scheduler):
    request = prefill_request()
    request.status = RequestStatus.FINISHED_ABORTED
    assert producer_scheduler.request_finished(request, ([], [])) == (False, None)
    meta = producer_scheduler.build_connector_meta(step(request, None, computed=16, scheduled=0))
    assert meta.push_aborted_requests == {request.request_id: "xfer-123"}
    assert not meta.push_send_requests


def test_scheduler_registration_is_queryable_over_http(producer_scheduler, monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = push_scheduler.AscendMooncakeBootstrapServer("127.0.0.1", port)
    server.start()
    try:
        monkeypatch.setattr(push_scheduler, "get_mooncake_bootstrap_addr", lambda config: ("127.0.0.1", port))
        producer_scheduler.set_xfer_handshake_metadata_from_workers(workers())
        with httpx.Client(timeout=3) as client:
            response = client.get(f"http://127.0.0.1:{port}/query")
            response.raise_for_status()
        entry = response.json()["2"]
        assert entry["engine_id"] == producer_scheduler.engine_id
        assert (entry["tp_size"], entry["pp_size"], entry["pcp_size"], entry["dcp_size"]) == (2, 2, 2, 4)
        for (pp, pcp, tp), worker in workers().items():
            assert (
                entry["worker_addr"][str(pcp)][str(tp)][str(pp)] == f"tcp://{worker.local_ip}:{worker.handshake_port}"
            )
    finally:
        server.shutdown()
