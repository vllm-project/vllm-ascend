# SPDX-License-Identifier: Apache-2.0

import json
import socket
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import msgspec
import pytest
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.request import RequestStatus
from zmq.constants import SocketOption, SocketType
from zmq.sugar.context import Context

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import utils
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
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import MooncakeMetaserverClient


def workers():
    return {
        (pp, pcp, tp): MooncakeTransferMetadata(
            engine_id="d",
            te_rpc_port=9000 + pcp * 2 + tp,
            block_size=16,
            cp_kv_cache_interleave_size=16,
            num_blocks=8,
            layer_names=[f"layer.{pp}"],
            layer_block_sizes=[16],
            group_indices=[0],
            kv_caches_base_addr=[[4096 + pp * 1024 + pcp * 256 + tp * 128]],
            block_strides=[[128]],
            block_lens=[[128]],
            block_shapes=[[(16, 8)]],
            block_size_scales=[[1]],
            indexer_replication_sizes=[0],
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
    obj.metaserver_client = None
    obj._remote_worker_addresses = ()
    obj.group_block_size = [16, 16]
    obj.group_unique_specs = [[Mock(spec=FullAttentionSpec, is_circular=False)] for _ in range(2)]
    obj.num_speculative_tokens = 0
    obj._reqs_pending_recv = {}
    obj._reqs_need_send_layerwise = {}
    obj._reqs_sent_to_worker = set()
    obj._reqs_need_send = {}
    obj._reqs_finished_sending = set()
    obj._reqs_pending_send = {}
    obj._reqs_aborted = {}
    obj.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=2))
    yield obj
    obj.shutdown()


@pytest.fixture
def metaserver(monkeypatch):
    def create(handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(utils.httpx, "Client", lambda **kwargs: client)
        return MooncakeMetaserverClient()

    return create


def query(thread, message):
    with Context.instance().socket(SocketType.REQ) as sock:
        sock.setsockopt(SocketOption.LINGER, 0)
        sock.setsockopt(SocketOption.RCVTIMEO, 2000)
        sock.setsockopt(SocketOption.SNDTIMEO, 2000)
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


def test_bind_failure_is_reported_and_thread_is_cleaned_up(scheduler):
    with Context.instance().socket(SocketType.REP) as occupied:
        occupied.bind("tcp://127.0.0.1:*")
        scheduler.side_channel_port = int(occupied.getsockopt(SocketOption.LAST_ENDPOINT).decode().rsplit(":", 1)[1])
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
        thread.add_delayed_request("request")
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
    scheduler.dcp_size = 1
    return scheduler


def prefill_request(request_id="p-request"):
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=48,
        kv_transfer_params={
            "do_remote_decode": True,
            "remote_request_id": "d-request-123",
            "remote_engine_id": "d",
            "remote_host": "127.0.0.1",
            "remote_port": 9000,
            "remote_block_ids": [[10, 11, 12], [20, 21, 22]],
            "remote_num_computed_tokens": 0,
            "remote_num_external_tokens": 48,
            "remote_worker_addresses": [[0, 0, 0, "tcp://127.0.0.1:9001"]],
        },
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
        has_sync_kv_loads=False,
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
    assert chunk.request_id == "d-request-123"
    assert not second.push_requests
    third = scheduler.build_connector_meta(step(request, ([3], [13]), computed=32, scheduled=20))
    assert third.push_send_requests[request.request_id].num_scheduled_tokens == 16
    assert first.push_send_requests[request.request_id].local_block_ids == ((1,), (11,))
    assert not scheduler.build_connector_meta(step(request, None, computed=48, scheduled=1)).push_send_requests


@pytest.mark.parametrize("num_speculative_blocks", [0, 2])
def test_producer_mamba_excludes_speculative_tail(producer_scheduler, num_speculative_blocks):
    producer_scheduler.group_unique_specs[1] = [
        Mock(spec=MambaSpec, num_speculative_blocks=num_speculative_blocks, is_circular=False)
    ]
    request = prefill_request()
    blocks = ([1, 2, 3], [10, 11, 12])
    allocate(producer_scheduler, request, blocks)
    meta = producer_scheduler.build_connector_meta(step(request, blocks, scheduled=48))
    state_blocks = meta.push_send_requests[request.request_id].local_block_ids[1]
    assert state_blocks == ((10,) if num_speculative_blocks else (10, 11, 12))
    # Preserve the allocator's complete table for subsequent scheduling.
    assert producer_scheduler._reqs_need_send_layerwise[request.request_id][1] == blocks


def test_prefix_hit_publishes_cached_prefix_and_new_token_range(producer_scheduler):
    request = prefill_request()
    allocate(producer_scheduler, request, ([1, 2, 3], [11, 12, 13]))
    output = step(request, ([1, 2, 3], [11, 12, 13]))
    output.scheduled_new_reqs[0].num_computed_tokens = 32
    chunk = producer_scheduler.build_connector_meta(output).push_send_requests[request.request_id]
    assert (chunk.num_computed_tokens, chunk.num_scheduled_tokens) == (32, 16)
    assert chunk.local_block_ids == ((1, 2, 3), (11, 12, 13))


@pytest.mark.parametrize("aborted", [False, True])
def test_producer_delays_release_only_for_unfinished_abort(producer_scheduler, aborted):
    scheduler = producer_scheduler
    request = prefill_request()
    allocate(scheduler, request)
    scheduler.build_connector_meta(step(request, ([1], [11])))
    if aborted:
        request.status = RequestStatus.FINISHED_ABORTED
    assert scheduler.request_finished(request, ([1], [11])) == (aborted, None)
    terminal_step = step(request, None, computed=16, scheduled=0)
    meta = scheduler.build_connector_meta(terminal_step)
    assert set(meta.requests_to_send) == {request.request_id}
    assert terminal_step.has_sync_kv_loads
    assert set(meta.push_aborted_requests) == ({request.request_id} if aborted else set())
    assert not meta.push_send_requests
    assert not scheduler.build_connector_meta(terminal_step).requests_to_send
    output = SimpleNamespace(finished_sending={request.request_id}, finished_recving=None)
    scheduler.update_connector_output(output)
    assert output.finished_sending == ({request.request_id} if aborted else None)
    assert not scheduler._reqs_need_send
    assert not scheduler._reqs_need_send_layerwise
    # A duplicate completion cannot cause core to free an already-freed request.
    scheduler.update_connector_output(output)
    assert not output.finished_sending


@pytest.mark.parametrize("allocated", [False, True])
def test_abort_before_dispatch_preserves_routing_without_pinning_blocks(producer_scheduler, allocated):
    request = prefill_request()
    if allocated:
        allocate(producer_scheduler, request)
    request.status = RequestStatus.FINISHED_ABORTED
    assert producer_scheduler.request_finished(request, ([1], [11]) if allocated else ([], [])) == (False, None)
    assert not producer_scheduler._reqs_need_send_layerwise
    output = step(request, scheduled=0)
    output.num_scheduled_tokens = {}
    meta = producer_scheduler.build_connector_meta(output)
    assert output.has_sync_kv_loads
    assert not meta.push_send_requests
    assert meta.push_aborted_requests[request.request_id] == dict(
        remote_request_id="d-request-123",
        remote_engine_id="d",
        remote_worker_addresses=((0, 0, 0, "tcp://127.0.0.1:9001"),),
    )


@pytest.mark.parametrize("external_tokens", [0, 16])
def test_metaserver_dispatch_is_concurrent_and_snapshots_blocks(scheduler, external_tokens, metaserver):
    scheduler.dcp_size = 1
    scheduler.set_xfer_handshake_metadata_from_workers(workers())
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()
    messages = []

    def handle(http_request):
        message = json.loads(http_request.content)
        messages.append(message)
        entered[int(message["request_id"].rsplit("-", 1)[1])].set()
        assert release.wait(3)
        return httpx.Response(200, json={"status": "ok"})

    scheduler.metaserver_client = metaserver(handle)
    try:
        for i in range(2):
            request = SimpleNamespace(
                request_id=f"cmpl-parent-{i}-12345678",
                num_prompt_tokens=32,
                kv_transfer_params={
                    "do_remote_prefill": True,
                    "metaserver": "http://proxy/v1/metaserver",
                    "num_computed_tokens": 16,
                },
            )
            blocks = ([1, 2], [3, 4])
            scheduler.update_state_after_alloc(
                request, SimpleNamespace(get_block_ids=lambda blocks=blocks: blocks), external_tokens
            )
            blocks[0][0] = 99
        assert all(event.wait(3) for event in entered)
        # Both HTTP calls are blocked, while scheduler metadata can still progress.
        output = SimpleNamespace()
        meta = scheduler.build_connector_meta(output)
        assert len(meta.push_requests) == (2 if external_tokens else 0)
        assert {message["request_id"] for message in messages} == {"cmpl-parent-0", "cmpl-parent-1"}
        for message in messages:
            assert message["remote_block_ids"] == [[1, 2], [3, 4]]
            assert message["remote_request_id"] == message["request_id"] + "-12345678"
            assert len(message["remote_worker_addresses"]) == 8
            assert message["remote_num_external_tokens"] == external_tokens
            assert message["remote_num_computed_tokens"] == 16
    finally:
        release.set()
        scheduler.metaserver_client.shutdown()
    assert scheduler.metaserver_client.client.is_closed
    assert not scheduler.metaserver_client.get_failed_requests()


@pytest.mark.parametrize("failure", ["connect", "response", "read_timeout"])
def test_metaserver_failure_is_propagated_without_redispatch(scheduler, failure, metaserver):
    calls = []

    def handle(request):
        calls.append(request)
        if failure == "connect":
            raise httpx.ConnectError("unreachable", request=request)
        if failure == "read_timeout":
            raise httpx.ReadTimeout("P may already be running", request=request)
        return httpx.Response(502)

    scheduler.metaserver_client = metaserver(handle)
    scheduler.metaserver_client.submit("request", "http://proxy/v1/metaserver", {})
    scheduler.metaserver_client.executor.shutdown(wait=True)
    assert len(calls) == (3 if failure == "connect" else 1)
    # The next metadata-only step carries failures; other scheduler hooks
    # neither raise nor consume the result before the worker can see it.
    scheduler.update_connector_output(SimpleNamespace(finished_recving=set()))
    scheduler.update_state_after_alloc(SimpleNamespace(kv_transfer_params=None), None, 0)
    meta = scheduler.build_connector_meta(SimpleNamespace())
    assert meta.push_failed_requests == {"request": failure == "connect"}
    assert not scheduler.build_connector_meta(SimpleNamespace()).push_failed_requests


def test_d_full_prefix_hit_needs_no_p_transfer(producer_scheduler):
    request = prefill_request()
    request.kv_transfer_params["remote_num_external_tokens"] = 0
    allocate(producer_scheduler, request)
    assert not producer_scheduler.build_connector_meta(step(request)).push_send_requests
    assert producer_scheduler.request_finished(request, ([1],)) == (False, None)


@pytest.mark.parametrize("initial_flag", [False, True])
def test_push_load_order_covers_chunks_pending_sends_and_idle(producer_scheduler, initial_flag):
    scheduler = producer_scheduler
    request = prefill_request()
    idle = step(request, scheduled=0)
    idle.scheduled_new_reqs = []
    idle.num_scheduled_tokens = {}
    idle.has_sync_kv_loads = initial_flag
    scheduler.build_connector_meta(idle)
    assert idle.has_sync_kv_loads is initial_flag

    allocate(scheduler, request)
    for computed in (0, 16):
        output = step(request, computed=computed)
        scheduler.build_connector_meta(output)
        assert output.has_sync_kv_loads

    request.status = RequestStatus.FINISHED_ABORTED
    assert scheduler.request_finished(request, ([1], [11]))[0]
    for _ in range(2):
        idle.has_sync_kv_loads = False
        scheduler.build_connector_meta(idle)
        assert idle.has_sync_kv_loads  # Includes steps after the finish notice was consumed.

    scheduler.update_connector_output(SimpleNamespace(finished_sending={request.request_id}))
    idle.has_sync_kv_loads = False
    scheduler.build_connector_meta(idle)
    assert not idle.has_sync_kv_loads


@pytest.mark.parametrize("use_replayssm", [False, True])
def test_metaserver_mamba_blocks_exclude_only_speculative_tail(scheduler, use_replayssm, metaserver):
    # MambaSpec derives this count from use_replayssm at model registration.
    num_speculative_blocks = 0 if use_replayssm else 2
    scheduler.group_unique_specs[1] = [
        Mock(spec=MambaSpec, num_speculative_blocks=num_speculative_blocks, is_circular=False)
    ]
    scheduler.dcp_size = 1
    scheduler._metadata_thread = Mock(error=None)
    messages = []

    def handle(request):
        messages.append(json.loads(request.content))
        return httpx.Response(200)

    scheduler.metaserver_client = metaserver(handle)
    request = SimpleNamespace(
        request_id="cmpl-test-12345678",
        num_prompt_tokens=32,
        kv_transfer_params={
            "do_remote_prefill": True,
            "num_computed_tokens": 0,
            "metaserver": "http://proxy/v1/metaserver",
        },
    )
    blocks = ([1, 2], [10, 11, 12])
    scheduler.update_state_after_alloc(request, SimpleNamespace(get_block_ids=lambda: blocks), 32)
    scheduler.metaserver_client.executor.shutdown(wait=True)
    assert messages[0]["remote_block_ids"] == [[1, 2], [10, 11, 12] if use_replayssm else [10]]
    assert "remote_num_speculative_tokens" not in messages[0]
    assert blocks == ([1, 2], [10, 11, 12])
    assert scheduler.build_connector_meta(None).push_requests[request.request_id].local_block_ids == (
        (1, 2),
        (10, 11, 12),
    )


@pytest.mark.parametrize("url", [None, "ftp://proxy/metaserver", "http://proxy:bad"])
def test_invalid_metaserver_url_fails_before_http(scheduler, metaserver, url):
    handler = Mock(return_value=httpx.Response(200))
    scheduler.metaserver_client = metaserver(handler)
    scheduler.metaserver_client.submit("d-request", url, {})
    assert scheduler.build_connector_meta(None).push_failed_requests == {"d-request": True}
    handler.assert_not_called()


def test_executor_rejection_cannot_publish_enqueued_metadata(scheduler, metaserver, monkeypatch):
    handler = Mock(return_value=httpx.Response(200))
    scheduler.metaserver_client = metaserver(handler)
    enqueued = []

    def reject_after_enqueue(task):
        enqueued.append(task)
        raise RuntimeError("cannot start new thread")

    monkeypatch.setattr(scheduler.metaserver_client.executor, "submit", reject_after_enqueue)
    scheduler.metaserver_client.submit("d-request", "http://proxy/v1/metaserver", {})
    # Simulate another pool thread picking up the rejected work item later.
    enqueued[0]()
    handler.assert_not_called()
    assert scheduler.build_connector_meta(None).push_failed_requests == {"d-request": True}


def test_scheduler_shutdown_stops_metadata_even_if_client_cleanup_fails(scheduler):
    scheduler.metaserver_client = Mock()
    scheduler.metaserver_client.shutdown.side_effect = RuntimeError("client cleanup failed")
    thread = Mock()
    scheduler._metadata_thread = thread
    with pytest.raises(RuntimeError, match="client cleanup failed"):
        scheduler.shutdown()
    thread.shutdown.assert_called_once_with()
    assert scheduler.metaserver_client is None
    assert scheduler._metadata_thread is None
