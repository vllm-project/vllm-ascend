# SPDX-License-Identifier: Apache-2.0

import socket
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
import zmq

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


def test_p_does_not_start_metadata_service(scheduler):
    scheduler.kv_transfer_config.is_kv_producer = True
    scheduler.set_xfer_handshake_metadata_from_workers(workers())
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
