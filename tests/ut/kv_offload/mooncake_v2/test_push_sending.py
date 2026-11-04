# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""P control, chunk planning and WRITE boundaries without an NPU."""

import ctypes
import queue
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker, push_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakePCPTransferMetadata,
    MooncakePPTransferMetadata,
    MooncakeTPTransferMetadata,
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
)


class Event:
    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True

    def synchronize(self):
        assert self.recorded


def descriptor():
    return MooncakeTransferMetadata(
        engine_id="p",
        te_rpc_port=1,
        block_size=4,
        num_blocks=100,
        layer_names=["layer"],
        layer_block_sizes=[4],
        group_indices=[0],
        kv_caches_base_addr=[[1000]],
        block_strides=[[32]],
        block_lens=[[32]],
        block_shapes=[[(2, 4, 2)]],
        block_size_scales=[[1]],
    )


def remote_metadata(pcp_size=1, tp_size=1, heads=2):
    return MooncakeTransferMetadataGroups(
        engine_id="d",
        scheduler_host="127.0.0.1",
        scheduler_port=9000,
        pp_size=1,
        pcp_size=pcp_size,
        dcp_size=1,
        tp_size=tp_size,
        use_kv_pp=False,
        metadata_by_pp_rank={
            0: MooncakePPTransferMetadata(
                block_size=4,
                num_blocks=100,
                layer_names=["layer"],
                layer_block_sizes=[4],
                group_indices=[0],
                block_strides=[[heads * 16]],
                block_lens=[[heads * 16]],
                block_shapes=[[(heads, 4, 2)]],
                block_size_scales=[[1]],
                metadata_by_pcp_rank={
                    pcp: MooncakePCPTransferMetadata(
                        {tp: MooncakeTPTransferMetadata(2, [0], [[10000]], "127.0.0.1", 9001) for tp in range(tp_size)}
                    )
                    for pcp in range(pcp_size)
                },
            )
        },
    )


def source_meta(computed=0, scheduled=4, prompt=8):
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_send_req("p-request", ([1, 2],), prompt, computed, scheduled, {"transfer_id": "t"})
    return meta


def ready(rank=(0, 0, 0)):
    return dict(
        kind="ready",
        transfer_id="t",
        request_id="d-request",
        producer_engine_id="p",
        producer_rank=[0, 0, 0],
        consumer_engine_id="d",
        consumer_rank=list(rank),
        scheduler_address="tcp://127.0.0.1:9000",
        completion_address="tcp://127.0.0.1:9001",
        block_ids=[[10, 11]],
        num_computed_tokens=0,
        num_external_tokens=8,
    )


def drive(worker, predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        worker.get_transfer_results()
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("Push worker did not make progress")


@pytest.fixture
def worker(monkeypatch):
    spec = Mock(spec=push_worker.FullAttentionSpec)
    spec.sliding_window = None
    spec.is_circular = False
    config = SimpleNamespace(speculative_config=None)

    def initialize(self, *args):
        self.engine_id = "p"
        self.vllm_config = config
        self.pp_rank = self.pcp_rank = self.tp_rank = 0
        self.pp_size = self.pcp_size = self.tp_size = self.dcp_size = 1
        self.side_channel_host = "127.0.0.1"
        self.handshake_port = 0
        self.kv_transfer_config = SimpleNamespace(
            is_kv_producer=True, get_from_extra_config=lambda name, default: default
        )
        self.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
        self.transfer_metadata = descriptor()
        self.kv_cache_specs = [spec]
        self.layer_name_to_spec_index = {"layer": 0}
        self.engine = SimpleNamespace(batch_transfer_sync_write=Mock(return_value=0))

    monkeypatch.setattr(MooncakeBaseConnectorWorker, "__init__", initialize)
    monkeypatch.setattr(MooncakeBaseConnectorWorker, "register_kv_caches", lambda *args: None)
    monkeypatch.setattr(
        push_worker,
        "torch",
        SimpleNamespace(npu=SimpleNamespace(current_device=lambda: 0, set_device=lambda device: None, Event=Event)),
    )
    obj = push_worker.MooncakePushConnectorWorker(
        SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=1)), "p", None
    )
    obj.register_kv_caches({})
    monkeypatch.setattr(obj, "_get_remote_metadata", lambda message: remote_metadata())
    monkeypatch.setattr(obj, "_send_done", Mock(return_value=True))
    yield obj
    obj.shutdown()
    assert not obj._recving_thread.is_alive()


def test_ready_listener_and_two_chunks(worker):
    with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.connect(worker._recving_thread.address)
        sock.send(msgspec.msgpack.encode(ready()))
        assert msgspec.msgpack.decode(sock.recv())["kind"] == "ready_ack"
    worker.start_load_kv(source_meta())
    worker.save_kv_layer("layer")
    worker.wait_for_save()
    drive(worker, lambda: worker.engine.batch_transfer_sync_write.call_count == 1)
    worker.wait_for_save()
    assert not worker._send_done.called
    assert worker.engine.batch_transfer_sync_write.call_args.args[1:] == ([1032], [10320], [32])
    worker.start_load_kv(source_meta(4, 4))
    worker.save_kv_layer("layer")
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker.engine.batch_transfer_sync_write.call_count == 2
    assert worker.engine.batch_transfer_sync_write.call_args.args[1:] == ([1064], [10352], [32])
    terminal = MooncakeConnectorMetadata()
    terminal.requests_to_send = {"p-request": time.monotonic()}
    worker.start_load_kv(terminal)
    results = worker.get_transfer_results()
    assert results.finished_sending == {"p-request"}


def test_late_ready_sends_all_resident_chunks(worker):
    worker.start_load_kv(source_meta())
    worker.wait_for_save()
    worker.start_load_kv(source_meta(4, 4))
    worker.wait_for_save()
    worker._ready_queue.put(ready())
    drive(worker, lambda: worker._send_done.called)
    assert worker.engine.batch_transfer_sync_write.call_args.args[1:] == ([1032, 1064], [10320, 10352], [32, 32])


def test_partial_block_waits_for_later_chunk(worker):
    worker._ready_queue.put(ready())
    worker.start_load_kv(source_meta(0, 3))
    worker.wait_for_save()
    drive(worker, lambda: worker._reqs_send_state["t"].destinations[(0, 0, 0)].required_layers is not None)
    assert not worker.engine.batch_transfer_sync_write.called
    worker.start_load_kv(source_meta(3, 5))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)


def test_swa_is_final_chunk_only_and_pcp_replicated(worker):
    spec = Mock(spec=push_worker.SlidingWindowSpec)
    spec.sliding_window = 4
    spec.num_retained_tokens = 3
    spec.is_circular = False
    worker.kv_cache_specs = [spec]
    worker.pcp_size = 2
    assert worker._get_transfer_partition("layer", remote_metadata(), (0, 0, 0)) == (0, 0, 2)
    worker.pcp_rank = 1
    assert worker._get_transfer_partition("layer", remote_metadata(), (0, 0, 0)) is None
    worker.pcp_rank = 0
    worker._ready_queue.put(ready())
    worker.start_load_kv(source_meta())
    worker.wait_for_save()
    drive(worker, lambda: worker._reqs_send_state["t"].destinations[(0, 0, 0)].required_layers is not None)
    assert not worker.engine.batch_transfer_sync_write.called
    worker.start_load_kv(source_meta(4, 4))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker.engine.batch_transfer_sync_write.call_args.args[1:] == ([1064], [10352], [32])


def test_tp_head_slices_and_replicas(worker):
    assert worker._get_transfer_partition("layer", remote_metadata(tp_size=2, heads=1), (0, 0, 1)) == (1, 0, 1)
    worker.tp_size = 4
    worker.transfer_metadata.block_shapes[0][0] = (1, 4, 2)
    worker.tp_rank = 2
    assert worker._get_transfer_partition("layer", remote_metadata(tp_size=2, heads=1), (0, 0, 1)) is None
    worker.tp_rank = 3
    assert worker._get_transfer_partition("layer", remote_metadata(tp_size=2, heads=1), (0, 0, 1)) == (0, 0, 1)


def test_mamba_flattens_pcp_tp(worker):
    worker.kv_cache_specs = [Mock(spec=push_worker.MambaSpec, tp_replicated=False)]
    worker.pcp_size = 2
    worker.pcp_rank = 1
    assert worker._get_transfer_partition("layer", remote_metadata(tp_size=2), (0, 0, 1)) is not None
    assert worker._get_transfer_partition("layer", remote_metadata(tp_size=2), (0, 0, 0)) is None


def test_non_owner_sends_done_without_write(worker):
    worker.pcp_size = 2
    worker.pcp_rank = 1
    worker._ready_queue.put(ready())
    drive(worker, lambda: worker._send_done.called)
    assert worker._send_done.call_args.args[1] is None
    assert not worker.engine.batch_transfer_sync_write.called


def test_callback_failure_after_success_does_not_pin_p(worker):
    worker._send_done.return_value = False
    worker._ready_queue.put(ready())
    worker.start_load_kv(source_meta(0, 8))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker._send_done.call_args.args[1] is None
    terminal = MooncakeConnectorMetadata()
    terminal.requests_to_send = {"p-request": time.monotonic()}
    worker.start_load_kv(terminal)
    assert worker.get_transfer_results().finished_sending == {"p-request"}


@pytest.mark.parametrize("native_exception", [False, True])
def test_write_failure_does_not_certify_drain(worker, native_exception):
    if native_exception:
        worker.engine.batch_transfer_sync_write.side_effect = RuntimeError("native exception")
    else:
        worker.engine.batch_transfer_sync_write.return_value = -1
    worker._ready_queue.put(ready())
    worker.start_load_kv(source_meta(0, 8))
    deadline = time.monotonic() + 3
    with pytest.raises(RuntimeError, match="Push control failed"):
        while time.monotonic() < deadline:
            worker.wait_for_save()
            time.sleep(0.005)
    assert worker._send_done.call_args.args[2] is False


def test_abort_before_source_metadata(worker):
    worker._ready_queue.put(ready())
    meta = MooncakeConnectorMetadata()
    meta.push_aborted_requests = {"p-request": "t"}
    worker.start_load_kv(meta)
    drive(worker, lambda: worker._send_done.called)
    assert worker._send_done.call_args.args[1] == "P request aborted"
    assert not worker.engine.batch_transfer_sync_write.called


def test_done_socket_timeout_is_bounded(worker, monkeypatch):
    monkeypatch.setattr(push_worker.envs, "VLLM_MOONCAKE_CONNECTOR_TIMEOUT", 0.02)
    message = ready()
    with zmq.Context() as ctx, ctx.socket(zmq.REP) as sock:
        sock.bind("tcp://127.0.0.1:*")
        message["completion_address"] = sock.getsockopt(zmq.LAST_ENDPOINT).decode()
        assert not push_worker.MooncakePushConnectorWorker._send_done(worker, message, "failed")


def test_dcp_is_rejected_before_initialization():
    with pytest.raises(ValueError, match="DCP=1"):
        push_worker.MooncakePushConnectorWorker(
            SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=2)), "p", None
        )


def test_metadata_query_and_done_real_sockets(worker, monkeypatch):
    """Run P's actual metadata query and completion against D protocol servers."""

    result_queue = queue.SimpleQueue()
    d = SimpleNamespace(
        _stop_event=threading.Event(),
        engine_id="d",
        pp_rank=0,
        pcp_rank=0,
        tp_rank=0,
        side_channel_host="127.0.0.1",
        handshake_port=0,
        _result_queue=result_queue,
    )
    completion = push_worker.MooncakeCompletionThread(d)
    completion.start()
    started = threading.Event()
    address = []

    def serve_metadata():
        with zmq.Context() as ctx, ctx.socket(zmq.REP) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 3000)
            sock.bind("tcp://127.0.0.1:*")
            address.append(sock.getsockopt(zmq.LAST_ENDPOINT).decode())
            started.set()
            assert msgspec.msgpack.decode(sock.recv()) == [b"get_meta_msg", "d"]
            sock.send(msgspec.msgpack.encode(remote_metadata()))

    thread = threading.Thread(target=serve_metadata)
    thread.start()
    assert started.wait(3)
    monkeypatch.setattr(
        worker, "_get_remote_metadata", push_worker.MooncakePushConnectorWorker._get_remote_metadata.__get__(worker)
    )
    monkeypatch.setattr(worker, "_send_done", push_worker.MooncakePushConnectorWorker._send_done.__get__(worker))
    message = ready()
    message.update(scheduler_address=address[0], completion_address=completion.address)
    try:
        worker._ready_queue.put(message)
        worker.start_load_kv(source_meta(0, 8))
        worker.wait_for_save()
        drive(worker, lambda: not result_queue.empty())
        kind, transfer_id, payload = result_queue.get_nowait()
        assert (kind, transfer_id, payload["status"], payload["writes_drained"]) == ("done", "t", "ok", True)
    finally:
        completion.shutdown()
        thread.join(4)
        assert not thread.is_alive()


def test_circular_waits_for_final_chunk(worker):
    worker.kv_cache_specs = [SimpleNamespace(is_circular=True, sliding_window=None)]
    message = ready()
    message["block_ids"] = [[10]]
    worker._ready_queue.put(message)
    first = source_meta()

    first.push_send_requests["p-request"] = replace(first.push_send_requests["p-request"], local_block_ids=((1,),))
    worker.start_load_kv(first)
    worker.wait_for_save()
    drive(worker, lambda: worker._reqs_send_state["t"].destinations[(0, 0, 0)].required_layers is not None)
    assert not worker.engine.batch_transfer_sync_write.called
    final = source_meta(4, 4)
    final.push_send_requests["p-request"] = replace(final.push_send_requests["p-request"], local_block_ids=((1,),))
    worker.start_load_kv(final)
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker.engine.batch_transfer_sync_write.call_count == 1


def test_discovery_failure_reports_failure_without_write(worker, monkeypatch):
    def fail(message):
        raise TimeoutError("D scheduler unavailable")

    monkeypatch.setattr(worker, "_get_remote_metadata", fail)
    worker._ready_queue.put(ready())
    worker.start_load_kv(source_meta(0, 8))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker._send_done.call_args.args[1] == "D scheduler unavailable"
    assert not worker.engine.batch_transfer_sync_write.called


def test_missing_ready_expires_and_rejects_late_write(worker, monkeypatch):
    monkeypatch.setattr(push_worker.envs, "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", 0)
    worker.start_load_kv(source_meta(0, 8))
    worker.wait_for_save()
    terminal = MooncakeConnectorMetadata()
    terminal.requests_to_send = {"p-request": time.monotonic()}
    worker.start_load_kv(terminal)
    assert worker.get_transfer_results().finished_sending == {"p-request"}
    worker._ready_queue.put(ready())
    drive(worker, lambda: worker._send_done.called)
    assert worker._send_done.call_args.args[1] == "Push transfer has already retired"
    assert not worker.engine.batch_transfer_sync_write.called


def test_no_release_until_every_d_rank_is_terminal(worker, monkeypatch):
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda message: remote_metadata(pcp_size=2))
    worker._ready_queue.put(ready())
    worker.start_load_kv(source_meta(0, 8))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    terminal = MooncakeConnectorMetadata()
    terminal.requests_to_send = {"p-request": time.monotonic()}
    worker.start_load_kv(terminal)
    assert not worker.get_transfer_results().finished_sending
    worker._ready_queue.put(ready((0, 1, 0)))
    drive(worker, lambda: "t" not in worker._reqs_send_state)
    assert worker._send_done.call_count == 2


def test_head_split_copies_correct_bytes(worker, monkeypatch):

    source = ctypes.create_string_buffer(bytes(range(96)))
    destination = ctypes.create_string_buffer(b"\xff" * 64)
    worker.transfer_metadata.kv_caches_base_addr[0][0] = ctypes.addressof(source)
    remote = remote_metadata(tp_size=2, heads=1)
    remote.metadata_by_pp_rank[0].metadata_by_pcp_rank[0].metadata_by_tp_rank[1].kv_caches_base_addr[0][0] = (
        ctypes.addressof(destination)
    )
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda message: remote)

    def copy(session, src, dst, lengths):
        for s, d, size in zip(src, dst, lengths):
            ctypes.memmove(d, s, size)
        return 0

    worker.engine.batch_transfer_sync_write.side_effect = copy
    message = ready((0, 0, 1))
    message["block_ids"] = [[1, 2]]
    worker._ready_queue.put(message)
    worker.start_load_kv(source_meta(0, 8))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert destination.raw[:16] == b"\xff" * 16
    assert destination.raw[16:32] == source.raw[48:64]
    assert destination.raw[32:48] == source.raw[80:96]
    assert destination.raw[48:64] == b"\xff" * 16


def test_mamba_final_chunk_merges_pcp_conv_projections_and_state(worker, monkeypatch):

    backends = SimpleNamespace(MAMBA1="mamba1", MAMBA2="mamba2", GDN_ATTN="gdn")
    monkeypatch.setattr(base_worker, "MambaAttentionBackendEnum", backends, raising=False)
    monkeypatch.setattr(
        base_worker, "torch", SimpleNamespace(tensor=lambda *args, **kwargs: SimpleNamespace(element_size=lambda: 1))
    )
    spec = Mock(spec=push_worker.MambaSpec)
    spec.tp_replicated = False
    spec.mamba_type = backends.GDN_ATTN
    spec.dtypes = ("byte", "byte")
    spec.sliding_window = None
    spec.is_circular = False
    worker.kv_cache_specs = [spec]
    worker.pcp_size = 2
    worker.pcp_rank = 1
    local = worker.transfer_metadata
    local.kv_caches_base_addr[:] = [[1000, 2000]]
    local.block_strides[:] = [[12, 4]]
    local.block_lens[:] = [[12, 4]]
    local.block_shapes[:] = [[(2, 6), (1, 2, 2)]]
    local.block_size_scales[:] = [[1, 1]]
    remote = remote_metadata()
    pp = remote.metadata_by_pp_rank[0]
    pp.block_strides[:] = [[24, 8]]
    pp.block_lens[:] = [[24, 8]]
    pp.block_shapes[:] = [[(2, 12), (2, 2, 2)]]
    pp.block_size_scales[:] = [[1, 1]]
    pp.metadata_by_pcp_rank[0].metadata_by_tp_rank[0].kv_caches_base_addr[:] = [[10000, 20000]]
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda message: remote)
    worker._ready_queue.put(ready())
    worker.start_load_kv(source_meta())
    worker.wait_for_save()
    drive(worker, lambda: worker._reqs_send_state["t"].destinations[(0, 0, 0)].required_layers is not None)
    assert not worker.engine.batch_transfer_sync_write.called
    worker.start_load_kv(source_meta(4, 4))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    _, src, dst, sizes = worker.engine.batch_transfer_sync_write.call_args.args
    assert src == [1024, 1030, 1026, 1032, 1028, 1034, 2008]
    assert dst == [10266, 10278, 10270, 10282, 10274, 10286, 20092]
    assert sizes == [2, 2, 2, 2, 2, 2, 4]


def test_wait_for_save_drains_write_without_blocking_ready_listener(worker):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def blocked_write(*args):
        entered.set()
        assert release.wait(3)
        return 0

    worker.engine.batch_transfer_sync_write.side_effect = blocked_write
    worker._ready_queue.put(ready())
    worker.start_load_kv(source_meta(0, 8))
    worker.save_kv_layer("layer")
    drive(worker, entered.is_set)

    def wait():
        worker.wait_for_save()
        returned.set()

    thread = threading.Thread(target=wait)
    thread.start()
    try:
        with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            sock.connect(worker._recving_thread.address)
            message = ready()
            message["transfer_id"] = "new-ready"
            sock.send(msgspec.msgpack.encode(message))
            assert msgspec.msgpack.decode(sock.recv())["kind"] == "ready_ack"
        assert not returned.is_set()
    finally:
        release.set()
        thread.join(4)
    assert returned.is_set()


def test_layer_batch_groups_requests_for_one_destination(worker):
    first, second = ready(), ready()
    second.update(transfer_id="t2", request_id="d-request-2", block_ids=[[12, 13]])
    worker._ready_queue.put(first)
    worker._ready_queue.put(second)
    meta = source_meta(0, 8)
    meta.add_new_push_send_req("p-request-2", ([3, 4],), 8, 0, 8, {"transfer_id": "t2"})
    worker.start_load_kv(meta)
    drive(
        worker,
        lambda: all(
            state.destinations[(0, 0, 0)].required_layers is not None for state in worker._reqs_send_state.values()
        ),
    )
    worker.save_kv_layer("layer")
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.call_count == 2)
    assert worker.engine.batch_transfer_sync_write.call_count == 1
    assert worker.engine.batch_transfer_sync_write.call_args.args[1:] == (
        [1032, 1064, 1096, 1128],
        [10320, 10352, 10384, 10416],
        [32] * 4,
    )


def test_cached_destination_prefix_is_not_written(worker):
    message = ready()
    message.update(num_computed_tokens=4, num_external_tokens=4)
    worker._ready_queue.put(message)
    worker.start_load_kv(source_meta(4, 4))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker.engine.batch_transfer_sync_write.call_args.args[1:] == ([1064], [10352], [32])
