# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""P control, chunk planning and WRITE boundaries without an NPU."""

import ctypes
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
from zmq.constants import SocketOption, SocketType
from zmq.sugar.context import Context

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
        cp_kv_cache_interleave_size=4,
        num_blocks=100,
        layer_names=["model.layers.0.attn"],
        layer_block_sizes=[4],
        group_indices=[0],
        kv_caches_base_addr=[[1000]],
        block_strides=[[32]],
        block_lens=[[32]],
        block_shapes=[[(2, 4, 2)]],
        block_size_scales=[[1]],
        indexer_replication_sizes=[0],
        dcp_sharded=[True],
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
                cp_kv_cache_interleave_size=4,
                num_blocks=100,
                layer_names=["model.layers.0.attn"],
                layer_block_sizes=[4],
                group_indices=[0],
                block_strides=[[heads * 16]],
                block_lens=[[heads * 16]],
                block_shapes=[[(heads, 4, 2)]],
                block_size_scales=[[1]],
                indexer_replication_sizes=[0],
                dcp_sharded=[True],
                metadata_by_pcp_rank={
                    pcp: MooncakePCPTransferMetadata(
                        {tp: MooncakeTPTransferMetadata(2, [0], [[10000]], "127.0.0.1", 9001) for tp in range(tp_size)}
                    )
                    for pcp in range(pcp_size)
                },
            )
        },
    )


def target_params(message=None):
    message = message or destination_message()
    host, port = message["scheduler_address"].removeprefix("tcp://").rsplit(":", 1)
    return dict(
        remote_request_id=message["request_id"],
        remote_engine_id="d",
        remote_host=host,
        remote_port=int(port),
        remote_block_ids=message["block_ids"],
        remote_num_computed_tokens=message["num_computed_tokens"],
        remote_num_external_tokens=message["num_external_tokens"],
        remote_worker_addresses=[[*message["consumer_rank"], message["completion_address"]]],
    )


def source_meta(computed=0, scheduled=4, prompt=8, target=None):
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_send_req("p-request", ([1, 2],), prompt, computed, scheduled, target_params(target))
    return meta


def destination_message(rank=(0, 0, 0)):
    return dict(
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
    """Advance WRITE work without finalizing the step or sending DONE."""
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        worker._process_sending_requests()
        worker.get_transfer_results()
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("Push worker did not make progress")


@pytest.fixture
def worker(monkeypatch, request):
    spec = Mock(spec=push_worker.FullAttentionSpec)
    spec.sliding_window = None
    spec.is_circular = False
    config = SimpleNamespace(
        speculative_config=None, model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="test"))
    )

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
        layout = getattr(request, "param", None)
        if layout is not None:
            names = ["model.layers.0.attn", "model.layers.0.indexer", "model.layers.1.attn"]
            if layout == "longcat":
                config.model_config.hf_text_config.model_type = "longcat_flash"
                names = ["model.layers.0.attn.0", "model.layers.0.attn.0.indexer", "model.layers.0.attn.1"]
            elif layout == "three_layers":
                names = [f"model.layers.{i}.attn" for i in range(3)]
            elif layout in ("mtp", "eagle3", "dspark"):
                names[-1] = f"{layout}.layer.0.self_attn"
            self.transfer_metadata = replace(
                self.transfer_metadata,
                layer_names=names,
                layer_block_sizes=[4, 4, 4],
                group_indices=[0, 1, 0],
                kv_caches_base_addr=[[1000], [2000], [3000]],
                block_strides=[[32], [32], [32]],
                block_lens=[[32], [32], [32]],
                block_shapes=[[(2, 4, 2)]] * 3,
                block_size_scales=[[1], [1], [1]],
                indexer_replication_sizes=[0, 0, 0],
                dcp_sharded=[True] * 3,
            )
        self.kv_cache_specs = [spec]
        self.layer_name_to_spec_index = {name: 0 for name in self.transfer_metadata.layer_names}
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
    assert obj._sender_executor is None
    assert obj._zmq_context is None


def test_proxy_destinations_and_two_chunks(worker):
    worker.start_load_kv(source_meta())
    # Deferred finalization may collect results before all cache hooks run.
    # Polling must not publish unfinished caches or reset this step's state.
    assert not worker.get_transfer_results().finished_sending
    assert not worker._reqs_send_state["d-request"].available_layers
    assert worker._prefilling_request_ids == ("d-request",)
    worker.save_kv_layer("model.layers.0.attn")
    worker.wait_for_save()
    drive(worker, lambda: worker.engine.batch_transfer_sync_write.call_count == 1)
    worker.wait_for_save()
    assert not worker._send_done.called
    assert worker.engine.batch_transfer_sync_write.call_args.args[1:] == ([1032], [10320], [32])
    worker.start_load_kv(source_meta(4, 4))
    worker.save_kv_layer("model.layers.0.attn")
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker.engine.batch_transfer_sync_write.call_count == 2
    assert worker.engine.batch_transfer_sync_write.call_args.args[1:] == ([1064], [10352], [32])
    # Sending DONE does not retire a request until scheduler reports its end.
    assert not worker.get_transfer_results().finished_sending
    terminal = MooncakeConnectorMetadata()
    terminal.requests_to_send = {"p-request": time.monotonic()}
    worker.start_load_kv(terminal)
    results = worker.get_transfer_results()
    assert results.finished_sending == {"p-request"}


def test_partial_block_waits_for_later_chunk(worker):
    worker.start_load_kv(source_meta(0, 3))
    worker.wait_for_save()
    drive(worker, lambda: worker._reqs_send_state["d-request"].destinations[(0, 0, 0)].required_layers is not None)
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
    assert worker._get_transfer_partition("model.layers.0.attn", remote_metadata(), (0, 0, 0)) == (0, 0, 2)
    worker.pcp_rank = 1
    assert worker._get_transfer_partition("model.layers.0.attn", remote_metadata(), (0, 0, 0)) is None
    worker.pcp_rank = 0
    worker.start_load_kv(source_meta())
    worker.wait_for_save()
    drive(worker, lambda: worker._reqs_send_state["d-request"].destinations[(0, 0, 0)].required_layers is not None)
    assert not worker.engine.batch_transfer_sync_write.called
    worker.start_load_kv(source_meta(4, 4))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker.engine.batch_transfer_sync_write.call_args.args[1:] == ([1064], [10352], [32])


def test_tp_head_slices_and_replicas(worker):
    assert worker._get_transfer_partition("model.layers.0.attn", remote_metadata(tp_size=2, heads=1), (0, 0, 1)) == (
        1,
        0,
        1,
    )
    worker.tp_size = 4
    worker.transfer_metadata.block_shapes[0][0] = (1, 4, 2)
    worker.tp_rank = 2
    assert worker._get_transfer_partition("model.layers.0.attn", remote_metadata(tp_size=2, heads=1), (0, 0, 1)) is None
    worker.tp_rank = 3
    assert worker._get_transfer_partition("model.layers.0.attn", remote_metadata(tp_size=2, heads=1), (0, 0, 1)) == (
        0,
        0,
        1,
    )


def test_mamba_flattens_pcp_tp(worker):
    worker.kv_cache_specs = [Mock(spec=push_worker.MambaSpec, tp_replicated=False)]
    worker.pcp_size = 2
    worker.pcp_rank = 1
    assert worker._get_transfer_partition("model.layers.0.attn", remote_metadata(tp_size=2), (0, 0, 1)) is not None
    assert worker._get_transfer_partition("model.layers.0.attn", remote_metadata(tp_size=2), (0, 0, 0)) is None


def test_non_owner_sends_done_without_write(worker):
    worker.pcp_size = 2
    worker.pcp_rank = 1
    worker.start_load_kv(source_meta())
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker._send_done.call_args.args[1] is None
    assert not worker.engine.batch_transfer_sync_write.called


def test_callback_failure_after_success_does_not_pin_p(worker):
    worker._send_done.return_value = False
    worker.start_load_kv(source_meta(0, 8))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker._send_done.call_args.args[1] is None
    terminal = MooncakeConnectorMetadata()
    terminal.requests_to_send = {"p-request": time.monotonic()}
    worker.start_load_kv(terminal)
    assert worker.get_transfer_results().finished_sending == {"p-request"}


@pytest.mark.parametrize("failure", ["negative", "exception", "empty_exception", "future"])
def test_write_failure_notifies_d_and_keeps_p_running(worker, monkeypatch, failure):
    if failure == "exception":
        worker.engine.batch_transfer_sync_write.side_effect = RuntimeError("native exception")
    elif failure == "empty_exception":
        worker.engine.batch_transfer_sync_write.side_effect = RuntimeError()
    elif failure == "future":

        def fail(batch):
            raise RuntimeError("sender task exception")

        monkeypatch.setattr(worker, "_execute_layer_transfer", fail)
    else:
        worker.engine.batch_transfer_sync_write.return_value = -1
    worker.start_load_kv(source_meta(0, 8))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker._send_done.call_args.args[1]
    assert not worker._pending_sends
    assert not worker._stop_event.is_set()
    terminal = MooncakeConnectorMetadata()
    terminal.requests_to_send = {"p-request": time.monotonic()}
    worker.start_load_kv(terminal)
    assert worker.get_transfer_results().finished_sending == {"p-request"}
    assert not worker._reqs_send_state

    # A subsequent request can still transfer and notify normally.
    worker.engine.batch_transfer_sync_write.side_effect = None
    worker.engine.batch_transfer_sync_write.return_value = 0
    monkeypatch.setattr(
        worker,
        "_execute_layer_transfer",
        push_worker.MooncakePushConnectorWorker._execute_layer_transfer.__get__(worker),
    )
    next_target = dict(destination_message(), request_id="next-d")
    worker.start_load_kv(source_meta(0, 8, target=next_target))
    worker.wait_for_save()
    assert worker._send_done.call_args.args[0]["request_id"] == "next-d"
    assert worker._send_done.call_args.args[1] is None


def test_abort_before_dispatch_notifies_every_d_rank(worker, monkeypatch):
    discovery = Mock(side_effect=AssertionError("Abort must not query cache metadata"))
    monkeypatch.setattr(worker, "_get_remote_metadata", discovery)
    params = target_params()
    params["remote_worker_addresses"].append([0, 0, 1, "tcp://127.0.0.1:9002"])
    meta = MooncakeConnectorMetadata()
    meta.push_aborted_requests = {"p-request": params}
    worker.start_load_kv(meta)
    worker.wait_for_save()
    assert worker._send_done.call_count == 2
    assert {
        (tuple(call.args[0]["consumer_rank"]), call.args[0]["completion_address"])
        for call in worker._send_done.call_args_list
    } == {
        ((0, 0, 0), "tcp://127.0.0.1:9001"),
        ((0, 0, 1), "tcp://127.0.0.1:9002"),
    }
    assert all(call.args[1] == "P request aborted" for call in worker._send_done.call_args_list)
    assert worker.get_transfer_results().finished_sending == {"p-request"}

    discovery.assert_not_called()
    worker.engine.batch_transfer_sync_write.assert_not_called()
    assert all(call.args[0]["request_id"] == "d-request" for call in worker._send_done.call_args_list)
    assert all(call.args[0]["consumer_engine_id"] == "d" for call in worker._send_done.call_args_list)


def test_abort_waits_for_submitted_write(worker):
    entered, release = threading.Event(), threading.Event()

    def blocked_write(*args):
        entered.set()
        assert release.wait(3)
        return 0

    worker.engine.batch_transfer_sync_write.side_effect = blocked_write
    worker.start_load_kv(source_meta(0, 4))
    worker.save_kv_layer("model.layers.0.attn")
    drive(worker, entered.is_set)
    meta = MooncakeConnectorMetadata()
    meta.push_aborted_requests = {"p-request": target_params()}
    worker.start_load_kv(meta)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker.wait_for_save)
        try:
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
            worker._send_done.assert_not_called()
            assert not worker.get_transfer_results().finished_sending
        finally:
            release.set()
        future.result(timeout=3)
    worker._send_done.assert_called_once()
    assert worker._send_done.call_args.args[1] == "P request aborted"
    assert worker.get_transfer_results().finished_sending == {"p-request"}


def test_done_socket_timeout_is_bounded(worker, monkeypatch):
    monkeypatch.setattr(push_worker.envs, "VLLM_MOONCAKE_CONNECTOR_TIMEOUT", 0.02)
    message = destination_message()
    with Context() as ctx, ctx.socket(SocketType.REP) as sock:
        sock.bind("tcp://127.0.0.1:*")
        message["completion_address"] = sock.getsockopt(SocketOption.LAST_ENDPOINT).decode()
        assert not push_worker.MooncakePushConnectorWorker._send_done(worker, message, "failed")


def test_dcp_is_rejected_before_initialization():
    with pytest.raises(ValueError, match="DCP=1"):
        push_worker.MooncakePushConnectorWorker(
            SimpleNamespace(parallel_config=SimpleNamespace(decode_context_parallel_size=2)), "p", None
        )


@pytest.mark.parametrize("write_fails", [False, True])
def test_metadata_query_and_done_real_sockets(worker, monkeypatch, write_fails):
    """Run P's actual metadata query and completion against D protocol servers."""

    result_queue: queue.SimpleQueue[tuple[str, str, dict[str, object]]] = queue.SimpleQueue()
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
    completion = push_worker.MooncakeDecodeListenerThread(d)
    completion.start()
    started = threading.Event()
    address = []

    def serve_metadata():
        with Context() as ctx, ctx.socket(SocketType.REP) as sock:
            sock.setsockopt(SocketOption.RCVTIMEO, 3000)
            sock.bind("tcp://127.0.0.1:*")
            address.append(sock.getsockopt(SocketOption.LAST_ENDPOINT).decode())
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
    message = destination_message()
    message.update(scheduler_address=address[0], completion_address=completion.address)
    if write_fails:
        worker.engine.batch_transfer_sync_write.side_effect = RuntimeError("WRITE failed")
    try:
        worker.start_load_kv(source_meta(0, 8, target=message))
        worker.wait_for_save()
        drive(worker, lambda: not result_queue.empty())
        kind, request_id, payload = result_queue.get_nowait()
        assert (kind, request_id, payload["status"]) == ("done", "d-request", "error" if write_fails else "ok")
    finally:
        completion.shutdown()
        thread.join(4)
        assert not thread.is_alive()


def test_circular_waits_for_final_chunk(worker):
    worker.kv_cache_specs = [SimpleNamespace(is_circular=True, sliding_window=None)]
    message = destination_message()
    message["block_ids"] = [[10]]
    first = source_meta(target=message)

    first.push_send_requests["p-request"] = replace(first.push_send_requests["p-request"], local_block_ids=((1,),))
    worker.start_load_kv(first)
    worker.wait_for_save()
    drive(worker, lambda: worker._reqs_send_state["d-request"].destinations[(0, 0, 0)].required_layers is not None)
    assert not worker.engine.batch_transfer_sync_write.called
    final = source_meta(4, 4, target=message)
    final.push_send_requests["p-request"] = replace(final.push_send_requests["p-request"], local_block_ids=((1,),))
    worker.start_load_kv(final)
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker.engine.batch_transfer_sync_write.call_count == 1


def test_discovery_failure_reports_failure_and_next_request_retries(worker, monkeypatch):
    query = Mock(side_effect=[TimeoutError("D scheduler unavailable"), remote_metadata()])
    monkeypatch.setattr(worker, "_get_remote_metadata", query)
    worker.start_load_kv(source_meta(0, 8))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker._send_done.call_args.args[1] == "D scheduler unavailable"
    assert not worker.engine.batch_transfer_sync_write.called
    target = dict(destination_message(), request_id="next-d")
    worker.start_load_kv(source_meta(0, 8, target=target))
    worker.wait_for_save()
    assert query.call_count == 2
    assert worker.engine.batch_transfer_sync_write.call_count == 1
    assert worker._send_done.call_args.args[0]["request_id"] == "next-d"
    assert worker._send_done.call_args.args[1] is None


def test_head_split_copies_correct_bytes(worker, monkeypatch):
    source = ctypes.create_string_buffer(bytes(range(96)))
    destination = ctypes.create_string_buffer(b"\xff" * 64)
    worker.transfer_metadata.kv_caches_base_addr[0][0] = ctypes.addressof(source)
    other_destination = ctypes.create_string_buffer(b"\xff" * 64)
    remote = remote_metadata(tp_size=2, heads=1)
    remote.metadata_by_pp_rank[0].metadata_by_pcp_rank[0].metadata_by_tp_rank[0].kv_caches_base_addr[0][0] = (
        ctypes.addressof(other_destination)
    )
    remote.metadata_by_pp_rank[0].metadata_by_pcp_rank[0].metadata_by_tp_rank[1].kv_caches_base_addr[0][0] = (
        ctypes.addressof(destination)
    )
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda message: remote)

    def copy(session, src, dst, lengths):
        for s, d, size in zip(src, dst, lengths):
            ctypes.memmove(d, s, size)
        return 0

    worker.engine.batch_transfer_sync_write.side_effect = copy
    message = destination_message((0, 0, 1))
    message["block_ids"] = [[1, 2]]
    meta = source_meta(0, 8, target=message)
    target = meta.push_send_requests["p-request"]
    meta.push_send_requests["p-request"] = replace(
        target, remote_worker_addresses=((0, 0, 0, "tcp://127.0.0.1:9001"), *target.remote_worker_addresses)
    )
    worker.start_load_kv(meta)
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
    worker.start_load_kv(source_meta())
    worker.wait_for_save()
    drive(worker, lambda: worker._reqs_send_state["d-request"].destinations[(0, 0, 0)].required_layers is not None)
    assert not worker.engine.batch_transfer_sync_write.called
    worker.start_load_kv(source_meta(4, 4))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    _, src, dst, sizes = worker.engine.batch_transfer_sync_write.call_args.args
    assert src == [1024, 1030, 1026, 1032, 1028, 1034, 2008]
    assert dst == [10266, 10278, 10270, 10282, 10274, 10286, 20092]
    assert sizes == [2, 2, 2, 2, 2, 2, 4]


@pytest.mark.parametrize("invalid_second", [False, True])
def test_layer_batch_groups_valid_requests_and_isolates_invalid_blocks(worker, invalid_second):
    second = dict(destination_message(), request_id="d-request-2", block_ids=[[12, 13]])
    meta = source_meta(0, 8)
    local_blocks = ([0, 4],) if invalid_second else ([3, 4],)
    meta.add_new_push_send_req("p-request-2", local_blocks, 8, 0, 8, target_params(second))
    worker.start_load_kv(meta)
    worker.wait_for_save()
    assert worker.engine.batch_transfer_sync_write.call_count == 1
    expected_src = [1032, 1064] if invalid_second else [1032, 1064, 1096, 1128]
    expected_dst = [10320, 10352] if invalid_second else [10320, 10352, 10384, 10416]
    assert worker.engine.batch_transfer_sync_write.call_args.args[1:] == (
        expected_src,
        expected_dst,
        [32] * len(expected_src),
    )
    outcomes = {call.args[0]["request_id"]: call.args[1] for call in worker._send_done.call_args_list}
    assert outcomes == {
        "d-request": None,
        "d-request-2": "Required Push block is not resident" if invalid_second else None,
    }


def test_cached_destination_prefix_is_not_written(worker):
    message = destination_message()
    message.update(num_computed_tokens=4, num_external_tokens=4)
    worker.start_load_kv(source_meta(4, 4, target=message))
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker.engine.batch_transfer_sync_write.call_args.args[1:] == ([1064], [10352], [32])


def multi_cache_remote(worker):
    remote = remote_metadata()
    pp = remote.metadata_by_pp_rank[0]
    # P/D groups intentionally have different ordering.
    remote.metadata_by_pp_rank[0] = replace(
        pp,
        layer_names=worker.transfer_metadata.layer_names,
        layer_block_sizes=[4, 4, 4],
        group_indices=[1, 0, 1],
        block_strides=[[32], [32], [32]],
        block_lens=[[32], [32], [32]],
        block_shapes=[[(2, 4, 2)]] * 3,
        block_size_scales=[[1], [1], [1]],
        indexer_replication_sizes=[0, 0, 0],
        dcp_sharded=[True] * 3,
        metadata_by_pcp_rank={
            0: MooncakePCPTransferMetadata(
                {0: MooncakeTPTransferMetadata(2, [0, 1, 2], [[10000], [20000], [30000]], "127.0.0.1", 9001)}
            )
        },
    )
    return remote


@pytest.mark.parametrize("worker", ["multi_cache"], indirect=True)
def test_failed_layer_waits_for_other_submitted_writes_before_done(worker, monkeypatch):
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda message: multi_cache_remote(worker))
    entered, release = threading.Event(), threading.Event()

    def write(session, src, dst, lengths):
        if src[0] == 1032:
            raise RuntimeError("first layer failed")
        entered.set()
        assert release.wait(3)
        return 0

    worker.engine.batch_transfer_sync_write.side_effect = write
    message = dict(destination_message(), block_ids=[[20, 21], [10, 11]])
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_send_req("p-request", ([1, 2], [3, 4]), 8, 0, 8, target_params(message))
    worker.start_load_kv(meta)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker.wait_for_save)
        try:
            assert entered.wait(3)
            worker._send_done.assert_not_called()
            assert not future.done()
        finally:
            release.set()
        future.result(timeout=3)
    worker._send_done.assert_called_once()
    assert worker._send_done.call_args.args[1] == "first layer failed"
    assert not worker._pending_sends
    assert worker.engine.batch_transfer_sync_write.call_count == 3


@pytest.mark.parametrize("worker", ["multi_cache", "longcat"], indirect=True)
def test_layer_hook_sends_all_caches_across_groups(worker, monkeypatch):
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda message: multi_cache_remote(worker))
    message = destination_message()
    message["block_ids"] = [[20, 21], [10, 11]]
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_send_req("p-request", ([1, 2], [3, 4]), 8, 0, 8, target_params(message))
    worker.start_load_kv(meta)
    state = worker._reqs_send_state["d-request"]
    drive(worker, lambda: state.destinations[(0, 0, 0)].required_layers is not None)
    attention, indexer, next_layer = worker.transfer_metadata.layer_names
    worker.save_kv_layer(attention)
    drive(worker, lambda: len(state.destinations[(0, 0, 0)].sent_tokens) == 2)
    assert set(state.available_layers) == {attention, indexer}
    assert state.available_layers[attention][1] is state.available_layers[indexer][1]
    assert not worker._send_done.called
    transfers = {
        tuple(call.args[1]): (call.args[2], call.args[3])
        for call in worker.engine.batch_transfer_sync_write.call_args_list
    }
    assert transfers == {
        (1032, 1064): ([10320, 10352], [32, 32]),
        (2096, 2128): ([20640, 20672], [32, 32]),
    }
    # Another hook for the same physical layer must not send the caches twice.
    worker.save_kv_layer(indexer)
    assert worker.engine.batch_transfer_sync_write.call_count == 2
    worker.save_kv_layer(next_layer)
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert worker.engine.batch_transfer_sync_write.call_count == 3


@pytest.mark.parametrize("worker", ["mtp", "eagle3", "dspark"], indirect=True)
@pytest.mark.parametrize("draft_hook", [False, True])
def test_draft_namespace_does_not_share_target_readiness(worker, monkeypatch, draft_hook):
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda message: multi_cache_remote(worker))
    message = dict(destination_message(), block_ids=[[20, 21], [10, 11]])
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_send_req("p-request", ([1, 2], [3, 4]), 8, 0, 8, target_params(message))
    worker.start_load_kv(meta)
    state = worker._reqs_send_state["d-request"]
    destination = state.destinations[(0, 0, 0)]
    destination.metadata_future.result(timeout=3)
    attention, indexer, draft = worker.transfer_metadata.layer_names

    worker.save_kv_layer(attention)
    assert set(state.available_layers) == {attention, indexer}
    assert state.available_layers[attention][1] is state.available_layers[indexer][1]
    drive(worker, lambda: len(destination.sent_tokens) == 2)
    worker.save_kv_layer(indexer)
    assert draft not in state.available_layers
    assert destination.completion is None

    if draft_hook:
        worker.save_kv_layer(draft)
        drive(worker, lambda: len(destination.sent_tokens) == 3)
        assert state.available_layers[draft][1] is not state.available_layers[attention][1]
        # No namespace's numeric last layer can finalize the entire step.
        assert destination.completion is None
    worker._send_done.assert_not_called()

    # The deferred runner finalization also covers draft caches without hooks.
    worker.wait_for_save()
    assert set(destination.sent_tokens) == {attention, indexer, draft}
    assert worker.engine.batch_transfer_sync_write.call_count == 3
    worker._send_done.assert_called_once()
    assert worker._send_done.call_args.args[1] is None


@pytest.mark.parametrize("worker", ["multi_cache"], indirect=True)
def test_shared_layer_event_preserves_final_chunk_only_state(worker, monkeypatch):
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda message: multi_cache_remote(worker))
    attention, circular, _ = worker.transfer_metadata.layer_names
    worker.kv_cache_specs.append(SimpleNamespace(is_circular=True, sliding_window=None))
    worker.layer_name_to_spec_index[circular] = 1
    message = destination_message()
    message["block_ids"] = [[20], [10, 11]]
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_send_req("p-request", ([1, 2], [3]), 8, 0, 4, target_params(message))
    worker.start_load_kv(meta)
    state = worker._reqs_send_state["d-request"]
    drive(worker, lambda: state.destinations[(0, 0, 0)].required_layers is not None)
    worker.save_kv_layer(attention)
    drive(worker, lambda: attention in state.destinations[(0, 0, 0)].sent_tokens)
    assert circular in state.available_layers
    assert circular not in state.destinations[(0, 0, 0)].sent_tokens
    assert worker.engine.batch_transfer_sync_write.call_count == 1
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_send_req("p-request", ([1, 2], [3]), 8, 4, 4, target_params(message))
    worker.start_load_kv(meta)
    worker.save_kv_layer(attention)
    drive(
        worker,
        lambda: all(state.destinations[(0, 0, 0)].sent_tokens.get(layer) == 8 for layer in (attention, circular)),
    )
    assert worker.engine.batch_transfer_sync_write.call_count == 3
    assert any(
        call.args[1:] == ([2096], [20640], [32]) for call in worker.engine.batch_transfer_sync_write.call_args_list
    )


def test_new_progress_while_write_pending_is_not_lost(worker):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked_write(session, src, dst, lengths):
        calls.append((src, dst, lengths))
        if len(calls) == 1:
            entered.set()
            assert release.wait(3)
        return 0

    worker.engine.batch_transfer_sync_write.side_effect = blocked_write
    worker.start_load_kv(source_meta(4, 4))
    drive(worker, entered.is_set)
    try:
        # The cached prefix is still being sent when this forward finishes.
        worker.save_kv_layer("model.layers.0.attn")
        assert len(calls) == 1
    finally:
        release.set()
    worker.wait_for_save()
    drive(worker, lambda: worker._send_done.called)
    assert calls == [([1032], [10320], [32]), ([1064], [10352], [32])]


def test_layers_ready_before_metadata_query_finishes_are_backfilled(worker, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def delayed_metadata(message):
        entered.set()
        assert release.wait(3)
        return remote_metadata()

    monkeypatch.setattr(worker, "_get_remote_metadata", delayed_metadata)
    worker.start_load_kv(source_meta(0, 8))
    assert entered.wait(3)
    worker.save_kv_layer("model.layers.0.attn")
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker.wait_for_save)
        try:
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
            assert not worker.engine.batch_transfer_sync_write.called
        finally:
            release.set()
        future.result(timeout=3)
    assert worker.engine.batch_transfer_sync_write.call_count == 1
    assert worker._send_done.called


def test_no_release_until_every_d_rank_is_terminal(worker, monkeypatch):
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda message: remote_metadata(pcp_size=2))
    entered, release = threading.Event(), threading.Event()

    def done(params, error):
        if params["consumer_rank"] == [0, 1, 0]:
            entered.set()
            assert release.wait(3)
        return True

    monkeypatch.setattr(worker, "_send_done", done)
    meta = source_meta(0, 8)
    target = meta.push_send_requests["p-request"]
    meta.push_send_requests["p-request"] = replace(
        target, remote_worker_addresses=(*target.remote_worker_addresses, (0, 1, 0, "tcp://127.0.0.1:9002"))
    )
    worker.start_load_kv(meta)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker.wait_for_save)
        try:
            assert entered.wait(3)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
        finally:
            release.set()
        future.result(timeout=3)
    terminal = MooncakeConnectorMetadata()
    terminal.requests_to_send = {"p-request": time.monotonic()}
    worker.start_load_kv(terminal)
    drive(worker, lambda: "d-request" not in worker._reqs_send_state)


@pytest.mark.parametrize("worker", ["multi_cache"], indirect=True)
def test_last_layer_flushes_backlog_after_metadata_arrives(worker, monkeypatch):
    release = threading.Event()

    def metadata_query(params):
        assert release.wait(3)
        return multi_cache_remote(worker)

    monkeypatch.setattr(worker, "_get_remote_metadata", metadata_query)
    message = dict(destination_message(), block_ids=[[20, 21], [10, 11]])
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_send_req("p-request", ([1, 2], [3, 4]), 8, 0, 8, target_params(message))
    worker.start_load_kv(meta)
    attention, indexer, next_layer = worker.transfer_metadata.layer_names
    try:
        worker.save_kv_layer(attention)
        assert not worker.engine.batch_transfer_sync_write.called
    finally:
        release.set()
    state = worker._reqs_send_state["d-request"]
    state.destinations[(0, 0, 0)].metadata_future.result(timeout=3)
    # Polling results must not backfill old layers or submit any WRITE.
    worker.get_transfer_results()
    assert not worker.engine.batch_transfer_sync_write.called
    worker.save_kv_layer(next_layer)
    assert {batch.layer_name for batch, _ in worker._pending_sends} == {attention, indexer, next_layer}
    for batch, future in worker._pending_sends:
        future.result(timeout=3)
    state.destinations[(0, 0, 0)].completion.result(timeout=3)
    assert worker.engine.batch_transfer_sync_write.call_count == 3
    worker._send_done.assert_called_once()
    worker.wait_for_save()
    assert worker.engine.batch_transfer_sync_write.call_count == 3
    assert set(state.destinations[(0, 0, 0)].sent_tokens) == {attention, indexer, next_layer}
    worker._send_done.assert_called_once()


def test_cancel_only_step_finishes_without_waiting_for_metadata(worker, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def metadata_query(params):
        entered.set()
        assert release.wait(3)
        return remote_metadata()

    monkeypatch.setattr(worker, "_get_remote_metadata", metadata_query)
    worker.start_load_kv(source_meta(0, 0))
    assert entered.wait(3)
    events = Mock(side_effect=Event)
    monkeypatch.setattr(push_worker.torch.npu, "Event", events)
    try:
        cancelled = MooncakeConnectorMetadata()
        cancelled.push_aborted_requests = {"p-request": target_params()}
        worker.start_load_kv(cancelled)
        assert not worker._prefilling_request_ids
        worker._send_done.assert_not_called()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(worker.wait_for_save)
            try:
                future.result(timeout=1)
            finally:
                release.set()
        events.assert_not_called()
        assert not worker.engine.batch_transfer_sync_write.called
        worker._send_done.assert_called_once()
        assert worker._send_done.call_args.args[1] == "P request aborted"
        assert worker.get_transfer_results().finished_sending == {"p-request"}
    finally:
        release.set()


def test_send_timeout_is_processed_in_a_no_forward_step(worker):
    worker.start_load_kv(source_meta(0, 0))
    state = worker._reqs_send_state["d-request"]
    state.start_time -= push_worker.envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 1
    worker.start_load_kv(MooncakeConnectorMetadata())
    assert not worker._prefilling_request_ids
    assert not worker.engine.batch_transfer_sync_write.called
    worker._send_done.assert_not_called()
    worker.wait_for_save()
    worker._send_done.assert_called_once()
    assert worker._send_done.call_args.args[1] == "Push transfer timed out"


def test_metadata_wait_crossing_deadline_does_not_start_write(worker, monkeypatch):
    release = threading.Event()

    def metadata_query(params):
        assert release.wait(3)
        # Model a query that completes only after the request deadline.
        worker._reqs_send_state["d-request"].start_time -= push_worker.envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 1
        return remote_metadata()

    class EndOfForwardEvent(Event):
        def record(self):
            super().record()
            release.set()

    monkeypatch.setattr(worker, "_get_remote_metadata", metadata_query)
    monkeypatch.setattr(push_worker.torch.npu, "Event", EndOfForwardEvent)
    try:
        worker.start_load_kv(source_meta(0, 8))
        worker.wait_for_save()
        worker.engine.batch_transfer_sync_write.assert_not_called()
        worker._send_done.assert_called_once()
        assert worker._send_done.call_args.args[1] == "Push transfer timed out"
    finally:
        release.set()


@pytest.mark.parametrize("worker", ["three_layers"], indirect=True)
@pytest.mark.parametrize("failure", [None, "negative"])
def test_busy_engine_defers_middle_layer_but_last_layer_submits(worker, monkeypatch, failure):
    monkeypatch.setattr(worker, "_get_remote_metadata", lambda params: multi_cache_remote(worker))
    entered, release, last_written = threading.Event(), threading.Event(), threading.Event()

    def write(session, src, dst, lengths):
        if src[0] == 1032:
            entered.set()
            assert release.wait(3)
            if failure == "negative":
                return -1
        if src[0] == 3032:
            last_written.set()
        return 0

    worker.engine.batch_transfer_sync_write.side_effect = write
    message = dict(destination_message(), block_ids=[[20, 21], [10, 11]])
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_send_req("p-request", ([1, 2], [3, 4]), 8, 0, 8, target_params(message))
    worker.start_load_kv(meta)
    destination = worker._reqs_send_state["d-request"].destinations[(0, 0, 0)]
    destination.metadata_future.result(timeout=3)
    first, middle, last = worker.transfer_metadata.layer_names
    try:
        worker.save_kv_layer(first)
        assert entered.wait(3)
        worker.save_kv_layer(middle)
        assert [batch.layer_name for batch, _ in worker._pending_sends] == [first]
        worker.save_kv_layer(last)
        assert last_written.wait(3)
        assert {batch.layer_name for batch, _ in worker._pending_sends} == {first, middle, last}
        assert destination.completion is not None
        assert not destination.completion.done()
        worker._send_done.assert_not_called()
    finally:
        release.set()
    # DONE is emitted by the pool without a wait_for_save or results poll.
    destination.completion.result(timeout=3)
    worker._send_done.assert_called_once()
    assert bool(worker._send_done.call_args.args[1]) == (failure is not None)
    worker.save_kv_layer(last)
    worker.wait_for_save()
    worker.wait_for_save()
    assert worker.engine.batch_transfer_sync_write.call_count == 3
    assert not worker._sending_futures
    worker._send_done.assert_called_once()


@pytest.mark.parametrize("worker", ["three_layers"], indirect=True)
def test_busy_engine_does_not_block_another_engine(worker, monkeypatch):
    monkeypatch.setattr(
        worker,
        "_get_remote_metadata",
        lambda params: replace(multi_cache_remote(worker), engine_id=params["consumer_engine_id"]),
    )
    entered, release = threading.Event(), threading.Event()

    def write(session, src, dst, lengths):
        if src[0] == 1032:
            entered.set()
            assert release.wait(3)
        return 0

    worker.engine.batch_transfer_sync_write.side_effect = write
    message = dict(destination_message(), block_ids=[[20, 21], [10, 11]])
    meta = MooncakeConnectorMetadata()
    meta.add_new_push_send_req("p-request", ([1, 2], [3, 4]), 8, 0, 8, target_params(message))
    params = dict(target_params(message), remote_engine_id="other", remote_request_id="other-request")
    meta.add_new_push_send_req("other-p", ([5, 6], [7, 8]), 8, 0, 8, params)
    worker.start_load_kv(meta)
    for state in worker._reqs_send_state.values():
        state.destinations[(0, 0, 0)].metadata_future.result(timeout=3)
    first, middle, _ = worker.transfer_metadata.layer_names
    try:
        worker.save_kv_layer(first)
        assert entered.wait(3)
        for future in worker._sending_futures["other"]:
            future.result(timeout=3)
        worker.save_kv_layer(middle)
        for future in worker._sending_futures["other"]:
            future.result(timeout=3)
        assert any(call.args[1] == [2224, 2256] for call in worker.engine.batch_transfer_sync_write.call_args_list)
        assert not any(call.args[1] == [2096, 2128] for call in worker.engine.batch_transfer_sync_write.call_args_list)
    finally:
        release.set()
    worker.wait_for_save()
    assert worker._send_done.call_count == 2
