# SPDX-License-Identifier: Apache-2.0
"""Keep draft-KV acknowledgement bytes and receiver lifecycle unchanged."""

import ast
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h import send_thread as sending
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    DSparkContextDescriptor,
    DSparkContextReceiver,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_kv import DraftKVCacheMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.protocol import (
    DSPARK_DRAFT_KV,
    DSPARK_DRAFT_KV_ACK,
    DSparkDraftKVStatus,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.read_thread import ConsumerReadState, MembPullReadThread


def _descriptor(generation="allocation-1"):
    return DSparkContextDescriptor("request", generation, 6, (2, 22, 38, 58, 74), 4)


def _metadata(group_id, base):
    return DraftKVCacheMetadata(
        group_id=group_id,
        block_size=4,
        num_blocks=8,
        base_addrs=(base,),
        block_strides=(16,),
        block_lens=(16,),
        block_scales=(1,),
        shapes=((4, 2),),
        dtypes=("torch.bfloat16",),
    )


def _reader(receiver):
    reader = MembPullReadThread.__new__(MembPullReadThread)
    reader._state = ConsumerReadState(
        num_blocks=8,
        tp_size=1,
        layer_metadata={},
        main_name_to_idx={},
        cpu_pools=[],
        main_gva_bases=[],
        main_block_lens=[],
        indexer_tensors=[],
        indexer_scale_tensors=[],
        dest_blocks_by_req={},
        get_offload_layer_id=lambda name: 0,
        dspark_context_receiver=receiver,
        dspark_draft_kv_metadata={"draft": _metadata(2, 4000)},
        dspark_draft_blocks_by_req={"request": {2: (4, 5)}},
    )
    reader._p_sessions = {b"p": "p-session"}
    reader._p_pp_topology = {b"p": (1, 2)}
    reader._lock = threading.Lock()
    reader._failed_requests = set()
    reader.engine = SimpleNamespace(batch_transfer_sync_read=Mock(return_value=0))
    return reader


def _read_ack(reader, descriptor):
    source = _metadata(7, 1000)
    source_wire = {name: getattr(source, name) for name in source.__dataclass_fields__}
    message = (
        DSPARK_DRAFT_KV,
        descriptor.request_id,
        descriptor.generation,
        descriptor.prompt_tokens,
        descriptor.aux_layer_ids,
        descriptor.hidden_size,
        {"draft": source_wire},
        {7: (1, 2)},
    )
    sock = Mock()
    reader._handle_dspark_draft_kv(b"p", message, sock, msgspec.msgpack.Encoder())
    sock.send_multipart.assert_called_once()
    identity, separator, payload = sock.send_multipart.call_args.args[0]
    assert (identity, separator) == (b"p", b"")
    return msgspec.msgpack.decode(payload)


@pytest.mark.parametrize(
    ("status", "wire"),
    [
        (DSparkDraftKVStatus.ACCEPTED, b"accepted"),
        (DSparkDraftKVStatus.BACKPRESSURE, b"backpressure"),
        (DSparkDraftKVStatus.STALE, b"stale"),
        (DSparkDraftKVStatus.FAILED, b"failed"),
    ],
)
def test_status_enum_preserves_existing_msgpack_bytes(status, wire):
    assert type(status.value) is bytes
    assert status.value == wire
    assert DSparkDraftKVStatus(wire) is status
    assert msgspec.msgpack.encode((DSPARK_DRAFT_KV_ACK, "request", "allocation-1", status.value)) == (
        msgspec.msgpack.encode((DSPARK_DRAFT_KV_ACK, "request", "allocation-1", wire))
    )


@pytest.mark.parametrize("admission", ["not-bound", "not-admitted", "in-flight", "stale"])
def test_reader_wait_or_stale_ack_never_copies_or_fails_request(admission):
    descriptor = _descriptor()
    receiver = DSparkContextReceiver(max_requests=1)
    if admission == "in-flight":
        receiver.register_request(descriptor)
        receiver.begin_direct_transfer(descriptor)
    elif admission == "stale":
        receiver.register_request(_descriptor("allocation-2"))
    reader = _reader(None if admission == "not-bound" else receiver)

    ack = _read_ack(reader, descriptor)

    expected = b"stale" if admission == "stale" else b"backpressure"
    assert ack == [DSPARK_DRAFT_KV_ACK, "request", "allocation-1", expected]
    reader.engine.batch_transfer_sync_read.assert_not_called()
    assert reader._failed_requests == set()


def test_reader_completed_and_retired_duplicate_ack_does_not_copy_twice():
    descriptor = _descriptor()
    receiver = DSparkContextReceiver(max_requests=1)
    receiver.register_request(descriptor)
    reader = _reader(receiver)
    expected = [DSPARK_DRAFT_KV_ACK, "request", "allocation-1", b"accepted"]

    assert _read_ack(reader, descriptor) == expected
    assert receiver.ready_requests() == set()
    assert _read_ack(reader, descriptor) == expected
    receiver.mark_target_kv_done("request", "allocation-1")
    assert receiver.ready_requests() == {"request"}
    receiver.retire_ready_request("request", "allocation-1")
    assert _read_ack(reader, descriptor) == expected
    receiver.discard_request_id("request")
    assert _read_ack(reader, descriptor) == expected
    reader.engine.batch_transfer_sync_read.assert_called_once_with("p-session", [4064, 4080], [1016, 1032], [16, 8])
    assert reader._failed_requests == set()


def test_reader_failed_ack_quarantines_allocation_and_does_not_retry_copy():
    descriptor = _descriptor()
    receiver = DSparkContextReceiver(max_requests=1)
    receiver.register_request(descriptor)
    reader = _reader(receiver)
    reader.engine.batch_transfer_sync_read.return_value = 1
    expected = [DSPARK_DRAFT_KV_ACK, "request", "allocation-1", b"failed"]

    assert _read_ack(reader, descriptor) == expected
    assert _read_ack(reader, descriptor) == expected
    reader.engine.batch_transfer_sync_read.assert_called_once()
    receiver.mark_target_kv_done("request", "allocation-1")
    assert receiver.ready_requests() == set()
    assert reader._failed_requests == {"request"}


def _sender(monkeypatch, replies):
    descriptor = _descriptor()
    path = "tcp://127.0.0.1:1234"
    sender = sending.MembPullSendingThread.__new__(sending.MembPullSendingThread)
    dealer = Mock()
    dealer.poll.return_value = True
    dealer.recv_multipart.side_effect = [
        [b"", msgspec.msgpack.encode((DSPARK_DRAFT_KV_ACK, "request", "allocation-1", status))] for status in replies
    ]
    monkeypatch.setattr(sender, "_ensure_peer_dealer", Mock(return_value=dealer))
    sender._mf_meta_sent_paths = {path}
    send_meta = Mock()
    monkeypatch.setattr(sender, "_send_mf_meta", send_meta)
    monkeypatch.setattr(sending, "make_zmq_path", lambda *args: path)
    monkeypatch.setattr(sending.time, "monotonic", lambda: 0.0)
    sleep = Mock()
    monkeypatch.setattr(sending.time, "sleep", sleep)
    task = SimpleNamespace(
        host="127.0.0.1",
        port=1234,
        remote_engine_id="decode",
        descriptor=descriptor,
        timeout=1.0,
        draft_kv_metadata={"draft": {"base_addrs": (1000,)}},
        source_blocks_by_group={7: (1, 2)},
    )
    return sender, dealer, task, sleep, send_meta


def test_sender_backpressure_retries_identical_payload_until_accepted(monkeypatch):
    sender, dealer, task, sleep, send_meta = _sender(monkeypatch, [b"backpressure", b"accepted"])

    sender._process_dspark_draft_kv_task(task, msgspec.msgpack.Encoder(), msgspec.msgpack.Decoder())

    assert dealer.send.call_count == 2
    assert dealer.send.call_args_list[0] == dealer.send.call_args_list[1]
    sleep.assert_called_once_with(0.01)
    send_meta.assert_not_called()
    payload = msgspec.msgpack.decode(dealer.send.call_args.args[0])
    assert payload[:6] == [DSPARK_DRAFT_KV, "request", "allocation-1", 6, [2, 22, 38, 58, 74], 4]


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (b"stale", "D discarded stale DSpark draft-KV allocation request/allocation-1"),
        (b"failed", "D rejected DSpark draft-KV transfer: b'failed'"),
        (b"unsupported", "D rejected DSpark draft-KV transfer: b'unsupported'"),
    ],
)
def test_sender_rejection_keeps_original_error_and_does_not_retry(monkeypatch, status, message):
    sender, dealer, task, sleep, _ = _sender(monkeypatch, [status])

    with pytest.raises(RuntimeError) as error:
        sender._process_dspark_draft_kv_task(task, msgspec.msgpack.Encoder(), msgspec.msgpack.Decoder())

    assert str(error.value) == message
    dealer.send.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("class_name", "method_name", "message"),
    [
        (
            "SFAPDRD2HConsumerWorker",
            "get_finished",
            "DSpark remote prompt ready: req=%s tokens=%d aux_layers=%s draft_groups=%s tp_rank=%d",
        ),
        (
            "SFAPDRD2HProducerWorker",
            "send_dspark_draft_kv",
            "DSpark P draft KV transferred: req=%s tokens=%d draft_layers=%d draft_groups=%s tp_rank=%d",
        ),
    ],
)
def test_per_request_dspark_transfer_log_is_debug(class_name, method_name, message):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logger"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == message
    ]
    assert len(calls) == 1
    assert isinstance(calls[0].func, ast.Attribute)
    assert calls[0].func.attr == "debug"
