# SPDX-License-Identifier: Apache-2.0
"""Real control sockets and real byte buffers; only the device WRITE is doubled."""

import ctypes
import queue
import socket
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import msgspec
import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_config import PushConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_metadata import (
    MooncakePushMetadata,
    PushReceive,
    PushSource,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_worker import (
    MooncakePushConnectorWorker,
    ProducerPeer,
    PushCompletion,
    Registration,
    SourceState,
)

from .helpers import make_transfer_metadata


def worker(metadata, producer=False, engine=None):
    instance = MooncakePushConnectorWorker.__new__(MooncakePushConnectorWorker)
    instance.engine_id = metadata.engine_id
    instance.tp_rank = 0
    instance.tp_size = 1
    instance.engine = engine or MagicMock()
    instance._metadata = metadata
    instance._dtype = "torch.float16"
    instance._device_index = 0
    instance.side_channel_host = "127.0.0.1"
    instance.handshake_port = metadata.handshake_port
    instance.kv_transfer_config = SimpleNamespace(is_kv_producer=producer)
    instance.vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"))
    instance.push_config = PushConfig(timeout=3, handshake_timeout=1, max_pending=8)
    instance._inbox = queue.SimpleQueue()
    instance._completions = queue.SimpleQueue()
    instance._invalid_blocks = set()
    instance._pending_results = KVConnectorTransferResults()
    instance._ready = threading.Event()
    instance._stop = threading.Event()
    instance._epoch = uuid4().hex
    instance._producer_peers = {}
    instance._consumer_peers = {}
    instance._sources = {}
    instance._registrations = {}
    instance._receives = {}
    instance._target_quarantine = {}
    instance._peer_receives = {}
    instance._router = None
    instance._context = None
    instance._poller = None
    instance._startup_error = None
    instance._thread = None
    return instance


def enqueue_receive(instance, receive, event=None):
    instance._inbox.put(("receive", (receive, event if event is not None else MagicMock())))


def free_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for push protocol progress")
        time.sleep(0.005)


def start(instance):
    instance._thread = threading.Thread(target=instance._run, daemon=True)
    instance._thread.start()
    assert instance._ready.wait(2)
    assert instance._startup_error is None


@pytest.mark.parametrize("registration_first", [True, False])
def test_real_control_rendezvous_copies_fragmented_pages(registration_first):
    source = (ctypes.c_ubyte * 4096)(*(index % 251 for index in range(4096)))
    target = (ctypes.c_ubyte * 4096)()
    source_metadata = make_transfer_metadata(
        engine_id="p", handshake_port=free_port(), base_addrs=[[ctypes.addressof(source)]]
    )
    target_metadata = make_transfer_metadata(engine_id="d", base_addrs=[[ctypes.addressof(target)]])
    engine = MagicMock()

    def write(_session, sources, destinations, lengths):
        for src, dst, length in zip(sources, destinations, lengths):
            ctypes.memmove(dst, src, length)
        return 0

    engine.batch_transfer_sync_write.side_effect = write
    producer = worker(source_metadata, True, engine)
    consumer = worker(target_metadata)
    start(producer)
    start(consumer)
    key = str(uuid4())
    source_task = PushSource("p-internal-random-id", key, ([1, 3],), 31, True)
    receive = PushReceive(
        "d-different-random-id", key, ([2, 5],), 31, "127.0.0.1", source_metadata.handshake_port, "p", 1
    )
    try:
        if registration_first:
            enqueue_receive(consumer, receive)
            wait_until(lambda: key in producer._registrations)
            producer._inbox.put(("source", (source_task, MagicMock())))
        else:
            producer._inbox.put(("source", (source_task, MagicMock())))
            wait_until(lambda: key in producer._sources)
            enqueue_receive(consumer, receive)
        wait_until(lambda: not consumer._completions.empty())
        assert consumer.get_transfer_results().finished_recving == {"d-different-random-id"}
        wait_until(lambda: not producer._completions.empty())
        assert producer.get_transfer_results().finished_sending == {"p-internal-random-id"}
        assert bytes(target[256:384]) == bytes(source[128:256])
        assert bytes(target[640:768]) == bytes(source[384:512])
        assert not any(target[:256])
        assert not any(target[384:640])
        assert engine.batch_transfer_sync_write.call_count == 1
        assert not consumer.get_transfer_results().finished_recving
    finally:
        consumer.shutdown()
        producer.shutdown()


def test_cancel_during_write_does_not_complete_until_write_is_fenced():
    started = threading.Event()
    release = threading.Event()
    engine = MagicMock()

    def write(*_args):
        started.set()
        assert release.wait(5)
        return 0

    engine.batch_transfer_sync_write.side_effect = write
    metadata = make_transfer_metadata(engine_id="p", handshake_port=free_port())
    producer = worker(metadata, True, engine)
    consumer = worker(replace(metadata, engine_id="d"))
    start(producer)
    start(consumer)
    key = str(uuid4())
    try:
        producer._inbox.put(("source", (PushSource("p", key, ([1],), 16, True), MagicMock())))
        enqueue_receive(consumer, PushReceive("d", key, ([2],), 16, "127.0.0.1", metadata.handshake_port, "p", 1))
        assert started.wait(3)
        consumer._inbox.put(("cancel", "d"))
        wait_until(lambda: consumer._receives["d"].cancelling)
        assert not consumer.get_transfer_results().finished_recving
        assert not producer.get_transfer_results().finished_sending
        release.set()
        wait_until(lambda: not consumer._completions.empty())
        assert consumer.get_transfer_results().finished_recving == {"d"}
    finally:
        release.set()
        consumer.shutdown()
        producer.shutdown()


def test_failed_write_quarantines_source_and_rejects_cancel_ack():
    metadata = make_transfer_metadata()
    instance = worker(metadata, True)
    instance._router = MagicMock()
    instance.engine.batch_transfer_sync_write.return_value = -1
    instance._producer_peers[b"d"] = ProducerPeer(replace(metadata, engine_id="d"), 1)
    key = str(uuid4())
    instance._sources[key] = SourceState(PushSource("p", key, ([1],), 16, True), MagicMock(), 0)
    instance._registrations[key] = Registration(b"d", 1, key, ([2],), 16, 0)
    instance._try_push(key)
    instance._handle_producer_message(
        b"d", {"kind": "CANCEL", "producer_epoch": instance._epoch, "sequence": 1, "transfer_id": key}
    )
    instance._expire()
    assert instance._sources[key].quarantined
    assert instance._registrations[key].quarantined
    assert not instance.get_transfer_results().finished_sending
    replies = [
        msgspec.msgpack.decode(call.args[0][1])["kind"] for call in instance._router.send_multipart.call_args_list
    ]
    assert replies == ["QUARANTINED", "QUARANTINED"]


def test_retired_sequence_cannot_reauthorize_destination_after_cancel():
    metadata = make_transfer_metadata()
    instance = worker(metadata, True)
    instance._router = MagicMock()
    instance._producer_peers[b"d"] = ProducerPeer(metadata)
    key = str(uuid4())
    registration = {
        "kind": "REGISTER",
        "producer_epoch": instance._epoch,
        "sequence": 1,
        "transfer_id": key,
        "block_ids": [[2]],
        "num_tokens": 16,
    }
    instance._handle_producer_message(b"d", registration)
    instance._handle_producer_message(b"d", {**registration, "kind": "CANCEL"})
    instance._handle_producer_message(b"d", registration)
    assert key not in instance._registrations
    assert not instance.engine.batch_transfer_sync_write.called
    replies = [
        msgspec.msgpack.decode(call.args[0][1])["kind"] for call in instance._router.send_multipart.call_args_list
    ]
    assert replies == ["CANCELLED", "RETIRED"]


def test_invalid_pages_and_completions_share_one_snapshot():
    instance = worker(make_transfer_metadata())
    instance._completions.put(PushCompletion(received={"first"}, failed={"first"}, invalid_blocks={2}))
    first = instance.get_transfer_results()
    instance._completions.put(PushCompletion(received={"second"}, failed={"second"}, invalid_blocks={3}))
    assert first.finished_recving == first.failed_recving == {"first"}
    assert instance.get_block_ids_with_load_errors() == {2}
    second = instance.get_transfer_results()
    assert second.finished_recving == second.failed_recving == {"second"}
    assert instance.get_block_ids_with_load_errors() == {3}


def test_source_expiry_without_registration_is_safe_to_release():
    instance = worker(make_transfer_metadata(), True)
    key = str(uuid4())
    instance._sources[key] = SourceState(PushSource("p", key, ([1],), 16, True), MagicMock(), 0)
    instance._expire()
    assert instance.get_transfer_results().finished_sending == {"p"}
    assert not instance.engine.batch_transfer_sync_write.called


def test_stale_epoch_never_authorizes_a_write():
    metadata = make_transfer_metadata()
    instance = worker(metadata, True)
    instance._router = MagicMock()
    instance._producer_peers[b"d"] = ProducerPeer(metadata)
    instance._handle_producer_message(
        b"d", {"kind": "REGISTER", "producer_epoch": "old-process", "sequence": 1, "transfer_id": str(uuid4())}
    )
    assert not instance._registrations
    assert not instance.engine.batch_transfer_sync_write.called


def test_destination_is_not_published_before_initialization_fence():
    entered = threading.Event()
    initialized = threading.Event()
    event = MagicMock()
    event.synchronize.side_effect = lambda: (entered.set(), initialized.wait(5))
    metadata = make_transfer_metadata(engine_id="p", handshake_port=free_port())
    producer = worker(metadata, True)
    producer.engine.batch_transfer_sync_write.return_value = 0
    consumer = worker(replace(metadata, engine_id="d"))
    start(producer)
    start(consumer)
    key = str(uuid4())
    try:
        producer._inbox.put(("source", (PushSource("p", key, ([1],), 16, True), MagicMock())))
        enqueue_receive(
            consumer, PushReceive("d", key, ([2],), 16, "127.0.0.1", metadata.handshake_port, "p", 1), event
        )
        assert entered.wait(2)
        assert not producer._registrations
        assert not producer.engine.batch_transfer_sync_write.called
        initialized.set()
        wait_until(lambda: not consumer._completions.empty())
        assert consumer.get_transfer_results().finished_recving == {"d"}
    finally:
        initialized.set()
        consumer.shutdown()
        producer.shutdown()


def test_failed_target_fence_does_not_release_destination():
    instance = worker(make_transfer_metadata(engine_id="d"))
    key = str(uuid4())
    receive = PushReceive("d", key, ([2],), 16, "127.0.0.1", 5600, "p", 1)
    event = MagicMock()
    event.synchronize.side_effect = RuntimeError("device fence failed")
    instance._register_receive(receive, event)
    assert instance._target_quarantine["d"] == receive
    assert not instance.get_transfer_results().finished_recving


def test_runner_records_destination_fence_before_queueing(monkeypatch):
    instance = worker(make_transfer_metadata(engine_id="d"))
    event = MagicMock()
    monkeypatch.setattr("torch.npu.Event", lambda: event)
    metadata = MooncakePushMetadata()
    metadata.push_receives = [PushReceive("d", str(uuid4()), ([2],), 16, "127.0.0.1", 5600, "p", 1)]
    instance.start_load_kv(metadata)
    event.record.assert_called_once_with()
    kind, (receive, fence) = instance._inbox.get_nowait()
    assert kind == "receive" and fence is event and receive is metadata.push_receives[0]
