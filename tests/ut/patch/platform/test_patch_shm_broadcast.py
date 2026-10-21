# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pickle
import re
import threading
import time
import zlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.distributed.device_communicators import shm_broadcast
from zmq.error import Again
from zmq.sugar.frame import Frame

import vllm_ascend.patch.platform.patch_shm_broadcast as patch_module

_MIB = 1024 * 1024


def _make_queue(n_local_reader, n_remote_reader, *, max_chunk_bytes):
    shm_buffer = bytearray(max_chunk_bytes)
    queue = SimpleNamespace(
        _is_writer=True,
        _is_local_reader=True,
        _is_remote_reader=False,
        n_local_reader=n_local_reader,
        n_remote_reader=n_remote_reader,
        buffer=SimpleNamespace(max_chunk_bytes=max_chunk_bytes),
        local_socket=MagicMock(),
        remote_socket=MagicMock(),
        acquire_write=MagicMock(return_value=nullcontext(shm_buffer)),
        acquire_read=MagicMock(return_value=nullcontext(shm_buffer)),
        _spin_condition=MagicMock(),
    )
    return queue, shm_buffer


@pytest.fixture
def diagnostic_logger(monkeypatch):
    mock_logger = MagicMock()
    monkeypatch.setattr(patch_module, "logger", mock_logger)
    return mock_logger


def _error_summary(mock_logger):
    mock_logger.error.assert_called_once()
    call = mock_logger.error.call_args
    # No second traceback or payload dump alongside the caller's error.
    assert call.kwargs == {}
    summary = call.args[0] % call.args[1:]
    assert "\n" not in summary
    return summary


@pytest.mark.parametrize("payload_type", ["tensor", "numpy"])
@pytest.mark.parametrize("size", [_MIB - 1, _MIB, _MIB + 1])
@pytest.mark.parametrize("readers", [(0, 1), (1, 0), (1, 1)], ids=["remote", "local-overflow", "both"])
def test_enqueue_snapshots_buffers_before_async_send(payload_type, size, readers):
    queue, shm_buffer = _make_queue(*readers, max_chunk_bytes=128)
    if payload_type == "tensor":
        source = torch.ones(size, dtype=torch.uint8)
    else:
        source = np.ones(size, dtype=np.uint8)
    pattern = re.compile("snapshot", re.IGNORECASE)

    shm_broadcast.MessageQueue.enqueue(queue, {"data": source, "pattern": pattern}, timeout=1.5)
    # MagicMock retains the actual buffers without copying them. Mutating the
    # source before deserialization deterministically models a delayed ZMQ send.
    source[:] = 0

    for reader_count, socket in zip(readers, (queue.local_socket, queue.remote_socket)):
        if not reader_count:
            socket.send_multipart.assert_not_called()
            continue
        socket.send_multipart.assert_called_once()
        assert socket.send_multipart.call_args.kwargs == {"copy": False}
        buffers = socket.send_multipart.call_args.args[0]
        if socket is queue.remote_socket:
            header, *buffers = buffers
            assert header[:4] == patch_module._FRAME_MAGIC
            assert len(header) == patch_module._FRAME_HEADER.size
        assert len(buffers) == (1 if size < _MIB else 2)
        if size >= _MIB:
            assert isinstance(buffers[1], bytes)
            assert len(buffers[1]) == size
        restored = pickle.loads(buffers[0], buffers=buffers[1:])
        assert isinstance(restored["data"], type(source))
        assert restored["data"].shape == source.shape
        assert restored["data"].dtype == source.dtype
        assert (restored["data"] == 1).all()
        # copyreg reducers must remain available alongside the tensor reducer.
        assert restored["pattern"].pattern == pattern.pattern
        assert restored["pattern"].flags == pattern.flags

    if readers[0]:
        assert shm_buffer[0] == 1
        queue.acquire_write.assert_called_once_with(1.5)
        queue._spin_condition.notify.assert_called_once_with()
    else:
        queue.acquire_write.assert_not_called()
        queue._spin_condition.notify.assert_not_called()


@pytest.mark.parametrize("size", [_MIB - 1, _MIB, _MIB + 1])
@pytest.mark.parametrize("n_remote_reader", [0, 1])
def test_enqueue_preserves_shared_memory_format(size, n_remote_reader):
    queue, shm_buffer = _make_queue(1, n_remote_reader, max_chunk_bytes=2 * _MIB)
    source = np.ones(size, dtype=np.uint8)
    shm_broadcast.MessageQueue.enqueue(queue, source, timeout=2.0)
    source[:] = 0

    assert shm_buffer[0] == 0
    queue.local_socket.send_multipart.assert_not_called()
    queue.acquire_write.assert_called_once_with(2.0)
    queue._spin_condition.notify.assert_called_once_with()
    # Use the upstream reader to verify compatibility with its wire format.
    restored = shm_broadcast.MessageQueue.dequeue(queue)
    np.testing.assert_array_equal(restored, np.ones(size, dtype=np.uint8))

    if n_remote_reader:
        frames = queue.remote_socket.send_multipart.call_args.args[0]
        _, *buffers = frames
        remote_restored = pickle.loads(buffers[0], buffers=buffers[1:])
        np.testing.assert_array_equal(remote_restored, restored)
    else:
        queue.remote_socket.send_multipart.assert_not_called()


def test_enqueue_rejects_readers():
    queue, _ = _make_queue(0, 1, max_chunk_bytes=128)
    queue._is_writer = False
    with pytest.raises(AssertionError, match="Only writers can enqueue"):
        shm_broadcast.MessageQueue.enqueue(queue, "message")
    queue.remote_socket.send_multipart.assert_not_called()


def test_enqueue_propagates_write_timeout(diagnostic_logger):
    queue, _ = _make_queue(1, 1, max_chunk_bytes=128)
    queue.acquire_write.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        shm_broadcast.MessageQueue.enqueue(queue, "message", timeout=0.1)
    queue.acquire_write.assert_called_once_with(0.1)
    queue.local_socket.send_multipart.assert_not_called()
    queue.remote_socket.send_multipart.assert_not_called()
    queue._spin_condition.notify.assert_not_called()
    diagnostic_logger.error.assert_not_called()


def test_patch_is_installed_on_upstream_class():
    assert shm_broadcast.MessageQueue.enqueue is patch_module.enqueue
    assert shm_broadcast.MessageQueue.recv is patch_module.recv
    assert shm_broadcast.MessageQueue.dequeue is patch_module.dequeue


@pytest.mark.parametrize("readers,transport", [((0, 1), "remote"), ((1, 0), "local"), ((1, 1), "remote")])
def test_enqueue_failure_reports_type_and_frame_sizes(diagnostic_logger, readers, transport):
    queue, _ = _make_queue(*readers, max_chunk_bytes=128)
    socket = getattr(queue, f"{transport}_socket")
    error = RuntimeError("private error details")
    socket.send_multipart.side_effect = error
    payload = {"private payload": np.ones(_MIB, dtype=np.uint8)}

    with pytest.raises(RuntimeError) as caught:
        shm_broadcast.MessageQueue.enqueue(queue, payload)

    assert caught.value is error
    buffers = socket.send_multipart.call_args.args[0]
    if transport == "remote":
        # The integrity header is added after serialization and must not show
        # up in the payload summary.
        assert buffers[0][:4] == patch_module._FRAME_MAGIC
        buffers = buffers[1:]
    sizes = [len(buffer) for buffer in buffers]
    summary = _error_summary(diagnostic_logger)
    assert f"stage={transport}_zmq_send" in summary
    assert "message_type=builtins.dict" in summary
    assert "error_type=RuntimeError" in summary
    assert "frame_count=2" in summary
    assert f"total_bytes={sum(sizes)}" in summary
    assert f"frame_sizes={sizes}" in summary
    assert "private" not in summary


def test_serialization_failure_is_logged_without_masking_error(diagnostic_logger):
    error = pickle.PicklingError("private error details")

    class Unpicklable:
        def __reduce_ex__(self, protocol):
            raise error

    queue, _ = _make_queue(0, 1, max_chunk_bytes=128)
    with pytest.raises(pickle.PicklingError) as caught:
        shm_broadcast.MessageQueue.enqueue(queue, Unpicklable())

    assert caught.value is error
    summary = _error_summary(diagnostic_logger)
    assert "stage=serialize" in summary
    assert "error_type=PicklingError" in summary
    assert "total_bytes=0" in summary
    assert "private" not in summary
    queue.remote_socket.send_multipart.assert_not_called()


@pytest.mark.parametrize("route", ["direct", "remote", "local-overflow"])
@pytest.mark.parametrize("frame_type", [bytes, Frame])
def test_corrupt_recv_logs_once_through_dequeue(diagnostic_logger, route, frame_type):
    queue, shm_buffer = _make_queue(1, 1, max_chunk_bytes=128)
    socket = queue.local_socket if route == "local-overflow" else queue.remote_socket
    socket.poll.return_value = True
    buffers = [frame_type(pickle.dumps({"private": "payload"})[:-3]), frame_type(b"oob")]
    socket.recv_multipart.return_value = buffers
    queue._is_local_reader = route == "local-overflow"
    queue._is_remote_reader = not queue._is_local_reader
    shm_buffer[0] = 1

    with pytest.raises(pickle.UnpicklingError):
        if route == "direct":
            shm_broadcast.MessageQueue.recv(socket, timeout=0.5)
        else:
            shm_broadcast.MessageQueue.dequeue(queue, timeout=0.5)

    socket.poll.assert_called_once_with(timeout=500)
    socket.recv_multipart.assert_called_once_with(copy=False)
    summary = _error_summary(diagnostic_logger)
    assert "stage=zmq_deserialize" in summary
    assert "message_type=unknown" in summary
    assert "error_type=UnpicklingError" in summary
    assert "frame_count=2" in summary
    assert f"total_bytes={sum(len(buffer) for buffer in buffers)}" in summary
    assert "private" not in summary


def test_failure_summary_limits_frame_details(diagnostic_logger):
    buffers = [b"abc"] * 20
    patch_module._log_message_queue_failure("zmq_deserialize", buffers, ValueError("private"))
    summary = _error_summary(diagnostic_logger)
    assert "frame_count=20" in summary
    assert "total_bytes=60" in summary
    assert "frame_sizes=[3, 3, 3, 3, 3, 3, 3, 3]" in summary
    assert "omitted_frames=12" in summary
    assert "private" not in summary


def test_logging_failure_does_not_replace_send_error(diagnostic_logger):
    queue, _ = _make_queue(0, 1, max_chunk_bytes=128)
    error = RuntimeError("original send failure")
    queue.remote_socket.send_multipart.side_effect = error
    diagnostic_logger.error.side_effect = RuntimeError("logging failed")
    with pytest.raises(RuntimeError) as caught:
        shm_broadcast.MessageQueue.enqueue(queue, "message")
    assert caught.value is error


def test_successful_send_and_recv_are_quiet(diagnostic_logger):
    queue, _ = _make_queue(0, 1, max_chunk_bytes=128)
    payload = {"private": "payload"}
    shm_broadcast.MessageQueue.enqueue(queue, payload)
    socket = queue.remote_socket
    socket.poll.return_value = True
    socket.recv_multipart.return_value = socket.send_multipart.call_args.args[0]
    assert shm_broadcast.MessageQueue.recv(socket, timeout=None) == payload
    diagnostic_logger.error.assert_not_called()


@pytest.mark.parametrize("timeout,timeout_ms", [(None, None), (-0.1, 0), (0.25, 250)])
def test_recv_poll_timeouts_are_quiet(diagnostic_logger, timeout, timeout_ms):
    socket = MagicMock()
    socket.poll.return_value = False
    with pytest.raises(TimeoutError):
        shm_broadcast.MessageQueue.recv(socket, timeout)
    socket.poll.assert_called_once_with(timeout=timeout_ms)
    socket.recv_multipart.assert_not_called()
    diagnostic_logger.error.assert_not_called()


@pytest.mark.parametrize("error", [TimeoutError(), Again()])
def test_retryable_send_failures_are_quiet(diagnostic_logger, error):
    queue, _ = _make_queue(0, 1, max_chunk_bytes=128)
    queue.remote_socket.send_multipart.side_effect = error
    with pytest.raises(type(error)) as caught:
        shm_broadcast.MessageQueue.enqueue(queue, "message")
    assert caught.value is error
    diagnostic_logger.error.assert_not_called()


def test_remote_frames_carry_integrity_header():
    queue, _ = _make_queue(0, 1, max_chunk_bytes=128)
    payload = {"data": np.arange(8, dtype=np.uint8)}
    shm_broadcast.MessageQueue.enqueue(queue, payload)

    frames = queue.remote_socket.send_multipart.call_args.args[0]
    header, main_frame, *_ = frames
    magic, length, checksum, sequence = patch_module._FRAME_HEADER.unpack(header)
    assert magic == patch_module._FRAME_MAGIC
    assert length == len(main_frame)
    assert checksum == zlib.crc32(main_frame) & 0xFFFFFFFF
    assert sequence == 0
    # A second message on the same queue advances the sequence.
    shm_broadcast.MessageQueue.enqueue(queue, payload)
    header = queue.remote_socket.send_multipart.call_args.args[0][0]
    assert patch_module._FRAME_HEADER.unpack(header)[3] == 1


def test_recv_rejects_corrupted_remote_frame(diagnostic_logger):
    queue, _ = _make_queue(0, 1, max_chunk_bytes=128)
    shm_broadcast.MessageQueue.enqueue(queue, {"private": "payload"})
    header, main_frame = queue.remote_socket.send_multipart.call_args.args[0][:2]
    corrupted = bytearray(main_frame)
    corrupted[0] = 0x20  # the field signature: invalid load key, ' '

    socket = queue.remote_socket
    socket.poll.return_value = True
    socket.recv_multipart.return_value = [header, bytes(corrupted)]

    with pytest.raises(patch_module.MessageQueueFrameCorruptionError) as caught:
        shm_broadcast.MessageQueue.recv(socket, timeout=0.5)

    detail = str(caught.value)
    assert "seq=0" in detail
    assert "head=" in detail
    summary = _error_summary(diagnostic_logger)
    assert "frame integrity failure" in summary
    assert "frame_count=2" in summary
    assert "private" not in summary


def test_recv_accepts_legacy_frame_without_header(diagnostic_logger):
    socket = MagicMock()
    socket.poll.return_value = True
    socket.recv_multipart.return_value = [pickle.dumps({"value": 1}), b"oob"]

    assert shm_broadcast.MessageQueue.recv(socket, timeout=None) == {"value": 1}
    diagnostic_logger.error.assert_not_called()


def test_dequeue_is_serialized_per_queue(monkeypatch):
    queue, _ = _make_queue(1, 0, max_chunk_bytes=128)
    state = {"active": 0, "max_active": 0}
    guard = threading.Lock()

    def slow_dequeue(self, timeout=None, indefinite=False):
        with guard:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.01)
        with guard:
            state["active"] -= 1
        return "ok"

    monkeypatch.setattr(patch_module, "_upstream_dequeue", slow_dequeue)
    results = []

    def reader():
        results.append(shm_broadcast.MessageQueue.dequeue(queue, timeout=0))

    threads = [threading.Thread(target=reader) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert state["max_active"] == 1
    assert results == ["ok", "ok"]
