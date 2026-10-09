# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pickle
import re
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.distributed.device_communicators import shm_broadcast

from vllm_ascend.patch.platform.patch_shm_broadcast import enqueue

_MIB = 1024 * 1024


def _make_queue(n_local_reader, n_remote_reader, max_chunk_bytes):
    shm_buffer = bytearray(max_chunk_bytes)
    queue = SimpleNamespace(
        _is_writer=True,
        _is_local_reader=True,
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
        buffers = queue.remote_socket.send_multipart.call_args.args[0]
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


def test_enqueue_propagates_write_timeout():
    queue, _ = _make_queue(1, 1, max_chunk_bytes=128)
    queue.acquire_write.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        shm_broadcast.MessageQueue.enqueue(queue, "message", timeout=0.1)
    queue.acquire_write.assert_called_once_with(0.1)
    queue.local_socket.send_multipart.assert_not_called()
    queue.remote_socket.send_multipart.assert_not_called()
    queue._spin_condition.notify.assert_not_called()


def test_patch_is_installed_on_upstream_class():
    assert shm_broadcast.MessageQueue.enqueue is enqueue
