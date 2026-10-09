# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Backport https://github.com/vllm-project/vllm/pull/53217.

Remove this patch once the supported vLLM version copies out-of-band buffers
before asynchronous ZMQ sends and provides equivalent failure diagnostics.
"""

import copyreg
import io
import pickle
from pickle import PickleBuffer
from typing import TYPE_CHECKING, Any

import torch
import zmq
from vllm.distributed.device_communicators import shm_broadcast
from vllm.logger import logger

if TYPE_CHECKING:
    from _typeshed import SizedBuffer

_OOB_BUFFER_THRESHOLD = 1024 * 1024
_MAX_LOGGED_FRAME_SIZES = 8
_MAX_LOGGED_TYPE_LENGTH = 128


def _log_message_queue_failure(stage: str, buffers, error: Exception, obj_type: type | None = None) -> None:
    """Emit one bounded summary; leave the traceback to the caller."""
    try:
        message_type = "unknown" if obj_type is None else f"{obj_type.__module__}.{obj_type.__qualname__}"
        logger.error(
            "MessageQueue failure: stage=%s message_type=%s error_type=%s "
            "frame_count=%d total_bytes=%d frame_sizes=%s omitted_frames=%d",
            stage,
            message_type[:_MAX_LOGGED_TYPE_LENGTH],
            type(error).__name__[:_MAX_LOGGED_TYPE_LENGTH],
            len(buffers),
            sum(len(buffer) for buffer in buffers),
            [len(buffer) for buffer in buffers[:_MAX_LOGGED_FRAME_SIZES]],
            max(0, len(buffers) - _MAX_LOGGED_FRAME_SIZES),
        )
    except Exception:
        # Diagnostics must never replace the original exception.
        pass


def enqueue(self: shm_broadcast.MessageQueue, obj, timeout: float | None = None):
    """Write to message queue with optional timeout (in seconds)."""
    assert self._is_writer, "Only writers can enqueue"
    all_buffers: list[SizedBuffer] = [b""]
    total_bytes = 6  # 2 bytes for oob buffer count, 4 for main buffer size

    def oob_callback(buf: PickleBuffer) -> bool:
        raw_buf = buf.raw()
        if len(raw_buf) < _OOB_BUFFER_THRESHOLD:
            # In-line buffers smaller than 1MiB.
            return True
        # raw_buf can alias live tensor/array memory that is reused after
        # enqueue returns. Own a snapshot before the asynchronous ZMQ send.
        oob_buf = bytes(raw_buf)
        all_buffers.append(oob_buf)
        nonlocal total_bytes
        total_bytes += len(oob_buf) + 4
        return False

    stage = "serialize"
    try:
        # Preserve globally registered reducers and upstream CPU tensor handling.
        dispatch_table = dict(copyreg.dispatch_table)
        dispatch_table[torch.Tensor] = shm_broadcast._reduce_tensor
        with io.BytesIO() as bio:
            pickler = pickle.Pickler(
                bio,
                protocol=pickle.HIGHEST_PROTOCOL,
                buffer_callback=oob_callback,
            )
            pickler.dispatch_table = dispatch_table
            pickler.dump(obj)
            all_buffers[0] = bio.getvalue()
        if self.n_local_reader > 0:
            stage = "shm_write"
            if total_bytes + len(all_buffers[0]) >= self.buffer.max_chunk_bytes:
                with self.acquire_write(timeout) as buf:
                    buf[0] = 1  # overflow
                stage = "local_zmq_send"
                self.local_socket.send_multipart(all_buffers, copy=False)
            else:
                # Byte 0: overflow flag; bytes 1-2: buffer count. Each buffer
                # follows its length encoded as a 4-byte big-endian integer.
                with self.acquire_write(timeout) as buf:
                    buf[0] = 0  # not overflow
                    offset = 3
                    buf[1:offset] = shm_broadcast.to_bytes_big(len(all_buffers), 2)
                    for buffer in all_buffers:
                        buf_len = len(buffer)
                        buf_offset = offset + 4
                        buf[offset:buf_offset] = shm_broadcast.to_bytes_big(buf_len, 4)
                        buf[buf_offset : (offset := buf_offset + buf_len)] = buffer

            stage = "shm_notify"
            self._spin_condition.notify()

        if self.n_remote_reader > 0:
            stage = "remote_zmq_send"
            self.remote_socket.send_multipart(all_buffers, copy=False)
    except (TimeoutError, zmq.Again):
        # Queue backpressure and retryable sends are not corruption reports.
        raise
    except Exception as error:
        _log_message_queue_failure(stage, all_buffers, error, type(obj))
        raise


def recv(socket: zmq.Socket, timeout: float | None) -> Any:
    """Keep polling quiet and add frame metadata only if decoding fails."""
    timeout_ms = None if timeout is None else max(0, int(timeout * 1000))
    if not socket.poll(timeout=timeout_ms):
        raise TimeoutError
    all_buffers = socket.recv_multipart(copy=False)
    try:
        main_buffer, *oob_buffers = all_buffers
        return pickle.loads(main_buffer, buffers=oob_buffers)
    except Exception as error:
        # The object type is unknown until decoding succeeds. Never dump the
        # pickle/payload or repeat the traceback that the caller will report.
        _log_message_queue_failure("zmq_deserialize", all_buffers, error)
        raise


shm_broadcast.MessageQueue.enqueue = enqueue
shm_broadcast.MessageQueue.recv = staticmethod(recv)
