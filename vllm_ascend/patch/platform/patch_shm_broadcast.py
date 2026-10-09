# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Backport https://github.com/vllm-project/vllm/pull/53217.

Remove this patch once the supported vLLM version copies out-of-band buffers
in MessageQueue.enqueue before handing them to asynchronous ZMQ sends.
"""

import copyreg
import io
import pickle
from pickle import PickleBuffer
from typing import TYPE_CHECKING

import torch
from vllm.distributed.device_communicators import shm_broadcast

if TYPE_CHECKING:
    from _typeshed import SizedBuffer

_OOB_BUFFER_THRESHOLD = 1024 * 1024


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
        if total_bytes + len(all_buffers[0]) >= self.buffer.max_chunk_bytes:
            with self.acquire_write(timeout) as buf:
                buf[0] = 1  # overflow
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

        self._spin_condition.notify()

    if self.n_remote_reader > 0:
        self.remote_socket.send_multipart(all_buffers, copy=False)


shm_broadcast.MessageQueue.enqueue = enqueue
