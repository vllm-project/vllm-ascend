# SPDX-License-Identifier: Apache-2.0
"""Transport helpers shared by the mooncake v1 KV connectors.

The helpers in this module are byte-for-byte copies of the definitions that used
to live in ``mooncake_connector``. They are collected here so that both V1
connectors (pull and layerwise) share one implementation and the unit tests have
a single patch target.

They are intentionally **not** imported from the V2 ``mooncake/utils.py``:

* V2 ``mooncake/__init__.py`` eagerly imports the whole V2 package
  (base_scheduler/base_worker/connector/metadata/pull_scheduler/pull_worker/
  stats), so reusing its utils would drag V2 into the V1 import graph;
* the two copies have already diverged: ``zmq_ctx`` here accepts ``DEALER``
  while the V2 version only accepts ``ROUTER``/``REQ``, and V2 ``SizedDict`` is
  a generic version;
* keeping the copy local gives the V1 unit tests a stable ``mooncake_v1.common``
  patch target instead of coupling them to the V2 module.

This module must not import any ``vllm_ascend.*`` module.
"""

import contextlib
import hashlib
import struct
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket


@dataclass
class SizedDict(OrderedDict):
    def __init__(self, max_size=16000, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            value: dict[int, list[int]] = {}
            self[key] = value
            return value


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:  # type: ignore
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def group_concurrent_contiguous(
    src: list[int],
    dst: list[int],
    src_block_stride: int = 1,
    dst_block_stride: int = 1,
    block_len: int = 1,
) -> tuple[list[list[int]], list[list[int]]]:
    """Group block ids that are contiguous in both id space and memory."""
    src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
    dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

    if src_indices.size == 0:
        return [], []

    src_byte_contiguous = np.diff(src_indices) * src_block_stride == block_len
    dst_byte_contiguous = np.diff(dst_indices) * dst_block_stride == block_len
    brk = np.where(~(src_byte_contiguous & dst_byte_contiguous))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def split_if_not_byte_contiguous(
    src_groups: list[list[int]],
    dst_groups: list[list[int]],
    src_block_stride: int,
    dst_block_stride: int,
    block_len: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if src_block_stride == block_len and dst_block_stride == block_len:
        return src_groups, dst_groups

    src = [bid for group in src_groups for bid in group]
    dst = [bid for group in dst_groups for bid in group]
    return group_concurrent_contiguous(
        src,
        dst,
        src_block_stride=src_block_stride,
        dst_block_stride=dst_block_stride,
        block_len=block_len,
    )


def string_to_int64_hash(input_str):
    """
    Hash the string using SHA-256 and convert it into an int64 integer.
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def ensure_zmq_send(
    socket: zmq.Socket,  # type: ignore
    data: bytes,
    path: str,
    max_retries: int = 3,
):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Send failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Send failed after all retries. error=%s. ", e)
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def ensure_zmq_recv(
    socket: zmq.Socket,  # type: ignore
    path: str,
    max_retries: int = 3,
) -> bytes:
    retries_left = max_retries
    while True:
        try:
            return socket.recv()
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Receive failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Receive failed after all retries. source=%s, error=%s. ", path, e)
                raise RuntimeError(f"Failed to receive data after {max_retries} retries: {e}")


__all__ = [
    "SizedDict",
    "ensure_zmq_recv",
    "ensure_zmq_send",
    "group_concurrent_contiguous",
    "split_if_not_byte_contiguous",
    "string_to_int64_hash",
    "zmq_ctx",
]
