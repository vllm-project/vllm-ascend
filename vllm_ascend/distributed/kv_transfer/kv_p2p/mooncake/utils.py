# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared utilities for Mooncake KV transfer connectors."""

import hashlib
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import zmq
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SizedDict(OrderedDict):
    """Insertion-ordered mapping with a bounded number of entries."""

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
]
