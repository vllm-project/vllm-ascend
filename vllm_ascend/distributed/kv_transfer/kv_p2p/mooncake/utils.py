# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Shared utilities for Mooncake KV transfer connectors."""

import contextlib
import hashlib
import struct
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import numpy.typing as npt
import torch
import zmq
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket

from vllm_ascend.distributed.kv_transfer.utils.utils import (
    RegisterRegions,
    tensor_storage_key,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[Any]:
    """Create a Mooncake ROUTER or REQ socket and clean up its context."""
    if socket_type not in (zmq.ROUTER, zmq.REQ):  # type: ignore[attr-defined]
        raise ValueError(f"Unexpected socket type: {socket_type}")

    context: Any | None = None
    try:
        context = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=context,
            path=addr,
            socket_type=socket_type,
            bind=socket_type == zmq.ROUTER,  # type: ignore[attr-defined]
        )
    finally:
        if context is not None:
            context.destroy(linger=0)


def as_kv_cache_tensors(cache_or_caches: Any) -> tuple[torch.Tensor, ...]:
    """Normalize one layer's KV cache into a tuple of tensors."""
    if isinstance(cache_or_caches, torch.Tensor):
        return (cache_or_caches,)
    if isinstance(cache_or_caches, (list, tuple)) and all(isinstance(cache, torch.Tensor) for cache in cache_or_caches):
        return tuple(cache_or_caches)
    raise TypeError(
        f"A layer KV cache must be a tensor or a list/tuple of tensors, but got {type(cache_or_caches).__name__}."
    )


def _get_storage_nbytes(tensor: torch.Tensor) -> int:
    """Return the byte size of the allocation backing tensor."""
    try:
        return tensor.untyped_storage().nbytes()
    except Exception:
        try:
            return tensor.storage().nbytes()
        except Exception:
            return tensor.nbytes


def collect_configured_register_regions(
    kv_cache_config: "KVCacheConfig",
    kv_caches: dict[str, Any],
) -> RegisterRegions:
    """Collect one registration range per configured backing storage."""
    ranges_by_storage: dict[int, tuple[int, int]] = {}
    configured_tensor_count = 0

    def merge_storage_range(
        storage_key: int,
        register_start: int,
        register_end: int,
    ) -> None:
        previous_range = ranges_by_storage.get(storage_key)
        if previous_range is None:
            ranges_by_storage[storage_key] = (register_start, register_end)
        else:
            ranges_by_storage[storage_key] = (
                min(previous_range[0], register_start),
                max(previous_range[1], register_end),
            )

    for tensor_config in kv_cache_config.kv_cache_tensors:
        if not tensor_config.shared_by:
            continue

        cache_tensors: list[torch.Tensor] = []
        for layer_name in tensor_config.shared_by:
            cache_tensors.extend(as_kv_cache_tensors(kv_caches.get(layer_name)))

        caches_by_storage: dict[int, list[torch.Tensor]] = {}
        for cache in cache_tensors:
            storage_key = tensor_storage_key(cache)
            caches_by_storage.setdefault(storage_key, []).append(cache)

        if len(caches_by_storage) == 1:
            # KVCacheTensor.size is authoritative when all layer views belong
            # to its one configured allocation. Packed layouts expose each
            # view at allocation_base + offset.
            storage_key, storage_caches = next(iter(caches_by_storage.items()))
            register_start = min(cache.data_ptr() for cache in storage_caches) - tensor_config.offset
            merge_storage_range(
                storage_key,
                register_start,
                register_start + tensor_config.size,
            )
        else:
            # Some cache types initialize component tensors independently.
            # The config size cannot be applied to every storage, so register
            # the actual range of each allocation.
            for storage_key, storage_caches in caches_by_storage.items():
                register_start = min(cache.data_ptr() for cache in storage_caches)
                register_end = max(storage_key + _get_storage_nbytes(cache) for cache in storage_caches)
                if register_end <= register_start:
                    raise ValueError(f"Invalid KV cache storage range: start={register_start}, end={register_end}.")
                merge_storage_range(storage_key, register_start, register_end)

        configured_tensor_count += 1

    if not ranges_by_storage:
        raise ValueError("KV cache config contains no registerable tensors.")

    ptrs = [region[0] for region in ranges_by_storage.values()]
    lengths = [region[1] - region[0] for region in ranges_by_storage.values()]
    return RegisterRegions(
        ptrs=ptrs,
        lengths=lengths,
        logical_tensor_count=configured_tensor_count,
        logical_total_bytes=sum(lengths),
    )


_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


@dataclass
class SizedDict(OrderedDict[_KT, _VT]):
    """Insertion-ordered mapping with a bounded number of entries."""

    def __init__(self, max_size: int = 16000, *args: Any, **kwargs: Any) -> None:
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: _KT, value: _VT) -> None:
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key: _KT) -> _VT:
        try:
            return super().__getitem__(key)
        except KeyError:
            value = cast(_VT, {})
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
    "as_kv_cache_tensors",
    "collect_configured_register_regions",
    "ensure_zmq_recv",
    "ensure_zmq_send",
    "group_concurrent_contiguous",
    "split_if_not_byte_contiguous",
    "string_to_int64_hash",
    "zmq_ctx",
]
