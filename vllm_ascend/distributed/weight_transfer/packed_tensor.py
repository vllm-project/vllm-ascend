# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from threading import Lock
from typing import Any

import torch
from vllm.distributed.weight_transfer import packed_tensor as upstream
from vllm.logger import logger

DEFAULT_PACKED_BUFFER_SIZE_BYTES = upstream.DEFAULT_PACKED_BUFFER_SIZE_BYTES
DEFAULT_PACKED_NUM_BUFFERS = upstream.DEFAULT_PACKED_NUM_BUFFERS

# Upstream caches streams across communicators on each device.
# Reject overlaps through cleanup; callers must order transfers consistently across ranks.
_HCCL_TRANSFER_LOCK = Lock()
_ACTIVE_HCCL_DEVICES: set[int] = set()


@contextmanager
def _serialized_hccl_transfer(device_idx: int):
    with _HCCL_TRANSFER_LOCK:
        if device_idx in _ACTIVE_HCCL_DEVICES:
            raise RuntimeError(
                f"Packed HCCL transfers on npu:{device_idx} must be serialized across communicators; "
                "wait for the active transfer to finish on every rank before starting another"
            )
        _ACTIVE_HCCL_DEVICES.add(device_idx)
    try:
        yield
    finally:
        with _HCCL_TRANSFER_LOCK:
            _ACTIVE_HCCL_DEVICES.remove(device_idx)


def _get_hccl_streams(device_idx: int, num_buffers: int) -> list[Any]:
    """Isolate the private vLLM stream-cache dependency."""
    try:
        streams = list(upstream._get_streams(device_idx, num_buffers))
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("Incompatible vLLM packed stream API; check the supported vLLM pairing") from exc
    if len(streams) != num_buffers:
        raise RuntimeError("Incompatible vLLM packed stream count; check the supported vLLM pairing")
    return streams


# On a failed stream drain, keep storage alive until process exit. Its async
# users may still be running, so returning it to the allocator is unsafe.
# torch_npu's getStreamFromPool uses process-global native streams (see
# NPUStream.cpp / LeakyStreamInternals). Dropping a Python Stream wrapper is
# not a completion barrier. This failed-drain path has no completion proof.
# This failure-only list is intentionally unbounded: retries after a drain
# failure can exhaust device memory. Do not clear it to recover memory while
# work may still be pending. Transfers must not concurrently share a communicator
# or its buffers; this list only holds references, it does not serialize them.
_UNSYNCHRONIZED_BUFFERS: list[torch.Tensor] = []


def _drain_packed_streams(
    streams: list[Any], packed_tensors: list[torch.Tensor], body_error: BaseException | None
) -> None:
    """Drain every slot, retaining failed storage and prioritizing termination."""
    cleanup_error: BaseException | None = None
    retained_slots: list[tuple[int, int]] = []
    for index, stream in enumerate(streams):
        try:
            stream.synchronize()
        except BaseException as exc:
            # Empty placeholders have no payload storage to protect.
            if packed_tensors[index].numel():
                _UNSYNCHRONIZED_BUFFERS.append(packed_tensors[index])
                retained_slots.append((index, packed_tensors[index].numel() * packed_tensors[index].element_size()))
            if cleanup_error is None or (isinstance(cleanup_error, Exception) and not isinstance(exc, Exception)):
                cleanup_error = exc
    if cleanup_error is not None:
        with suppress(Exception):
            logger.error(
                "Failed to drain a packed HCCL stream; "
                "retaining buffers until process exit (slot, bytes): %s; cleanup error: %s",
                retained_slots,
                cleanup_error,
            )
        # A pending body exception propagates unless cleanup raised a higher-
        # priority termination signal. Never skip draining other slots for it.
        if body_error is None or (isinstance(body_error, Exception) and not isinstance(cleanup_error, Exception)):
            raise cleanup_error


class _HCCLPackedGroup:
    def __init__(self, group, streams, *, is_producer=False):
        self.group = group
        self.streams = streams
        self.buffers = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in streams]
        self.is_producer = is_producer
        self.previous_broadcast_stream = None

    def broadcast(self, tensor, src):
        stream = torch.npu.current_stream()
        try:
            index = self.streams.index(stream)
        except ValueError as exc:
            raise RuntimeError("vLLM packed stream cache changed; check the supported vLLM pairing") from exc
        self.buffers[index] = tensor
        # Finish torch packing before native HCCL; serialize collectives across slots.
        stream.synchronize()
        if self.previous_broadcast_stream is not None and self.previous_broadcast_stream is not stream:
            self.previous_broadcast_stream.synchronize()
        self.group.broadcast(tensor, src=src, stream=stream)
        if self.is_producer:
            self.previous_broadcast_stream = stream
        else:
            # Native receive work must finish before torch unpacks the buffer.
            stream.synchronize()


@contextmanager
def _hccl_group(group, num_buffers, *, is_producer=False):
    # Standalone trainers also need NPU API adaptation.
    from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

    device_idx = torch.accelerator.current_device_index()
    with _serialized_hccl_transfer(device_idx), torch_cuda_wrapper():
        # Use the same accelerator index/cache key as the upstream loops.
        streams = _get_hccl_streams(device_idx, num_buffers)
        transport = _HCCLPackedGroup(group, streams, is_producer=is_producer)
        try:
            yield transport
        except BaseException as exc:
            _drain_packed_streams(streams, transport.buffers, exc)
            raise


def _checked_callback(callback):
    def checked(item):
        tensor = callback(item)
        original = item[1]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"post_iter_func for {item[0]!r} must return a torch.Tensor")
        if tensor.dtype != original.dtype or tensor.numel() != original.numel():
            raise ValueError(
                f"post_iter_func returned tensor {item[0]!r} with dtype={tensor.dtype}, "
                f"numel={tensor.numel()}; expected dtype={original.dtype}, numel={original.numel()}"
            )
        return tensor

    return checked


def packed_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Broadcast tensors in a packed manner from trainer to workers.

    Transfers must be serialized per device on every rank.

    Args:
        iterator: Iterator of model parameters. Returns a tuple of (name, tensor)
        group: Process group (PyHcclCommunicator)
        src: Source rank (0 in current implementation)
        post_iter_func: Function to apply to each (name, tensor) pair before
                       packing. Must return a Tensor with the source dtype and
                       element count, or raises ValueError. Flattened element
                       positions must still match the source metadata: values,
                       device and strides may change, but elements must not be
                       reordered (e.g. by transposing). Order is a caller contract,
                       not checked here.
        buffer_size_bytes: Size in bytes for each packed tensor buffer.
                          Both producer and consumer must use the same value.
        num_buffers: Number of buffers for double/triple buffering.
                    Both producer and consumer must use the same value.
    """
    if buffer_size_bytes <= 0 or num_buffers <= 0:
        raise ValueError(
            f"buffer_size_bytes and num_buffers must be positive, got "
            f"buffer_size_bytes={buffer_size_bytes}, num_buffers={num_buffers}"
        )
    with _hccl_group(group, num_buffers, is_producer=True) as transport:
        upstream.packed_nccl_broadcast_producer(
            iterator, transport, src, _checked_callback(post_iter_func), buffer_size_bytes, num_buffers
        )


def packed_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Consume packed tensors and unpack them into a list of tensors.

    Unpacking clones one actual chunk's payload in addition to the receive
    buffer and loader copies. A HCCL chunk may exceed buffer_size_bytes; peak
    memory also depends on in-flight slots and tensors retained by the loader.

    Transfers must be serialized per device on every rank.
    Additional reader streams must arrange their own readiness and storage tracking.

    Args:
        iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
        group: Process group (PyHcclCommunicator)
        src: Source rank (0 in current implementation)
        post_unpack_func: Function to apply to each list of (name, tensor) after
                         unpacking. Tensors own independent storage and may be
                         retained or modified without aliasing the packed buffer.
        buffer_size_bytes: Size in bytes for each packed tensor buffer.
                          Both producer and consumer must use the same value.
        num_buffers: Number of buffers for double/triple buffering.
                    Both producer and consumer must use the same value.
    """
    if buffer_size_bytes <= 0 or num_buffers <= 0:
        raise ValueError(
            f"buffer_size_bytes and num_buffers must be positive, got "
            f"buffer_size_bytes={buffer_size_bytes}, num_buffers={num_buffers}"
        )

    # Byte metadata avoids upstream dtype views at unaligned offsets.
    # Clone each byte slice before restoring its dtype and shape.
    expected: list[tuple[list[int], torch.dtype]] = []
    previous_load: Any = None

    def owning(tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Restore aligned, independent storage and order cross-slot loader reads."""
        nonlocal previous_load
        stream = torch.npu.current_stream()
        # Upstream's wait_event covers receives, but our async clones run later.
        # Chain callback completion to order cross-slot reads of those clones.
        if previous_load is not None:
            stream.wait_event(previous_load)
        restored = []
        for (name, raw), (shape, dtype) in zip(tensors, expected):
            tensor = raw.clone().view(dtype).reshape(shape)
            # Retained clones may be read on later slot or caller streams.
            for reader in reader_streams:
                tensor.record_stream(reader)
            restored.append((name, tensor))
        del expected[: len(tensors)]
        try:
            post_unpack_func(restored)
        except StopIteration as exc:
            # Do not let upstream treat loader failure as metadata exhaustion.
            raise RuntimeError("post_unpack_func raised StopIteration") from exc
        previous_load = torch.npu.Event()
        previous_load.record(stream)

    def metadata():
        for name, (shape, dtype) in iterator:
            if any(dimension < 0 for dimension in shape):
                raise ValueError(f"tensor {name!r} has a negative shape: {shape}")
            expected.append((shape, dtype))
            yield name, ([math.prod(shape) * dtype.itemsize], torch.uint8)

    with _hccl_group(group, num_buffers) as transport:
        reader_streams = [*transport.streams, torch.npu.current_stream()]
        upstream.packed_nccl_broadcast_consumer(
            metadata(),
            transport,
            src,
            owning,
            buffer_size_bytes,
            num_buffers,
            device=f"npu:{torch.npu.current_device()}",
        )
