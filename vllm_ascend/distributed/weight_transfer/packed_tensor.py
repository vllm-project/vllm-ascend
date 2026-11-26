# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from threading import Lock
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor
from vllm.distributed.weight_transfer import packed_tensor as upstream
from vllm.distributed.weight_transfer.packed_tensor import PackedBufferImporter
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


def packed_npu_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    npu_uuid: str,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
) -> Iterator[upstream.PackedIpcChunk]:
    """Pack tensors into a reusable NPU IPC buffer and yield chunks.

    Allocates a single NPU buffer of ``buffer_size_bytes`` and registers
    it for IPC once via ``reduce_tensor``.  Each chunk's packed data is
    copied into this buffer before yielding, so only one IPC-shared
    allocation exists for the lifetime of the transfer.
    A nonempty iterator containing only zero-element tensors still yields a
    chunk so its names, shapes and dtypes are delivered; an empty iterator does not.

    Args:
        iterator: Iterator of (name, tensor) pairs.
        npu_uuid: Physical NPU UUID string for this rank.
        post_iter_func: Applied to each (name, tensor) before packing. Must return
            a Tensor with the source dtype and element count, or raises ValueError.
            Flattened element positions must match the source metadata: values,
            device and strides may change, but elements must not be reordered
            (e.g. by transposing). Order is a caller contract, not checked here.
        buffer_size_bytes: Exact capacity of the reusable IPC buffer.
    """
    if buffer_size_bytes <= 0:
        raise ValueError(f"buffer_size_bytes must be positive, got {buffer_size_bytes}")
    ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device="npu")
    # Store only the rebuild args (drop the func); the consumer rebuilds with
    # the well-known ``rebuild_npu_tensor``, mirroring upstream's CUDA IPC engine.
    _, ipc_args = reduce_tensor(ipc_buffer)

    names: list[str] = []
    shapes: list[list[int]] = []
    dtypes: list[torch.dtype] = []
    tensor_sizes: list[int] = []
    total_bytes = 0

    checked = _checked_callback(post_iter_func)
    for name, orig_tensor in iterator:
        flat = checked((name, orig_tensor)).contiguous().reshape(-1).view(torch.uint8)

        if flat.numel() > buffer_size_bytes:
            raise ValueError(
                f"Tensor '{name}' has size {flat.numel()} bytes, "
                f"which exceeds buffer_size_bytes={buffer_size_bytes}. "
                f"Increase buffer_size_bytes to at least {flat.numel()}."
            )

        if total_bytes and total_bytes + flat.numel() > buffer_size_bytes:
            torch.npu.current_stream().synchronize()
            yield upstream.PackedIpcChunk(
                names=names,
                shapes=shapes,
                dtype_names=[str(d).split(".")[-1] for d in dtypes],
                tensor_sizes=tensor_sizes,
                ipc_handle={npu_uuid: ipc_args},
            )
            names, shapes, dtypes, tensor_sizes = [], [], [], []
            total_bytes = 0

        ipc_buffer[total_bytes : total_bytes + flat.numel()].copy_(flat)
        names.append(name)
        shapes.append(list(orig_tensor.shape))
        dtypes.append(orig_tensor.dtype)
        tensor_sizes.append(flat.numel())
        total_bytes += flat.numel()

    if names:
        torch.npu.current_stream().synchronize()
        yield upstream.PackedIpcChunk(
            names=names,
            shapes=shapes,
            dtype_names=[str(d).split(".")[-1] for d in dtypes],
            tensor_sizes=tensor_sizes,
            ipc_handle={npu_uuid: ipc_args},
        )


class NPUPackedBufferImporter(PackedBufferImporter):
    """Cache one rebuilt NPU mapping per packed IPC export."""

    _entry: tuple[tuple, torch.Tensor] | None

    def rebuild(self, list_args: list) -> torch.Tensor:
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        key = tuple(list_args)
        if self._entry is not None and self._entry[0] == key:
            return self._entry[1]
        packed = rebuild_npu_tensor(*list_args)
        self._entry = (key, packed)
        return packed


def packed_npu_ipc_consumer(
    ipc_handle: dict[str, tuple],
    physical_npu_id: str,
    names: list[str],
    shapes: list[list[int]],
    dtype_names: list[str],
    tensor_sizes: list[int],
    device_index: int,
    importer: PackedBufferImporter | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Unpack a single packed IPC chunk into named tensors.

    Reconstructs the packed buffer via the IPC handle, unpacks into
    individual tensors, and clones each into independent storage before
    returning.  The clone is required because the producer reuses one
    IPC buffer across chunks.
    Clones add one chunk's payload storage on top of the shared buffer and any
    loader-owned copies. Returned tensors may be retained or modified without
    aliasing that buffer.

    Args:
        ipc_handle: Mapping of NPU UUID to a ``rebuild_npu_tensor`` args tuple
            from ``reduce_tensor``.
        physical_npu_id: Physical NPU UUID string for the current process.
        names: Parameter names in the packed buffer.
        shapes: Parameter shapes.
        dtype_names: Parameter dtype name strings (e.g. "float16").
        tensor_sizes: Size in bytes of each parameter in the packed buffer.
        device_index: Local NPU device index.
        importer: ``None`` is safe only for single-chunk exports. For a
            multi-chunk export, pass the same importer for every chunk so
            the shared mapping is rebuilt only once.
    """
    if not isinstance(device_index, int) or isinstance(device_index, bool) or device_index < 0:
        raise ValueError(f"device_index must be a non-negative integer, got {device_index!r}")
    if physical_npu_id not in ipc_handle:
        raise ValueError(
            f"IPC handle not found for NPU UUID {physical_npu_id}. Available UUIDs: {list(ipc_handle.keys())}"
        )

    dtypes = []
    for dtype_name in dtype_names:
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unknown torch dtype name {dtype_name!r}")
        dtypes.append(dtype)
    if not (len(names) == len(shapes) == len(dtypes) == len(tensor_sizes)):
        raise ValueError("packed IPC metadata lengths disagree")
    for name, shape, dtype, size in zip(names, shapes, dtypes, tensor_sizes):
        if any(dimension < 0 for dimension in shape) or size < 0:
            raise ValueError(f"invalid metadata for tensor {name!r}")
        expected = math.prod(shape) * dtype.itemsize
        if size != expected:
            raise ValueError(f"tensor {name!r} metadata says {size} bytes but requires {expected} bytes")

    args = ipc_handle[physical_npu_id]
    list_args = list(args)
    # Index 6 of the args from reduce_tensor is the device_index.
    # Overwrite it with the receiver's device index.
    if len(list_args) <= 6:
        raise ValueError(f"NPU IPC rebuild arguments do not contain a device index: got {len(list_args)} values")
    list_args[6] = device_index
    if importer is None:
        importer = NPUPackedBufferImporter()
    packed = importer.rebuild(list_args)

    content_size = sum(tensor_sizes)
    if content_size > packed.numel():
        raise ValueError(f"packed IPC metadata declares {content_size} bytes, buffer has {packed.numel()}")
    packed = packed[:content_size]

    weights: list[tuple[str, torch.Tensor]] = []
    offset = 0
    for name, shape, dtype, size in zip(names, shapes, dtypes, tensor_sizes):
        raw = packed[offset : offset + size]
        tensor = raw.clone().view(dtype).reshape(shape)
        weights.append((name, tensor))
        offset += size

    return weights
