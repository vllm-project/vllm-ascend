# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed weight transfer: vLLM HCCL loops with NPU boundaries, and NPU IPC.

HCCL delegates packing and receive loops to vLLM. Local hooks provide owning
unpack, callback completion ordering and native HCCL stream boundaries.
"""

import math
from collections.abc import Callable, Iterator
from contextlib import suppress
from contextvars import ContextVar
from functools import cache
from threading import Lock
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor
from vllm.distributed.weight_transfer import packed_tensor as upstream
from vllm.logger import logger

# Default values for packed tensor configuration.
# These are imported by HCCLWeightTransferUpdateInfo and trainer_send_weights.
DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_PACKED_NUM_BUFFERS = 2

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


# Per-context state prevents callbacks in other threads from sharing a transfer.
# vLLM caches streams per device; a per-device lock prevents simultaneous use.
_ACTIVE: ContextVar[Any] = ContextVar("ascend_packed_transfer", default=None)
_INSTALL_LOCK = Lock()


@cache
def _device_lock(device: int) -> Any:
    return Lock()


def _install_packed_hooks() -> None:
    with _INSTALL_LOCK:
        # Standalone trainers need not initialize platform patches.
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.current_stream = torch.npu.current_stream
        if getattr(upstream.unpack_tensor, "_ascend_packed_hook", False):
            return
        unpack = upstream.unpack_tensor

        def owning_unpack(packed, names, shapes, dtypes, sizes):
            state = _ACTIVE.get()
            if state is None:
                return unpack(packed, names, shapes, dtypes, sizes)
            return state.unpack(packed, names, shapes, dtypes, sizes)

        owning_unpack._ascend_packed_hook = True
        upstream.unpack_tensor = owning_unpack


class _HCCLPackedTransfer:
    def __init__(self, group, callback, receiving):
        self.group = group
        self.callback = callback
        self.receiving = receiving
        self.caller = torch.npu.current_stream()
        self.streams = []
        self.buffers = []
        self.readers = ()
        self.events = []
        self.previous_ready = None
        self.previous_broadcast = None

    def initialize(self, streams):
        self.streams = list(streams)
        self.buffers = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in streams]
        self.readers = (*streams, self.caller)
        self.events = [torch.npu.Event() for _ in streams] if self.receiving else []
        for stream in streams:
            stream.wait_stream(self.caller)

    def unpack(self, packed, names, shapes, dtypes, sizes):
        stream = torch.npu.current_stream()
        if self.previous_ready is not None:
            stream.wait_event(self.previous_ready)
        weights = [
            (name, raw.clone().view(dtype).reshape(shape))
            for name, shape, dtype, raw in zip(names, shapes, dtypes, packed.split(sizes))
        ]
        for _, tensor in weights:
            for reader in self.readers:
                tensor.record_stream(reader)
        return weights

    def slot(self, stream):
        return self.streams.index(stream)

    def broadcast(self, tensor, src, **kwargs):
        stream = torch.npu.current_stream()
        slot = self.slot(stream)
        self.buffers[slot] = tensor
        if not self.receiving and self.previous_broadcast is not None and self.previous_broadcast != stream:
            self.previous_broadcast.synchronize()
        # torch_npu packing/allocation must reach the native stream before HCCL.
        stream.synchronize()
        self.group.broadcast(tensor, src=src, stream=stream)
        self.previous_broadcast = stream
        if self.receiving:
            stream.synchronize()
            # Raw storage is read only on this slot; only owning clones
            # escape to callbacks and require cross-stream lifetime tracking.

    def load(self, weights):
        self.callback(weights)
        stream = torch.npu.current_stream()
        self.previous_ready = self.events[self.slot(stream)]
        self.previous_ready.record(stream)


def _packed_broadcast(iterator, group, src, callback, buffer_size_bytes, num_buffers, *, receiving):
    """Delegate the actual loop to the installed (main or release) vLLM."""
    if buffer_size_bytes <= 0 or num_buffers <= 0:
        raise ValueError("buffer_size_bytes and num_buffers must be positive")
    if _ACTIVE.get() is not None:
        raise RuntimeError("nested packed transfers cannot share vLLM's cached streams")
    _install_packed_hooks()
    device = torch.npu.current_device()
    # functools.cache can evaluate a missing key twice on concurrent first use.
    # Serialize lookup too, so both transfers obtain the same device lock.
    with _INSTALL_LOCK:
        lock = _device_lock(device)
    with lock:
        state = _HCCLPackedTransfer(group, callback, receiving)
        token = _ACTIVE.set(state)
        error = None
        try:
            # Both supported vLLM versions cache this tuple; the loop retrieves
            # the same streams. Initialize explicitly instead of hooking it.
            state.initialize(upstream._get_streams(device, num_buffers))
            if receiving:

                def checked_metadata():
                    for name, (shape, dtype) in iterator:
                        if any(dimension < 0 for dimension in shape):
                            raise ValueError(f"tensor {name!r} has a negative shape: {shape}")
                        yield name, (shape, dtype)

                upstream.packed_nccl_broadcast_consumer(
                    checked_metadata(),
                    state,
                    src,
                    state.load,
                    buffer_size_bytes,
                    num_buffers,
                    device=f"npu:{device}",
                )
            else:

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

                upstream.packed_nccl_broadcast_producer(
                    iterator,
                    state,
                    src,
                    checked,
                    buffer_size_bytes,
                    num_buffers,
                )
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                _drain_packed_streams(state.streams, state.buffers, error)
            finally:
                _ACTIVE.reset(token)


def packed_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Run vLLM's producer with NPU stream and native HCCL adapters.

    Join additional source writers onto the caller stream before entry.
    Lazy sources must produce tensors ready on the active packing stream.
    """
    _packed_broadcast(iterator, group, src, post_iter_func, buffer_size_bytes, num_buffers, receiving=False)


def packed_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Run vLLM's consumer with owning clones and callback completion ordering.

    Additional callback streams must join work needed by later callbacks onto
    the active slot before returning. Returned weights own their storage.
    """
    _packed_broadcast(iterator, group, src, post_unpack_func, buffer_size_bytes, num_buffers, receiving=True)


# ── NPU IPC packed transfer ────────────────────────────────────────────


def packed_npu_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    npu_uuid: str,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
) -> Iterator[dict[str, Any]]:
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

    for name, orig_tensor in iterator:
        tensor = post_iter_func((name, orig_tensor))
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"post_iter_func for tensor {name!r} must return a torch.Tensor")
        tensor = tensor.contiguous()
        if tensor.dtype != orig_tensor.dtype or tensor.numel() != orig_tensor.numel():
            raise ValueError(
                f"post_iter_func returned tensor {name!r} with dtype={tensor.dtype}, "
                f"numel={tensor.numel()}; expected dtype={orig_tensor.dtype}, numel={orig_tensor.numel()}"
            )
        flat = tensor.reshape(-1).view(torch.uint8)

        if flat.numel() > buffer_size_bytes:
            raise ValueError(
                f"Tensor '{name}' has size {flat.numel()} bytes, "
                f"which exceeds buffer_size_bytes={buffer_size_bytes}. "
                f"Increase buffer_size_bytes to at least {flat.numel()}."
            )

        if total_bytes and total_bytes + flat.numel() > buffer_size_bytes:
            torch.npu.current_stream().synchronize()
            yield {
                "names": names,
                "shapes": shapes,
                "dtype_names": [str(d).split(".")[-1] for d in dtypes],
                "tensor_sizes": tensor_sizes,
                "ipc_handle": {npu_uuid: ipc_args},
            }
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
        yield {
            "names": names,
            "shapes": shapes,
            "dtype_names": [str(d).split(".")[-1] for d in dtypes],
            "tensor_sizes": tensor_sizes,
            "ipc_handle": {npu_uuid: ipc_args},
        }


def packed_npu_ipc_consumer(
    ipc_handle: dict[str, tuple],
    physical_npu_id: str,
    names: list[str],
    shapes: list[list[int]],
    dtype_names: list[str],
    tensor_sizes: list[int],
    device_index: int,
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
    """
    # Lazy import: ``rebuild_npu_tensor`` lives in ``torch_npu`` and must not be
    # imported at module load time on non-NPU hosts.
    from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

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
    packed = rebuild_npu_tensor(*list_args)

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
