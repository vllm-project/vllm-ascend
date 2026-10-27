# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterator
from contextlib import suppress
from functools import cache
from types import FunctionType
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor
from vllm.distributed.weight_transfer import packed_tensor as upstream_packed
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


@cache
def _get_npu_streams(device_index: int, num_buffers: int) -> tuple[Any, ...]:
    """Cache slots per current NPU, as upstream does for allocator reuse.

    The engines bind the assigned device before calling these helpers. Calls
    sharing a communicator must not run concurrently, as in the upstream lane.
    """
    return tuple(torch.npu.Stream() for _ in range(num_buffers))


class _PackedNPUTorch:
    """Resolve only upstream's CUDA stream/Event operations to NPU APIs."""

    def __getattr__(self, name: str) -> Any:
        if name == "cuda":
            return torch.npu
        if name == "Event":
            return torch.npu.Event
        return getattr(torch, name)


class _PackedHCCLGroup:
    """Bind each upstream round-robin broadcast to its active NPU slot.

    Keep one raw buffer per slot until its next broadcast or the final drain,
    including when a collective or loader raises before upstream's final wait.
    """

    def __init__(self, group: Any, streams: tuple[Any, ...], *, producer: bool) -> None:
        self.group = group
        self.streams = streams
        self.buffers = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in streams]
        self.slot = 0
        self.producer = producer
        self.previous_broadcast_stream = None

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self.buffers[self.slot] = tensor
        stream = self.streams[self.slot]
        if self.producer:
            # Native HCCL calls bypass torch_npu's submission queue. Preserve
            # collective order across slots and finish packing before exposing
            # the raw pointer to HCCL; this is not a loader/clone host barrier.
            if self.previous_broadcast_stream is not None and self.previous_broadcast_stream is not stream:
                self.previous_broadcast_stream.synchronize()
            stream.synchronize()
        self.group.broadcast(tensor, src=src, stream=stream)
        if not self.producer:
            # Preserve main's #18078 native receive-to-torch boundary. Loader
            # ordering is established separately with device-side events.
            stream.synchronize()
        self.previous_broadcast_stream = stream
        self.slot = (self.slot + 1) % len(self.streams)


def _unpack_hccl_tensor(packed_tensor, names, shapes, dtypes, tensor_sizes):
    """Preserve Ascend's owning-storage and unaligned-byte-offset contract."""
    return [
        (name, tensor.clone().view(dtype).reshape(shape))
        for name, shape, dtype, tensor in zip(names, shapes, dtypes, packed_tensor.split(tensor_sizes))
    ]


class _PackedReload:
    """Extend upstream's receive ordering to owning clones and loader work.

    The upstream ready event precedes unpacking. Record completion after the
    loader callback instead, so the next slot can safely consume retained
    clones and state written by earlier callbacks without a post-load host wait.
    """

    def __init__(self, load: Callable, streams: tuple[Any, ...], caller_stream: Any) -> None:
        self.load_weights = load
        self.streams = streams
        self.caller_stream = caller_stream
        self.slot = 0
        self.previous_ready: Any = None

    def _record_clones(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        # A later callback can queue a read and release a retained clone before
        # returning. Its owning stream may then reuse that block immediately.
        # Protect every supported reader before handing any clone to the loader.
        for _, tensor in weights:
            for stream in (*self.streams, self.caller_stream):
                tensor.record_stream(stream)

    def unpack(self, packed_tensor, names, shapes, dtypes, tensor_sizes):
        stream = self.streams[self.slot]
        if self.previous_ready is not None:
            stream.wait_event(self.previous_ready)
        weights = _unpack_hccl_tensor(packed_tensor, names, shapes, dtypes, tensor_sizes)
        self._record_clones(weights)
        return weights

    def load(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        self.load_weights(weights)
        ready = torch.npu.Event()
        ready.record(self.streams[self.slot])
        self.previous_ready = ready
        self.slot = (self.slot + 1) % len(self.streams)


def _run_upstream_packed(function: Any, group: Any, num_buffers: int, **kwargs: Any) -> None:
    """Reuse upstream bytecode with isolated backend bindings, never global patches.

    Upstream 544f182d has no stream/backend injection API. Copying the globals
    dictionary keeps NCCL and other concurrent users untouched while reusing
    the producer/consumer loops, ready-event chain and raw-storage tracking.
    Remove this binding shim when upstream exposes device-neutral hooks.
    """
    streams = _get_npu_streams(torch.accelerator.current_device_index(), num_buffers)
    caller_stream = torch.npu.current_stream()
    # Join work already submitted by this caller without synchronizing the host.
    # Additional producer/callback streams must be joined by their owner.
    for stream in streams:
        stream.wait_stream(caller_stream)
    adapter = _PackedHCCLGroup(group, streams, producer="post_iter_func" in kwargs)
    globals_ = dict(function.__globals__)
    globals_.update(torch=_PackedNPUTorch(), _get_streams=_get_npu_streams, unpack_tensor=_unpack_hccl_tensor)
    if "post_unpack_func" in kwargs:
        reload = _PackedReload(kwargs["post_unpack_func"], streams, caller_stream)
        globals_["unpack_tensor"] = reload.unpack
        kwargs["post_unpack_func"] = reload.load
    bound = FunctionType(function.__code__, globals_, function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    try:
        bound(group=adapter, num_buffers=num_buffers, **kwargs)
    except BaseException as exc:
        # Upstream drains on success only. Preserve the existing Ascend failure
        # cleanup without adding post-load host barriers.
        _drain_packed_streams(list(streams), adapter.buffers, exc)
        raise


def packed_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Use upstream packing/broadcast orchestration with NPU stream bindings.

    Already-submitted caller-stream writes are joined onto every packing slot.
    Join writes from any additional streams onto the caller before entry. Lazy
    iterators must make newly produced tensors ready on their active slot.
    """
    if buffer_size_bytes <= 0 or num_buffers <= 0:
        raise ValueError("buffer_size_bytes and num_buffers must be positive")

    def checked_post(item):
        tensor = post_iter_func(item)
        original = item[1]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"post_iter_func for tensor {item[0]!r} must return a torch.Tensor")
        if tensor.dtype != original.dtype or tensor.numel() != original.numel():
            raise ValueError(
                f"post_iter_func returned tensor {item[0]!r} with dtype={tensor.dtype}, numel={tensor.numel()}; "
                f"expected dtype={original.dtype}, numel={original.numel()}"
            )
        return tensor

    _run_upstream_packed(
        upstream_packed.packed_nccl_broadcast_producer,
        group,
        num_buffers,
        iterator=iterator,
        src=src,
        post_iter_func=checked_post,
        buffer_size_bytes=buffer_size_bytes,
    )


def packed_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Reuse upstream receives while ordering and protecting owning NPU clones.

    Each callback is ordered after earlier callbacks' slot-stream work. Owning
    clones are protected on all slots and the caller stream before loading.
    Callbacks using additional streams must establish their own dependencies;
    join any work needed by later callbacks back onto the active slot before
    returning. No per-chunk post-load host barrier is added.
    """
    if buffer_size_bytes <= 0 or num_buffers <= 0:
        raise ValueError("buffer_size_bytes and num_buffers must be positive")

    def checked_metadata():
        for name, (shape, dtype) in iterator:
            if any(dimension < 0 for dimension in shape):
                raise ValueError(f"tensor {name!r} has a negative shape: {shape}")
            yield name, (shape, dtype)

    _run_upstream_packed(
        upstream_packed.packed_nccl_broadcast_consumer,
        group,
        num_buffers,
        iterator=checked_metadata(),
        src=src,
        post_unpack_func=post_unpack_func,
        buffer_size_bytes=buffer_size_bytes,
        device="npu",
    )


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
