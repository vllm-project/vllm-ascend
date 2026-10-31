# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterator
from contextlib import suppress
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor
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


def packed_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Broadcast tensors in a packed manner from trainer to workers.

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
    target_packed_tensor_size = buffer_size_bytes

    streams = [torch.npu.Stream() for _ in range(num_buffers)]
    buffer_idx = 0
    previous_broadcast_stream = None

    packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
    packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
    packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]

    body_error: BaseException | None = None
    try:
        done = False
        while not done:
            # Synchronize the current stream (waits for previous
            # iteration's work on this buffer to finish)
            streams[buffer_idx].synchronize()
            # Start tasks for the new buffer in a new stream
            with torch.npu.stream(streams[buffer_idx]):
                # Initialize the packing tensor list and sizes
                packing_tensor_list[buffer_idx] = []
                packing_tensor_sizes[buffer_idx] = 0
                # Pack the tensors
                while True:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        done = True
                        break
                    # Apply post processing and convert to linearized uint8 tensor
                    tensor = post_iter_func(item)
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError(f"post_iter_func for tensor {item[0]!r} must return a torch.Tensor")
                    tensor = tensor.contiguous()
                    original = item[1]
                    if tensor.dtype != original.dtype or tensor.numel() != original.numel():
                        raise ValueError(
                            f"post_iter_func returned tensor {item[0]!r} with dtype={tensor.dtype}, "
                            f"numel={tensor.numel()}; expected dtype={original.dtype}, numel={original.numel()}"
                        )
                    tensor = tensor.reshape(-1).view(torch.uint8)
                    packing_tensor_list[buffer_idx].append(tensor)
                    packing_tensor_sizes[buffer_idx] += tensor.numel()
                    if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                        break
                if len(packing_tensor_list[buffer_idx]) > 0:
                    # Pack the tensors
                    packed_tensors[buffer_idx] = torch.cat(packing_tensor_list[buffer_idx], dim=0)
                    # Packing can overlap the previous transfer, but collectives
                    # on this communicator must not overlap across streams.
                    if previous_broadcast_stream is not None and previous_broadcast_stream is not streams[buffer_idx]:
                        previous_broadcast_stream.synchronize()
                    # Match the active packing stream and the explicit native
                    # HCCL stream; wait for both before reusing this slot.
                    group.broadcast(packed_tensors[buffer_idx], src=src, stream=streams[buffer_idx])
                    previous_broadcast_stream = streams[buffer_idx]

            if len(packing_tensor_list[buffer_idx]) == 0:
                # No more tensors — nothing left to broadcast
                break

            # Move to the next buffer
            buffer_idx = (buffer_idx + 1) % num_buffers
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        _drain_packed_streams(streams, packed_tensors, body_error)


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

    def unpack_tensor(
        packed_tensor: torch.Tensor,
        names: list[str],
        shapes: list[list[int]],
        dtypes: list[torch.dtype],
        tensor_sizes: list[int],
    ) -> list[tuple[str, torch.Tensor]]:
        """Unpack named tensors with their declared dtypes and shapes."""
        unpacked_tensors = packed_tensor.split(tensor_sizes)
        unpacked_list = [
            # Preserve independent storage for callbacks that retain or mutate
            # inputs. Cloning bytes first also fixes unaligned dtype views.
            (name, tensor.clone().view(dtype).reshape(shape))
            for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
        ]
        return unpacked_list

    if buffer_size_bytes <= 0 or num_buffers <= 0:
        raise ValueError(
            f"buffer_size_bytes and num_buffers must be positive, got "
            f"buffer_size_bytes={buffer_size_bytes}, num_buffers={num_buffers}"
        )
    target_packed_tensor_size = buffer_size_bytes

    streams = [torch.npu.Stream() for _ in range(num_buffers)]
    buffer_idx = 0

    packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [[] for _ in range(num_buffers)]
    packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
    packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]

    body_error: BaseException | None = None
    try:
        done = False
        while not done:
            # Synchronize the current stream (waits for previous
            # iteration's load_weights on this buffer to finish)
            streams[buffer_idx].synchronize()
            with torch.npu.stream(streams[buffer_idx]):
                # Collect parameter metadata for this buffer
                packing_tensor_meta_data[buffer_idx] = []
                packing_tensor_sizes[buffer_idx] = 0
                while True:
                    try:
                        name, (shape, dtype) = next(iterator)
                    except StopIteration:
                        done = True
                        break
                    if any(dimension < 0 for dimension in shape):
                        raise ValueError(f"tensor {name!r} has a negative shape: {shape}")
                    tensor_size = math.prod(shape) * dtype.itemsize
                    packing_tensor_meta_data[buffer_idx].append((name, shape, dtype, tensor_size))
                    packing_tensor_sizes[buffer_idx] += tensor_size
                    if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                        break
                if len(packing_tensor_meta_data[buffer_idx]) > 0:
                    packed_tensors[buffer_idx] = torch.empty(
                        packing_tensor_sizes[buffer_idx],
                        dtype=torch.uint8,
                        device="npu",
                    )

            if len(packing_tensor_meta_data[buffer_idx]) == 0:
                break

            # Broadcast, unpack clones and load_weights share one ordered stream.
            group.broadcast(packed_tensors[buffer_idx], src=src, stream=streams[buffer_idx])
            # Match #18078: external HCCL work must finish before torch unpacks.
            streams[buffer_idx].synchronize()

            # Unpack and load weights on the custom stream
            with torch.npu.stream(streams[buffer_idx]):
                names, shapes, dtypes, tensor_sizes = zip(*packing_tensor_meta_data[buffer_idx])
                post_unpack_func(
                    unpack_tensor(
                        packed_tensors[buffer_idx],
                        list(names),
                        list(shapes),
                        list(dtypes),
                        list(tensor_sizes),
                    )
                )

            # Move to the next buffer
            buffer_idx = (buffer_idx + 1) % num_buffers

    except BaseException as exc:
        body_error = exc
        raise
    finally:
        _drain_packed_streams(streams, packed_tensors, body_error)


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
