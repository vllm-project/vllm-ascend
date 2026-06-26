# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for efficient weight transfer."""

import math
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor

# Default values for packed tensor configuration.
# These are imported by HCCLWeightTransferUpdateInfo and trainer_send_weights.
DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
DEFAULT_PACKED_NUM_BUFFERS = 2


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
                       packing, should return a tensor
        buffer_size_bytes: Size in bytes for each packed tensor buffer.
                          Both producer and consumer must use the same value.
        num_buffers: Number of buffers for double/triple buffering.
                    Both producer and consumer must use the same value.
    """
    target_packed_tensor_size = buffer_size_bytes

    streams = [torch.npu.Stream() for _ in range(num_buffers)]
    buffer_idx = 0

    packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
    packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
    packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]

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
                tensor = post_iter_func(item).contiguous().view(torch.uint8).view(-1)
                packing_tensor_list[buffer_idx].append(tensor)
                packing_tensor_sizes[buffer_idx] += tensor.numel()
                if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                    break
            if len(packing_tensor_list[buffer_idx]) > 0:
                # Pack the tensors
                packed_tensors[buffer_idx] = torch.cat(packing_tensor_list[buffer_idx], dim=0)

        if len(packing_tensor_list[buffer_idx]) == 0:
            # No more tensors — nothing left to broadcast
            break

        # torch.cat runs on the custom stream.  Synchronize before
        # broadcasting on the default stream so the packed data is ready.
        streams[buffer_idx].synchronize()
        group.broadcast(packed_tensors[buffer_idx], src=src)

        # Move to the next buffer
        buffer_idx = (buffer_idx + 1) % num_buffers

    # Ensure the last broadcast on the default stream has completed
    # before returning, so NPU tensor cleanup at exit doesn't hang.
    torch.npu.current_stream().synchronize()


def packed_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
) -> None:
    """Consume packed tensors and unpack them into a list of tensors.

    Args:
        iterator: Iterator of parameter metadata. Returns (name, (shape, dtype))
        group: Process group (PyHcclCommunicator)
        src: Source rank (0 in current implementation)
        post_unpack_func: Function to apply to each list of (name, tensor) after
                         unpacking
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
        """Unpack a packed uint8 tensor into a list of typed tensors."""
        unpacked_tensors = packed_tensor.split(tensor_sizes)
        unpacked_list = [
            (name, tensor.contiguous().view(dtype).view(*shape))
            for name, shape, dtype, tensor in zip(names, shapes, dtypes, unpacked_tensors)
        ]
        return unpacked_list

    target_packed_tensor_size = buffer_size_bytes

    streams = [torch.npu.Stream() for _ in range(num_buffers)]
    default_stream = torch.npu.current_stream()
    buffer_idx = 0

    packing_tensor_meta_data: list[list[tuple[str, list[int], torch.dtype, int]]] = [[] for _ in range(num_buffers)]
    packing_tensor_sizes: list[int] = [0 for _ in range(num_buffers)]
    packed_tensors: list[torch.Tensor] = [torch.empty(0, dtype=torch.uint8, device="npu") for _ in range(num_buffers)]

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

        # Broadcast on the default stream.
        group.broadcast(packed_tensors[buffer_idx], src=src)

        # Synchronize the default stream so broadcast completes before
        # load_weights (running on the custom stream) reads the data.
        default_stream.synchronize()

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

    # Wait for all in-flight load_weights (on custom streams) to finish.
    # Otherwise NPU tensor cleanup at exit may hang.
    for s in streams:
        s.synchronize()


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

    Args:
        iterator: Iterator of (name, tensor) pairs.
        npu_uuid: Physical NPU UUID string for this rank.
        post_iter_func: Applied to each (name, tensor) before packing.
        buffer_size_bytes: Exact capacity of the reusable IPC buffer.
    """
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
        flat = post_iter_func((name, orig_tensor)).contiguous().view(torch.uint8).view(-1)

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

    if total_bytes:
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

    if physical_npu_id not in ipc_handle:
        raise ValueError(
            f"IPC handle not found for NPU UUID {physical_npu_id}. Available UUIDs: {list(ipc_handle.keys())}"
        )

    args = ipc_handle[physical_npu_id]
    list_args = list(args)
    # Index 6 of the args from reduce_tensor is the device_index.
    # Overwrite it with the receiver's device index.
    list_args[6] = device_index
    packed = rebuild_npu_tensor(*list_args)

    content_size = sum(tensor_sizes)
    packed = packed[:content_size]

    dtypes = [getattr(torch, dn) for dn in dtype_names]
    weights: list[tuple[str, torch.Tensor]] = []
    offset = 0
    for name, shape, dtype, size in zip(names, shapes, dtypes, tensor_sizes):
        raw = packed[offset : offset + size]
        tensor = raw.contiguous().view(dtype).view(*shape).clone()
        weights.append((name, tensor))
        offset += size

    return weights
