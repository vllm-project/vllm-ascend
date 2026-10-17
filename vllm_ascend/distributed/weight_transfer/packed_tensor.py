# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for HCCL and NPU IPC weight transfer."""

import logging
import math
import warnings
from collections.abc import Callable, Iterator
from contextlib import suppress
from functools import cache
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor
from vllm.distributed.weight_transfer.packed_tensor import (
    PackedChunk,
    PackedIpcChunk,
)

DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024
DEFAULT_PACKED_NUM_BUFFERS = 2
logger = logging.getLogger(__name__)

# A failed stream cannot prove that its asynchronous readers have stopped.
# Retain those buffers until process exit instead of returning their storage to
# the allocator.  This list is populated only on a stream-drain failure.
_UNSYNCHRONIZED_BUFFERS: list[Any] = []


@cache
def _get_streams(device_index: int, num_buffers: int) -> tuple[Any, ...]:
    """Reuse streams so the caching allocator can reuse packed buffers."""
    with torch.npu.device(torch.device("npu", device_index)):
        return tuple(torch.npu.Stream() for _ in range(num_buffers))


_get_npu_streams = _get_streams


def unpack_tensor(
    packed_tensor: torch.Tensor,
    names: list[str],
    shapes: list[list[int]],
    dtypes: list[torch.dtype],
    tensor_sizes: list[int],
) -> list[tuple[str, torch.Tensor]]:
    """Clone packed byte slices into independently owned typed tensors."""
    if not (len(names) == len(shapes) == len(dtypes) == len(tensor_sizes)):
        raise ValueError(
            "packed metadata lengths disagree: "
            f"names={len(names)}, shapes={len(shapes)}, "
            f"dtypes={len(dtypes)}, tensor_sizes={len(tensor_sizes)}"
        )
    for name, shape, dtype, size in zip(names, shapes, dtypes, tensor_sizes):
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"tensor {name!r} has a negative shape: {shape}")
        if size < 0:
            raise ValueError(f"tensor {name!r} has a negative byte size: {size}")
        expected = math.prod(shape) * dtype.itemsize
        if size != expected:
            raise ValueError(f"tensor {name!r} metadata says {size} bytes but its shape/dtype require {expected} bytes")
    unpacked = packed_tensor.split(tensor_sizes)
    return [
        # Clone before the dtype view: a mixed-dtype packed layout may place a
        # tensor at a byte offset that is valid for the wire format but not
        # naturally aligned for the destination dtype.
        (name, raw.clone().view(dtype).reshape(shape))
        for name, shape, dtype, raw in zip(names, shapes, dtypes, unpacked)
    ]


def pack_tensors(
    iterator: Iterator[tuple[str, torch.Tensor]],
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int,
    tensor_list: list[torch.Tensor] | None = None,
    current_size: int = 0,
) -> PackedChunk | None:
    """Pack until the accumulated bytes are greater than the threshold.

    The ``>`` rule is intentional and matches the existing HCCL wire behavior:
    a tensor that crosses the threshold is sent as part of that same chunk.
    """
    if buffer_size_bytes <= 0:
        raise ValueError(f"buffer_size_bytes must be positive, got {buffer_size_bytes}")
    tensors = tensor_list if tensor_list is not None else []
    names: list[str] = []
    shapes: list[list[int]] = []
    dtypes: list[torch.dtype] = []
    tensor_sizes: list[int] = []
    total_bytes = current_size

    while True:
        try:
            name, original = next(iterator)
        except StopIteration:
            break

        tensor = post_iter_func((name, original)).contiguous()
        if tensor.dtype != original.dtype:
            raise ValueError(
                f"tensor {name!r} materialized with dtype {tensor.dtype} but metadata requires {original.dtype}"
            )
        if tensor.numel() != original.numel():
            raise ValueError(
                f"tensor {name!r} materialized with {tensor.numel()} elements but metadata requires {original.numel()}"
            )
        flat = tensor.reshape(-1).view(torch.uint8)
        expected = math.prod(original.shape) * original.dtype.itemsize
        if flat.numel() != expected:
            raise ValueError(
                f"tensor {name!r} materialized as {flat.numel()} bytes but metadata requires {expected} bytes"
            )
        tensors.append(flat)
        names.append(name)
        shapes.append(list(original.shape))
        dtypes.append(original.dtype)
        tensor_sizes.append(flat.numel())
        total_bytes += flat.numel()

        if flat.numel() > buffer_size_bytes:
            warnings.warn(
                f"Tensor {name!r} has size {flat.numel()} bytes, which exceeds "
                f"buffer_size_bytes={buffer_size_bytes}; sending it as one chunk.",
                stacklevel=2,
            )
        if total_bytes > buffer_size_bytes:
            break

    if not tensors:
        return None
    packed = torch.cat(tensors, dim=0)
    return PackedChunk(
        packed_tensor=packed,
        names=names,
        shapes=shapes,
        dtypes=dtypes,
        tensor_sizes=tensor_sizes,
    )


def packed_hccl_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    device: torch.device | str | None = None,
) -> None:
    """Broadcast packed HCCL chunks from the source rank."""
    if buffer_size_bytes <= 0:
        raise ValueError(f"buffer_size_bytes must be positive, got {buffer_size_bytes}")
    if num_buffers < 1:
        raise ValueError(f"num_buffers must be at least 1, got {num_buffers}")
    if device is None:
        device_index = torch.accelerator.current_device_index()
        if device_index is None:
            raise RuntimeError("NPU packed transfer requires an active NPU device")
        resolved = torch.device("npu", device_index)
    else:
        resolved = torch.device(device)
        if resolved.type != "npu" or resolved.index is None:
            raise ValueError(f"packed HCCL transfer requires an indexed NPU device, got {resolved}")
        device_index = resolved.index
    streams = _get_streams(device_index, num_buffers)
    in_flight: list[PackedChunk | None] = [None] * num_buffers
    buffer_idx = 0

    body_error: BaseException | None = None
    try:
        while True:
            stream = streams[buffer_idx]
            stream.synchronize()
            in_flight[buffer_idx] = None
            with torch.npu.device(resolved), torch.npu.stream(stream):
                chunk = pack_tensors(iterator, post_iter_func, buffer_size_bytes)
                if chunk is None:
                    break
                in_flight[buffer_idx] = chunk
                group.broadcast(chunk.packed_tensor, src=src, stream=stream)
            buffer_idx = (buffer_idx + 1) % num_buffers
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        cleanup_error: BaseException | None = None
        for index, stream in enumerate(streams):
            try:
                stream.synchronize()
            except BaseException as exc:
                if in_flight[index] is not None:
                    _UNSYNCHRONIZED_BUFFERS.append(in_flight[index])
                if cleanup_error is None:
                    cleanup_error = exc
            else:
                in_flight[index] = None
        if cleanup_error is not None and body_error is None:
            raise cleanup_error
        if cleanup_error is not None and body_error is not None:
            with suppress(BaseException):
                logger.warning(
                    "Failed to drain a packed HCCL stream while handling another exception; "
                    "retaining its buffer until process exit: %s",
                    cleanup_error,
                )


def packed_hccl_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    device: torch.device | str | None = None,
) -> None:
    """Receive packed HCCL chunks and load them on the worker."""
    if buffer_size_bytes <= 0:
        raise ValueError(f"buffer_size_bytes must be positive, got {buffer_size_bytes}")
    if num_buffers < 1:
        raise ValueError(f"num_buffers must be at least 1, got {num_buffers}")
    if device is None:
        device_index = torch.accelerator.current_device_index()
        if device_index is None:
            raise RuntimeError("NPU packed transfer requires an active NPU device")
        resolved = torch.device("npu", device_index)
    else:
        resolved = torch.device(device)
        if resolved.type != "npu" or resolved.index is None:
            raise ValueError(f"packed HCCL transfer requires an indexed NPU device, got {resolved}")
        device_index = resolved.index
    streams = _get_streams(device_index, num_buffers)
    in_flight: list[torch.Tensor | None] = [None] * num_buffers
    buffer_idx = 0

    body_error: BaseException | None = None
    try:
        while True:
            stream = streams[buffer_idx]
            stream.synchronize()
            in_flight[buffer_idx] = None
            metadata: list[tuple[str, list[int], torch.dtype, int]] = []
            total_bytes = 0
            while True:
                try:
                    name, (shape, dtype) = next(iterator)
                except StopIteration:
                    break
                if any(dimension < 0 for dimension in shape):
                    raise ValueError(f"tensor {name!r} has a negative shape: {shape}")
                size = math.prod(shape) * dtype.itemsize
                metadata.append((name, list(shape), dtype, size))
                total_bytes += size
                if total_bytes > buffer_size_bytes:
                    break

            if not metadata:
                break

            names, shapes, dtypes, tensor_sizes = zip(*metadata)
            packed = torch.empty(total_bytes, dtype=torch.uint8, device=resolved)
            # Keep each receive slot alive until the stream using that slot has
            # completed its broadcast, unpack clones, and model loading work.
            in_flight[buffer_idx] = packed
            with torch.npu.device(resolved):
                group.broadcast(packed, src=src, stream=stream)
            stream.synchronize()
            with torch.npu.device(resolved), torch.npu.stream(stream):
                post_unpack_func(
                    unpack_tensor(
                        packed,
                        list(names),
                        list(shapes),
                        list(dtypes),
                        list(tensor_sizes),
                    )
                )
            buffer_idx = (buffer_idx + 1) % num_buffers
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        cleanup_error: BaseException | None = None
        for index, stream in enumerate(streams):
            try:
                stream.synchronize()
            except BaseException as exc:
                if in_flight[index] is not None:
                    _UNSYNCHRONIZED_BUFFERS.append(in_flight[index])
                if cleanup_error is None:
                    cleanup_error = exc
            else:
                in_flight[index] = None
        if cleanup_error is not None and body_error is None:
            raise cleanup_error
        if cleanup_error is not None and body_error is not None:
            with suppress(BaseException):
                logger.warning(
                    "Failed to drain a packed HCCL stream while handling another exception; "
                    "retaining its buffer until process exit: %s",
                    cleanup_error,
                )


def packed_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    npu_uuid: str,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    device: torch.device | str | None = None,
) -> Iterator[PackedIpcChunk]:
    """Yield chunks backed by one reusable NPU IPC buffer."""
    if buffer_size_bytes <= 0:
        raise ValueError(f"buffer_size_bytes must be positive, got {buffer_size_bytes}")
    if device is None:
        device_index = torch.accelerator.current_device_index()
        if device_index is None:
            raise RuntimeError("NPU packed transfer requires an active NPU device")
        resolved = torch.device("npu", device_index)
    else:
        resolved = torch.device(device)
        if resolved.type != "npu" or resolved.index is None:
            raise ValueError(f"packed IPC transfer requires an indexed NPU device, got {resolved}")
    with torch.npu.device(resolved):
        ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device=resolved)
        _, ipc_args = reduce_tensor(ipc_buffer)

        names: list[str] = []
        shapes: list[list[int]] = []
        dtypes: list[torch.dtype] = []
        tensor_sizes: list[int] = []
        total_bytes = 0
        has_tensors = False

        for name, original in iterator:
            flat_tensor = post_iter_func((name, original)).contiguous()
            if flat_tensor.dtype != original.dtype:
                raise ValueError(
                    f"tensor {name!r} materialized with dtype {flat_tensor.dtype} "
                    f"but metadata requires {original.dtype}"
                )
            if flat_tensor.numel() != original.numel():
                raise ValueError(
                    f"tensor {name!r} materialized with {flat_tensor.numel()} elements "
                    f"but metadata requires {original.numel()}"
                )
            flat = flat_tensor.reshape(-1).view(torch.uint8)
            expected = math.prod(original.shape) * original.dtype.itemsize
            if flat.numel() != expected:
                raise ValueError(
                    f"tensor {name!r} materialized as {flat.numel()} bytes but metadata requires {expected} bytes"
                )
            if flat.numel() > buffer_size_bytes:
                raise ValueError(
                    f"Tensor {name!r} has size {flat.numel()} bytes, which exceeds "
                    f"buffer_size_bytes={buffer_size_bytes}. "
                    "Increase the buffer."
                )
            if total_bytes and total_bytes + flat.numel() > buffer_size_bytes:
                torch.npu.current_stream().synchronize()
                yield PackedIpcChunk(
                    names=names,
                    shapes=shapes,
                    dtype_names=[str(dtype).split(".")[-1] for dtype in dtypes],
                    tensor_sizes=tensor_sizes,
                    ipc_handle={npu_uuid: ipc_args},
                )
                names, shapes, dtypes, tensor_sizes = [], [], [], []
                total_bytes = 0

            ipc_buffer[total_bytes : total_bytes + flat.numel()].copy_(flat)
            names.append(name)
            shapes.append(list(original.shape))
            dtypes.append(original.dtype)
            tensor_sizes.append(flat.numel())
            total_bytes += flat.numel()
            has_tensors = True

        if has_tensors:
            torch.npu.current_stream().synchronize()
            yield PackedIpcChunk(
                names=names,
                shapes=shapes,
                dtype_names=[str(dtype).split(".")[-1] for dtype in dtypes],
                tensor_sizes=tensor_sizes,
                ipc_handle={npu_uuid: ipc_args},
            )


def _default_rebuild_npu_tensor(
    ipc_args: tuple[Any, ...],
    device_index: int,
) -> torch.Tensor:
    """Rebuild a packed IPC buffer for the legacy compatibility path."""
    from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

    list_args = list(ipc_args)
    # reduce_tensor stores the sender device in tuple slot 6. The
    # compatibility path keeps this ABI local; the IPC engine supplies a
    # callback so its cached importer owns the same adjustment.
    if len(list_args) <= 6:
        raise ValueError(f"NPU IPC rebuild arguments do not contain a device index: got {len(list_args)} values")
    list_args[6] = device_index
    return rebuild_npu_tensor(*list_args)


PackedIpcRebuildFunc = Callable[[tuple[Any, ...], int], torch.Tensor]


def packed_ipc_consumer(
    ipc_handle: dict[str, tuple],
    names: list[str],
    shapes: list[list[int]],
    dtype_names: list[str],
    tensor_sizes: list[int],
    device_index: int,
    rebuild_func: PackedIpcRebuildFunc | None = None,
    device: torch.device | str | None = None,
    physical_npu_id: str | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Import one packed NPU IPC chunk and return owned tensor slices.

    rebuild_func is supplied by the NPU IPC engine when it needs to reuse
    a cached importer. Omitting it preserves the original one-shot rebuild
    behavior for the legacy entry point.
    """
    if device_index < 0:
        raise ValueError(f"device_index must be non-negative, got {device_index}")
    if physical_npu_id is None:
        raise ValueError("physical_npu_id is required for packed IPC consumption")
    if physical_npu_id not in ipc_handle:
        raise ValueError(
            f"IPC handle not found for NPU UUID {physical_npu_id}. Available UUIDs: {list(ipc_handle.keys())}"
        )

    dtypes: list[torch.dtype] = []
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

    if device is None:
        resolved = torch.device("npu", device_index)
    else:
        resolved = torch.device(device)
        if resolved.type != "npu" or resolved.index is None:
            raise ValueError(f"packed IPC transfer requires an indexed NPU device, got {resolved}")
    rebuild = _default_rebuild_npu_tensor if rebuild_func is None else rebuild_func
    with torch.npu.device(resolved):
        packed = rebuild(ipc_handle[physical_npu_id], device_index)
        packed = packed[: sum(tensor_sizes)]
        return unpack_tensor(
            packed,
            names,
            shapes,
            dtypes,
            tensor_sizes,
        )


# Compatibility aliases for the names used by the original Ascend PR.
packed_broadcast_producer = packed_hccl_broadcast_producer
packed_broadcast_consumer = packed_hccl_broadcast_consumer


def packed_npu_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    npu_uuid: str,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    device: torch.device | str | None = None,
) -> Iterator[dict[str, Any]]:
    """Preserve the legacy dict wire protocol for NPU IPC callers."""
    for chunk in packed_ipc_producer(
        iterator=iterator,
        npu_uuid=npu_uuid,
        post_iter_func=post_iter_func,
        buffer_size_bytes=buffer_size_bytes,
        device=device,
    ):
        # Keep the lists and handle object from the typed chunk; the wrapper
        # changes only the outer protocol and does not deep-copy the payload.
        yield {
            "names": chunk.names,
            "shapes": chunk.shapes,
            "dtype_names": chunk.dtype_names,
            "tensor_sizes": chunk.tensor_sizes,
            "ipc_handle": chunk.ipc_handle,
        }


def packed_npu_ipc_consumer(
    ipc_handle: dict[str, tuple],
    physical_npu_id: str,
    names: list[str],
    shapes: list[list[int]],
    dtype_names: list[str],
    tensor_sizes: list[int],
    device_index: int,
    rebuild_func: PackedIpcRebuildFunc | None = None,
    device: torch.device | str | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Compatibility entry point with the original IPC argument order."""
    return packed_ipc_consumer(
        ipc_handle=ipc_handle,
        names=names,
        shapes=shapes,
        dtype_names=dtype_names,
        tensor_sizes=tensor_sizes,
        device_index=device_index,
        rebuild_func=rebuild_func,
        device=device,
        physical_npu_id=physical_npu_id,
    )
