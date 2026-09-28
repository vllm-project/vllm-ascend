# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed tensor utilities for HCCL and NPU IPC weight transfer."""

import logging
import math
import warnings
from collections.abc import Callable, Iterator
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm_ascend.distributed.weight_transfer.npu_ipc_utils import (
    NpuPackedBufferImporter,
    rewrite_rebuild_device,
)

DEFAULT_PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024
DEFAULT_PACKED_NUM_BUFFERS = 2
logger = logging.getLogger(__name__)

# A failed stream cannot prove that its asynchronous readers have stopped.
# Retain those buffers until process exit instead of returning their storage to
# the allocator.  This list is populated only on a stream-drain failure.
_UNSYNCHRONIZED_BUFFERS: list[Any] = []


def _drain_in_flight_buffers(
    streams: list[Any] | tuple[Any, ...],
    in_flight: list[Any | None],
    *,
    preserve_original_error: bool,
) -> None:
    """Synchronize submitted work without masking an active caller error."""
    cleanup_error: BaseException | None = None
    for idx, stream in enumerate(streams):
        try:
            stream.synchronize()
        except BaseException as exc:
            if in_flight[idx] is not None:
                _UNSYNCHRONIZED_BUFFERS.append(in_flight[idx])
            if cleanup_error is None:
                cleanup_error = exc
        else:
            in_flight[idx] = None

    if cleanup_error is None:
        return
    if preserve_original_error:
        # Diagnostics must never replace the active source/loader exception,
        # even when an application installs a logging handler that raises.
        with suppress(BaseException):
            logger.warning(
                "Failed to drain a packed transfer stream while handling "
                "another exception; retaining its buffers until process "
                "exit: %s",
                cleanup_error,
            )
        return
    raise cleanup_error


def _resolve_device(device: torch.device | str | None = None) -> torch.device:
    # CPU-only unit tests replace ``torch.npu`` with a minimal namespace.  The
    # real runtime always provides ``torch.npu.device``; use CPU only for that
    # explicit test double so the byte-level packing logic remains testable.
    if not _has_npu_device_context():
        return torch.device("cpu")
    if device is None:
        index = torch.accelerator.current_device_index()
        return torch.device("npu", index)
    resolved = torch.device(device)
    if resolved.type != "npu":
        raise ValueError(f"packed NPU/HCCL transfer requires an NPU device, got {resolved}")
    if resolved.index is None:
        return torch.device("npu", torch.accelerator.current_device_index())
    return resolved


def _has_npu_device_context() -> bool:
    npu = getattr(torch, "npu", None)
    return npu is not None and callable(getattr(npu, "device", None))


def _npu_device_context(device: torch.device | int):
    if not _has_npu_device_context():
        return nullcontext()
    return torch.npu.device(device)


def _npu_stream_context(stream: Any):
    npu = getattr(torch, "npu", None)
    stream_func = getattr(npu, "stream", None)
    if not callable(stream_func):
        return nullcontext()
    return stream_func(stream)


def _validate_config(buffer_size_bytes: int, num_buffers: int | None = None) -> None:
    if buffer_size_bytes <= 0:
        raise ValueError(f"buffer_size_bytes must be positive, got {buffer_size_bytes}")
    if num_buffers is not None and num_buffers < 1:
        raise ValueError(f"num_buffers must be at least 1, got {num_buffers}")


def _validate_tensor_meta(
    names: list[str],
    shapes: list[list[int]],
    dtypes: list[torch.dtype],
    tensor_sizes: list[int],
) -> None:
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


def _dtype_from_name(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unknown torch dtype name {name!r}") from exc
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{name!r} is not a torch dtype")
    return dtype


def _flatten_to_bytes(
    name: str,
    original: torch.Tensor,
    materialized: torch.Tensor,
) -> torch.Tensor:
    """Return a 1-D byte view while preserving the source wire metadata."""
    tensor = materialized.contiguous()
    if tensor.dtype != original.dtype:
        raise ValueError(
            f"tensor {name!r} materialized with dtype {tensor.dtype} but metadata requires {original.dtype}"
        )
    if tensor.numel() != original.numel():
        raise ValueError(
            f"tensor {name!r} materialized with {tensor.numel()} elements but metadata requires {original.numel()}"
        )
    # dtype view rejects zero-dimensional tensors. Flattening first preserves
    # scalar data while the separately carried metadata keeps shape ``[]``.
    return tensor.reshape(-1).view(torch.uint8)


@cache
def _get_npu_streams(device_index: int, num_buffers: int) -> tuple[Any, ...]:
    """Reuse streams so the caching allocator can reuse packed buffers."""
    with _npu_device_context(device_index):
        return tuple(torch.npu.Stream() for _ in range(num_buffers))


def unpack_tensor(
    packed_tensor: torch.Tensor,
    names: list[str],
    shapes: list[list[int]],
    dtypes: list[torch.dtype],
    tensor_sizes: list[int],
) -> list[tuple[str, torch.Tensor]]:
    """Clone packed byte slices into independently owned typed tensors."""
    _validate_tensor_meta(names, shapes, dtypes, tensor_sizes)
    unpacked = packed_tensor.split(tensor_sizes)
    return [
        # Clone before the dtype view: a mixed-dtype packed layout may place a
        # tensor at a byte offset that is valid for the wire format but not
        # naturally aligned for the destination dtype.
        (name, raw.clone().view(dtype).reshape(shape))
        for name, shape, dtype, raw in zip(names, shapes, dtypes, unpacked)
    ]


@dataclass
class PackedChunk:
    packed_tensor: torch.Tensor
    names: list[str]
    shapes: list[list[int]]
    dtypes: list[torch.dtype]
    tensor_sizes: list[int]


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
    _validate_config(buffer_size_bytes)
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

        tensor = post_iter_func((name, original))
        flat = _flatten_to_bytes(name, original, tensor)
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
    return PackedChunk(packed, names, shapes, dtypes, tensor_sizes)


def packed_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    group: Any,
    src: int,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    device: torch.device | str | None = None,
) -> None:
    """Broadcast packed HCCL chunks from the source rank."""
    _validate_config(buffer_size_bytes, num_buffers)
    resolved = _resolve_device(device)
    streams = _get_npu_streams(resolved.index, num_buffers)
    in_flight: list[PackedChunk | None] = [None] * num_buffers
    buffer_idx = 0

    failed = False
    try:
        while True:
            stream = streams[buffer_idx]
            stream.synchronize()
            in_flight[buffer_idx] = None
            with _npu_device_context(resolved), _npu_stream_context(stream):
                chunk = pack_tensors(iterator, post_iter_func, buffer_size_bytes)
                if chunk is None:
                    break
                in_flight[buffer_idx] = chunk
                group.broadcast(chunk.packed_tensor, src=src, stream=stream)
            buffer_idx = (buffer_idx + 1) % num_buffers
    except BaseException:
        failed = True
        raise
    finally:
        _drain_in_flight_buffers(
            streams,
            in_flight,
            preserve_original_error=failed,
        )


def packed_broadcast_consumer(
    iterator: Iterator[tuple[str, tuple[list[int], torch.dtype]]],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    device: torch.device | str | None = None,
) -> None:
    """Receive packed HCCL chunks and load them on the worker."""
    _validate_config(buffer_size_bytes, num_buffers)
    resolved = _resolve_device(device)
    streams = _get_npu_streams(resolved.index, num_buffers)
    in_flight: list[torch.Tensor | None] = [None] * num_buffers
    buffer_idx = 0

    failed = False
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
            with _npu_device_context(resolved):
                group.broadcast(packed, src=src, stream=stream)
            stream.synchronize()
            with _npu_device_context(resolved), _npu_stream_context(stream):
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
    except BaseException:
        failed = True
        raise
    finally:
        _drain_in_flight_buffers(
            streams,
            in_flight,
            preserve_original_error=failed,
        )


def packed_npu_ipc_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    npu_uuid: str,
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor],
    buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    device: torch.device | str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield chunks backed by one reusable NPU IPC buffer."""
    _validate_config(buffer_size_bytes)
    resolved = _resolve_device(device)
    with _npu_device_context(resolved):
        ipc_buffer = torch.empty(buffer_size_bytes, dtype=torch.uint8, device=resolved)
        _, ipc_args = reduce_tensor(ipc_buffer)

        names: list[str] = []
        shapes: list[list[int]] = []
        dtypes: list[torch.dtype] = []
        tensor_sizes: list[int] = []
        total_bytes = 0
        has_tensors = False

        for name, original in iterator:
            flat_tensor = post_iter_func((name, original))
            flat = _flatten_to_bytes(name, original, flat_tensor)
            expected = math.prod(original.shape) * original.dtype.itemsize
            if flat.numel() != expected:
                raise ValueError(
                    f"tensor {name!r} materialized as {flat.numel()} bytes but metadata requires {expected} bytes"
                )
            if flat.numel() > buffer_size_bytes:
                raise ValueError(
                    f"Tensor {name!r} has size {flat.numel()} bytes, which exceeds "
                    f"buffer_size_bytes={buffer_size_bytes}. Increase the buffer."
                )
            if total_bytes and total_bytes + flat.numel() > buffer_size_bytes:
                torch.npu.current_stream().synchronize()
                yield {
                    "names": names,
                    "shapes": shapes,
                    "dtype_names": [str(dtype).split(".")[-1] for dtype in dtypes],
                    "tensor_sizes": tensor_sizes,
                    "ipc_handle": {npu_uuid: ipc_args},
                }
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
            yield {
                "names": names,
                "shapes": shapes,
                "dtype_names": [str(dtype).split(".")[-1] for dtype in dtypes],
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
    importer: NpuPackedBufferImporter | None = None,
    device: torch.device | str | None = None,
) -> list[tuple[str, torch.Tensor]]:
    """Import one packed NPU IPC chunk and return owned tensor slices."""
    if physical_npu_id not in ipc_handle:
        raise ValueError(
            f"IPC handle not found for NPU UUID {physical_npu_id}. Available UUIDs: {list(ipc_handle.keys())}"
        )
    dtypes = [_dtype_from_name(name) for name in dtype_names]
    _validate_tensor_meta(names, shapes, dtypes, tensor_sizes)

    resolved = _resolve_device(device or torch.device("npu", device_index))
    with _npu_device_context(resolved):
        args = rewrite_rebuild_device(ipc_handle[physical_npu_id], device_index)
        if importer is None:
            importer = NpuPackedBufferImporter()
        packed = importer.rebuild(args, device_index)
        packed = packed[: sum(tensor_sizes)]
        return unpack_tensor(
            packed,
            names,
            shapes,
            dtypes,
            tensor_sizes,
        )

