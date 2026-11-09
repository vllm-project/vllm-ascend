# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-bounded HCCL transfers with independently owned loader inputs."""

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu

from vllm_ascend.distributed.weight_transfer.packed_tensor import _drain_packed_streams

WeightMetadata = tuple[str, tuple[list[int] | tuple[int, ...], torch.dtype]]


@dataclass(frozen=True)
class TensorFragment:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    parameter_offset: int
    buffer_offset: int
    nbytes: int
    complete: bool


@dataclass(frozen=True)
class TransferChunk:
    fragments: tuple[TensorFragment, ...]
    nbytes: int


def plan_transfer(metadata: Iterator[WeightMetadata], capacity: int) -> Iterator[TransferChunk]:
    """Partition ordered parameters without exceeding a collective's capacity.

    Zero-element parameters have a zero-byte fragment, including when the
    entire update is empty. A scalar has one element. Fragment offsets are
    byte offsets; mixed dtypes need no padding on the wire.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    fragments: list[TensorFragment] = []
    cursor = 0
    for name, (shape, dtype) in metadata:
        shape = tuple(shape)
        if any(dimension < 0 for dimension in shape):
            raise ValueError(f"negative dimension for parameter {name!r}: {shape}")
        size = math.prod(shape) * dtype.itemsize
        offset = 0
        while True:
            count = min(size - offset, capacity - cursor)
            fragments.append(TensorFragment(name, shape, dtype, offset, cursor, count, offset + count == size))
            offset += count
            cursor += count
            if cursor == capacity:
                yield TransferChunk(tuple(fragments), cursor)
                fragments = []
                cursor = 0
            if offset == size:
                break
    if fragments:
        yield TransferChunk(tuple(fragments), cursor)


def _source_bytes(item: tuple[str, torch.Tensor], fragment: TensorFragment, device: torch.device) -> torch.Tensor:
    name, tensor = item
    if name != fragment.name or tuple(tensor.shape) != fragment.shape or tensor.dtype != fragment.dtype:
        raise ValueError(f"source parameter {name!r} disagrees with metadata for {fragment.name!r}")
    if tensor.device != device or tensor.layout != torch.strided or tensor.is_quantized or not tensor.is_contiguous():
        raise ValueError("bounded transfers require contiguous, non-quantized tensors on the current NPU")
    if torch_npu.get_npu_format(tensor) not in (0, 2):
        raise ValueError("bounded transfers require NCHW or ND source storage; convert internal NPU formats beforehand")
    return tensor.detach().reshape(-1).view(torch.uint8)


def bounded_broadcast_producer(
    iterator: Iterator[tuple[str, torch.Tensor]],
    metadata: Iterator[WeightMetadata],
    group: Any,
    src: int,
    buffer_size_bytes: int,
    num_buffers: int,
    *,
    streams: tuple[Any, ...] | None = None,
) -> None:
    """Copy source byte views into fixed-capacity slots and broadcast them.

    Source writes must be ready on the caller stream. Lazy sources run on the
    active packing stream. Noncontiguous or off-device inputs are rejected
    before any conversion can allocate an entire parameter copy.
    """
    if num_buffers <= 0:
        raise ValueError("num_buffers must be positive")
    metadata_items = tuple(metadata)
    capacity = min(buffer_size_bytes, sum(math.prod(shape) * dtype.itemsize for _, (shape, dtype) in metadata_items))
    chunks = plan_transfer(iter(metadata_items), buffer_size_bytes)
    chunk = next(chunks, None)
    if chunk is None:
        return
    caller = torch.npu.current_stream()
    if streams is None:
        streams = tuple(torch.npu.Stream() for _ in range(num_buffers))
    buffers = [torch.empty(capacity, dtype=torch.uint8, device="npu") for _ in streams]
    for stream in streams:
        stream.wait_stream(caller)
    current = None
    previous = None
    body_error = None
    index = 0
    try:
        while chunk is not None:
            slot = index % num_buffers
            stream, buffer = streams[slot], buffers[slot]
            stream.synchronize()
            with torch.npu.stream(stream):
                for fragment in chunk.fragments:
                    if fragment.parameter_offset == 0:
                        try:
                            item = next(iterator)
                        except StopIteration as exc:
                            raise ValueError(
                                f"source is missing parameter {fragment.name!r} declared by metadata"
                            ) from exc
                        current = _source_bytes(item, fragment, buffer.device)
                    assert current is not None
                    start, source, count = fragment.buffer_offset, fragment.parameter_offset, fragment.nbytes
                    buffer[start : start + count].copy_(current[source : source + count])
                    current.record_stream(stream)
                    if fragment.complete:
                        current = None
            if chunk.nbytes:
                # Packing on this slot may overlap the previous collective.
                # Native HCCL must see completed torch_npu packing operations.
                stream.synchronize()
                if previous is not None and previous is not stream:
                    previous.synchronize()
                group.broadcast(buffer[: chunk.nbytes], src=src, stream=stream)
                previous = stream
            index += 1
            chunk = next(chunks, None)
        if next(iterator, None) is not None:
            raise ValueError("source contains parameters absent from the metadata")
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        _drain_packed_streams(list(streams), buffers, body_error)


def bounded_broadcast_consumer(
    metadata: Iterator[WeightMetadata],
    group: Any,
    src: int,
    post_unpack_func: Callable[[list[tuple[str, torch.Tensor]]], None],
    buffer_size_bytes: int,
    num_buffers: int,
    *,
    streams: tuple[Any, ...] | None = None,
    copy_stream: Any = None,
) -> None:
    """Reassemble owning parameters within a fixed transport storage budget.

    Transport-held tensor storage is at most K * B + Lmax: K receive slots of
    B bytes and owning inputs totalling at most Lmax bytes. When an entire
    update is smaller than B, slots shrink to its payload size and unused
    slot capacity can hold additional complete loader inputs. At most one
    parameter is incomplete; completed small parameters share loader calls.
    Loader-retained tensors,
    model/source weights, runtime workspaces and allocator rounding/reserved
    memory are separate. Loader inputs never alias a reusable receive slot.

    K=1 waits serially. K>=2 receives the next chunk while copying/loading the
    current chunk. Loader work on other streams must join the active stream
    before returning, matching the packed loader contract.
    """
    if num_buffers <= 0:
        raise ValueError("num_buffers must be positive")
    metadata_items = tuple(metadata)
    largest = max((math.prod(shape) * dtype.itemsize for _, (shape, dtype) in metadata_items), default=0)
    capacity = min(buffer_size_bytes, sum(math.prod(shape) * dtype.itemsize for _, (shape, dtype) in metadata_items))
    weight_budget = largest + num_buffers * (buffer_size_bytes - capacity)
    chunks = plan_transfer(iter(metadata_items), buffer_size_bytes)
    chunk = next(chunks, None)
    if chunk is None:
        return
    caller = torch.npu.current_stream()
    if streams is None:
        streams = tuple(torch.npu.Stream() for _ in range(num_buffers))
    if copy_stream is None:
        copy_stream = torch.npu.Stream()
    buffers = [torch.empty(capacity, dtype=torch.uint8, device="npu") for _ in streams]
    for stream in (*streams, copy_stream):
        stream.wait_stream(caller)
        stream.synchronize()
    weight = None
    raw = None
    pending: list[tuple[str, torch.Tensor]] = []
    pending_bytes = 0
    body_error = None

    def load_pending() -> None:
        nonlocal pending, pending_bytes
        if pending:
            post_unpack_func(pending)
            # Release transport ownership only after the loader's reads and
            # derived-state writes complete. Retention then belongs to it.
            copy_stream.synchronize()
            pending = []
            pending_bytes = 0

    def receive(next_chunk: TransferChunk, slot: int) -> None:
        if next_chunk.nbytes:
            group.broadcast(buffers[slot][: next_chunk.nbytes], src=src, stream=streams[slot])

    try:
        index = 0
        receive(chunk, 0)
        while chunk is not None:
            slot = index % num_buffers
            # External HCCL completion precedes reads by torch_npu.
            streams[slot].synchronize()
            following = next(chunks, None)
            if following is not None and num_buffers > 1:
                receive(following, (index + 1) % num_buffers)
            with torch.npu.stream(copy_stream):
                for fragment in chunk.fragments:
                    if fragment.parameter_offset == 0:
                        size = math.prod(fragment.shape) * fragment.dtype.itemsize
                        if pending_bytes + size > weight_budget:
                            load_pending()
                        weight = torch.empty(fragment.shape, dtype=fragment.dtype, device="npu")
                        raw = weight.reshape(-1).view(torch.uint8)
                    assert weight is not None and raw is not None
                    start, target, count = fragment.buffer_offset, fragment.parameter_offset, fragment.nbytes
                    raw[target : target + count].copy_(buffers[slot][start : start + count])
                    if fragment.complete:
                        weight.record_stream(copy_stream)
                        weight.record_stream(caller)
                        pending.append((fragment.name, weight))
                        pending_bytes += weight.numel() * weight.element_size()
                        weight = None
                        raw = None
                        if pending_bytes == weight_budget:
                            load_pending()
            # Slot reuse must wait even when a parameter spans many chunks.
            copy_stream.synchronize()
            if following is not None and num_buffers == 1:
                receive(following, slot)
            chunk = following
            index += 1
        with torch.npu.stream(copy_stream):
            load_pending()
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        owned = [tensor for _, tensor in pending]
        if weight is not None:
            owned.append(weight)
        if not owned:
            owned.append(torch.empty(0, dtype=torch.uint8, device="npu"))
        _drain_packed_streams([copy_stream] * len(owned) + list(streams), owned + buffers, body_error)
