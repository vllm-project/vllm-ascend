# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cross-parameter batching for the HCCL M2N transport."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from itertools import islice
from typing import TYPE_CHECKING

import torch

from vllm_ascend.distributed.weight_transfer.hccl_m2n_transport import TransferPiece, transfer_parameter

if TYPE_CHECKING:
    from vllm.distributed.weight_transfer.m2n_common import M2NParamMeta

    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


# Byte alignment for mixed-dtype views packed into a single uint8 staging buffer.
_PACK_ALIGNMENT_BYTES = 16


def _aligned_nbytes(nbytes: int) -> int:
    return ((nbytes + _PACK_ALIGNMENT_BYTES - 1) // _PACK_ALIGNMENT_BYTES) * _PACK_ALIGNMENT_BYTES


def plan_batches(metas: Sequence[M2NParamMeta], max_bytes: int) -> tuple[range, ...]:
    """Partition the shared parameter order without using rank-local sizes."""
    batches = []
    start = 0
    total_bytes = 0
    for index, meta in enumerate(metas):
        size = math.prod(meta.shape) * meta.dtype.itemsize
        if start < index and total_bytes + size > max_bytes:
            batches.append(range(start, index))
            start = index
            total_bytes = 0
        total_bytes += size
    if start < len(metas):
        batches.append(range(start, len(metas)))
    return tuple(batches)


class _PackedSlot:
    def __init__(self) -> None:
        self.buffers: dict[int, torch.Tensor] = {}
        self.tensors: list[torch.Tensor] = []
        self.needs_sync: bool = False
        self._ready: torch.npu.Event | None = None

    def wait(self) -> None:
        if self._ready is not None:
            self._ready.synchronize()
            self._ready = None

    def submit(
        self,
        group: PyHcclCommunicator,
        stream: torch.npu.Stream,
        comm_stream: torch.npu.Stream,
        tensors: list[torch.Tensor],
        pieces: Sequence[tuple[TransferPiece, ...]],
        *,
        send: bool,
        needs_sync: bool = False,
    ) -> None:
        self.wait()
        self.tensors = tensors
        self.needs_sync = needs_sync

        peer_views: dict[int, list[torch.Tensor]] = defaultdict(list)
        for tensor, tensor_pieces in zip(tensors, pieces, strict=True):
            for piece in tensor_pieces:
                peer_views[piece.peer].append(tensor[piece.local_slice])

        buffers = []
        copies = []
        # Allocate uint8 staging buffers on the first stream to access them so
        # mixed-dtype parameters can share one P2P transfer per peer and a
        # receive can proceed while the caller stream unpacks the previous slot.
        with torch.npu.stream(stream if send else comm_stream):
            for peer, views in sorted(peer_views.items()):
                total_bytes = sum(_aligned_nbytes(view.numel() * view.element_size()) for view in views)
                buffer = self.buffers.get(peer)
                if buffer is None or buffer.numel() < total_bytes:
                    buffer = torch.empty((total_bytes,), dtype=torch.uint8, device=views[0].device)
                    self.buffers[peer] = buffer
                buffers.append((peer, buffer[:total_bytes]))
                offset = 0
                for view in views:
                    nbytes = view.numel() * view.element_size()
                    end = offset + nbytes
                    copies.append((view, buffer[offset:end].view(view.dtype).view(view.shape)))
                    offset += _aligned_nbytes(nbytes)

        if not buffers:
            self._ready = stream.record_event()
            return

        if send:
            with torch.npu.stream(stream):
                for view, payload in copies:
                    payload.copy_(view)
            comm_stream.wait_event(stream.record_event())
        group.batch_send_recv(buffers, send=send, stream=comm_stream)
        comm_done = comm_stream.record_event()
        if not send:
            stream.wait_event(comm_done)
            with torch.npu.stream(stream):
                for view, payload in copies:
                    view.copy_(payload)
        self._ready = comm_done if send else stream.record_event()


# A failed drain does not prove that HCCL released the raw device pointers.
# Keep affected slots alive until process exit instead of recycling storage.
_UNSYNCHRONIZED_SLOTS: list[_PackedSlot] = []


def transfer_packed_parameters(
    group: PyHcclCommunicator,
    stream: torch.npu.Stream,
    tensors: Iterator[torch.Tensor],
    pieces: Sequence[tuple[TransferPiece, ...]],
    batches: Sequence[range],
    *,
    send: bool,
    comm_stream: torch.npu.Stream | None = None,
    needs_host_sync: Sequence[bool] | None = None,
) -> Iterator[torch.Tensor]:
    """Alternate two staging slots while preserving loader parameter order."""
    if comm_stream is None:
        comm_stream = torch.npu.Stream(device=group.device)
    slots = (_PackedSlot(), _PackedSlot())
    keepalive: list[torch.Tensor] = []
    next_slot = 0
    pending: _PackedSlot | None = None
    try:
        for batch in batches:
            batch_needs_sync = not send and (needs_host_sync is None or any(needs_host_sync[i] for i in batch))
            if len(batch) == 1:
                if pending is not None:
                    # Drain comm_stream before switching the communicator to stream.
                    pending.wait()
                    yielded = pending.tensors
                    pending.tensors = []
                    yield from yielded
                    pending = None
                tensor = next(tensors)
                if batch_needs_sync:
                    transfer_parameter(group, stream, tensor, pieces[batch.start], send=send, sync=True)
                    keepalive.clear()
                else:
                    transfer_parameter(
                        group,
                        stream,
                        tensor,
                        pieces[batch.start],
                        send=send,
                        sync=False,
                        keepalive=keepalive,
                    )
                    # Keep the next comm_stream collective ordered after this stream op.
                    comm_stream.wait_event(stream.record_event())
                yield tensor
                continue

            batch_tensors = list(islice(tensors, len(batch)))
            if len(batch_tensors) != len(batch):
                raise RuntimeError("hccl_m2n packed update ended before the negotiated parameter list")
            slot = slots[next_slot]
            next_slot = 1 - next_slot
            slot.submit(
                group,
                stream,
                comm_stream,
                batch_tensors,
                pieces[batch.start : batch.stop],
                send=send,
                needs_sync=batch_needs_sync,
            )
            if pending is not None:
                if not send and pending.needs_sync:
                    pending.wait()
                yielded = pending.tensors
                pending.tensors = []
                yield from yielded
            pending = slot

        if pending is not None:
            if not send and pending.needs_sync:
                pending.wait()
            yielded = pending.tensors
            pending.tensors = []
            yield from yielded
        if next(tensors, None) is not None:
            raise RuntimeError("hccl_m2n packed update exceeded the negotiated parameter list")
    finally:
        try:
            try:
                comm_stream.synchronize()
            finally:
                stream.synchronize()
            keepalive.clear()
        except BaseException:
            _UNSYNCHRONIZED_SLOTS.extend(slots)
            if keepalive:
                failed_slot = _PackedSlot()
                failed_slot.tensors = keepalive
                _UNSYNCHRONIZED_SLOTS.append(failed_slot)
            raise
