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


def plan_batches(metas: Sequence[M2NParamMeta], max_bytes: int) -> tuple[range, ...]:
    """Partition the shared parameter order without using rank-local sizes."""
    batches = []
    start = 0
    total_bytes = 0
    dtype = None
    for index, meta in enumerate(metas):
        size = math.prod(meta.shape) * meta.dtype.itemsize
        if start < index and (dtype != meta.dtype or total_bytes + size > max_bytes):
            batches.append(range(start, index))
            start = index
            total_bytes = 0
        total_bytes += size
        dtype = meta.dtype
    if start < len(metas):
        batches.append(range(start, len(metas)))
    return tuple(batches)


class _PackedSlot:
    def __init__(self) -> None:
        self.buffers: dict[int, torch.Tensor] = {}
        self.tensors: list[torch.Tensor] = []
        self._ready: torch.npu.Event | None = None

    def wait(self) -> None:
        if self._ready is not None:
            self._ready.synchronize()

    def submit(
        self,
        group: PyHcclCommunicator,
        stream: torch.npu.Stream,
        comm_stream: torch.npu.Stream,
        tensors: list[torch.Tensor],
        pieces: Sequence[tuple[TransferPiece, ...]],
        *,
        send: bool,
    ) -> None:
        self.wait()
        self.tensors = tensors

        peer_views: dict[int, list[torch.Tensor]] = defaultdict(list)
        for tensor, tensor_pieces in zip(tensors, pieces, strict=True):
            for piece in tensor_pieces:
                peer_views[piece.peer].append(tensor[piece.local_slice])

        buffers = []
        copies = []
        # Allocate on the first stream to access the staging memory. A receive
        # can then proceed while the caller stream unpacks the previous slot.
        with torch.npu.stream(stream if send else comm_stream):
            for peer, views in sorted(peer_views.items()):
                size = sum(view.numel() for view in views)
                buffer = self.buffers.get(peer)
                if buffer is None or buffer.numel() < size or buffer.dtype != views[0].dtype:
                    buffer = views[0].new_empty((size,))
                    self.buffers[peer] = buffer
                buffers.append((peer, buffer[:size]))
                offset = 0
                for view in views:
                    end = offset + view.numel()
                    copies.append((view, buffer[offset:end]))
                    offset = end

        if not buffers:
            self._ready = stream.record_event()
            return

        if send:
            for view, payload in copies:
                payload.view(view.shape).copy_(view)
            comm_stream.wait_event(stream.record_event())
        group.batch_send_recv(buffers, send=send, stream=comm_stream)
        comm_done = comm_stream.record_event()
        if not send:
            stream.wait_event(comm_done)
            for view, payload in copies:
                view.copy_(payload.view(view.shape))
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
) -> Iterator[torch.Tensor]:
    """Alternate two staging slots while preserving loader parameter order."""
    comm_stream = torch.npu.Stream(device=group.device)
    slots = (_PackedSlot(), _PackedSlot())
    next_slot = 0
    pending = None
    try:
        for batch in batches:
            if len(batch) == 1:
                if pending is not None:
                    pending.wait()
                    yield from pending.tensors
                    pending = None
                tensor = next(tensors)
                transfer_parameter(group, stream, tensor, pieces[batch.start], send=send)
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
            )
            if pending is not None:
                if not send:
                    pending.wait()
                yield from pending.tensors
            pending = slot

        if pending is not None:
            if not send:
                pending.wait()
            yield from pending.tensors
        if next(tensors, None) is not None:
            raise RuntimeError("hccl_m2n packed update exceeded the negotiated parameter list")
    finally:
        try:
            try:
                comm_stream.synchronize()
            finally:
                stream.synchronize()
        except BaseException:
            _UNSYNCHRONIZED_SLOTS.extend(slots)
            raise
