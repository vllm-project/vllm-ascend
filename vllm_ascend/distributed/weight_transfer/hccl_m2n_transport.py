# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Rank-local transfer plans for uniform M2N shards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.weight_transfer.m2n_common import REPLICATE, M2NLayout, resolve_layout, validate_layout

if TYPE_CHECKING:
    import torch

    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


@dataclass(frozen=True)
class TransferPiece:
    peer: int
    local_slice: tuple[slice, ...]


def _rank_regions(
    layout: M2NLayout, shape: tuple[int, ...]
) -> tuple[tuple[int, ...], dict[tuple[int, ...], list[int]]]:
    """Group ranks with identical logical shards by their global origin."""
    mesh, placements = resolve_layout(layout.mesh, layout.placements)
    local_shape = validate_layout(mesh, placements, shape, "HCCL M2N layout")
    regions: dict[tuple[int, ...], list[int]] = {}
    for index in range(mesh.size):
        origin = [0] * len(shape)
        for coordinate, dim in zip(divmod(index, mesh.dims[1]), placements):
            if dim != REPLICATE:
                origin[dim] = coordinate * local_shape[dim]
        regions.setdefault(tuple(origin), []).append(mesh.start_rank + index)
    return local_shape, regions


def plan_rank_transfers(
    shape: tuple[int, ...], source: M2NLayout, destination: M2NLayout, rank: int
) -> tuple[TransferPiece, ...]:
    """Intersect logical shards and pick one source replica per destination tile."""
    src_shape, src_regions = _rank_regions(source, shape)
    dst_shape, dst_regions = _rank_regions(destination, shape)
    sending = source.mesh.start_rank <= rank < source.mesh.start_rank + source.mesh.size
    if sending:
        src_regions = {origin: ranks for origin, ranks in src_regions.items() if rank in ranks}
    else:
        # Preserve the destination tile index below for deterministic replica selection.
        if not destination.mesh.start_rank <= rank < destination.mesh.start_rank + destination.mesh.size:
            raise ValueError(f"Rank {rank} is outside both HCCL M2N meshes")

    pieces = []
    for dst_index, (dst_origin, dst_ranks) in enumerate(dst_regions.items()):
        if not sending and rank not in dst_ranks:
            continue
        for src_origin, src_ranks in src_regions.items():
            start = tuple(max(src, dst) for src, dst in zip(src_origin, dst_origin))
            stop = tuple(
                min(src + src_size, dst + dst_size)
                for src, src_size, dst, dst_size in zip(src_origin, src_shape, dst_origin, dst_shape)
            )
            if any(lo >= hi for lo, hi in zip(start, stop)):
                continue
            local_origin = src_origin if sending else dst_origin
            local_slice = tuple(slice(lo - base, hi - base) for lo, hi, base in zip(start, stop, local_origin))
            for replica, dst_rank in enumerate(dst_ranks):
                src_rank = src_ranks[(dst_index + replica) % len(src_ranks)]
                if sending and rank == src_rank:
                    pieces.append(TransferPiece(dst_rank, local_slice))
                elif not sending and rank == dst_rank:
                    pieces.append(TransferPiece(src_rank, local_slice))
    return tuple(sorted(pieces, key=lambda piece: piece.peer))


def transfer_parameter(
    group: PyHcclCommunicator,
    stream: torch.npu.Stream,
    tensor: torch.Tensor,
    pieces: tuple[TransferPiece, ...],
    *,
    send: bool,
) -> None:
    """Transfer one parameter. Stage only views that are not contiguous."""
    if not pieces:
        return

    views = [tensor[piece.local_slice] for piece in pieces]
    buffers = []
    for view in views:
        if view.is_contiguous():
            buffers.append(view)
        elif send:
            buffers.append(view.contiguous())
        else:
            buffers.append(view.new_empty(view.shape))
    # Raw HCCL pointers are invisible to PyTorch's allocator. Keep tensors
    # alive through the stream drain, including when HCCL reports an error.
    try:
        group.batch_send_recv(
            [(piece.peer, buffer) for piece, buffer in zip(pieces, buffers)], send=send, stream=stream
        )
        if not send:
            for view, buffer in zip(views, buffers):
                if view is not buffer:
                    view.copy_(buffer)
    finally:
        stream.synchronize()
