# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Execution helpers for policy-provided explicit migration plans."""

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch_npu
from vllm.distributed.eplb.rebalance_execute import TransferMetadata


def copy_expert_tensor_(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Copy one expert tensor while preserving its Ascend storage format.

    Ascend 950 does not implement device-to-device ``Tensor.copy_`` when
    both tensors use an internal format such as FRACTAL_NZ. EPLB buffers are
    created with ``empty_like``, so matching internal-format tensors have the
    same physical layout and can be copied byte-for-byte without another
    format conversion.
    """
    if destination.device.type == "npu" and source.device.type == "npu":
        destination_format = int(torch_npu.get_npu_format(destination))
        source_format = int(torch_npu.get_npu_format(source))
        if destination.shape != source.shape or destination.dtype != source.dtype:
            raise ValueError("EPLB tensors must have matching shape and dtype")
        if destination_format != source_format:
            format_cast_kwargs = {}
            if destination_format != int(torch_npu.Format.ND):
                format_cast_kwargs["customize_dtype"] = destination.dtype
            source = torch_npu.npu_format_cast(source, destination_format, **format_cast_kwargs)
            source_format = destination_format
        if destination_format == source_format and destination_format != int(torch_npu.Format.ND):
            torch_npu.copy_memory_(destination, source)
            return
    destination.copy_(source, non_blocking=True)


def requires_format_aware_copy(
    expert_weights: Sequence[torch.Tensor | Sequence[torch.Tensor]],
) -> bool:
    """Return whether any EPLB weight uses a non-ND NPU format."""
    for weight in expert_weights:
        if isinstance(weight, (torch.Tensor, Sequence)):
            tensors = weight
        else:
            continue
        for tensor in tensors:
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.device.type == "npu"
                and int(torch_npu.get_npu_format(tensor)) != int(torch_npu.Format.ND)
            ):
                return True
    return False


def commit_staged_expert_weights(
    expert_weights: Sequence[torch.Tensor | Sequence[torch.Tensor]],
    expert_weight_buffers: Sequence[torch.Tensor | Sequence[torch.Tensor]],
    transfer_metadata: TransferMetadata,
    new_indices: np.ndarray,
    ep_rank: int,
) -> None:
    """Commit staged EPLB weights using an Ascend-format-aware copy."""
    is_unchanged = transfer_metadata.is_unchanged
    is_received_locally = transfer_metadata.is_received_locally
    recv_primary_mask = transfer_metadata.recv_primary_mask
    recv_count = transfer_metadata.recv_count
    recv_expert_ids = transfer_metadata.recv_expert_ids
    recv_dst_rows = transfer_metadata.recv_dst_rows
    num_local_experts = is_unchanged.shape[0]

    copy_mask = np.logical_or(is_received_locally, recv_primary_mask)
    destination_mask = np.logical_and(~is_unchanged, copy_mask)
    for destination in np.nonzero(destination_mask)[0].tolist():
        for weight, buffer in zip(expert_weights, expert_weight_buffers):
            copy_expert_tensor_(weight[destination], buffer[destination])

    if recv_count == 0:
        return

    base = ep_rank * num_local_experts
    local_experts = new_indices[base + np.arange(num_local_experts, dtype=np.int32)]
    duplicate_mask = np.logical_and(
        np.logical_and(~is_unchanged, ~is_received_locally),
        np.logical_and(~recv_primary_mask, local_experts != -1),
    )
    if not bool(duplicate_mask.any()):
        return

    duplicate_destinations = np.nonzero(duplicate_mask)[0]
    duplicate_experts = local_experts[duplicate_destinations]
    primary_experts = recv_expert_ids[:recv_count]
    primary_destinations = recv_dst_rows[:recv_count]
    order = np.argsort(primary_experts, kind="stable")
    primary_experts = primary_experts[order]
    primary_destinations = primary_destinations[order]
    positions = np.asarray(
        np.searchsorted(primary_experts, duplicate_experts),
        dtype=np.intp,
    )
    valid = np.logical_and(
        positions < primary_experts.shape[0],
        primary_experts[np.minimum(positions, primary_experts.shape[0] - 1)] == duplicate_experts,
    )
    for destination, source in zip(
        duplicate_destinations[valid].tolist(),
        primary_destinations[positions[valid]].tolist(),
    ):
        for weight in expert_weights:
            copy_expert_tensor_(weight[destination], weight[source])


def stage_explicit_layer_transfer(
    old_layer_indices: torch.Tensor,
    new_layer_indices: torch.Tensor,
    source_rank_ids: np.ndarray,
    source_slot_ids: np.ndarray,
    expert_weights: Sequence[torch.Tensor | Sequence[torch.Tensor]],
    expert_weight_buffers: Sequence[torch.Tensor | Sequence[torch.Tensor]],
    ep_group: Any,
    communicator: Any,
    stream: torch.Stream | None = None,
    layer_idx: int = 0,
) -> TransferMetadata:
    """Stage one layer using its exact ``[ranks, slots]`` source plan.

    Old and new indices are flattened ``[ranks * slots]`` CPU tensors for a
    dense, equal-capacity placement with no rank-local duplicate experts. The
    source arrays index the old placement. Weight and buffer sequences are
    non-empty and aligned; each pair has ``slots`` expert tensors with matching
    shape, dtype, and device at each slot. Mappings, source coordinates, and
    buffer schemas are validated before any copy or transfer is registered.
    Callers must validate the policy's complete plan separately. This function
    stages data into buffers and returns metadata for the current rank; live
    weights are not committed here.
    """
    if old_layer_indices.device.type != "cpu" or new_layer_indices.device.type != "cpu":
        raise ValueError("explicit EPLB layer mappings must be CPU tensors")
    old = old_layer_indices.numpy()
    new = new_layer_indices.numpy()
    if (
        old.ndim != 1
        or new.shape != old.shape
        or old.size == 0
        or not np.issubdtype(old.dtype, np.integer)
        or not np.issubdtype(new.dtype, np.integer)
        or np.any(old < 0)
        or np.any(new < 0)
    ):
        raise ValueError("explicit EPLB layer mappings must be aligned non-empty non-negative integer vectors")

    num_ranks = ep_group.size()
    ep_rank = ep_group.rank()
    if num_ranks < 1 or not 0 <= ep_rank < num_ranks:
        raise ValueError("explicit EPLB transfer received an invalid EP group rank or size")
    if old.size % num_ranks:
        raise ValueError("explicit EPLB mapping size must divide evenly across EP ranks")

    slots_per_rank = old.size // num_ranks
    source_ranks = np.asarray(source_rank_ids)
    source_slots = np.asarray(source_slot_ids)
    expected_shape = (num_ranks, slots_per_rank)
    if (
        source_ranks.shape != expected_shape
        or source_slots.shape != expected_shape
        or not np.issubdtype(source_ranks.dtype, np.integer)
        or not np.issubdtype(source_slots.dtype, np.integer)
    ):
        raise ValueError("explicit EPLB source plans must be integer [ranks, slots] arrays")
    if (
        np.any(source_ranks < 0)
        or np.any(source_ranks >= num_ranks)
        or np.any(source_slots < 0)
        or np.any(source_slots >= slots_per_rank)
    ):
        raise ValueError("explicit EPLB source plan contains an out-of-range coordinate")
    if len(expert_weights) != len(expert_weight_buffers) or not expert_weights:
        raise ValueError("EPLB expert weights and buffers must be non-empty and aligned")
    for weight, buffer in zip(expert_weights, expert_weight_buffers):
        if (
            (isinstance(weight, torch.Tensor) and weight.ndim == 0)
            or (isinstance(buffer, torch.Tensor) and buffer.ndim == 0)
            or len(weight) != slots_per_rank
            or len(buffer) != slots_per_rank
        ):
            raise ValueError("each EPLB expert weight and buffer pair must have the same slot-aligned schema")
        for source_row, buffer_row in zip(weight, buffer):
            if (
                not isinstance(source_row, torch.Tensor)
                or not isinstance(buffer_row, torch.Tensor)
                or source_row.shape != buffer_row.shape
                or source_row.dtype != buffer_row.dtype
                or source_row.device != buffer_row.device
            ):
                raise ValueError("each EPLB expert weight and buffer pair must have the same slot-aligned schema")

    old_placement = old.reshape(expected_shape)
    new_placement = new.reshape(expected_shape)
    if not np.array_equal(old_placement[source_ranks, source_slots], new_placement):
        raise RuntimeError("explicit EPLB source plan does not own every target expert")

    is_unchanged = np.zeros(slots_per_rank, dtype=np.bool_)
    is_received_locally = np.zeros(slots_per_rank, dtype=np.bool_)
    recv_primary_mask = np.zeros(slots_per_rank, dtype=np.bool_)
    recv_expert_ids = np.full(slots_per_rank, -1, dtype=np.int64)
    recv_dst_rows = np.full(slots_per_rank, -1, dtype=np.int32)
    recv_count = 0
    communicator.set_transfer_context(old, layer_idx)
    destination_ranks = (ep_rank,) if getattr(communicator, "receiver_initiated", False) else range(num_ranks)

    with stream if stream is not None else nullcontext():
        for dst_rank in destination_ranks:
            for dst_slot in range(slots_per_rank):
                expert = int(new_placement[dst_rank, dst_slot])
                src_rank = int(source_ranks[dst_rank, dst_slot])
                src_slot = int(source_slots[dst_rank, dst_slot])
                if src_rank == dst_rank:
                    if ep_rank == dst_rank:
                        is_received_locally[dst_slot] = True
                        is_unchanged[dst_slot] = src_slot == dst_slot
                        if src_slot != dst_slot:
                            for weight, buffer in zip(expert_weights, expert_weight_buffers):
                                copy_expert_tensor_(buffer[dst_slot], weight[src_slot])
                    continue
                if ep_rank == src_rank:
                    communicator.add_send([weight[src_slot] for weight in expert_weights], dst_rank, expert)
                if ep_rank == dst_rank:
                    communicator.add_recv([buffer[dst_slot] for buffer in expert_weight_buffers], src_rank, expert)
                    recv_primary_mask[dst_slot] = True
                    recv_expert_ids[recv_count] = expert
                    recv_dst_rows[recv_count] = dst_slot
                    recv_count += 1

    communicator.execute()
    return TransferMetadata(
        is_unchanged=is_unchanged,
        is_received_locally=is_received_locally,
        recv_primary_mask=recv_primary_mask,
        recv_count=recv_count,
        recv_expert_ids=recv_expert_ids,
        recv_dst_rows=recv_dst_rows,
    )
