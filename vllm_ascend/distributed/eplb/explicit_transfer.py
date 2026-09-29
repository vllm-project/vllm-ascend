# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Execution helpers for policy-provided explicit migration plans."""

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from vllm.distributed.eplb.rebalance_execute import TransferMetadata


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

    with stream if stream is not None else nullcontext():
        for dst_rank in range(num_ranks):
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
                                buffer[dst_slot].copy_(weight[src_slot], non_blocking=True)
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
