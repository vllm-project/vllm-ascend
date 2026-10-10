# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend-specific expert weight copy helpers for EPLB."""

from collections.abc import Sequence

import numpy as np
import torch
import torch_npu
from vllm.distributed.eplb.rebalance_execute import TransferMetadata


def copy_expert_tensor_(
    dst: torch.Tensor,
    src: torch.Tensor,
    *,
    non_blocking: bool = False,
) -> torch.Tensor:
    """Copy one EPLB expert tensor while preserving its NPU format."""
    if dst.shape != src.shape:
        raise ValueError(
            f"EPLB expert copy requires matching shapes, but got dst={tuple(dst.shape)} and src={tuple(src.shape)}"
        )
    if dst.dtype != src.dtype:
        raise ValueError(f"EPLB expert copy requires matching dtypes, but got dst={dst.dtype} and src={src.dtype}")
    if dst.device != src.device:
        raise ValueError(
            f"EPLB expert copy requires tensors on the same device, but got dst={dst.device} and src={src.device}"
        )
    if dst.nbytes != src.nbytes:
        raise ValueError(
            "EPLB expert copy requires matching tensor sizes, "
            f"but got dst={dst.nbytes} bytes and src={src.nbytes} bytes"
        )

    if dst.device.type != "npu":
        return dst.copy_(src, non_blocking=non_blocking)

    dst_format = int(torch_npu.get_npu_format(dst))
    src_format = int(torch_npu.get_npu_format(src))
    if dst_format != src_format:
        raise ValueError(
            f"EPLB expert copy requires matching NPU formats, but got dst={dst_format} and src={src_format}"
        )

    if dst_format == int(torch_npu.Format.ND):
        return dst.copy_(src, non_blocking=non_blocking)

    if dst.storage_offset() != 0 or src.storage_offset() != 0:
        raise ValueError(
            "Internal-format EPLB expert copies require offset-0 tensors, "
            f"but got dst={dst.storage_offset()} and "
            f"src={src.storage_offset()}"
        )

    return torch_npu.copy_memory_(
        dst,
        src,
        non_blocking=non_blocking,
    )


def move_from_buffer(
    expert_weights: Sequence[torch.Tensor | Sequence[torch.Tensor]],
    expert_weights_buffers: Sequence[torch.Tensor | Sequence[torch.Tensor]],
    transfer_metadata: TransferMetadata,
    new_indices: np.ndarray,
    ep_rank: int,
) -> None:
    """Commit transferred experts using Ascend-safe copy operations."""
    is_unchanged = transfer_metadata.is_unchanged
    is_received_locally = transfer_metadata.is_received_locally
    recv_primary_mask = transfer_metadata.recv_primary_mask
    recv_count = transfer_metadata.recv_count
    recv_expert_ids = transfer_metadata.recv_expert_ids
    recv_dst_rows = transfer_metadata.recv_dst_rows
    num_local_experts = is_unchanged.shape[0]

    copy_mask = np.logical_or(is_received_locally, recv_primary_mask)
    dest_mask = np.logical_and(~is_unchanged, copy_mask)
    if bool(dest_mask.any()):
        for dst in np.nonzero(dest_mask)[0].tolist():
            for weight, buffer in zip(
                expert_weights,
                expert_weights_buffers,
            ):
                copy_expert_tensor_(
                    weight[dst],
                    buffer[dst],
                    non_blocking=True,
                )

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

    duplicate_rows = np.nonzero(duplicate_mask)[0]
    duplicate_experts = local_experts[duplicate_rows]

    primary_experts = recv_expert_ids[:recv_count]
    primary_rows = recv_dst_rows[:recv_count]
    order = np.argsort(primary_experts, kind="stable")
    primary_experts = primary_experts[order]
    primary_rows = primary_rows[order]

    positions = np.asarray(
        np.searchsorted(primary_experts, duplicate_experts),
        dtype=np.intp,
    )
    valid = np.logical_and(
        positions < primary_experts.shape[0],
        primary_experts[np.minimum(positions, primary_experts.shape[0] - 1)] == duplicate_experts,
    )
    if not bool(valid.any()):
        return

    matched_dst_rows = duplicate_rows[valid]
    matched_src_rows = primary_rows[positions[valid]]
    for dst, src in zip(
        matched_dst_rows.tolist(),
        matched_src_rows.tolist(),
    ):
        for weight in expert_weights:
            copy_expert_tensor_(
                weight[dst],
                weight[src],
                non_blocking=True,
            )
