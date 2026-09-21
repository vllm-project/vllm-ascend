# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 mixed-quant sparse MLA adapter."""

from __future__ import annotations

import torch

from vllm_ascend.ops.triton.build_window_indices import (
    build_window_indices_triton,
)
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    wait_for_device_metadata,
)

from .package_loader import import_packaged_a5_module


def build_window_indices(
    positions: torch.Tensor,
    window_size: int,
    *,
    indices_output: torch.Tensor | None = None,
    lengths_output: torch.Tensor | None = None,
):
    return build_window_indices_triton(
        positions,
        window_size,
        indices_output=indices_output,
        lengths_output=lengths_output,
    )


def build_smla_metadata(length_rows: torch.Tensor) -> torch.Tensor:
    """Build the fixed A5 mixed-quant SMLA launch metadata."""
    import_packaged_a5_module("cann_ops_transformer.ops.attention.mixed_quant_sparse_flash_mla_dsl")
    return torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla_metadata(
        length_rows,
        length_rows,
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        quant_mode=1,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_win_kv=True,
        has_cmp_kv=True,
    )


def _resolve_window_indices(q, metadata, window_size):
    indices = metadata.swa.ori_sparse_indices
    lengths = metadata.swa.ori_topk_length
    if indices is None or lengths is None:
        return build_window_indices(
            metadata.positions[: q.shape[0]],
            window_size,
        )
    return indices[: q.shape[0]], lengths[: q.shape[0]]


def qsmla(
    q,
    win_kv,
    cmp_kv,
    metadata,
    compressed_indices,
    *,
    window_size,
    sinks,
    softmax_scale,
    compressed_lengths=None,
):
    import_packaged_a5_module("cann_ops_transformer.ops.attention.mixed_quant_sparse_flash_mla_dsl")

    win_indices, win_lengths = _resolve_window_indices(q, metadata, window_size)
    has_cmp = cmp_kv is not None
    if has_cmp:
        cmp_indices = compressed_indices[:, None, :].to(torch.int32).contiguous()
        cmp_lengths = compressed_lengths
    else:
        cmp_indices = None
        cmp_lengths = torch.zeros_like(win_lengths)
    task_metadata = metadata.swa.smla_metadata
    wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(task_metadata))
    output, _ = torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla(
        q,
        win_kv=win_kv,
        cmp_kv=cmp_kv,
        win_sparse_indices=win_indices,
        cmp_sparse_indices=cmp_indices,
        win_block_table=metadata.swa.block_table,
        cmp_block_table=metadata.attention.block_table if has_cmp else None,
        cu_seqlens_q=metadata.swa.query_start_loc,
        seqused_win_kv=metadata.swa.seq_lens,
        seqused_cmp_kv=metadata.attention.cache_seq_lens if has_cmp else None,
        win_topk_length=win_lengths,
        cmp_topk_length=cmp_lengths if has_cmp else None,
        sinks=sinks.detach().float().contiguous(),
        metadata=task_metadata,
        quant_mode=1,
        softmax_scale=softmax_scale,
        layout_q="TND",
        layout_kv="PA_BBND",
        return_softmax_lse=False,
    )
    return output
