# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

import torch


class GBSAMaskMode(IntEnum):
    NO_MASK = 0
    CAUSAL = 1
    WINDOW = 2


class GBSAQuantMode(IntEnum):
    NO_QUANT = 0
    FP8_E4M3_STATIC_PER_GROUP = 1
    FP8_E4M3_DYNAMIC_MX = 2
    FP4_E2M1_DYNAMIC_OCP = 3
    FP4_E2M1_DYNAMIC_CX = 4
    FP8_E4M3_STATIC_CAST_P = 5


def generic_block_sparse_attention_metadata(
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    block_shape: list[int],
    *,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_kv: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_kv: torch.Tensor | None = None,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "TND",
    layout_kv: str = "PA_BBND",
    layout_sparse_pattern: int = 4,
    mask_mode: int = GBSAMaskMode.CAUSAL,
    quant_mode: int = GBSAQuantMode.NO_QUANT,
    softmax_precision: int = 1,
    win_left: int = -1,
    win_right: int = -1,
    residual_block_mode: int = 0,
    is_consistent_topk: bool = False,
) -> torch.Tensor:
    """Build the per-invocation metadata required by GBSA."""
    return torch.ops._C_ascend.npu_generic_block_sparse_attention_metadata(
        sparse_block_idx,
        sparse_block_count,
        num_heads_q,
        num_heads_kv,
        head_dim,
        block_shape,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_sparse_pattern=layout_sparse_pattern,
        mask_mode=int(mask_mode),
        quant_mode=int(quant_mode),
        softmax_precision=softmax_precision,
        win_left=win_left,
        win_right=win_right,
        residual_block_mode=residual_block_mode,
        is_consistent_topk=is_consistent_topk,
    )


def generic_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    block_shape: list[int],
    *,
    metadata: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    q_dequant_scale: torch.Tensor | None = None,
    k_dequant_scale: torch.Tensor | None = None,
    v_dequant_scale: torch.Tensor | None = None,
    p_quant_scale: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_kv: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_kv: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    layout_q: str = "TND",
    layout_kv: str = "PA_BBND",
    layout_sparse_pattern: int = 4,
    softmax_scale: float = 0.0,
    mask_mode: int = GBSAMaskMode.CAUSAL,
    quant_mode: int = GBSAQuantMode.NO_QUANT,
    dst_type_max: float = 0.0,
    softmax_precision: int = 1,
    win_left: int = -1,
    win_right: int = -1,
    return_softmax_lse: bool = False,
    residual_block_mode: int = 0,
    is_consistent_topk: bool = False,
    attention_out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GBSA using metadata generated for the same inputs and attributes."""
    return torch.ops._C_ascend.npu_generic_block_sparse_attention(
        q,
        k,
        v,
        sparse_block_idx,
        sparse_block_count,
        block_shape,
        metadata=metadata,
        attn_mask=attn_mask,
        q_dequant_scale=q_dequant_scale,
        k_dequant_scale=k_dequant_scale,
        v_dequant_scale=v_dequant_scale,
        p_quant_scale=p_quant_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        block_table=block_table,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_sparse_pattern=layout_sparse_pattern,
        softmax_scale=softmax_scale,
        mask_mode=int(mask_mode),
        quant_mode=int(quant_mode),
        dst_type_max=dst_type_max,
        softmax_precision=softmax_precision,
        win_left=win_left,
        win_right=win_right,
        return_softmax_lse=return_softmax_lse,
        residual_block_mode=residual_block_mode,
        is_consistent_topk=is_consistent_topk,
        attention_out_dtype=attention_out_dtype,
    )
