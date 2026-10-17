# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from enum import IntEnum
from typing import List, Optional, Union
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

GBSA_METADATA_SIZE = 1024
GBSA_METADATA_OP_NAME = "generic_block_sparse_attention_metadata"


class MaskMode(IntEnum):
    """mask_mode 取值，与 aclnn maskType 一致。可直接当 int 传入。"""

    NO_MASK = 0
    CAUSAL = 1
    WINDOW = 2


class QuantMode(IntEnum):
    """quant_mode 取值，与 aclnn quantType 一致。可直接当 int 传入。"""

    NO_QUANT = 0
    FP8_E4M3_STATIC_PER_GROUP = 1
    FP8_E4M3_DYNAMIC_MX = 2
    FP4_E2M1_DYNAMIC_OCP = 3
    FP4_E2M1_DYNAMIC_CX = 4
    FP8_E4M3_STATIC_CAST_P = 5


class GenericBlockSparseAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("generic_block_sparse_attention", category="attention")

    def sources(self):
        return ["csrc/attention/generic_block_sparse_attention.cpp"]

    def schema(self):
        return [
            "generic_block_sparse_attention_metadata("
            "Tensor sparse_block_idx, Tensor sparse_block_count, "
            "int num_heads_q, int num_heads_kv, int head_dim, "
            "int[] block_shape, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "int? max_seqlen_q=-1, int? max_seqlen_kv=-1, "
            'str? layout_q="TND", str? layout_kv="PA_BBND", int layout_sparse_pattern=4, '
            "int? mask_mode=1, int? quant_mode=0, int? softmax_precision=1, "
            "int? win_left=-1, int? win_right=-1, "
            "int residual_block_mode=0, bool is_consistent_topk=False) -> Tensor",
            "generic_block_sparse_attention("
            "Tensor q, Tensor k, Tensor v, "
            "Tensor sparse_block_idx, Tensor sparse_block_count, "
            "int[] block_shape, *, "
            "Tensor? metadata=None, "
            "Tensor? attn_mask=None, "
            "Tensor? q_dequant_scale=None, "
            "Tensor? k_dequant_scale=None, "
            "Tensor? v_dequant_scale=None, "
            "Tensor? p_quant_scale=None, "
            "Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_kv=None, "
            "Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "Tensor? block_table=None, "
            'str layout_q="TND", '
            'str layout_kv="PA_BBND", '
            "int layout_sparse_pattern=4, "
            "float softmax_scale=0.0, "
            "int mask_mode=1, "
            "int quant_mode=0, "
            "float dst_type_max=0.0, "
            "int softmax_precision=1, "
            "int win_left=-1, "
            "int win_right=-1, "
            "bool return_softmax_lse=False, "
            "int residual_block_mode=0, "
            "bool is_consistent_topk=False, "
            "ScalarType? attention_out_dtype=None"
            ") -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        @torch.library.register_fake("cann_ops_transformer::" + GBSA_METADATA_OP_NAME)
        def generic_block_sparse_attention_metadata_meta(
            sparse_block_idx: torch.Tensor,
            sparse_block_count: torch.Tensor,
            num_heads_q: int,
            num_heads_kv: int,
            head_dim: int,
            block_shape: List[int],
            *,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "TND",
            layout_kv: Optional[str] = "PA_BBND",
            layout_sparse_pattern: Optional[int] = 4,
            mask_mode: Optional[Union[MaskMode, int]] = MaskMode.CAUSAL,
            quant_mode: Optional[Union[QuantMode, int]] = QuantMode.NO_QUANT,
            softmax_precision: Optional[int] = 1,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            residual_block_mode: Optional[int] = 0,
            is_consistent_topk: Optional[bool] = False,
        ):
            return torch.empty((GBSA_METADATA_SIZE,), dtype=torch.int32, device="meta")

        @impl(get_as_library(), self.name, "Meta")
        def generic_block_sparse_attention_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            sparse_block_idx: torch.Tensor,
            sparse_block_count: torch.Tensor,
            block_shape: List[int],
            *,
            metadata: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            q_dequant_scale: Optional[torch.Tensor] = None,
            k_dequant_scale: Optional[torch.Tensor] = None,
            v_dequant_scale: Optional[torch.Tensor] = None,
            p_quant_scale: Optional[torch.Tensor] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            block_table: Optional[torch.Tensor] = None,
            layout_q: Optional[str] = "TND",
            layout_kv: Optional[str] = "PA_BBND",
            layout_sparse_pattern: Optional[int] = 4,
            softmax_scale: Optional[float] = 0.0,
            mask_mode: Optional[Union[MaskMode, int]] = MaskMode.CAUSAL,
            quant_mode: Optional[Union[QuantMode, int]] = QuantMode.NO_QUANT,
            dst_type_max: Optional[float] = 0.0,
            softmax_precision: Optional[int] = 1,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            return_softmax_lse: Optional[bool] = False,
            residual_block_mode: Optional[int] = 0,
            is_consistent_topk: Optional[bool] = False,
            attention_out_dtype: Optional[torch.dtype] = None,
        ):
            # 与 C++ 实现保持一致：quant_mode != NO_QUANT 时 attention_out_dtype 必填
            if attention_out_dtype is not None:
                out_dtype = attention_out_dtype
            elif quant_mode == QuantMode.NO_QUANT:
                out_dtype = q.dtype
            else:
                raise ValueError(
                    "attention_out_dtype must be specified when quant_mode != NO_QUANT"
                )

            attn = q.new_empty(q.shape, dtype=out_dtype)
            if return_softmax_lse:
                # 与 C++ GenericBlockSparseAttention LSE 分配一致
                if layout_q == "TND":
                    lse_shape = (q.size(0), q.size(1), 1)
                elif layout_q == "BNSD":
                    lse_shape = (q.size(0), q.size(1), q.size(2), 1)
                else:
                    # BSND
                    lse_shape = (q.size(0), q.size(2), q.size(1), 1)
                lse = q.new_empty(lse_shape, dtype=torch.float32)
            else:
                lse = q.new_empty((0,), dtype=torch.float32)
            return attn, lse


generic_block_sparse_attention_op_builder = GenericBlockSparseAttentionOpBuilder()
generic_block_sparse_attention_op_builder._ensure_initialized()


@impl(get_as_library(), GBSA_METADATA_OP_NAME, "PrivateUse1")
def generic_block_sparse_attention_metadata(
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    block_shape: List[int],
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "TND",
    layout_kv: Optional[str] = "PA_BBND",
    layout_sparse_pattern: Optional[int] = 4,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.CAUSAL,
    quant_mode: Optional[Union[QuantMode, int]] = QuantMode.NO_QUANT,
    softmax_precision: Optional[int] = 1,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    residual_block_mode: Optional[int] = 0,
    is_consistent_topk: Optional[bool] = False,
):
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    layout_sparse_pattern = (
        4 if layout_sparse_pattern is None else layout_sparse_pattern
    )
    layout_q = "TND" if layout_q is None else layout_q
    layout_kv = "PA_BBND" if layout_kv is None else layout_kv
    mask_mode = int(MaskMode.CAUSAL) if mask_mode is None else int(mask_mode)
    quant_mode = int(QuantMode.NO_QUANT) if quant_mode is None else int(quant_mode)
    softmax_precision = 1 if softmax_precision is None else softmax_precision
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    op_module = generic_block_sparse_attention_op_builder.load()
    output = torch.empty(
        (GBSA_METADATA_SIZE,),
        dtype=torch.int32,
        device=sparse_block_idx.device,
    )
    return op_module.generic_block_sparse_attention_metadata(
        sparse_block_idx,
        sparse_block_count,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        max_seqlen_q,
        max_seqlen_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        block_shape,
        layout_q,
        layout_kv,
        layout_sparse_pattern,
        mask_mode,
        quant_mode,
        softmax_precision,
        win_left,
        win_right,
        residual_block_mode,
        is_consistent_topk,
        output,
    )


_generic_block_sparse_attention_metadata = generic_block_sparse_attention_metadata


@torch.library.register_kernel("cann_ops_transformer::" + GBSA_METADATA_OP_NAME, None)
def generic_block_sparse_attention_metadata_fallback(
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    block_shape: List[int],
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "TND",
    layout_kv: Optional[str] = "PA_BBND",
    layout_sparse_pattern: Optional[int] = 4,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.CAUSAL,
    quant_mode: Optional[Union[QuantMode, int]] = QuantMode.NO_QUANT,
    softmax_precision: Optional[int] = 1,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    residual_block_mode: Optional[int] = 0,
    is_consistent_topk: Optional[bool] = False,
):
    return _generic_block_sparse_attention_metadata(
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
        layout_sparse_pattern=layout_sparse_pattern,
        layout_q=layout_q,
        layout_kv=layout_kv,
        mask_mode=mask_mode,
        quant_mode=quant_mode,
        softmax_precision=softmax_precision,
        win_left=win_left,
        win_right=win_right,
        residual_block_mode=residual_block_mode,
        is_consistent_topk=is_consistent_topk,
    )


@impl(get_as_library(), generic_block_sparse_attention_op_builder.name, "PrivateUse1")
def generic_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    block_shape: List[int],
    *,
    metadata: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    q_dequant_scale: Optional[torch.Tensor] = None,
    k_dequant_scale: Optional[torch.Tensor] = None,
    v_dequant_scale: Optional[torch.Tensor] = None,
    p_quant_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    layout_q: Optional[str] = "TND",
    layout_kv: Optional[str] = "PA_BBND",
    layout_sparse_pattern: Optional[int] = 4,
    softmax_scale: Optional[float] = 0.0,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.CAUSAL,
    quant_mode: Optional[Union[QuantMode, int]] = QuantMode.NO_QUANT,
    dst_type_max: Optional[float] = 0.0,
    softmax_precision: Optional[int] = 1,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    return_softmax_lse: Optional[bool] = False,
    residual_block_mode: Optional[int] = 0,
    is_consistent_topk: Optional[bool] = False,
    attention_out_dtype: Optional[torch.dtype] = None,
):
    op_module = generic_block_sparse_attention_op_builder.load()
    return op_module.generic_block_sparse_attention(
        q,
        k,
        v,
        sparse_block_idx,
        sparse_block_count,
        block_shape,
        metadata,
        attn_mask,
        q_dequant_scale,
        k_dequant_scale,
        v_dequant_scale,
        p_quant_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        block_table,
        layout_q,
        layout_kv,
        layout_sparse_pattern,
        softmax_scale,
        mask_mode,
        quant_mode,
        dst_type_max,
        softmax_precision,
        win_left,
        win_right,
        return_softmax_lse,
        residual_block_mode,
        is_consistent_topk,
        attention_out_dtype,
    )


generic_block_sparse_attention = (
    torch.ops.cann_ops_transformer.generic_block_sparse_attention
)
generic_block_sparse_attention_metadata = (
    torch.ops.cann_ops_transformer.generic_block_sparse_attention_metadata
)

generic_block_sparse_attention.MaskMode = MaskMode
generic_block_sparse_attention.QuantMode = QuantMode
generic_block_sparse_attention_metadata.MaskMode = MaskMode
generic_block_sparse_attention_metadata.QuantMode = QuantMode
