#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import OpBuilder, get_as_library


class TurboQuantSparseAttnSharedkvOpBuilder(OpBuilder):
    def __init__(self):
        # Match the distributed torch-extension layout used by the base tree.
        super().__init__("turbo_quant_sparse_attn_sharedkv", category="attention")

    def sources(self):
        return ["csrc/attention/turbo_quant_sparse_attn_sharedkv.cpp"]

    def ensure_initialized(self):
        self._ensure_initialized()

    def schema(self):
        return (
            "turbo_quant_sparse_attn_sharedkv(Tensor q, *, "
            "Tensor? ori_kv=None, Tensor? cmp_kv=None, "
            "Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, "
            "Tensor? ori_block_table=None, Tensor? cmp_block_table=None, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, "
            "Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, Tensor? sinks=None, Tensor? metadata=None, "
            "float softmax_scale=1.0, int cmp_ratio=4, "
            "int ori_mask_mode=4, int cmp_mask_mode=3, "
            "int ori_win_left=127, int ori_win_right=0, "
            'str layout_q="TND", str layout_kv="PA_BSND", '
            "bool return_softmax_lse=False, int kv_quant_mode=3) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def turbo_quant_sparse_attn_sharedkv_meta(
            q,
            *,
            ori_kv=None,
            cmp_kv=None,
            ori_sparse_indices=None,
            cmp_sparse_indices=None,
            ori_block_table=None,
            cmp_block_table=None,
            cu_seqlens_q=None,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=None,
            sinks=None,
            metadata=None,
            softmax_scale=1.0,
            cmp_ratio=4,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BSND",
            return_softmax_lse=False,
            kv_quant_mode=3,
        ):
            if q.dim() != 3 or any(size <= 0 for size in q.shape[1:]):
                raise ValueError("query must be TND with positive N and D dimensions")

            attn_out = torch.empty(q.shape, dtype=q.dtype, device="meta")
            lse_shape = tuple(q.shape[:-1]) + (1,) if return_softmax_lse else (0,)
            softmax_lse = torch.empty(lse_shape, dtype=torch.float32, device="meta")
            return attn_out, softmax_lse


turbo_quant_sparse_attn_sharedkv_op_builder = TurboQuantSparseAttnSharedkvOpBuilder()
turbo_quant_sparse_attn_sharedkv_op_builder.ensure_initialized()


@impl(get_as_library(), turbo_quant_sparse_attn_sharedkv_op_builder.name, "PrivateUse1")
def turbo_quant_sparse_attn_sharedkv(
    q: torch.Tensor,
    *,
    ori_kv: Optional[torch.Tensor] = None,
    cmp_kv: Optional[torch.Tensor] = None,
    ori_sparse_indices: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    ori_block_table: Optional[torch.Tensor] = None,
    cmp_block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    cmp_ratio: int = 4,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "TND",
    layout_kv: str = "PA_BSND",
    return_softmax_lse: bool = False,
    kv_quant_mode: int = 3,
):
    op_module = turbo_quant_sparse_attn_sharedkv_op_builder.load()
    return op_module.turbo_quant_sparse_attn_sharedkv(
        q,
        ori_kv,
        cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_kv,
        sinks,
        metadata,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        return_softmax_lse,
        kv_quant_mode,
    )
