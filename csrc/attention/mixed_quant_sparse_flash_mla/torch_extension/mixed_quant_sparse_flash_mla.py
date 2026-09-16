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
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

MQSMLA_METADATA_SIZE = 1024
MQSMLA_METADATA_OP_NAME = "mixed_quant_sparse_flash_mla_metadata"


class MixedQuantSparseFlashMlaOpBuilder(OpBuilder):
    def __init__(self):
        super(MixedQuantSparseFlashMlaOpBuilder, self).__init__(
            "mixed_quant_sparse_flash_mla", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/mixed_quant_sparse_flash_mla.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "mixed_quant_sparse_flash_mla_metadata(int num_heads_q, int num_heads_kv, int head_dim,"
            "int quant_mode, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None,"
            "Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_ori_kv=None,"
            "Tensor? seqused_cmp_kv=None, Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None,"
            "Tensor? cmp_topk_length=None, int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_ori_kv=None,"
            "int? max_seqlen_cmp_kv=None, int? ori_topk=None, int? cmp_topk=None, int? rope_head_dim=None,"
            "int? cmp_ratio=None, int? ori_mask_mode=None, int? cmp_mask_mode=None, int? ori_win_left=None,"
            "int? ori_win_right=None, str? layout_q=None, str? layout_kv=None, bool? has_ori_kv=None,"
            "bool? has_cmp_kv=None) -> Tensor",
            "mixed_quant_sparse_flash_mla(Tensor q, *,"
            "Tensor? ori_kv=None, Tensor? cmp_kv=None, "
            "Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, "
            "Tensor? ori_block_table=None, Tensor? cmp_block_table=None, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, "
            "Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, "
            "Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, "
            "Tensor? cmp_residual_kv=None, "
            "Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, "
            "Tensor? sinks=None, Tensor? metadata=None, "
            "int quant_mode=None, int rope_head_dim=None, "
            "float softmax_scale=None, int cmp_ratio=1, "
            "int ori_mask_mode=0, int cmp_mask_mode=0, "
            "int ori_win_left=-1, int ori_win_right=-1, "
            'str layout_q="BSND", str layout_kv="BSND", '
            "int topk_value_mode=1, bool return_softmax_lse=False, "
            "int? key_dtype=None, int? value_dtype=None) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::" + MQSMLA_METADATA_OP_NAME)
        def mixed_quant_sparse_flash_mla_metadata_meta(
            num_heads_q: int,
            num_heads_kv: int,
            head_dim: int,
            quant_mode: int,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
            cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_ori_kv: Optional[torch.Tensor] = None,
            seqused_cmp_kv: Optional[torch.Tensor] = None,
            cmp_residual_kv: Optional[torch.Tensor] = None,
            ori_topk_length: Optional[torch.Tensor] = None,
            cmp_topk_length: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
            max_seqlen_q: Optional[int] = None,
            max_seqlen_ori_kv: Optional[int] = None,
            max_seqlen_cmp_kv: Optional[int] = None,
            ori_topk: Optional[int] = None,
            cmp_topk: Optional[int] = None,
            rope_head_dim: Optional[int] = None,
            cmp_ratio: Optional[int] = None,
            ori_mask_mode: Optional[int] = None,
            cmp_mask_mode: Optional[int] = None,
            ori_win_left: Optional[int] = None,
            ori_win_right: Optional[int] = None,
            layout_q: Optional[str] = None,
            layout_kv: Optional[str] = None,
            has_ori_kv: Optional[bool] = None,
            has_cmp_kv: Optional[bool] = None,
        ):
            return torch.empty((MQSMLA_METADATA_SIZE), dtype=torch.int32, device="npu")

        @impl(get_as_library(), self.name, "Meta")
        def mixed_quant_sparse_flash_mla_meta(
            q,
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
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            cmp_residual_kv=None,
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=None,
            metadata=None,
            quant_mode=None,
            rope_head_dim=None,
            softmax_scale=None,
            cmp_ratio=1,
            ori_mask_mode=0,
            cmp_mask_mode=0,
            ori_win_left=-1,
            ori_win_right=-1,
            layout_q="BSND",
            layout_kv="BSND",
            topk_value_mode=1,
            return_softmax_lse=False,
            key_dtype=None,
            value_dtype=None,
        ):
            if q.numel() == 0:
                raise ValueError("The shape size of q should not be 0")
            if ori_kv is not None and ori_kv.numel() == 0:
                raise ValueError("The shape size of ori_kv should not be 0")
            attn_out = torch.empty(q.shape, dtype=q.dtype, device="meta")
            if return_softmax_lse:
                if layout_q == "TND":
                    if layout_kv == "PA_BBND":
                        softmax_lse = torch.empty(
                            [
                                ori_kv.shape[2],
                                q.shape[0],
                                int(q.shape[1] / ori_kv.shape[2]),
                            ],
                            dtype=torch.float32,
                            device="meta",
                        )
                    else:
                        softmax_lse = torch.empty(
                            [
                                ori_kv.shape[1],
                                q.shape[0],
                                int(q.shape[1] / ori_kv.shape[1]),
                            ],
                            dtype=torch.float32,
                            device="meta",
                        )
                else:
                    softmax_lse = torch.empty(
                        [
                            q.shape[0],
                            ori_kv.shape[2],
                            q.shape[1],
                            int(q.shape[2] / ori_kv.shape[2]),
                        ],
                        dtype=torch.float32,
                        device="meta",
                    )
            else:
                softmax_lse = torch.empty([], dtype=torch.float32, device="meta")
            return (attn_out, softmax_lse)


# Instantiate the builder
mixed_quant_sparse_flash_mla_op_builder = MixedQuantSparseFlashMlaOpBuilder()
mixed_quant_sparse_flash_mla_op_builder._ensure_initialized()


@impl(get_as_library(), MQSMLA_METADATA_OP_NAME, "PrivateUse1")
def mixed_quant_sparse_flash_mla_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_ori_kv: Optional[torch.Tensor] = None,
    seqused_cmp_kv: Optional[torch.Tensor] = None,
    cmp_residual_kv: Optional[torch.Tensor] = None,
    ori_topk_length: Optional[torch.Tensor] = None,
    cmp_topk_length: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_ori_kv: Optional[int] = None,
    max_seqlen_cmp_kv: Optional[int] = None,
    ori_topk: Optional[int] = None,
    cmp_topk: Optional[int] = None,
    rope_head_dim: Optional[int] = None,
    cmp_ratio: Optional[int] = None,
    ori_mask_mode: Optional[int] = None,
    cmp_mask_mode: Optional[int] = None,
    ori_win_left: Optional[int] = None,
    ori_win_right: Optional[int] = None,
    layout_q: Optional[str] = None,
    layout_kv: Optional[str] = None,
    has_ori_kv: Optional[bool] = None,
    has_cmp_kv: Optional[bool] = None,
):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = 0 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_ori_kv = 0 if max_seqlen_ori_kv is None else max_seqlen_ori_kv
    max_seqlen_cmp_kv = 0 if max_seqlen_cmp_kv is None else max_seqlen_cmp_kv
    ori_topk = 0 if ori_topk is None else ori_topk
    cmp_topk = 0 if cmp_topk is None else cmp_topk
    rope_head_dim = 64 if rope_head_dim is None else rope_head_dim
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
    ori_mask_mode = 0 if ori_mask_mode is None else ori_mask_mode
    cmp_mask_mode = 0 if cmp_mask_mode is None else cmp_mask_mode
    ori_win_left = -1 if ori_win_left is None else ori_win_left
    ori_win_right = -1 if ori_win_right is None else ori_win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "BSND" if layout_kv is None else layout_kv
    has_ori_kv = True if has_ori_kv is None else has_ori_kv
    has_cmp_kv = True if has_cmp_kv is None else has_cmp_kv

    op_module = mixed_quant_sparse_flash_mla_op_builder.load()
    return op_module.mixed_quant_sparse_flash_mla_metadata(
        num_heads_q,
        num_heads_kv,
        head_dim,
        quant_mode,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        batch_size,
        max_seqlen_q,
        max_seqlen_ori_kv,
        max_seqlen_cmp_kv,
        ori_topk,
        cmp_topk,
        rope_head_dim,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        has_ori_kv,
        has_cmp_kv,
    )


@torch.library.register_kernel("cann_ops_transformer::" + MQSMLA_METADATA_OP_NAME, None)
def mixed_quant_sparse_flash_mla_metadata_fallback(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_ori_kv: Optional[torch.Tensor] = None,
    seqused_cmp_kv: Optional[torch.Tensor] = None,
    cmp_residual_kv: Optional[torch.Tensor] = None,
    ori_topk_length: Optional[torch.Tensor] = None,
    cmp_topk_length: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_ori_kv: Optional[int] = None,
    max_seqlen_cmp_kv: Optional[int] = None,
    ori_topk: Optional[int] = None,
    cmp_topk: Optional[int] = None,
    rope_head_dim: Optional[int] = None,
    cmp_ratio: Optional[int] = None,
    ori_mask_mode: Optional[int] = None,
    cmp_mask_mode: Optional[int] = None,
    ori_win_left: Optional[int] = None,
    ori_win_right: Optional[int] = None,
    layout_q: Optional[str] = None,
    layout_kv: Optional[str] = None,
    has_ori_kv: Optional[bool] = None,
    has_cmp_kv: Optional[bool] = None,
):
    # 处理所有 tensor 都为 None 的情况
    # 调用 NPU 实现
    return mixed_quant_sparse_flash_mla_metadata(
        num_heads_q,
        num_heads_kv,
        head_dim,
        quant_mode,
        cu_seqlens_q,
        cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv,
        seqused_q,
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        batch_size,
        max_seqlen_q,
        max_seqlen_ori_kv,
        max_seqlen_cmp_kv,
        ori_topk,
        cmp_topk,
        rope_head_dim,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        has_ori_kv,
        has_cmp_kv,
    )


torch.compiler.allow_in_graph(mixed_quant_sparse_flash_mla_metadata)


@impl(get_as_library(), mixed_quant_sparse_flash_mla_op_builder.name, "PrivateUse1")
def mixed_quant_sparse_flash_mla(
    q,
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
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode=None,
    rope_head_dim=None,
    softmax_scale=None,
    cmp_ratio=1,
    ori_mask_mode=0,
    cmp_mask_mode=0,
    ori_win_left=-1,
    ori_win_right=-1,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
    key_dtype=None,
    value_dtype=None,
):
    """
    dispatcher implementation for NPU
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    op_module = mixed_quant_sparse_flash_mla_op_builder.load()
    return op_module.mixed_quant_sparse_flash_mla(
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
        seqused_ori_kv,
        seqused_cmp_kv,
        cmp_residual_kv,
        ori_topk_length,
        cmp_topk_length,
        sinks,
        metadata,
        quant_mode,
        rope_head_dim,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
        topk_value_mode,
        return_softmax_lse,
        key_dtype,
        value_dtype,
    )
