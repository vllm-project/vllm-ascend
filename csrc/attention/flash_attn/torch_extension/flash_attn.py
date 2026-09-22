# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

FA_METADATA_OP_NAME = "flash_attn_metadata"
METADATA_STRIDE = 16
AIC_NUM = torch.npu.get_device_properties().cube_core_num
AIV_NUM = torch.npu.get_device_properties().vector_core_num
FAG_METADATA_SIZE = 121  # FAG (Flash Attn Grad) metadata size, in int32 elements


def _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q):
    if seqused_q is not None:
        return seqused_q.size(0)
    elif cu_seqlens_q is not None and cu_seqlens_q.size(0) > 0:
        return cu_seqlens_q.size(0) - 1

    if batch_size is None:
        raise ValueError(
            "When seqused_q and cu_seqlens_q are not provided, batch_size must be provided"
        )
    elif batch_size < 0:
        raise ValueError(
            f"When seqused_q and cu_seqlens_q are not provided, "
            f"batch_size must be greater than 0, but got {batch_size}"
        )
    return batch_size


def _calculate_metadata_size(batch_size, num_heads_kv):
    """计算 metadata tensor 的对齐后大小"""
    align_size = 4096
    head_size = 1 * METADATA_STRIDE
    fa_size = AIC_NUM * METADATA_STRIDE * batch_size * num_heads_kv
    fd_size = AIV_NUM * METADATA_STRIDE * batch_size * num_heads_kv

    metadata_size = head_size + fa_size + fd_size
    metadata_size += FAG_METADATA_SIZE  # FAG (Flash Attn Grad) metadata size
    return ((metadata_size + align_size - 1) // align_size) * align_size


class FlashAttenOpBuilder(OpBuilder):
    def __init__(self):
        super(FlashAttenOpBuilder, self).__init__("flash_attn", category="attention")

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/flash_attn.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "flash_attn_metadata(int num_heads_q, int num_heads_kv, int head_dim, *, "
            "int? head_dim_v=None, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None,"
            "int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_kv=None, "
            "int? mask_mode=None, int? win_left=None, int? win_right=None, "
            "str? layout_q=None, str? layout_kv=None, str? layout_out=None, "
            "bool is_grad_enabled=False) -> Tensor",
            "flash_attn(Tensor q, Tensor k, Tensor v,"
            "Tensor?block_table=None, Tensor?cu_seqlens_q=None,"
            "Tensor?cu_seqlens_kv=None, Tensor?seqused_q=None,"
            "Tensor?seqused_kv=None, Tensor?sinks=None, Tensor?attn_mask=None, Tensor?metadata=None,"
            "float softmax_scale=1.0, int mask_mode=0, int win_left=-1, int win_right=-1,"
            "int max_seqlen_q=-1, int max_seqlen_kv=-1,"
            'str layout_q="BSND", str layout_kv="BSND", str layout_out="BSND",'
            "bool return_softmax_lse=False) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::flash_attn_metadata")
        def flash_attn_metadata_meta(
            num_heads_q: int,
            num_heads_kv: int,
            head_dim: int,
            head_dim_v: Optional[int] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = -1,
            max_seqlen_q: Optional[int] = None,
            max_seqlen_kv: Optional[int] = None,
            mask_mode: Optional[int] = None,
            win_left: Optional[int] = None,
            win_right: Optional[int] = None,
            layout_q: Optional[str] = None,
            layout_kv: Optional[str] = None,
            layout_out: Optional[str] = None,
            is_grad_enabled: Optional[bool] = False,
        ):
            b_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q)
            metadata_size = _calculate_metadata_size(b_size, num_heads_kv)
            return torch.empty((metadata_size,), dtype=torch.int32, device="meta")

        @torch.library.register_fake("cann_ops_transformer::flash_attn")
        def flash_attn_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            block_table: Optional[torch.Tensor] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            sinks: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            softmax_scale: Optional[float] = 1.0,
            mask_mode: Optional[int] = 0,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_kv: Optional[str] = "BSND",
            layout_out: Optional[str] = "BSND",
            return_softmax_lse: Optional[bool] = False,
        ):
            if layout_q == "TND":
                t_size = q.size(0)
                n_size = q.size(1)
                softmax_out_size = (n_size, t_size)
            elif layout_q == "BSND":
                b_size = q.size(0)
                s_size = q.size(1)
                n_size = q.size(2)
                softmax_out_size = (b_size, n_size, s_size)
            else:
                b_size = q.size(0)
                n_size = q.size(1)
                s_size = q.size(2)
                softmax_out_size = (b_size, n_size, s_size)

            if layout_kv == "PA_NZ":
                d_size = v.size(-3) * v.size(-1)
            else:
                d_size = v.size(-1)

            if layout_out == "TND":
                torch._check(
                    layout_q == "TND",
                    lambda: f"When the layout of output is TND, the layout of query must be TND, but got {layout_q}",
                )
                attention_out_size = (t_size, n_size, d_size)
            elif layout_out == "BNSD":
                torch._check(
                    layout_q == "BNSD",
                    lambda: f"When the layout of output is BNSD, the layout of query must be BNSD, but got {layout_q}",
                )
                attention_out_size = (b_size, n_size, s_size, d_size)
            else:
                torch._check(
                    layout_q != "TND",
                    lambda: f"When the layout of output is BSND, the layout of query must be BNSD or BSND, but got {layout_q}",
                )
                attention_out_size = (b_size, s_size, n_size, d_size)

            return (
                torch.empty(attention_out_size, dtype=q.dtype, device=q.device),
                torch.empty(softmax_out_size, dtype=torch.float, device=q.device),
            )


# Instantiate the builder
flash_attn_op_builder = FlashAttenOpBuilder()
flash_attn_op_builder._ensure_initialized()


@impl(get_as_library(), FA_METADATA_OP_NAME, "PrivateUse1")
def flash_attn_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    head_dim_v: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = -1,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    mask_mode: Optional[int] = 0,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
    layout_out: Optional[str] = "BSND",
    is_grad_enabled: Optional[bool] = False,
):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    b_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q)

    batch_size = -1 if batch_size is None else batch_size
    head_dim_v = -1 if head_dim_v is None else head_dim_v
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    mask_mode = 1 if mask_mode is None else mask_mode
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "BSND" if layout_kv is None else layout_kv
    layout_out = "BSND" if layout_out is None else layout_out

    op_module = flash_attn_op_builder.load()
    metadata_size = _calculate_metadata_size(b_size, num_heads_kv)
    output = torch.empty((metadata_size,), dtype=torch.int32, device="npu")

    return op_module.flash_attn_metadata(
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        head_dim_v,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        mask_mode,
        win_left,
        win_right,
        layout_q,
        layout_kv,
        layout_out,
        is_grad_enabled,
        output,
    )


_flash_attn_metadata = flash_attn_metadata


@torch.library.register_kernel("cann_ops_transformer::" + FA_METADATA_OP_NAME, None)
def flash_attn_metadata_fallback(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    head_dim_v: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    mask_mode: Optional[int] = None,
    win_left: Optional[int] = None,
    win_right: Optional[int] = None,
    layout_q: Optional[str] = None,
    layout_kv: Optional[str] = None,
    layout_out: Optional[str] = None,
    is_grad_enabled: Optional[bool] = False,
):
    # 处理所有 tensor 都为 None 的情况
    return _flash_attn_metadata(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        is_grad_enabled=is_grad_enabled,
    )


@impl(get_as_library(), flash_attn_op_builder.name, "PrivateUse1")
def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = 1.0,
    mask_mode: Optional[int] = 0,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
    layout_out: Optional[str] = "BSND",
    return_softmax_lse: Optional[bool] = False,
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    op_module = flash_attn_op_builder.load()
    return op_module.flash_attn(
        q,
        k,
        v,
        block_table,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
        return_softmax_lse,
    )


flash_attn = torch.ops.cann_ops_transformer.flash_attn
flash_attn_metadata = torch.ops.cann_ops_transformer.flash_attn_metadata
