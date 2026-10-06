# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in this directory for the full text of the License.


"""DSV4.1 dispatcher contracts for the external A5 operator package.

Importing this module registers operators without importing the DSL compiler.
MQSMLA uses cannbot-arena-net-ops. QLI/QSLI use the local worker-count
fixes until the external package includes them. Compilation is deferred until
model warmup.
"""

import torch

torch.library.define(
    "vllm_ascend::quant_lightning_indexer_metadata",
    "(Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, "
    "Tensor? cmp_residual_k=None, *, int? batch_size=None, int max_seqlen_q=-1, int max_seqlen_k=-1, int "
    "num_heads_q, int num_heads_k, int head_dim, int topk, int mask_mode=0, int cmp_ratio=1, str "
    'layout_q="TND", str layout_k="TND", int candidate_topk_blocks=-1, int candidate_block_size=-1) -> Tensor',
)


torch.library.define(
    "vllm_ascend::quant_lightning_indexer",
    "(Tensor q, Tensor k, Tensor w, Tensor descale_q, Tensor descale_k, int topk, int quant_mode, *, Tensor? "
    "cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? "
    "cmp_residual_k=None, Tensor? block_table=None, Tensor? output_idx_offset=None, Tensor? metadata=None, int"
    ' max_seqlen_q=-1, int mask_mode=0, int cmp_ratio=1, str layout_q="TND", str layout_k="TND", bool '
    "return_value=False, int candidate_topk_blocks=-1, int candidate_block_size=-1) -> (Tensor, Tensor, "
    "Tensor, Tensor)",
)


@torch.library.register_fake("vllm_ascend::quant_lightning_indexer_metadata")
def _quant_lightning_indexer_metadata_fake(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    if layout_q != "TND" or layout_k not in ("PA_BBND", "TND"):
        raise ValueError("layout_q must be TND and layout_k must be PA_BBND or TND")
    device = next(
        (t.device for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k) if t is not None), None
    )
    if device is None:
        device = torch.device("npu", torch.npu.current_device())
    return torch.empty((1024,), dtype=torch.int32, device=device)


@torch.library.register_fake("vllm_ascend::quant_lightning_indexer")
def _quant_lightning_indexer_fake(
    q,
    k,
    w,
    descale_q,
    descale_k,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    return_value=False,
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    if layout_q != "TND" or layout_k not in ("PA_BBND", "TND"):
        raise ValueError("layout_q must be TND and layout_k must be PA_BBND or TND")
    heads = k.shape[1] if layout_k == "TND" else k.shape[2]
    output_shape = (q.shape[0], heads, topk)
    sparse_indices = torch.empty(output_shape, dtype=torch.int32, device=q.device)
    sparse_values_shape = output_shape if return_value else (0,)
    sparse_values = torch.empty(sparse_values_shape, dtype=torch.bfloat16, device=q.device)
    if candidate_topk_blocks > 0:
        candidate_indices_shape = (q.shape[0], heads, candidate_topk_blocks)
        candidate_length_shape = (q.shape[0], heads)
    else:
        candidate_indices_shape = (0,)
        candidate_length_shape = (0,)
    candidate_block_indices = torch.empty(candidate_indices_shape, dtype=torch.int32, device=q.device)
    candidate_block_length = torch.empty(candidate_length_shape, dtype=torch.int32, device=q.device)
    return (sparse_indices, sparse_values, candidate_block_indices, candidate_block_length)


@torch.library.impl("vllm_ascend::quant_lightning_indexer", "PrivateUse1")
def _quant_lightning_indexer_impl(
    q,
    k,
    w,
    descale_q,
    descale_k,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    return_value=False,
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    """Run the CANNBotDSL QLI kernel through the Torch NPU dispatcher."""
    from vllm_ascend.ops.pythondsl.quant_lightning_indexer_dsl import (
        quant_lightning_indexer as dsl_quant_lightning_indexer,
    )

    return dsl_quant_lightning_indexer(
        q,
        k,
        w,
        descale_q,
        descale_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        topk=topk,
        quant_mode=quant_mode,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        return_value=return_value,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


@torch.library.impl("vllm_ascend::quant_lightning_indexer_metadata", "CompositeExplicitAutograd")
def _quant_lightning_indexer_metadata_impl(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    from vllm_ascend.ops.pythondsl.quant_lightning_indexer_metadata_dsl import (
        quant_lightning_indexer_metadata as dsl_metadata,
    )

    return dsl_metadata(
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


torch.library.define(
    "vllm_ascend::quant_sparse_lightning_indexer_metadata",
    "(Tensor candidate_block_length, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? "
    "seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, *, int? batch_size=None, int "
    "max_seqlen_q=-1, int max_seqlen_k=-1, int num_heads_q, int num_heads_k, int head_dim, int topk, int "
    'quant_mode, int candidate_block_size, int mask_mode=0, int cmp_ratio=1, str layout_q="TND", str '
    'layout_k="TND") -> Tensor',
)


torch.library.define(
    "vllm_ascend::quant_sparse_lightning_indexer",
    "(Tensor q, Tensor k, Tensor w, Tensor descale_q, Tensor candidate_block_indices, Tensor "
    "candidate_block_length, int topk, int quant_mode, int candidate_block_size, *, Tensor? descale_k=None, "
    "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, "
    "Tensor? cmp_residual_k=None, Tensor? block_table=None, Tensor? output_idx_offset=None, Tensor? "
    'metadata=None, int max_seqlen_q=-1, int mask_mode=0, int cmp_ratio=1, str layout_q="TND", str '
    'layout_k="TND", bool return_value=False) -> (Tensor, Tensor)',
)


@torch.library.register_fake("vllm_ascend::quant_sparse_lightning_indexer_metadata")
def _quant_sparse_lightning_indexer_metadata_fake(
    candidate_block_length,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    quant_mode,
    candidate_block_size,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
):
    if layout_q != "TND" or layout_k not in ("PA_BBND", "TND"):
        raise ValueError("layout_q must be TND and layout_k must be PA_BBND or TND")
    device = candidate_block_length.device
    return torch.empty((1024,), dtype=torch.int32, device=device)


@torch.library.register_fake("vllm_ascend::quant_sparse_lightning_indexer")
def _quant_sparse_lightning_indexer_fake(
    q,
    k,
    w,
    descale_q,
    candidate_block_indices,
    candidate_block_length,
    topk,
    quant_mode,
    candidate_block_size,
    *,
    descale_k=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    return_value=False,
):
    if layout_q != "TND" or layout_k not in ("PA_BBND", "TND"):
        raise ValueError("layout_q must be TND and layout_k must be PA_BBND or TND")
    output_shape = (q.shape[0], candidate_block_indices.shape[1], topk)
    sparse_indices = torch.empty(output_shape, dtype=torch.int32, device=q.device)
    sparse_values_shape = output_shape if return_value else (0,)
    sparse_values = torch.empty(sparse_values_shape, dtype=torch.bfloat16, device=q.device)
    return (sparse_indices, sparse_values)


@torch.library.impl("vllm_ascend::quant_sparse_lightning_indexer", "PrivateUse1")
def _quant_sparse_lightning_indexer_impl(
    q,
    k,
    w,
    descale_q,
    candidate_block_indices,
    candidate_block_length,
    topk,
    quant_mode,
    candidate_block_size,
    *,
    descale_k=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    return_value=False,
):
    """Run the CANNBotDSL QSLI kernel through the Torch NPU dispatcher."""
    from vllm_ascend.ops.pythondsl.quant_sparse_lightning_indexer_dsl import (
        quant_sparse_lightning_indexer as dsl_quant_sparse_lightning_indexer,
    )

    return dsl_quant_sparse_lightning_indexer(
        q,
        k,
        w,
        descale_q,
        candidate_block_indices,
        candidate_block_length,
        descale_k=descale_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        topk=topk,
        quant_mode=quant_mode,
        candidate_block_size=candidate_block_size,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        return_value=return_value,
    )


@torch.library.impl("vllm_ascend::quant_sparse_lightning_indexer_metadata", "CompositeExplicitAutograd")
def _quant_sparse_lightning_indexer_metadata_impl(
    candidate_block_length,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    quant_mode,
    candidate_block_size,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
):
    from vllm_ascend.ops.pythondsl.quant_sparse_lightning_indexer_metadata_dsl import (
        quant_sparse_lightning_indexer_metadata as dsl_metadata,
    )

    return dsl_metadata(
        candidate_block_length,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        quant_mode=quant_mode,
        candidate_block_size=candidate_block_size,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
    )


torch.library.define(
    "vllm_ascend::mixed_quant_sparse_flash_mla_metadata",
    "(Tensor ori_topk_length, Tensor cmp_topk_length, *, Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, "
    "Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, int? batch_size=None, int? max_seqlen_q=None, "
    "int? max_seqlen_ori_kv=None, int? max_seqlen_cmp_kv=None, int num_heads_q, int num_heads_kv, int "
    'head_dim, int quant_mode, str layout_q="TND", str layout_kv="PA_BBND", bool has_ori_kv=True, bool '
    "has_cmp_kv=True) -> Tensor",
)


torch.library.define(
    "vllm_ascend::mixed_quant_sparse_flash_mla",
    "(Tensor q, *, Tensor? ori_kv=None, Tensor? cmp_kv=None, Tensor? ori_sparse_indices=None, Tensor? "
    "cmp_sparse_indices=None, Tensor? ori_block_table=None, Tensor? cmp_block_table=None, Tensor? "
    "cu_seqlens_q=None, Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, "
    "Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, Tensor? sinks=None, Tensor? metadata=None, "
    'int quant_mode, float? softmax_scale=None, str layout_q="TND", str layout_kv="PA_BBND", bool '
    "return_softmax_lse=False) -> (Tensor, Tensor)",
)


@torch.library.register_fake("vllm_ascend::mixed_quant_sparse_flash_mla_metadata")
def _mixed_quant_sparse_flash_mla_metadata_fake(
    ori_topk_length: torch.Tensor,
    cmp_topk_length: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_ori_kv: torch.Tensor | None = None,
    seqused_cmp_kv: torch.Tensor | None = None,
    batch_size: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_ori_kv: int | None = None,
    max_seqlen_cmp_kv: int | None = None,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: int,
    layout_q: str = "TND",
    layout_kv: str = "PA_BBND",
    has_ori_kv: bool = True,
    has_cmp_kv: bool = True,
):
    return torch.empty((1024,), dtype=torch.int32, device=ori_topk_length.device)


@torch.library.register_fake("vllm_ascend::mixed_quant_sparse_flash_mla")
def _mixed_quant_sparse_flash_mla_fake(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
):
    attn_out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    if return_softmax_lse:
        n2 = ori_kv.shape[2]
        if layout_q == "TND":
            lse_shape = (n2, q.shape[0], q.shape[1] // n2)
        else:
            lse_shape = (q.shape[0], n2, q.shape[1], q.shape[2] // n2)
    else:
        lse_shape = (0,)
    softmax_lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    return (attn_out, softmax_lse)


@torch.library.impl("vllm_ascend::mixed_quant_sparse_flash_mla_metadata", "CompositeExplicitAutograd")
def _mixed_quant_sparse_flash_mla_metadata_impl(
    ori_topk_length: torch.Tensor,
    cmp_topk_length: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_ori_kv: torch.Tensor | None = None,
    seqused_cmp_kv: torch.Tensor | None = None,
    batch_size: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_ori_kv: int | None = None,
    max_seqlen_cmp_kv: int | None = None,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: int,
    layout_q: str = "TND",
    layout_kv: str = "PA_BBND",
    has_ori_kv: bool = True,
    has_cmp_kv: bool = True,
):
    """Generate the core task table through the in-tree DSL metadata kernel."""
    from ops.mixed_quant_sparse_flash_mla_metadata import (
        mixed_quant_sparse_flash_mla_metadata as dsl_metadata,
    )

    return dsl_metadata(
        ori_topk_length,
        cmp_topk_length,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        quant_mode=quant_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
    )


@torch.library.impl("vllm_ascend::mixed_quant_sparse_flash_mla", "PrivateUse1")
def _mixed_quant_sparse_flash_mla_impl(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
):
    """Allocate outputs and pass the original inputs to the in-tree DSL."""
    from ops.mixed_quant_sparse_flash_mla import mixed_quant_sparse_flash_mla as dsl_attention

    attn_out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    if return_softmax_lse:
        n2 = ori_kv.shape[2]
        if layout_q == "TND":
            lse_shape = (n2, q.shape[0], q.shape[1] // n2)
        else:
            lse_shape = (q.shape[0], n2, q.shape[1], q.shape[2] // n2)
    else:
        lse_shape = (0,)
    softmax_lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    dsl_attention(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        quant_mode=quant_mode,
        softmax_scale=softmax_scale,
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_softmax_lse=return_softmax_lse,
        out=attn_out,
        lse=softmax_lse,
    )
    return (attn_out, softmax_lse)


quant_lightning_indexer_metadata = torch.ops.vllm_ascend.quant_lightning_indexer_metadata.default


quant_lightning_indexer = torch.ops.vllm_ascend.quant_lightning_indexer.default


quant_sparse_lightning_indexer_metadata = torch.ops.vllm_ascend.quant_sparse_lightning_indexer_metadata.default


quant_sparse_lightning_indexer = torch.ops.vllm_ascend.quant_sparse_lightning_indexer.default


mixed_quant_sparse_flash_mla_metadata = torch.ops.vllm_ascend.mixed_quant_sparse_flash_mla_metadata.default


mixed_quant_sparse_flash_mla = torch.ops.vllm_ascend.mixed_quant_sparse_flash_mla.default


__all__ = [
    "quant_lightning_indexer_metadata",
    "quant_lightning_indexer",
    "quant_sparse_lightning_indexer_metadata",
    "quant_sparse_lightning_indexer",
    "mixed_quant_sparse_flash_mla_metadata",
    "mixed_quant_sparse_flash_mla",
]
