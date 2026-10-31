# SPDX-License-Identifier: Apache-2.0
"""Thin bindings for CANN LightningIndexerV2 and its scheduling metadata.

Requires the CANN transformer op-api library exposing both V2 entry points.
Import the vllm_ascend_C extension before calling these functions. This module
performs no QSA-specific parameter conversion and never copies the key cache.
"""

import torch


def lightning_indexer_v2_metadata(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    topk: int,
    *,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    cmp_residual_k: torch.Tensor | None = None,
    batch_size: int | None = 0,
    max_seqlen_q: int | None = -1,
    max_seqlen_k: int | None = -1,
    layout_q: str | None = "BSND",
    layout_k: str | None = "BSND",
    mask_mode: int | None = 0,
    cmp_ratio: int | None = 1,
) -> torch.Tensor:
    """Return 2048 INT32 scheduling entries on the metadata inputs' NPU.

    At least one NPU length tensor is required for device dispatch. TND query
    boundaries include the initial zero. Length tensors must be INT32.
    """
    if all(tensor is None for tensor in (
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k
    )):
        raise ValueError("LightningIndexerV2 metadata requires at least one NPU length tensor")
    # Match upstream Python defaults before entering the typed C++ schema.
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = -1 if max_seqlen_k is None else max_seqlen_k
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
    return torch.ops._C_ascend.npu_lightning_indexer_v2_metadata(
        num_heads_q, num_heads_k, head_dim, topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )


def lightning_indexer_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    topk: int,
    *,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    cmp_residual_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    output_idx_offset: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    max_seqlen_q: int = -1,
    layout_q: str = "BSND",
    layout_k: str = "BSND",
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    return_value: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-k key positions using FP16/BF16 Q/K and FP32 weights.

    Indices are INT32 with shape [T,Nk,topk] (TND) or [B,Sq,Nk,topk]
    (BSND). Values are FP32 with the same shape when return_value=1,
    otherwise an empty [0] tensor. Invalid entries are -1 / -inf.
    PA_BBND keys may have a noncontiguous page stride. Hardware-specific
    parameter constraints are enforced by CANN. No candidate/V3 mode is used.
    """
    return torch.ops._C_ascend.npu_lightning_indexer_v2(
        q, k, w, topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=max_seqlen_q,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        return_value=return_value,
    )
