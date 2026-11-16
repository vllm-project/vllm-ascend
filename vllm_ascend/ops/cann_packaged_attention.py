# SPDX-License-Identifier: Apache-2.0
"""CANN ds41 ABI adapters; Torch dispatcher schemas live in packaged_attention."""

from functools import cache
from importlib import import_module

import torch


@cache
def has_ds41_ops() -> bool:
    try:
        import_module("cann_ops_transformer.ops.ds41")
    except ModuleNotFoundError as exc:
        if exc.name not in {"cann_ops_transformer", "cann_ops_transformer.ops", "cann_ops_transformer.ops.ds41"}:
            raise
        return False
    return True


@cache
def _get_ds41_compute(name: str):
    """Load a compute op from the installed cann_ops_transformer ds41 package."""
    module = import_module("cann_ops_transformer.ops.ds41")
    return getattr(module, name)


@cache
def _get_ds41_smla_metadata():
    """The SMLA scheduling op lives in the DSL attention module."""
    module = import_module(
        "cann_ops_transformer.ops.attention.mixed_quant_sparse_flash_mla_dsl.mixed_quant_sparse_flash_mla"
    )
    return module.mixed_quant_sparse_flash_mla_metadata


def _present(kwargs: dict) -> dict:
    """Drop optional arguments left at None before calling the ds41 API."""
    return {key: value for key, value in kwargs.items() if value is not None}


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
    """Run the ds41 QLI kernel through the Torch NPU dispatcher."""
    return _get_ds41_compute("quant_lightning_indexer")(
        q=q,
        k=k,
        w=w,
        descale_q=descale_q,
        descale_k=descale_k,
        topk=topk,
        quant_mode=quant_mode,
        metadata=metadata,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        return_value=return_value,
        **_present(
            {
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "seqused_q": seqused_q,
                "seqused_k": seqused_k,
                "cmp_residual_k": cmp_residual_k,
                "block_table": block_table,
                "output_idx_offset": output_idx_offset,
            }
        ),
    )


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
    """Run the ds41 QLI scheduling op through the dispatcher."""
    return _get_ds41_compute("quant_lightning_indexer_metadata")(
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
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        **_present(
            {
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "seqused_q": seqused_q,
                "seqused_k": seqused_k,
                "cmp_residual_k": cmp_residual_k,
                "batch_size": batch_size,
            }
        ),
    )


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
    """Run the ds41 QSLI kernel through the Torch NPU dispatcher."""
    return _get_ds41_compute("quant_sparse_lightning_indexer")(
        q=q,
        k=k,
        w=w,
        descale_q=descale_q,
        candidate_block_indices=candidate_block_indices,
        candidate_block_length=candidate_block_length,
        topk=topk,
        quant_mode=quant_mode,
        candidate_block_size=candidate_block_size,
        metadata=metadata,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        return_value=return_value,
        **_present(
            {
                "descale_k": descale_k,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "seqused_q": seqused_q,
                "seqused_k": seqused_k,
                "cmp_residual_k": cmp_residual_k,
                "block_table": block_table,
                "output_idx_offset": output_idx_offset,
            }
        ),
    )


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
    """Run the ds41 QSLI scheduling op through the dispatcher."""
    return _get_ds41_compute("quant_sparse_lightning_indexer_metadata")(
        candidate_block_length,
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
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        **_present(
            {
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "seqused_q": seqused_q,
                "seqused_k": seqused_k,
                "cmp_residual_k": cmp_residual_k,
                "batch_size": batch_size,
            }
        ),
    )


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
    """Generate the core task table through the ds41 operator package."""
    return _get_ds41_smla_metadata()(
        ori_topk_length,
        cmp_topk_length,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        quant_mode=quant_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
        **_present(
            {
                "cu_seqlens_q": cu_seqlens_q,
                "seqused_q": seqused_q,
                "seqused_ori_kv": seqused_ori_kv,
                "seqused_cmp_kv": seqused_cmp_kv,
                "batch_size": batch_size,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_ori_kv": max_seqlen_ori_kv,
                "max_seqlen_cmp_kv": max_seqlen_cmp_kv,
            }
        ),
    )


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
    """Run the ds41 SMLA kernel; it returns the (output, lse) pair itself."""
    return _get_ds41_compute("mixed_quant_sparse_flash_mla")(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
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
        **_present({"seqused_q": seqused_q}),
    )
