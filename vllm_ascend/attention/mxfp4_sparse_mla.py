# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 main-KV-cache adapters for the MixedQuantSparseFlashMla operator.

GLM-5.3-Flash keeps one 512-wide no-rope MLA latent cache per layer.
``quant_mode=4`` of the ops-transformer operator consumes that cache as its FP4
"cmp" pool with explicit top-k token positions (``ori_kv=None``), so the whole
layer runs as CMP-only sparse MLA over the packed MXFP4 pages.

``cann_ops_transformer`` ships with the CANN/ops-transformer toolkit rather than
with vllm-ascend, so it is imported lazily: models that never select the FP4
cache must not pay for it, and a missing package has to surface as an actionable
error only when the mode is actually used.
"""

from __future__ import annotations

from functools import lru_cache

import torch

# Packed MXFP4 row: nope[448] + rope[64] as E2M1 pairs (256 B) followed by one
# BF16 shared-exponent scale per 16 values (32 x 2 B).
FP4_ROW_BYTES = 320
FP4_GROUP_SIZE = 16
FP4_HEAD_DIM = 512
# Attention dispatch always uses the mixed FP8/FP4 DSL, which is mode 4 even
# though the cache is CMP-only. The library metadata contract stays mode 1.
FP4_QUANT_MODE = 4
FP4_METADATA_QUANT_MODE = 1


@lru_cache(maxsize=1)
def _fp4_ops():
    try:
        import cann_ops_transformer as ops
    except ImportError as exc:  # pragma: no cover - depends on the toolkit
        raise RuntimeError(
            "The GLM-5.3-Flash FP4 KV cache requires cann_ops_transformer from "
            "the matching CANN/ops-transformer package (CANN 1126 or newer)."
        ) from exc
    return (
        ops.kv_compress_epilog,
        ops.mixed_quant_sparse_flash_mla,
        ops.mixed_quant_sparse_flash_mla_metadata,
    )


def write_fp4_kv_cache(cache: torch.Tensor, values: torch.Tensor, slots: torch.Tensor) -> None:
    """Quantize the normalized MLA latent to packed MXFP4 and scatter it.

    ``cache`` is the paged uint8 view ``[blocks, block, 1, 320]``; ``values`` is
    the BF16 RMSNorm output ``[tokens, 512]``; ``slots`` carries ``-1`` for
    tokens that must be skipped.
    """
    writer, _, _ = _fp4_ops()
    writer(
        cache,
        values.contiguous(),
        slots.contiguous(),
        quant_group_size=FP4_GROUP_SIZE,
        quant_mode="nope_rope_fp4_bf16",
    )


def _build_metadata(num_heads, metadata, topk, cu_seqlens_q, topk_length):
    _, _, metadata_op = _fp4_ops()
    return metadata_op(
        int(num_heads),
        1,
        FP4_HEAD_DIM,
        FP4_METADATA_QUANT_MODE,
        cu_seqlens_q=cu_seqlens_q,
        seqused_cmp_kv=metadata.seq_lens,
        batch_size=int(metadata.seq_lens.numel()),
        max_seqlen_q=int(metadata.max_query_len),
        max_seqlen_cmp_kv=int(metadata.max_seq_len),
        cmp_topk=int(topk),
        cmp_topk_length=topk_length,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=False,
        has_cmp_kv=True,
    )


def fp4_sparse_mla(q, cache, indices, metadata, scale):
    """Attend to the FP4 latent cache at the selected token positions."""
    if indices.dim() == 2:
        indices = indices.unsqueeze(1)
    elif indices.dim() != 3 or indices.shape[1] != 1:
        raise ValueError(f"FP4 sparse indices must be [tokens, 1, K], got {tuple(indices.shape)}.")
    # Sorting keeps the operator's paired DMA on adjacent cache columns; the
    # left-aligned valid prefix then matches the BF16 SFA path.
    sentinel = torch.iinfo(torch.int32).max
    sorted_indices = torch.where(indices >= 0, indices, sentinel).sort(dim=-1).values
    sorted_indices = torch.where(sorted_indices == sentinel, -1, sorted_indices)
    topk_length = metadata.smla_topk_length[: q.shape[0]]
    topk_length.copy_((sorted_indices >= 0).sum(dim=-1, dtype=torch.int32))
    cu_seqlens_q = metadata.query_start_loc
    if q.shape[0] != topk_length.shape[0]:
        # Eager and piecewise execution trim graph-padding query rows.
        cu_seqlens_q = cu_seqlens_q.clamp(max=q.shape[0])
    task_metadata = _build_metadata(
        q.shape[1],
        metadata,
        sorted_indices.shape[-1],
        cu_seqlens_q,
        topk_length,
    )
    _, attention_op, _ = _fp4_ops()
    output, _ = attention_op(
        q.contiguous(),
        FP4_QUANT_MODE,
        cmp_kv=cache,
        cmp_sparse_indices=sorted_indices.contiguous(),
        cmp_block_table=metadata.block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_cmp_kv=metadata.seq_lens,
        cmp_topk_length=topk_length,
        metadata=task_metadata,
        layout_q="TND",
        layout_kv="PA_BBND",
        softmax_scale=scale,
    )
    return output
