# SPDX-License-Identifier: Apache-2.0
"""Native INT8 GBSA with dynamic Q/K and calibrated per-channel V."""

import torch
from vllm.utils.math_utils import cdiv

from vllm_ascend.ops.gbsa import GBSAQuantMode, generic_block_sparse_attention, generic_block_sparse_attention_metadata
from vllm_ascend.ops.triton.qwen4_exp.c8 import quantize_qsa_query
from vllm_ascend.ops.triton.qwen4_exp.gbsa import prepare_qsa_gbsa_inputs

from .attention_dispatch import QSARequestPlan
from .gbsa import GBSA_COMPRESS_RATIO, GBSA_PAGE_SIZE

C8_PROBABILITY_QUANT_SCALE = 127.0
# 910B replay showed a gain for large prefill chunks with a short KV prefix.
# Keep long, decode and mixed batches on the independent-row GBSA schedule.
_MIN_GROUPED_QUERY_TOKENS = 512
_MAX_GROUPED_KV_TOKENS = 10 * 1024


def c8_prefill_kv_length(request_plan: QSARequestPlan | None, tokens: int) -> int | None:
    """Return the scheduler's KV upper bound for an eligible prefill chunk."""
    if request_plan is None or len(request_plan.requests) != 1:
        return None
    request = request_plan.requests[0]
    if (
        request.is_prefill
        and request.query_start == 0
        and request.query_end == tokens
        and _MIN_GROUPED_QUERY_TOKENS <= tokens <= request.kv_length <= _MAX_GROUPED_KV_TOKENS
    ):
        return request.kv_length
    return None


def qsa_c8_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    compact_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: torch.Tensor,
    *,
    key_scales: torch.Tensor,
    prefill_kv_length: int | None = None,
) -> torch.Tensor:
    """Run native INT8 GBSA on paged KV and calibrated scales."""
    if query.ndim != 3 or query.dtype != torch.bfloat16 or output.shape != query.shape or output.dtype != query.dtype:
        raise ValueError("QSA C8 attention requires matching BF16 [tokens,heads,dim] query/output")
    if key_cache.ndim != 4 or key_cache.shape != value_cache.shape or key_cache.shape[1] != GBSA_PAGE_SIZE:
        raise ValueError("QSA C8 attention requires matching 128-token K/V pages")
    pages, _, heads, dim = key_cache.shape
    if (
        key_cache.dtype != torch.int8
        or value_cache.dtype != torch.int8
        or key_scales.shape != (pages, heads, GBSA_PAGE_SIZE)
        or key_scales.dtype != torch.float32
        or key_scales.device != key_cache.device
        or not key_scales.is_contiguous()
    ):
        raise ValueError("QSA C8 GBSA requires INT8 cache and dense FP32 [pages,heads,128] K scales")
    if not heads or query.shape[-1] != dim or query.shape[1] % heads:
        raise ValueError("QSA C8 attention has incompatible query/KV heads")
    if compact_indices.ndim != 2 or not 3 < compact_indices.shape[1] <= 515:
        raise ValueError("QSA C8 requires at most 512 compact groups plus three residual columns")
    if compact_indices.shape[0] != query.shape[0] or token_to_req.shape != (query.shape[0],):
        raise ValueError("QSA C8 selection/request mapping must match query rows")
    if logical_positions.shape != token_to_req.shape or block_table.ndim != 2 or not block_table.shape[1]:
        raise ValueError("QSA C8 requires row positions and a nonempty page table")
    if seq_lens.shape != (block_table.shape[0],):
        raise ValueError("QSA C8 sequence lengths must match page table requests")
    if v_scale.shape != (heads, query.shape[-1]) or v_scale.dtype != torch.float32 or not v_scale.is_contiguous():
        raise ValueError("QSA native INT8 GBSA requires dense FP32 [KV heads,dim] V scale")
    if p_scale.shape != (1,) or p_scale.dtype != torch.float32 or not p_scale.is_contiguous():
        raise ValueError("QSA native INT8 GBSA requires one FP32 P quantization factor")
    output.zero_()
    if not query.shape[0]:
        return output
    if prefill_kv_length is not None:
        if block_table.shape[0] != 1:
            raise ValueError("Grouped QSA C8 prefill requires one request")
        block_table = block_table[:, : cdiv(prefill_kv_length, GBSA_PAGE_SIZE)].contiguous()
    groups, counts, visible, cu_q, row_table = prepare_qsa_gbsa_inputs(
        compact_indices, logical_positions, seq_lens, token_to_req, block_table, heads
    )
    max_seqlen_q = 1
    if prefill_kv_length is not None:
        # The scheduler positions are contiguous and right-aligned to the live
        # KV length. GBSA's causal mask derives the same visibility per row
        # from those request bounds, enabling KV reuse across the chunk.
        max_seqlen_q = query.shape[0]
        cu_q = cu_q[::max_seqlen_q].contiguous()
        visible = seq_lens
        row_table = block_table
    common = dict(
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=4,
        mask_mode=1,
        quant_mode=GBSAQuantMode.INT8_QK_DYNAMIC_V_STATIC,
        softmax_precision=0,
        residual_block_mode=0,
        is_consistent_topk=False,
    )
    # GBSA follows the scheduler page table for both K/V codes and K scales.
    codes_q, scales_q = quantize_qsa_query(query)
    metadata = generic_block_sparse_attention_metadata(
        groups,
        counts,
        query.shape[1],
        heads,
        query.shape[-1],
        [1, GBSA_COMPRESS_RATIO],
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=row_table.shape[1] * GBSA_PAGE_SIZE,
        cu_seqlens_q=cu_q,
        seqused_kv=visible,
        **common,
    )
    result, _ = generic_block_sparse_attention(
        codes_q,
        key_cache,
        value_cache,
        groups,
        counts,
        [1, GBSA_COMPRESS_RATIO],
        metadata=metadata,
        q_dequant_scale=scales_q,
        k_dequant_scale=key_scales,
        v_dequant_scale=v_scale,
        p_quant_scale=p_scale,
        attention_out_dtype=query.dtype,
        block_table=row_table,
        softmax_scale=query.shape[-1] ** -0.5,
        cu_seqlens_q=cu_q,
        seqused_kv=visible,
        **common,
    )
    output.copy_(torch.where((counts[0] > 0).view(-1, 1, 1), result, 0))
    return output
