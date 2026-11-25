# SPDX-License-Identifier: Apache-2.0
"""Optional Union sparse prefill attention for MiniMax-M3 on Ascend."""

import torch

from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    _sort_valid_gbsa_prefix,
    _split_main_kv_cache,
    _to_fp8_e4m3,
)

# UnionSparseGen merges eight adjacent query rows, restarting at each request.
UNION_SIZE = 8
# Sparse selection width in KV tokens; this is not the paged KV-cache page size.
_SPARSE_KV_BLOCK_SIZE = 128

# CANN enum values shared by metadata preparation and attention execution.
_UNION_LAYOUT = 4  # Head-major compressed sparse query groups.
_UNION_MASK_MODE = 3  # Union's exact per-query block mask; disables FlashDecode.
_FP8_STATIC_CAST_P = 5  # CANN quant mode: FP8 inputs with static probability casting.
_NO_QUANT = 0  # CANN quant mode: unquantized attention.


@torch.no_grad()
def prepare_union_gbsa_metadata(
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    num_union_groups: int,
    max_kv_blocks: int,
    max_query_len: int,
    max_seq_len: int,
    num_heads: int,
    topk: int,
    kv_dtype: torch.dtype,
) -> torch.Tensor:
    """Prepare one capacity schedule for all layers with the same geometry."""
    from cann_ops_transformer import generic_block_sparse_attention_metadata

    capacity = min(UNION_SIZE * topk, max_kv_blocks)
    # One sparse-pattern head is shared by all query heads in MiniMax-M3.
    counts = torch.full((1, num_union_groups), capacity, dtype=torch.int32, device=seq_lens.device)
    # Metadata reads counts and the indices' shape, not selected block IDs.
    indices = torch.empty((*counts.shape, capacity), dtype=torch.int32, device=seq_lens.device)
    # Union disables FlashDecode. The exact-mask kernels use each layer's
    # runtime union_count to bound KV work, so capacity counts can share a
    # schedule even when layers select different numbers of distinct blocks.
    return generic_block_sparse_attention_metadata(
        indices,
        counts,
        num_heads,
        1,  # MiniMax-M3 has one KV head per rank.
        _SPARSE_KV_BLOCK_SIZE,  # Head dimension is also 128 in supported M3 geometry.
        [UNION_SIZE, _SPARSE_KV_BLOCK_SIZE],
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seq_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_kv=max_seq_len,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=_UNION_LAYOUT,
        mask_mode=_UNION_MASK_MODE,
        quant_mode=_FP8_STATIC_CAST_P if kv_dtype == torch.float8_e4m3fn else _NO_QUANT,
        softmax_precision=0,  # Union requires precision 0, including FP8.
        win_left=-1,  # No sliding-window limit.
        win_right=-1,
        residual_block_mode=0,  # No separate residual block.
        is_consistent_topk=False,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_union(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    num_union_groups: int,
    max_kv_blocks: int,
    max_query_len: int,
    max_seq_len: int,
    sm_scale: float,
    output: torch.Tensor,
    metadata: torch.Tensor,
) -> None:
    """Union eight adjacent queries while preserving their exact selected KV sets.

    The caller selects supported geometry and supplies host-side capacities.
    Group boundaries restart for each request.
    """
    from cann_ops_nn import union_sparse_gen_launch
    from cann_ops_transformer import generic_block_sparse_attention

    key, value = _split_main_kv_cache(kv_cache)
    quantized = torch.float8_e4m3fn == key.dtype
    # Reduce directly to the operator's int32 counts, avoiding an int64 result Cast.
    counts = (topk_idx >= 0).sum(dim=-1, dtype=torch.int32)
    indices = _sort_valid_gbsa_prefix(topk_idx, counts)
    union_idx, union_count, exact_mask = union_sparse_gen_launch(
        indices,
        counts,
        cu_seqlens_q,
        None,  # seqused_kv below already supplies each request's KV length.
        None,
        seq_lens,
        block_shape=[1, _SPARSE_KV_BLOCK_SIZE],  # One query row per input sparse pattern.
        union_size=UNION_SIZE,
        mask_type=1,  # Causal mask.
        max_q_seqlen=max_query_len,
        max_kv_seqlen=max_seq_len,
        layout_sparse_pattern=_UNION_LAYOUT,
        is_input_idx_sorted=True,
        residual_block_mode=0,  # No separate residual block.
        totalq_union_num=num_union_groups,
        max_logical_blocks=max_kv_blocks,
    )
    # Union mask mode requires precision 0, including FP8; the native
    # non-Union FP8 path's precision 1 is a different operator contract.
    result, _ = generic_block_sparse_attention(
        _to_fp8_e4m3(q) if quantized else q,
        key,
        value,
        union_idx,
        union_count,
        [UNION_SIZE, _SPARSE_KV_BLOCK_SIZE],
        metadata=metadata,
        block_table=block_table,
        attn_mask=exact_mask,
        softmax_scale=sm_scale,
        return_softmax_lse=False,
        attention_out_dtype=output.dtype,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=None,
        seqused_kv=seq_lens,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=_UNION_LAYOUT,
        mask_mode=_UNION_MASK_MODE,
        quant_mode=_FP8_STATIC_CAST_P if quantized else _NO_QUANT,
        softmax_precision=0,
        win_left=-1,  # No sliding-window limit.
        win_right=-1,
        residual_block_mode=0,  # No separate residual block.
        is_consistent_topk=False,
    )
    output.copy_(result)
