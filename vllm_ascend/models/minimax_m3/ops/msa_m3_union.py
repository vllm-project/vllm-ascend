# SPDX-License-Identifier: Apache-2.0
"""Optional FP8 Union sparse attention for MiniMax-M3 on Ascend 950."""

from functools import lru_cache
from importlib import import_module

import torch

from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    _sort_valid_gbsa_prefix,
    _split_main_kv_cache,
    _to_fp8_e4m3,
)

UNION_SIZE = 8
_PAGE_SIZE = 128


@lru_cache(maxsize=1)
def load_union_extensions():
    """Load the optional, matched CANN extensions before graph capture."""
    try:
        nn_ops = import_module("cann_ops_nn_custom")
        attention_ops = import_module("cann_ops_transformer_custom")
        import_module("cann_ops_nn_custom.ops.index.union_sparse_gen.union_sparse_gen").union_sparse_gen_builder.load()
        import_module(
            "cann_ops_transformer_custom.ops.attention.generic_block_sparse_attention.generic_block_sparse_attention"
        ).generic_block_sparse_attention_op_builder.load()
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "MiniMax-M3 Union requires matching UnionSparseGen and GBSA exactMask extensions. "
            "See the MiniMax-M3 Union deployment guide."
        ) from exc
    return nn_ops, attention_ops


@torch.no_grad()
def minimax_m3_sparse_attn_union(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor | None,
    seq_lens: torch.Tensor,
    num_union_groups: int,
    max_kv_blocks: int,
    max_query_len: int,
    max_seq_len: int,
    sm_scale: float,
    output: torch.Tensor,
    *,
    select_num_idx: torch.Tensor | None = None,
) -> None:
    """Union eight adjacent queries while preserving their exact selected KV sets.

    Host metadata supplies output capacities; neither allocation nor graph
    capture reads a device scalar. Group boundaries restart for each request.
    """
    key, value = _split_main_kv_cache(kv_cache)
    if q.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise ValueError("MiniMax-M3 FP8 Union requires BF16 query and output")
    if key.dtype != torch.float8_e4m3fn or value.dtype != torch.float8_e4m3fn:
        raise ValueError("MiniMax-M3 FP8 Union requires E4M3 K/V caches")
    if key.shape[1:3] != (_PAGE_SIZE, 1) or value.shape != key.shape:
        raise ValueError("MiniMax-M3 Union requires 128-token pages and one local KV head")
    if q.ndim != 3 or not 1 <= q.shape[1] <= 16 or q.shape[2] != 128:
        raise ValueError("MiniMax-M3 FP8 Union requires 1..16 local Q heads and head size 128")
    if not key.is_contiguous() or not value.is_contiguous():
        raise ValueError("MiniMax-M3 Union requires contiguous PA_BBND K/V caches")
    if topk_idx.ndim != 3 or topk_idx.shape[:2] != (1, q.shape[0]):
        raise ValueError("MiniMax-M3 Union expects token-wise sparse indices [1, totalQ, K]")
    if output.shape != q.shape:
        raise ValueError("MiniMax-M3 Union output must match the query shape")
    if num_union_groups < 0 or max_kv_blocks < 0:
        raise ValueError("MiniMax-M3 Union capacities must be nonnegative")
    if q.shape[0] == 0:
        return
    if num_union_groups == 0 or max_kv_blocks == 0:
        raise ValueError("MiniMax-M3 Union capacities must be positive for nonempty queries")

    nn_ops, attention_ops = load_union_extensions()
    seq_lens = seq_lens.to(dtype=torch.int32).contiguous()
    counts = (
        (topk_idx >= 0).sum(dim=-1).to(torch.int32)
        if select_num_idx is None
        else select_num_idx.to(dtype=torch.int32).contiguous()
    )
    indices = _sort_valid_gbsa_prefix(topk_idx, counts)
    union_idx, union_count, exact_mask = nn_ops.union_sparse_gen_launch(
        indices,
        counts,
        cu_seqlens_q.to(dtype=torch.int32).contiguous(),
        None if cu_seqlens_kv is None else cu_seqlens_kv.to(dtype=torch.int32).contiguous(),
        None,
        seq_lens,
        1,
        _PAGE_SIZE,
        UNION_SIZE,
        True,
        max_query_len,
        max_seq_len,
        4,
        True,
        0,
        num_union_groups,
        max_kv_blocks,
    )
    # Union mask mode requires precision 0, including FP8; the native
    # non-Union FP8 path's precision 1 is a different operator contract.
    common = dict(
        cu_seqlens_q=cu_seqlens_q.to(dtype=torch.int64),
        cu_seqlens_kv=None,
        seqused_kv=seq_lens,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_sparse_pattern=4,
        mask_mode=3,
        quant_mode=5,
        softmax_precision=0,
        win_left=-1,
        win_right=-1,
        residual_block_mode=0,
        is_consistent_topk=False,
    )
    metadata = attention_ops.generic_block_sparse_attention_metadata(
        union_idx,
        union_count,
        q.shape[1],
        1,
        q.shape[2],
        [UNION_SIZE, _PAGE_SIZE],
        max_seqlen_q=max_query_len,
        max_seqlen_kv=max_seq_len,
        **common,
    )
    result, _ = attention_ops.generic_block_sparse_attention(
        _to_fp8_e4m3(q),
        key,
        value,
        union_idx,
        union_count,
        [UNION_SIZE, _PAGE_SIZE],
        metadata=metadata,
        block_table=block_table,
        attn_mask=exact_mask,
        softmax_scale=sm_scale,
        return_softmax_lse=False,
        attention_out_dtype=torch.bfloat16,
        **common,
    )
    output.copy_(result)
