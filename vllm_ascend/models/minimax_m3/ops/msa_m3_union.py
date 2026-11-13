# SPDX-License-Identifier: Apache-2.0
"""Optional Union sparse attention for MiniMax-M3 on Ascend 950."""

from functools import lru_cache
from importlib import import_module

import torch
from vllm.logger import logger

from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import (
    _sort_valid_gbsa_prefix,
    _split_main_kv_cache,
    _to_fp8_e4m3,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
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
        for module, names in (
            (nn_ops, ("union_sparse_gen_launch",)),
            (attention_ops, ("generic_block_sparse_attention_metadata", "generic_block_sparse_attention")),
        ):
            if any(not callable(getattr(module, name, None)) for name in names):
                raise AttributeError("Union operator API missing")
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        logger.warning_once("MiniMax-M3 Union extensions unavailable; using native Q-gather-KV: %s", exc)
        return None
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
    d8_threshold_pct: float = 100,
    total_kv_blocks: int = -1,
    decode_query_len: int = 0,
) -> None:
    """Union eight adjacent queries while preserving their exact selected KV sets.

    Host metadata supplies output capacities; neither allocation nor graph
    capture reads a device scalar. Group boundaries restart for each request.
    """
    key, value = _split_main_kv_cache(kv_cache)
    quantized = key.dtype == torch.float8_e4m3fn
    if q.dtype not in (torch.bfloat16, torch.float16) or output.dtype != q.dtype:
        raise ValueError("MiniMax-M3 Union requires BF16/FP16 query and matching output")
    if value.dtype != key.dtype or (not quantized and key.dtype != q.dtype):
        raise ValueError("MiniMax-M3 Union requires matching Q/K/V dtypes or E4M3 K/V caches")
    if quantized and q.dtype != torch.bfloat16:
        raise ValueError("MiniMax-M3 FP8 Union requires BF16 query and output")
    if key.shape[1:3] != (_PAGE_SIZE, 1) or value.shape != key.shape:
        raise ValueError("MiniMax-M3 Union requires 128-token pages and one local KV head")
    if q.ndim != 3 or not 1 <= q.shape[1] <= 16 or q.shape[2] != 128:
        raise ValueError("MiniMax-M3 Union requires 1..16 local Q heads and head size 128")
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

    extensions = load_union_extensions()
    assert extensions is not None, "Union must be selected only after extension loading succeeds"
    nn_ops, attention_ops = extensions
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
    if d8_threshold_pct < 100:
        # Imported only for opt-in runtime routing. The default pure Union
        # path has no routing kernel, native launch, or output merge.
        from vllm_ascend.models.minimax_m3.union_policy import route_union_groups

        union_count, native_indices, native_counts, token_route = route_union_groups(
            union_count, indices, counts, cu_seqlens_q, d8_threshold_pct
        )
        native_output = torch.empty_like(output)
        if decode_query_len:
            # Routed counts change with actual selections; do not reuse the
            # native layer-independent decode schedule for this mixed path.
            minimax_m3_sparse_attn_decode(
                q,
                kv_cache,
                native_indices,
                block_table,
                seq_lens,
                1,
                sm_scale,
                native_output,
                decode_query_len,
                select_num_idx=native_counts,
                cu_seqlens_q=cu_seqlens_q.to(torch.int64),
                indices_are_sorted=True,
            )
        else:
            minimax_m3_sparse_attn(
                q,
                kv_cache,
                native_indices,
                block_table,
                cu_seqlens_q,
                seq_lens,
                seq_lens,
                max_query_len,
                1,
                sm_scale,
                native_output,
                total_kv_blocks=total_kv_blocks,
                max_kv_blocks=max_kv_blocks,
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
        quant_mode=5 if quantized else 0,
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
        _to_fp8_e4m3(q) if quantized else q,
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
        attention_out_dtype=output.dtype,
        **common,
    )
    if d8_threshold_pct < 100:
        from vllm_ascend.models.minimax_m3.union_policy import merge_union_groups

        merge_union_groups(result, native_output, token_route, counts, output)
    else:
        output.copy_(result)
