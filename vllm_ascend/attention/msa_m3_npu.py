# SPDX-License-Identifier: Apache-2.0
"""NPU sparse attention ops for MiniMax-M3 on Ascend."""

from __future__ import annotations

import torch

_SPARSE_ATTN_INNER_PRECISE = 4


def _split_main_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) < 2:
            raise ValueError("Main kv cache tuple must contain K and V tensors")
        k_cache, v_cache = kv_cache[0], kv_cache[1]
    else:
        if kv_cache.ndim != 5:
            raise ValueError(f"Unexpected main kv cache ndim: {kv_cache.ndim}")
        if kv_cache.shape[0] == 2:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
        elif kv_cache.shape[1] == 2:
            k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
        else:
            raise ValueError(f"Unexpected main kv cache shape: {tuple(kv_cache.shape)}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "Unexpected split main kv cache shapes: "
            f"{tuple(k_cache.shape)}, {tuple(v_cache.shape)}"
        )
    return k_cache, v_cache


def _select_num_idx_from_topk(topk_idx: torch.Tensor) -> torch.Tensor:
    return (topk_idx >= 0).sum(dim=-1).to(dtype=torch.int32)


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    block_size: int = 128,
) -> None:
    del prefix_lens, max_query_len
    key, value = _split_main_kv_cache(kv_cache)
    q_lens_t = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    out = torch.ops._C_ascend.npu_sparse_attention_score(
        q,
        key,
        value,
        topk_idx,
        block_table,
        select_num_idx=_select_num_idx_from_topk(topk_idx),
        actual_seq_lengths=q_lens_t,
        actual_seq_lengths_kv=seq_lens,
        num_key_value_heads=num_kv_heads,
        scale_value=sm_scale,
        block_size=block_size,
        top_k=topk_idx.shape[-1],
        inner_precise=_SPARSE_ATTN_INNER_PRECISE,
    )
    output.copy_(out)


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    block_size: int = 128,
) -> None:
    num_reqs = seq_lens.shape[0]
    active_tokens = num_reqs * decode_query_len
    q_active = q[:active_tokens]
    topk_active = topk_idx[:, :active_tokens]
    key, value = _split_main_kv_cache(kv_cache)
    q_lens_t = torch.full(
        (num_reqs,),
        decode_query_len,
        device=q.device,
        dtype=torch.int32,
    )
    out = torch.ops._C_ascend.npu_sparse_attention_score(
        q_active,
        key,
        value,
        topk_active,
        block_table,
        select_num_idx=_select_num_idx_from_topk(topk_active),
        actual_seq_lengths=q_lens_t,
        actual_seq_lengths_kv=seq_lens,
        num_key_value_heads=num_kv_heads,
        scale_value=sm_scale,
        block_size=block_size,
        top_k=topk_active.shape[-1],
        inner_precise=_SPARSE_ATTN_INNER_PRECISE,
    )
    output[:active_tokens].copy_(out)
