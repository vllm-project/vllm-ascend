# SPDX-License-Identifier: Apache-2.0
"""NPU sparse attention ops for MiniMax-M3 on Ascend."""

from __future__ import annotations

import torch

from vllm_ascend.utils import (
    AscendDeviceType,
    enable_custom_op,
    get_ascend_device_type,
)

_SPARSE_ATTN_INNER_PRECISE = 4
_ASCEND_DEVICE_TYPE = get_ascend_device_type()


def _k2q_csr_block_stats(cu_block_lens: torch.Tensor) -> tuple[int, int]:
    """Derive ``(total_rows, max_kv)`` from ``cu_block_lens`` on host."""
    cu = cu_block_lens.reshape(-1)
    if cu.numel() <= 1:
        return 0, 0
    block_lens = cu[1:] - cu[:-1]
    total_rows = int(cu[-1].item())
    max_kv = int(block_lens.max().item()) if block_lens.numel() else 0
    return total_rows, max_kv


@torch.no_grad()
def _npu_k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_lens: torch.Tensor,
    order_method: int = 0,
    total_rows: int = -1,
    max_kv: int = -1,
    use_simt: int | bool = 0,
    q_global_offset: int | bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert MiniMax-M3 q2k indices to k2q CSR on NPU."""
    # A5 disables the generic custom-op loader, so register the in-tree
    # MiniMax M3 operators lazily after the NPU runtime has been initialized.
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, PLC0415

    enable_custom_op()
    if total_rows < 0 or max_kv < 0:
        derived_total_rows, derived_max_kv = _k2q_csr_block_stats(cu_block_lens)
        if total_rows < 0:
            total_rows = derived_total_rows
        if max_kv < 0:
            max_kv = derived_max_kv
    return torch.ops._C_ascend.npu_k2q_csr(
        q2k,
        cu_seqlens,
        cu_block_lens,
        int(order_method),
        int(total_rows),
        int(max_kv),
        int(use_simt),
        int(q_global_offset),
    )


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
        raise ValueError(f"Unexpected split main kv cache shapes: {tuple(k_cache.shape)}, {tuple(v_cache.shape)}")
    return k_cache, v_cache


def _select_num_idx_from_topk(topk_idx: torch.Tensor) -> torch.Tensor:
    return (topk_idx >= 0).sum(dim=-1).to(dtype=torch.int32)


def _build_cu_block_lens(
    seq_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Build cumulative logical KV-block counts for each prefill request."""
    block_lens = torch.div(
        seq_lens.to(torch.int32) + block_size - 1,
        block_size,
        rounding_mode="floor",
    )
    cu_block_lens = torch.empty(
        block_lens.numel() + 1,
        dtype=torch.int32,
        device=seq_lens.device,
    )
    cu_block_lens[0] = 0
    torch.cumsum(block_lens, dim=0, out=cu_block_lens[1:])
    return cu_block_lens


def _minimax_m3_sparse_attn_a3(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    block_size: int,
) -> None:
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


def _minimax_m3_sparse_attn_a5(
    q: torch.Tensor,
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    block_size: int,
) -> None:
    key, value = _split_main_kv_cache(kv_cache)
    cu_block_lens = _build_cu_block_lens(seq_lens, block_size)
    k2q_row_ptr, k2q_q_indices, k2q_slot_indices = _npu_k2q_csr(
        topk_idx,
        cu_seqlens_q,
        cu_block_lens,
        order_method=1,
        use_simt=0,
        q_global_offset=True,
    )

    k2q_row_ptr = k2q_row_ptr.to(dtype=torch.int32).contiguous()
    k2q_q_indices = k2q_q_indices.to(dtype=torch.int32).contiguous()
    k2q_slot_indices = k2q_slot_indices.to(dtype=torch.int32).contiguous()
    q_lens_t = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32).contiguous()
    kv_lens_t = seq_lens.to(torch.int32).contiguous()
    out = torch.ops._C_ascend.npu_sparse_attention_score_prefill(
        q,
        key,
        value,
        block_table,
        k2q_row_ptr,
        k2q_q_indices,
        k2q_slot_indices,
        num_kv_heads,
        sm_scale,
        block_size,
        topk_idx.shape[-1],
        1,
        actual_seq_lengths=q_lens_t,
        actual_seq_lengths_kv=kv_lens_t,
    )
    output.copy_(out)


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
    sparse_attn_impl = (
        _minimax_m3_sparse_attn_a5 if _ASCEND_DEVICE_TYPE == AscendDeviceType.A5 else _minimax_m3_sparse_attn_a3
    )
    sparse_attn_impl(
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        block_size,
    )


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
    """Run sparse decode through the AscendC sparse-attention operator."""
    if q.shape[0] != seq_lens.shape[0] * decode_query_len:
        raise ValueError("Decode query tokens must equal request count times decode_query_len")

    key, value = _split_main_kv_cache(kv_cache)
    query_lens = torch.full_like(seq_lens, decode_query_len, dtype=torch.int32)
    out = torch.ops._C_ascend.npu_sparse_attention_score(
        q,
        key,
        value,
        topk_idx,
        block_table,
        select_num_idx=_select_num_idx_from_topk(topk_idx),
        actual_seq_lengths=query_lens,
        actual_seq_lengths_kv=seq_lens,
        num_key_value_heads=num_kv_heads,
        scale_value=sm_scale,
        block_size=block_size,
        top_k=topk_idx.shape[-1],
        inner_precise=_SPARSE_ATTN_INNER_PRECISE,
    )
    output.copy_(out)
