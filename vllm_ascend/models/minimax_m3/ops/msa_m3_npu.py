# SPDX-License-Identifier: Apache-2.0
"""NPU sparse attention ops for MiniMax-M3 on Ascend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_ascend.utils import (
    AscendDeviceType,
    enable_custom_op,
    get_ascend_device_type,
)

_SPARSE_ATTN_INNER_PRECISE = 4
_MSA_INDEX_BLOCK_SIZE = 128
_MSA_SCORE_BLOCK_ALIGNMENT = 16
_ASCEND_DEVICE_TYPE = get_ascend_device_type()
_FP8_E4M3_MAX = 448.0

if _ASCEND_DEVICE_TYPE != AscendDeviceType.A5:
    from vllm_ascend.models.minimax_m3.ops.msa_m3_triton import (
        minimax_m3_index_topk as _minimax_m3_index_prefill_topk,
    )


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


@dataclass
class MiniMaxM3TPDecodeScoreMetadata:
    """Graph-stable inputs for packed TP decode scoring."""

    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    context_lens: torch.Tensor
    max_block_count: int
    block_size: int
    block_offset: int
    block_count: int
    decode_query_len: int


def _as_ascendc_index_kv_cache(
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
) -> torch.Tensor:
    """Convert the runtime index K cache to the AscendC BBND layout.

    The model runner binds this K-only cache as a one-element tuple whose
    tensor is shaped ``[num_blocks, 128, head_dim]``. Direct calls may pass
    that tensor without the tuple. MsaIndexScore expects the BBND shape
    ``[num_blocks, 128, 1, head_dim]`` instead.
    """
    if isinstance(index_kv_cache, tuple):
        index_kv_cache = index_kv_cache[0]
    return index_kv_cache.unsqueeze(2)


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


def _to_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clamp(min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX).to(torch.float8_e4m3fn)


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


@torch.no_grad()
def _minimax_m3_index_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor | None,
    *,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> torch.Tensor:
    """Compute MSA index scores with the bundled AscendC operator.

    ``start_loc`` is the current query block in the *passed block table*.  For
    a TP-sharded table this is a per-request local block index, not the scalar
    global ``block_offset`` used by the Triton decode kernel.

    A causal mask selects sparse mode 3. Passing no mask selects dense mode 0
    for a TP chunk that is entirely before the current query positions.
    ``init_blocks`` and ``local_blocks`` are kept for parity with the index
    scoring interface; candidate forcing is applied by the TopK stage.
    """
    index_kv_cache = _as_ascendc_index_kv_cache(index_kv_cache)
    return torch.ops._C_ascend.npu_msa_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        start_loc,
        atten_mask=causal_mask,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_klen=seq_lens,
        layout_key="BBND",
        sparse_mode=3 if causal_mask is not None else 0,
    )


@torch.no_grad()
def minimax_m3_index_prefill(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor,
    *,
    max_query_len: int,
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Compute AscendC prefill scores and finalize their block TopK."""
    score = _minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        start_loc,
        causal_mask,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    logical_block_count = (max_seq_len + _MSA_INDEX_BLOCK_SIZE - 1) // _MSA_INDEX_BLOCK_SIZE
    logical_score_width = (
        (logical_block_count + _MSA_SCORE_BLOCK_ALIGNMENT - 1)
        // _MSA_SCORE_BLOCK_ALIGNMENT
        * _MSA_SCORE_BLOCK_ALIGNMENT
    )
    score = score[..., :logical_score_width]
    return _minimax_m3_index_prefill_topk(
        score,
        cu_seqlens_q,
        context_lens,
        max_query_len,
        topk,
        init_blocks,
        local_blocks,
    )


def _index_score_topk_candidates(
    score: torch.Tensor,
    context_lens: torch.Tensor,
    decode_query_len: int,
    topk: int,
    block_offset: int = 0,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global block IDs and scores after exactly one local TopK."""
    block_count = score.shape[-1]
    query_offsets = torch.arange(
        decode_query_len,
        dtype=context_lens.dtype,
        device=context_lens.device,
    ).repeat(context_lens.shape[0])
    visible_tokens = context_lens.repeat_interleave(decode_query_len) + query_offsets + 1
    global_valid_block_count = torch.div(
        visible_tokens + block_offset * _MSA_INDEX_BLOCK_SIZE + _MSA_INDEX_BLOCK_SIZE - 1,
        _MSA_INDEX_BLOCK_SIZE,
        rounding_mode="floor",
    ).clamp(min=0)
    valid_block_count = (global_valid_block_count - block_offset).clamp(min=0, max=block_count)
    local_block_ids = torch.arange(block_count, device=score.device)
    global_block_ids = local_block_ids + block_offset
    valid_blocks = global_block_ids[None, :] < global_valid_block_count[:, None]
    # A dense raw-score TP chunk can contain whole future speculative blocks
    # for a shorter request. Remove them before TopK so they cannot displace a
    # valid candidate and only then be discarded by output validation.
    score = torch.where(valid_blocks[None, :, :], score, float("-inf"))

    # Apply forced-block scores per token before the one and only TopK.
    if init_blocks > 0:
        init_mask = valid_blocks & (global_block_ids[None, :] < init_blocks)
        score = torch.where(init_mask[None, :, :], 1.0e30, score)
    if local_blocks > 0:
        local_start = (global_valid_block_count - local_blocks).clamp(min=0)
        local_mask = valid_blocks & (global_block_ids[None, :] >= local_start[:, None])
        score = torch.where(local_mask[None, :, :], 1.0e29, score)

    actual_topk_count = min(topk, block_count)
    raw_scores, raw_topk = torch.topk(
        score,
        k=actual_topk_count,
        dim=-1,
    )
    if actual_topk_count == topk:
        topk_indices = raw_topk.to(dtype=torch.int32)
        topk_scores = raw_scores
    else:
        topk_indices = torch.full(
            (*raw_topk.shape[:-1], topk),
            -1,
            dtype=torch.int32,
            device=raw_topk.device,
        )
        topk_scores = torch.full(
            (*raw_scores.shape[:-1], topk),
            float("-inf"),
            dtype=raw_scores.dtype,
            device=raw_scores.device,
        )
        topk_indices[..., :actual_topk_count].copy_(raw_topk)
        topk_scores[..., :actual_topk_count].copy_(raw_scores)

    valid_candidate = (topk_indices >= 0) & (topk_indices < valid_block_count[None, :, None])
    topk_indices = torch.where(
        valid_candidate,
        topk_indices + block_offset,
        -1,
    )
    topk_scores = torch.where(
        valid_candidate,
        topk_scores,
        float("-inf"),
    )
    return topk_indices, topk_scores


@torch.no_grad()
def _minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor | None,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    decode_query_len: int,
    block_offset: int = 0,
    block_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run AscendC decode score followed by one non-fused local TopK."""
    score = _minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        start_loc,
        causal_mask,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
    if block_count is None:
        block_count = block_table.shape[-1]
    return _index_score_topk_candidates(
        score[..., :block_count],
        context_lens,
        decode_query_len,
        topk,
        block_offset,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    decode_query_len: int,
) -> torch.Tensor:
    """Compute AscendC decode scores and return their block TopK."""
    topk_indices, _ = _minimax_m3_index_decode(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        context_lens,
        start_loc,
        causal_mask,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        decode_query_len=decode_query_len,
    )
    return topk_indices


@torch.no_grad()
def minimax_m3_index_tp_block_parallel_decode(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    metadata: MiniMaxM3TPDecodeScoreMetadata,
    causal_mask: torch.Tensor,
    *,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    tp_group: Any,
) -> torch.Tensor:
    """Run packed-query scoring and TopK over TP-sharded KV blocks."""
    full_idx_q = tp_group.all_gather(idx_q.contiguous(), dim=1).contiguous()
    tp_rank = tp_group.rank_in_group
    block_offset = metadata.block_offset
    block_count = metadata.block_count
    if block_count == 0:
        # MsaIndexScore requires a non-empty block-table width. Ranks without
        # logical blocks contribute neutral candidates to the collectives.
        candidate_shape = (
            full_idx_q.shape[1],
            full_idx_q.shape[0],
            topk,
        )
        local_topk = torch.full(
            candidate_shape,
            -1,
            dtype=torch.int32,
            device=full_idx_q.device,
        )
        local_scores = torch.full(
            candidate_shape,
            float("-inf"),
            dtype=torch.float32,
            device=full_idx_q.device,
        )
    else:
        # Keep all derived tensors in the model forward. FULL_DECODE_ONLY
        # captures and replays this work, whereas tensors allocated by the
        # metadata builder would leave the graph holding stale device pointers.
        halo_blocks = (metadata.decode_query_len - 1 + metadata.block_size - 1) // metadata.block_size
        score_block_end = min(
            block_offset + block_count + halo_blocks,
            metadata.max_block_count,
        )
        score_block_table = metadata.block_table[:, block_offset:score_block_end].contiguous()
        local_context_lens = metadata.context_lens - block_offset * metadata.block_size
        chunk_capacity = block_count * metadata.block_size
        score_capacity = score_block_table.shape[-1] * metadata.block_size
        max_score_k_len = min(
            chunk_capacity + metadata.decode_query_len - 1,
            score_capacity,
        )
        score_k_lens = torch.clamp(
            metadata.context_lens + metadata.decode_query_len - block_offset * metadata.block_size,
            min=0,
            max=max_score_k_len,
        )
        score_start_loc = (
            torch.div(
                metadata.context_lens,
                metadata.block_size,
                rounding_mode="floor",
            )
            .sub(block_offset)
            .clamp(
                min=0,
                max=score_block_table.shape[-1] - 1,
            )
        )
        local_topk, local_scores = _minimax_m3_index_decode(
            full_idx_q,
            index_kv_cache,
            score_block_table,
            metadata.cu_seqlens_q,
            score_k_lens,
            local_context_lens,
            score_start_loc,
            None if metadata.decode_query_len == 1 else causal_mask,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            decode_query_len=metadata.decode_query_len,
            block_offset=block_offset,
            block_count=block_count,
        )

    gathered_scores = tp_group.all_gather(local_scores.contiguous(), dim=-1)
    gathered_topk = tp_group.all_gather(local_topk.contiguous(), dim=-1)

    local_head_count = idx_q.shape[1]
    local_head_start = tp_rank * local_head_count
    local_gathered_scores = gathered_scores.narrow(0, local_head_start, local_head_count)
    _, merged_pos = torch.topk(
        local_gathered_scores,
        k=topk,
        dim=-1,
    )
    local_gathered_topk = gathered_topk.narrow(0, local_head_start, local_head_count)
    return torch.gather(local_gathered_topk, dim=-1, index=merged_pos)


@torch.no_grad()
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
    inner_precise = 1
    if key.dtype == torch.float8_e4m3fn:
        if value.dtype != torch.float8_e4m3fn:
            raise TypeError("MiniMax-M3 FP8 sparse attention requires both K and V caches in E4M3")
        q = _to_fp8_e4m3(q)
        inner_precise = _SPARSE_ATTN_INNER_PRECISE
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
        inner_precise,
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
    *,
    select_num_idx: torch.Tensor | None = None,
    query_lens: torch.Tensor | None = None,
    dequant_scale: torch.Tensor | None = None,
) -> None:
    """Run sparse decode through the AscendC sparse-attention operator."""
    if q.shape[0] != seq_lens.shape[0] * decode_query_len:
        raise ValueError("Decode query tokens must equal request count times decode_query_len")

    key, value = _split_main_kv_cache(kv_cache)
    if query_lens is None:
        query_lens = torch.full_like(seq_lens, decode_query_len, dtype=torch.int32)
    if select_num_idx is None:
        select_num_idx = _select_num_idx_from_topk(topk_idx)

    op_kwargs: dict[str, Any] = {}
    if key.dtype == torch.float8_e4m3fn:
        if value.dtype != torch.float8_e4m3fn:
            raise TypeError("MiniMax-M3 FP8 sparse attention requires both K and V caches in E4M3")
        q = _to_fp8_e4m3(q)
        if dequant_scale is None:
            dequant_scale = torch.ones((1, 1, 1, 1), dtype=torch.float32, device=q.device)
        op_kwargs = {
            "q_dequant_scale": dequant_scale,
            "k_dequant_scale": dequant_scale,
            "v_dequant_scale": dequant_scale,
            "attention_out_dtype": torch.bfloat16,
        }
    out = torch.ops._C_ascend.npu_sparse_attention_score(
        q,
        key,
        value,
        topk_idx,
        block_table,
        select_num_idx=select_num_idx,
        actual_seq_lengths=query_lens,
        actual_seq_lengths_kv=seq_lens,
        num_key_value_heads=num_kv_heads,
        scale_value=sm_scale,
        block_size=block_size,
        top_k=topk_idx.shape[-1],
        inner_precise=_SPARSE_ATTN_INNER_PRECISE,
        **op_kwargs,
    )
    output.copy_(out)
