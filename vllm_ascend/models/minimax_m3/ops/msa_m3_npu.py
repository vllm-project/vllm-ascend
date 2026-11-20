# SPDX-License-Identifier: Apache-2.0
"""NPU sparse attention ops for MiniMax-M3 on Ascend."""

from __future__ import annotations

from functools import lru_cache
from inspect import signature
from typing import Any

import torch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.msa.msa_m3_index_postprocess import index_decode_topk_fused

_MSA_INDEX_BLOCK_SIZE = 128
_GBSA_MASK_MODE_CAUSAL = 1
_GBSA_NO_QUANT = 0
_GBSA_FP8_STATIC_CAST_P = 5
_GBSA_SOFTMAX_PRECISION = 0
_GBSA_FP8_SOFTMAX_PRECISION = 1
_MSA_SCORE_BLOCK_ALIGNMENT = 16
_FP8_E4M3_MAX = 448.0

# CANN restricts ``topk`` to this range.
_INDEX_SCORE_TOPK_MAX = 128
# The operator produces incorrect results when topk is not a multiple of 8.
_INDEX_SCORE_TOPK_WRITE_ALIGNMENT = 8

if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
    from vllm_ascend.ops.triton.msa.msa_m3_index_topk import (
        minimax_m3_index_topk as _minimax_m3_index_prefill_topk,
    )
elif get_current_hardware_profile().supports(HardwareCapability.RUNTIME_CUSTOM_OPS):
    from vllm_ascend.models.minimax_m3.ops.msa_m3_triton import (
        minimax_m3_index_topk as _minimax_m3_index_prefill_topk,
    )


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


def _build_cu_seqlens_q(query_lens: torch.Tensor) -> torch.Tensor:
    cu_seqlens_q = torch.empty(
        query_lens.numel() + 1,
        dtype=torch.int64,
        device=query_lens.device,
    )
    cu_seqlens_q[0] = 0
    torch.cumsum(
        query_lens.to(dtype=torch.int64),
        dim=0,
        out=cu_seqlens_q[1:],
    )
    return cu_seqlens_q


def _gbsa_softmax_precision(kv_cache_dtype: torch.dtype) -> int:
    if kv_cache_dtype == torch.float8_e4m3fn:
        return _GBSA_FP8_SOFTMAX_PRECISION
    if kv_cache_dtype == torch.bfloat16:
        return get_current_hardware_profile().minimax_m3_gbsa_bf16_softmax_precision
    return _GBSA_SOFTMAX_PRECISION


def _decode_gbsa_metadata_key(
    num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    topk: int,
    query_len: int,
    kv_cache_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Any, ...]:
    return (
        (num_tokens, num_heads, head_size),
        (num_kv_heads, num_tokens, topk),
        (num_kv_heads, num_tokens),
        num_kv_heads,
        query_len,
        kv_cache_dtype,
        _gbsa_softmax_precision(kv_cache_dtype),
    )


def _build_generic_block_sparse_attention_metadata(
    topk_idx: torch.Tensor,
    select_num_idx: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    max_query_len: int,
    quant_mode: int,
    softmax_precision: int,
) -> torch.Tensor:
    from cann_ops_transformer.ops import generic_block_sparse_attention_metadata

    return generic_block_sparse_attention_metadata(
        topk_idx,
        select_num_idx,
        num_heads,
        num_kv_heads,
        head_size,
        [1, block_size],
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seq_lens,
        max_seqlen_q=max_query_len,
        layout_q="TND",
        layout_kv="PA_BBND",
        mask_mode=_GBSA_MASK_MODE_CAUSAL,
        quant_mode=quant_mode,
        softmax_precision=softmax_precision,
    )


def _decode_gbsa_block_counts(
    seq_lens: torch.Tensor,
    query_len: int,
    num_kv_heads: int,
    topk: int,
    logical_block_count: int,
) -> torch.Tensor:
    # Match the causal validity mask in index_decode_topk_fused, including
    # zero-length graph padding and each speculative token's query offset.
    offsets = torch.arange(query_len, dtype=torch.int32, device=seq_lens.device)
    visible_blocks = torch.div(
        seq_lens[:, None] - query_len + offsets + _MSA_INDEX_BLOCK_SIZE,
        _MSA_INDEX_BLOCK_SIZE,
        rounding_mode="floor",
    ).clamp_(0, min(topk, logical_block_count))
    return visible_blocks.reshape(1, -1).expand(num_kv_heads, -1).contiguous()


@torch.no_grad()
def prepare_decode_gbsa_metadata(
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    query_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    topk: int,
    logical_block_count: int,
    kv_cache_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Prepare a supported decode schedule before any layer's IndexScore."""
    counts = _decode_gbsa_block_counts(seq_lens, query_len, num_kv_heads, topk, logical_block_count)
    # CANN's metadata operator reads block counts, not selected block IDs;
    # sparse_block_indices supplies only the [heads, tokens, topk] shape.
    indices_shape = torch.empty((*counts.shape, topk), dtype=torch.int32, device=seq_lens.device)
    return _build_generic_block_sparse_attention_metadata(
        indices_shape,
        counts,
        cu_seqlens_q,
        seq_lens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=_MSA_INDEX_BLOCK_SIZE,
        max_query_len=query_len,
        quant_mode=_GBSA_FP8_STATIC_CAST_P if kv_cache_dtype == torch.float8_e4m3fn else _GBSA_NO_QUANT,
        softmax_precision=_gbsa_softmax_precision(kv_cache_dtype),
    )


def _sort_valid_gbsa_prefix(
    topk_idx: torch.Tensor,
    select_num_idx: torch.Tensor,
) -> torch.Tensor:
    """Sort valid logical blocks, preserving counts and trailing -1 padding."""
    # Logical block IDs are small exact FP32 integers. Sorting FP32 avoids
    # the slow integer Sort path on Ascend; GBSA still consumes int32 IDs.
    indices = topk_idx.to(dtype=torch.float32)
    positions = torch.arange(indices.shape[-1], device=indices.device, dtype=torch.int32)
    valid = (positions < select_num_idx.unsqueeze(-1)) & (indices >= 0)
    sentinel = float("inf")
    keys = torch.where(valid, indices, sentinel)
    ordered = torch.sort(keys, dim=-1).values
    return torch.where(ordered == sentinel, -1, ordered).to(dtype=torch.int32).contiguous()


@torch.no_grad()
def _run_generic_block_sparse_attention(
    q: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    select_num_idx: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_kv_heads: int,
    sm_scale: float,
    block_size: int,
    max_query_len: int = -1,
    dequant_scale: torch.Tensor | None = None,
    gbsa_metadata_cache: dict[tuple[Any, ...], torch.Tensor] | None = None,
    indices_are_sorted: bool = False,
) -> torch.Tensor:
    """Run MiniMax-M3 sparse attention through CANN Ops GBSA."""
    del dequant_scale
    select_num_idx = select_num_idx.to(dtype=torch.int32).contiguous()
    if indices_are_sorted:
        if topk_idx.dtype != torch.int32 or not topk_idx.is_contiguous():
            raise ValueError("Pre-sorted GBSA indices must be contiguous int32")
    else:
        topk_idx = _sort_valid_gbsa_prefix(topk_idx, select_num_idx)
    block_table = block_table.to(dtype=torch.int32).contiguous()
    cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int64).contiguous()
    seq_lens = seq_lens.to(dtype=torch.int32).contiguous()

    quant_mode = _GBSA_NO_QUANT
    softmax_precision = _gbsa_softmax_precision(key.dtype)
    quant_kwargs: dict[str, Any] = {}
    if key.dtype == torch.float8_e4m3fn:
        if value.dtype != torch.float8_e4m3fn:
            raise TypeError("MiniMax-M3 FP8 sparse attention requires both K and V caches in E4M3")
        q = _to_fp8_e4m3(q)
        quant_mode = _GBSA_FP8_STATIC_CAST_P
        # GBSA quant mode 5 requires precision 1. BF16 follows the hardware policy.
        quant_kwargs = {"attention_out_dtype": torch.bfloat16}
    elif q.dtype != key.dtype or value.dtype != key.dtype:
        raise TypeError("MiniMax-M3 sparse attention requires Q, K, and V to have the same dtype")

    block_shape = [1, block_size]
    from cann_ops_transformer.ops import generic_block_sparse_attention

    # Opt-in only from the model's supported uniform-decode metadata builder. Direct
    # callers and prefill must not assume that layer-dependent counts match.
    # The cache belongs to one decode metadata object, so its sequence/prefix
    # tensors are shared by all entries. Include all other scheduling inputs.
    reuse_metadata = (
        gbsa_metadata_cache is not None
        and (
            torch.bfloat16 == key.dtype
            or (
                torch.float8_e4m3fn == key.dtype
                and get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)
            )
        )
        and q.dtype == key.dtype
        and block_size == _MSA_INDEX_BLOCK_SIZE
        and max_query_len > 0
        and q.shape[0] == seq_lens.numel() * max_query_len
    )
    cache = gbsa_metadata_cache if reuse_metadata else None
    cache_key = (
        tuple(q.shape),
        tuple(topk_idx.shape),
        tuple(select_num_idx.shape),
        num_kv_heads,
        max_query_len,
        key.dtype,
        softmax_precision,
    )
    gbsa_metadata = cache.get(cache_key) if cache is not None else None
    if gbsa_metadata is None:
        gbsa_metadata = _build_generic_block_sparse_attention_metadata(
            topk_idx,
            select_num_idx,
            cu_seqlens_q,
            seq_lens,
            num_heads=q.shape[1],
            num_kv_heads=num_kv_heads,
            head_size=q.shape[2],
            block_size=block_size,
            max_query_len=max_query_len,
            quant_mode=quant_mode,
            softmax_precision=softmax_precision,
        )
        if cache is not None:
            cache[cache_key] = gbsa_metadata
    out, _ = generic_block_sparse_attention(
        q,
        key,
        value,
        topk_idx,
        select_num_idx,
        block_shape,
        metadata=gbsa_metadata,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seq_lens,
        block_table=block_table,
        layout_q="TND",
        layout_kv="PA_BBND",
        softmax_scale=sm_scale,
        mask_mode=_GBSA_MASK_MODE_CAUSAL,
        quant_mode=quant_mode,
        softmax_precision=softmax_precision,
        return_softmax_lse=False,
        **quant_kwargs,
    )
    return out


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
    """Compute MSA index scores with the CANN operator.

    ``start_loc`` is the current query block in the complete block table.
    A causal mask selects sparse mode 3; passing no mask selects dense mode 0.
    ``init_blocks`` and ``local_blocks`` are kept for parity with the index
    scoring interface; candidate forcing is applied by the TopK stage.
    """
    from cann_ops_transformer.ops import msa_index_score

    index_kv_cache = _as_ascendc_index_kv_cache(index_kv_cache)
    if index_kv_cache.dtype == torch.float8_e4m3fn and idx_q.dtype != index_kv_cache.dtype:
        idx_q = _to_fp8_e4m3(idx_q)
    return msa_index_score(
        idx_q,
        index_kv_cache,
        start_loc,
        block_table=block_table,
        atten_mask=causal_mask,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_klen=seq_lens,
        layout_key="BBND",
        sparse_mode=3 if causal_mask is not None else 0,
        init_blocks=0,
        local_blocks=0,
    )


@torch.no_grad()
def _minimax_m3_index_score_topk(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor | tuple[torch.Tensor],
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    start_loc: torch.Tensor,
    causal_mask: torch.Tensor | None,
    *,
    topk: int,
    init_blocks: int = 0,
    local_blocks: int = 0,
) -> torch.Tensor:
    """Run the CANN MsaIndexScore TopK fusion and return its block indices.

    Returns ``[num_index_heads, total_q, topk]`` int32 logical block IDs, already
    ordered by the operator.
    """
    from cann_ops_transformer.ops import msa_index_score

    index_kv_cache = _as_ascendc_index_kv_cache(index_kv_cache)
    if index_kv_cache.dtype == torch.float8_e4m3fn and idx_q.dtype != index_kv_cache.dtype:
        idx_q = _to_fp8_e4m3(idx_q)
    return msa_index_score(
        idx_q,
        index_kv_cache,
        start_loc,
        return_choice=1,  # 1: only ``sparse_indices``; 0: the plain score
        topk=topk,
        block_table=block_table,
        atten_mask=causal_mask,
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_klen=seq_lens,
        layout_key="BBND",  # paged key: [num_blocks, block_size, heads, dim]
        # 3 requires ``atten_mask`` and 0 forbids it, so both follow the same call.
        sparse_mode=3 if causal_mask is not None else 0,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )


@lru_cache
def _index_score_has_return_choice() -> bool:
    """Whether the installed CANN package exposes the fused TopK output.

    ``return_choice`` was added with the TopK fusion, so its presence is what
    separates a new package from an older one. Inspect the wrapper signature
    rather than the op name: a package without the keyword would reject the call
    with ``TypeError`` instead of reaching the kernel.
    """
    try:
        from cann_ops_transformer.ops import msa_index_score

        return "return_choice" in signature(msa_index_score).parameters
    except Exception:  # noqa: BLE001 -- optional toolchain, any failure -> no fusion
        return False


def _can_use_index_score_topk_fusion(topk: int) -> bool:
    """Whether the fused TopK can serve this ``topk`` on this device.

    ``topk`` must be aligned to the kernel's bulk write-back. The kernel collapses
    the row only for that alignment and otherwise stores slot by slot, which with
    more than one query token returns uninitialised memory for every row but the
    last. Measured on Ascend950DT: ``topk`` in {4, 12, 17} at ``q_len`` in
    {2, 4, 8} produced values like ``1050427392`` and ``-8388609``; ``topk`` in
    {8, 16, 24, 32} was exact for every shape measured.
    """
    if not get_current_hardware_profile().supports(HardwareCapability.MINIMAX_M3_INDEX_SCORE_TOPK_FUSION):
        return False
    if not 0 < topk <= _INDEX_SCORE_TOPK_MAX or topk % _INDEX_SCORE_TOPK_WRITE_ALIGNMENT != 0:
        return False
    return _index_score_has_return_choice()


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
    """Compute AscendC prefill scores and finalize their block TopK.

    The CANN operator fuses the TopK into its score pass when the hardware and
    toolchain support ``return_choice``; otherwise the scores feed the separate
    TopK and sort helpers.
    """
    if _can_use_index_score_topk_fusion(topk):
        return _minimax_m3_index_score_topk(
            idx_q,
            index_kv_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            start_loc,
            causal_mask,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )
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
    sort_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute AscendC decode scores and fuse their TopK helpers.

    ``sort_indices`` serves the GBSA consumer, which requires ascending block
    IDs. The fused output is already globally ascending, so the fused results go
    straight through and the separate sort helper is only used on the score path.

    """
    if sort_indices and _can_use_index_score_topk_fusion(topk):
        indices = _minimax_m3_index_score_topk(
            idx_q,
            index_kv_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            start_loc,
            causal_mask,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )
        return indices, _select_num_idx_from_topk(indices)
    score = _minimax_m3_index_score(
        idx_q,
        index_kv_cache,
        block_table,
        cu_seqlens_q,
        seq_lens,
        start_loc,
        causal_mask,
    )
    return index_decode_topk_fused(
        score,
        context_lens,
        decode_query_len,
        topk,
        block_table.shape[-1],
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sort_indices=sort_indices,
    )


@torch.no_grad()
def _minimax_m3_sparse_attn_gbsa(
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
    *,
    max_query_len: int = -1,
) -> None:
    key, value = _split_main_kv_cache(kv_cache)
    out = _run_generic_block_sparse_attention(
        q,
        key,
        value,
        topk_idx,
        block_table,
        _select_num_idx_from_topk(topk_idx),
        cu_seqlens_q,
        seq_lens,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
        block_size=block_size,
        max_query_len=max_query_len,
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
    """Run sparse Prefill through CANN Ops generic block sparse attention."""
    del prefix_lens
    common_args = (
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
    _minimax_m3_sparse_attn_gbsa(*common_args, max_query_len=max_query_len)


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
    dequant_scale: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    gbsa_metadata_cache: dict[tuple[Any, ...], torch.Tensor] | None = None,
    indices_are_sorted: bool = False,
) -> None:
    """Run sparse decode through CANN Ops generic block sparse attention."""
    if q.shape[0] != seq_lens.shape[0] * decode_query_len:
        raise ValueError("Decode query tokens must equal request count times decode_query_len")

    key, value = _split_main_kv_cache(kv_cache)
    # The model metadata builder prepares this immutable uniform prefix once
    # per query length, outside the per-layer attention/graph execution.
    # Keep the fallback for direct callers that do not supply model metadata.
    if cu_seqlens_q is None:
        query_lens = torch.full_like(seq_lens, decode_query_len, dtype=torch.int32)
        cu_seqlens_q = _build_cu_seqlens_q(query_lens)
    elif cu_seqlens_q.shape != (seq_lens.shape[0] + 1,) or cu_seqlens_q.dtype != torch.int64:
        raise ValueError("GBSA cu_seqlens_q must be int64 with request_count + 1 entries")
    if select_num_idx is None:
        select_num_idx = _select_num_idx_from_topk(topk_idx)

    out = _run_generic_block_sparse_attention(
        q,
        key,
        value,
        topk_idx,
        block_table,
        select_num_idx,
        cu_seqlens_q,
        seq_lens,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
        block_size=block_size,
        max_query_len=decode_query_len,
        dequant_scale=dequant_scale,
        gbsa_metadata_cache=gbsa_metadata_cache,
        indices_are_sorted=indices_are_sorted,
    )
    output.copy_(out)
