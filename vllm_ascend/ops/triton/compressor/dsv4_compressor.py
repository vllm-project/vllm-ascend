# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 C4/C128 compression after adapter-aware FP32 projections."""

import torch
from vllm.triton_utils import tl, triton

# Limit the C128 pooling tile to 128 x 32 FP32 elements per intermediate.
POOL_HEAD_BLOCK = 32


@triton.jit(do_not_specialize=["num_requests", "state_stride", "table_stride"])
def _pool_projected(
    kv_ptr,
    score_ptr,
    state_ptr,
    ape_ptr,
    table_ptr,
    cu_ptr,
    start_ptr,
    pooled_ptr,
    num_requests,
    state_stride,
    table_stride,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    COEFFICIENT: tl.constexpr,
    STATE_BLOCK_SIZE: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    output_row = tl.program_id(0)
    columns = tl.program_id(1) * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    requests = tl.arange(0, REQUEST_BLOCK)
    query_begin = tl.load(cu_ptr + requests, requests < num_requests, 0)
    query_end = tl.load(cu_ptr + requests + 1, requests < num_requests, 0)
    starts = tl.load(start_ptr + requests, requests < num_requests, -1)
    counts = tl.where(
        (requests < num_requests) & (starts >= 0),
        (starts + query_end - query_begin) // RATIO - starts // RATIO,
        0,
    )
    output_ends = tl.cumsum(counts)
    output_begins = output_ends - counts
    owner = (output_row >= output_begins) & (output_row < output_ends)
    valid_output = tl.sum(owner.to(tl.int32), 0) > 0
    pooled = tl.full((HEAD_BLOCK,), 0.0, tl.float32)
    if valid_output:
        request = tl.sum(tl.where(owner, requests, 0), 0)
        start = tl.sum(tl.where(owner, starts, 0), 0)
        input_begin = tl.sum(tl.where(owner, query_begin, 0), 0)
        output_begin = tl.sum(tl.where(owner, output_begins, 0), 0)
        group = start // RATIO + output_row - output_begin
        rows = tl.arange(0, COEFFICIENT * RATIO)
        # C4 joins the previous group's first head and current group's second
        # head; the first group has no previous head and masks it from softmax.
        positions = (group - COEFFICIENT + 1) * RATIO + rows
        head_offsets = rows // RATIO * HEAD_DIM
        valid = positions >= 0
        current = positions >= start
        width = COEFFICIENT * HEAD_DIM
        projection_rows = input_begin + tl.maximum(positions - start, 0)
        projection_offsets = projection_rows[:, None] * width + head_offsets[:, None] + columns[None, :]
        kv = tl.load(kv_ptr + projection_offsets, (valid & current)[:, None], 0)
        scores = tl.load(score_ptr + projection_offsets, (valid & current)[:, None], 0)
        ape = tl.load(
            ape_ptr + (tl.maximum(positions, 0) % RATIO)[:, None] * width + head_offsets[:, None] + columns[None, :],
            valid[:, None],
            0,
        )
        page = tl.load(
            table_ptr + request * table_stride + tl.maximum(positions, 0) // STATE_BLOCK_SIZE,
            valid & ~current,
            0,
        ).to(tl.int64)
        state_offsets = (
            page[:, None] * state_stride
            + (tl.maximum(positions, 0) % STATE_BLOCK_SIZE)[:, None] * 2 * width
            + head_offsets[:, None]
            + columns[None, :]
        )
        history_kv = tl.load(state_ptr + state_offsets, (valid & ~current)[:, None], 0)
        history_scores = tl.load(state_ptr + state_offsets + width, (valid & ~current)[:, None], 0)
        kv = tl.where(current[:, None], kv, history_kv)
        scores = tl.where(current[:, None], scores + ape, history_scores)
        scores = tl.where(valid[:, None], scores, -float("inf"))
        probabilities = tl.exp(scores - tl.max(scores, 0)[None, :])
        pooled = tl.sum(probabilities * kv, 0) / tl.sum(probabilities, 0)
    tl.store(pooled_ptr + output_row * HEAD_DIM + columns, pooled)


@triton.jit(do_not_specialize=["norm_eps"])
def _normalize_projected(
    pooled_ptr,
    norm_ptr,
    sin_ptr,
    cos_ptr,
    output_ptr,
    norm_eps,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    norm_columns = tl.arange(0, HEAD_DIM)
    values = tl.load(pooled_ptr + row * HEAD_DIM + norm_columns)
    inverse_rms = tl.rsqrt(tl.sum(values * values, 0) / HEAD_DIM + norm_eps)
    normalized = values * inverse_rms * tl.load(norm_ptr + norm_columns)
    rope_columns = tl.arange(0, ROPE_DIM)
    rope_start = HEAD_DIM - ROPE_DIM
    rope_values = tl.load(pooled_ptr + row * HEAD_DIM + rope_start + rope_columns)
    rope_values *= inverse_rms * tl.load(norm_ptr + rope_start + rope_columns)
    partner_columns = rope_columns ^ 1
    partners = tl.load(pooled_ptr + row * HEAD_DIM + rope_start + partner_columns)
    partners *= inverse_rms * tl.load(norm_ptr + rope_start + partner_columns)
    signs = tl.where(rope_columns % 2 == 0, -1.0, 1.0)
    rotated = rope_values * tl.load(cos_ptr + row * ROPE_DIM + rope_columns) + partners * signs * tl.load(
        sin_ptr + row * ROPE_DIM + rope_columns
    )
    tl.store(output_ptr + row * HEAD_DIM + norm_columns, normalized, norm_columns < rope_start)
    tl.store(output_ptr + row * HEAD_DIM + rope_start + rope_columns, rotated)


@triton.jit(do_not_specialize=["num_requests", "state_stride", "table_stride"])
def _write_projected_state(
    kv_ptr,
    score_ptr,
    state_ptr,
    ape_ptr,
    table_ptr,
    cu_ptr,
    start_ptr,
    num_requests,
    state_stride,
    table_stride,
    HEAD_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    COEFFICIENT: tl.constexpr,
    STATE_BLOCK_SIZE: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    requests = tl.arange(0, REQUEST_BLOCK)
    begin = tl.load(cu_ptr + requests, requests < num_requests, 0)
    end = tl.load(cu_ptr + requests + 1, requests < num_requests, 0)
    starts = tl.load(start_ptr + requests, requests < num_requests, -1)
    owner = (requests < num_requests) & (row >= begin) & (row < end) & (starts >= 0)
    if tl.sum(owner.to(tl.int32), 0) > 0:
        request = tl.sum(tl.where(owner, requests, 0), 0)
        position = tl.sum(tl.where(owner, starts + row - begin, 0), 0)
        page = tl.load(table_ptr + request * table_stride + position // STATE_BLOCK_SIZE).to(tl.int64)
        # Page zero denotes evicted/unallocated state and must stay intact.
        if page > 0:
            width = COEFFICIENT * HEAD_DIM
            state_columns = tl.arange(0, 2 * COEFFICIENT * HEAD_DIM)
            projection_columns = state_columns % width
            state_offset = page * state_stride + (position % STATE_BLOCK_SIZE) * 2 * width + state_columns
            kv = tl.load(kv_ptr + row * width + projection_columns)
            scores = tl.load(score_ptr + row * width + projection_columns)
            scores += tl.load(ape_ptr + (position % RATIO) * width + projection_columns)
            # Write KV and scores together as one contiguous state row.
            tl.store(state_ptr + state_offset, tl.where(state_columns < width, kv, scores))


def compressor_from_projected(
    kv: torch.Tensor,
    scores: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    *,
    state_block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    start_pos: torch.Tensor,
    compress_ratio: int,
    norm_eps: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Compress FP32 projections and update paged ``[kv, score+APE]`` state.

    Returns compact request-major BF16/FP16 rows followed by zero padding, with
    the row count supplied by the fused compressor's RoPE metadata. Controls stay on device;
    padded requests have zero query length or a negative start. State pages may
    have padding in their outer stride. Page zero is never modified. Pooling
    completes before cache writes, preserving leading history during prefill.
    """
    if compress_ratio not in (4, 128):
        raise ValueError("DeepSeek V4 compression requires ratio 4 or 128")
    coefficient = 2 if compress_ratio == 4 else 1
    head_dim = norm_weight.numel()
    if head_dim not in (128, 512) or kv.ndim != 2 or kv.shape[1] != coefficient * head_dim:
        raise ValueError("Unsupported DeepSeek V4 projection shape")
    if kv.shape != scores.shape or kv.dtype != torch.float32 or scores.dtype != torch.float32:
        raise ValueError("KV and score projections must have matching shapes and FP32 dtype")
    if not kv.is_contiguous() or not scores.is_contiguous():
        raise ValueError("KV and score projections must be contiguous")
    if state_cache.dtype != torch.float32 or state_cache.shape[-1] != 2 * coefficient * head_dim:
        raise ValueError("Compressor state must contain FP32 KV and score planes")
    if state_cache.stride(-1) != 1 or state_cache.stride(-2) != state_cache.shape[-1]:
        raise ValueError("Compressor state rows must be contiguous")
    if output_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Compressed output must be BF16 or FP16")
    if ape.shape != (compress_ratio, coefficient * head_dim) or ape.dtype != torch.float32:
        raise ValueError("APE must contain FP32 biases for every compression position")
    if norm_weight.dtype != torch.float32 or norm_weight.ndim != 1:
        raise ValueError("Compressor normalization weights must be a FP32 vector")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() != start_pos.numel() + 1:
        raise ValueError("Query offsets must delimit every request")
    if state_block_table.ndim != 2 or state_block_table.shape[0] != start_pos.numel():
        raise ValueError("State block table must contain one row per request")
    if any(tensor.dtype != torch.int32 for tensor in (state_block_table, cu_seqlens, start_pos)):
        raise ValueError("Compressor controls must use INT32")
    if rope_cos.ndim != 2 or rope_cos.shape[-1] != 64 or rope_cos.shape != rope_sin.shape:
        raise ValueError("Compressor RoPE metadata must contain matching 64-column tables")
    if rope_cos.dtype not in (torch.float32, torch.bfloat16, torch.float16) or rope_cos.dtype != rope_sin.dtype:
        raise ValueError("Unsupported compressor RoPE dtype")
    if any(not tensor.is_contiguous() for tensor in (ape, norm_weight, rope_sin, rope_cos, cu_seqlens, start_pos)):
        raise ValueError("Projection metadata tensors must be contiguous")
    if state_block_table.stride(-1) != 1:
        raise ValueError("State block table rows must be contiguous")
    num_tokens, num_requests = kv.shape[0], start_pos.numel()
    num_outputs = rope_sin.shape[0]
    output = torch.zeros((num_outputs, head_dim), dtype=output_dtype, device=kv.device)
    if not num_tokens or not num_requests:
        return output
    pooled = torch.empty((num_outputs, head_dim), dtype=torch.float32, device=kv.device)
    common = dict(
        num_requests=num_requests,
        state_stride=state_cache.stride(0),
        table_stride=state_block_table.stride(0),
        HEAD_DIM=head_dim,
        RATIO=compress_ratio,
        COEFFICIENT=coefficient,
        STATE_BLOCK_SIZE=state_cache.shape[1],
        REQUEST_BLOCK=triton.next_power_of_2(num_requests),
    )
    if num_outputs:
        _pool_projected[(num_outputs, triton.cdiv(head_dim, POOL_HEAD_BLOCK))](
            kv,
            scores,
            state_cache,
            ape,
            state_block_table,
            cu_seqlens,
            start_pos,
            pooled,
            HEAD_BLOCK=POOL_HEAD_BLOCK,
            **common,
        )
        _normalize_projected[(num_outputs,)](
            pooled,
            norm_weight,
            rope_sin,
            rope_cos,
            output,
            norm_eps=norm_eps,
            HEAD_DIM=head_dim,
            ROPE_DIM=rope_cos.shape[-1],
        )
    _write_projected_state[(num_tokens,)](
        kv,
        scores,
        state_cache,
        ape,
        state_block_table,
        cu_seqlens,
        start_pos,
        **common,
    )
    return output
