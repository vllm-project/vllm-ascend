# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm_ascend.utils import enable_custom_op

_BLOCK_SIZE = 64
_GROUP_SIZE = 64
_HEAD_DIM = 128
_BLOCKS_PER_QUERY_TILE = 2
_ALIGNMENT = 8
_MAX_BLOCKS = 1 << 16
_SCORE_BUFFER_BYTES = 64 * 1024 * 1024


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    sink_tokens: int = 0,
    sink_start: int | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run noncausal Sol-Attn with diagonal threshold estimation on Ascend 910B.

    Q, K, and V are contiguous BF16 tensors of shape [B, T, H, 128] on one
    NPU. The default scale is 128**-0.5. Larger tau values increase the
    threshold for exact block attention. A nonempty sink covers a contiguous
    range of exact KV blocks and defaults to the last sink_tokens input tokens.
    The optional FP32 base-2 log-sum-exp tensor has shape [B, H, T].
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Sol-Attn requires equal Q/K/V shapes [B, T, H, 128].")
    batch, tokens, heads, head_dim = q.shape
    if batch == 0 or tokens == 0 or heads == 0 or head_dim != _HEAD_DIM:
        raise ValueError("Sol-Attn requires positive B, T, H and a head dimension of 128.")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise TypeError("Sol-Attn requires BF16 Q, K, and V.")
    if q.device.type != "npu" or k.device != q.device or v.device != q.device:
        raise ValueError("Sol-Attn requires Q, K, and V on the same NPU.")
    if any(not t.is_contiguous() for t in (q, k, v)):
        raise ValueError("Sol-Attn requires contiguous BTHD tensors.")
    if type(sink_tokens) is not int or (sink_start is not None and type(sink_start) is not int):
        raise TypeError("Sol-Attn sink_tokens and sink_start must be integers.")
    if not 0 <= sink_tokens <= tokens:
        raise ValueError("Sol-Attn sink_tokens must be in [0, T].")
    start = tokens - sink_tokens if sink_start is None else sink_start
    if not 0 <= start <= tokens - sink_tokens:
        raise ValueError("Sol-Attn sink range must be within [0, T].")
    scale_log2 = (head_dim**-0.5 if scale is None else scale) * math.log2(math.e)
    if not math.isfinite(scale_log2) or not math.isfinite(tau):
        raise ValueError("Sol-Attn scale and tau must be finite.")

    num_blocks = (tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    if num_blocks > _MAX_BLOCKS:
        raise ValueError(f"Sol-Attn supports at most {_MAX_BLOCKS} KV blocks.")
    num_groups = (num_blocks + _GROUP_SIZE - 1) // _GROUP_SIZE
    padded_blocks = num_groups * _GROUP_SIZE
    padded_tokens = num_blocks * _BLOCK_SIZE
    query_tiles = (num_blocks + _BLOCKS_PER_QUERY_TILE - 1) // _BLOCKS_PER_QUERY_TILE
    padded_query_tiles = (query_tiles + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
    padded_lse = (tokens + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
    sink_start_block = start // _BLOCK_SIZE if sink_tokens else num_blocks
    sink_end_block = (start + sink_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE if sink_tokens else num_blocks

    if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "sol_attn_fwd_m128"):
        raise RuntimeError("Sol-Attn requires the vLLM Ascend custom operators for Ascend 910B.")
    ops = torch.ops._C_ascend
    device = q.device
    k_pad = torch.empty((batch, heads, padded_tokens, head_dim), device=device, dtype=q.dtype)
    v_pad = torch.empty_like(k_pad)
    k_summary = torch.empty((batch, heads, padded_blocks, head_dim), device=device, dtype=q.dtype)
    v_summary = torch.empty_like(k_summary)
    k_summary_fp32 = torch.empty_like(k_summary, dtype=torch.float32)
    q_summary = torch.empty((batch, heads, num_blocks, head_dim), device=device, dtype=torch.float32)
    tail_lengths = torch.empty((_GROUP_SIZE,), device=device, dtype=torch.float32)
    threshold = torch.empty((batch, heads, num_blocks), device=device, dtype=torch.float32)
    ops.sol_attn_prepare_qkv(q, k, v, k_pad, v_pad, k_summary, v_summary, k_summary_fp32, q_summary, tail_lengths)
    ops.sol_attn_prepare_diag_thresh(q_summary, k_summary_fp32, scale_log2, tau, threshold)

    head_count = batch * heads
    mask = torch.empty((head_count, query_tiles, num_groups), device=device, dtype=torch.int64)
    exact_count = torch.empty((head_count, padded_query_tiles), device=device, dtype=torch.int32)
    exact_index = torch.empty((head_count, query_tiles, padded_blocks), device=device, dtype=torch.int16)
    bytes_per_head = num_blocks * padded_blocks * q_summary.element_size()
    heads_per_chunk = min(head_count, max(1, _SCORE_BUFFER_BYTES // bytes_per_head))
    score_buffer = torch.empty((heads_per_chunk, num_blocks, padded_blocks), device=device, dtype=torch.float32)
    q_blocks = q_summary.flatten(0, 1)
    k_blocks = k_summary_fp32.flatten(0, 1).transpose(1, 2)
    thresholds = threshold.flatten(0, 1)

    # Each head chunk writes its slice of the final routing buffers directly.
    for first_head in range(0, head_count, heads_per_chunk):
        last_head = min(head_count, first_head + heads_per_chunk)
        chunk = slice(first_head, last_head)
        scores = score_buffer[: last_head - first_head]
        torch.bmm(q_blocks[chunk], k_blocks[chunk], out=scores)
        ops.sol_attn_prepare_route_pack(
            scores,
            thresholds[chunk],
            sink_start_block,
            sink_end_block,
            scale_log2,
            mask[chunk].view(torch.uint64),
            exact_count[chunk],
            exact_index[chunk].view(torch.uint16),
        )

    output = torch.empty_like(q)
    lse = torch.empty((batch, heads, padded_lse), device=device, dtype=torch.float32)
    ops.sol_attn_fwd_m128(
        q,
        k_pad,
        v_pad,
        k_summary,
        v_summary,
        exact_count.view(batch, heads, padded_query_tiles),
        exact_index.view(torch.uint16).view(batch, heads, query_tiles, padded_blocks),
        mask.view(torch.uint64).view(batch, heads, query_tiles, num_groups),
        tail_lengths,
        scale_log2,
        output,
        lse,
    )
    return (output, lse[..., :tokens]) if return_lse else output
