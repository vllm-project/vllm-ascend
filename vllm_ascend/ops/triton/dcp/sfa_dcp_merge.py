# SPDX-License-Identifier: Apache-2.0
"""FP32 local merge for the small-batch DCP exchange."""

from __future__ import annotations

import math

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_element, get_vectorcore_num, init_device_properties_triton


@triton.jit(
    do_not_specialize=[
        "lse_stride_axis1",
        "head_count",
        "total_tiles",
        "num_programs",
    ]
)
def _fused_merge_kernel(
    out_ptr,
    lse_ptr,
    result_ptr,
    out_stride_rank,
    out_stride_axis1,
    out_stride_axis2,
    LSE_RANK_STRIDE: tl.constexpr,
    lse_stride_axis1,
    head_count,
    head_dim,
    total_tiles,
    num_programs,
    RANKS: tl.constexpr,
    NATIVE_LAYOUT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    rank_ids = tl.arange(0, RANKS)
    for tile in tl.range(pid, total_tiles, num_programs):
        row = tile // tl.cdiv(head_dim, BLOCK_D)
        dim = (tile % tl.cdiv(head_dim, BLOCK_D)) * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = dim < head_dim
        token = row // head_count
        head = row % head_count
        if NATIVE_LAYOUT:
            lse_offsets = rank_ids * LSE_RANK_STRIDE + head * lse_stride_axis1 + token
        else:
            lse_offsets = rank_ids * LSE_RANK_STRIDE + token * lse_stride_axis1 + head
        raw_lse = tl.load(lse_ptr + lse_offsets).to(tl.float32)
        finite = (raw_lse == raw_lse) & (raw_lse > float("-inf")) & (raw_lse < float("inf"))
        safe_lse = tl.where(finite, raw_lse, float("-inf"))
        max_lse = tl.max(safe_lse, axis=0)
        # An all-invalid row must not evaluate -inf - (-inf), even in a
        # branch later discarded by tl.where.
        safe_max = tl.where(max_lse > float("-inf"), max_lse, 0.0)
        shifted = tl.where(finite, safe_lse - safe_max, 0.0)
        exponent = tl.where(finite, tl.exp(shifted), 0.0)
        denominator = tl.sum(exponent, axis=0)
        safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
        weights = exponent / safe_denominator
        result = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for rank in tl.static_range(0, RANKS):
            weight = get_element(weights, (rank,))
            if NATIVE_LAYOUT:
                out_offsets = rank * out_stride_rank + head * out_stride_axis1 + token * out_stride_axis2 + dim
            else:
                out_offsets = rank * out_stride_rank + token * out_stride_axis1 + head * out_stride_axis2 + dim
            values = tl.load(out_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
            # Match upstream: invalid-rank NaN/Inf outputs must not leak
            # through a zero LSE weight (NaN * 0 is still NaN).
            values = tl.where(get_element(finite, (rank,)), values, 0.0)
            result += values * weight
        tl.store(result_ptr + (token * head_count + head) * head_dim + dim, result, mask=mask)


def fused_merge(out_recv: torch.Tensor, lse_recv: torch.Tensor, token_dim: int) -> torch.Tensor:
    """Merge the validated rank/token/head payload produced by exchange."""
    ranks = out_recv.shape[0]
    if token_dim == 2:
        _, heads, tokens, head_dim = out_recv.shape
        native_layout = True
    else:
        _, tokens, heads, head_dim = out_recv.shape
        native_layout = False
    result = torch.empty((tokens, heads, head_dim), dtype=torch.float32, device=out_recv.device)
    # Bound each program to one feature accumulator and a rank-weight vector.
    block_d = min(triton.next_power_of_2(head_dim), 512)
    tiles = tokens * heads * math.ceil(head_dim / block_d)
    init_device_properties_triton()
    programs = min(tiles, get_vectorcore_num())
    _fused_merge_kernel[(programs,)](
        out_recv,
        lse_recv,
        result,
        # Eligibility guarantees unit innermost strides. The result is always
        # allocated contiguous, so those strides need no kernel arguments.
        out_recv.stride(0),
        out_recv.stride(1),
        out_recv.stride(2),
        lse_recv.stride(0),
        lse_recv.stride(1),
        heads,
        head_dim,
        total_tiles=tiles,
        num_programs=programs,
        RANKS=ranks,
        NATIVE_LAYOUT=native_layout,
        BLOCK_D=block_d,
    )
    return result
