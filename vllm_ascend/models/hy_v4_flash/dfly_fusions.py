# SPDX-License-Identifier: Apache-2.0
"""Fused draft input normalization and gated residual updates."""

import torch
from vllm.triton_utils import tl, triton

RESIDUAL_BLOCK = 1024


@triton.jit
def _compact_c8_indices_kernel(
    selected, order, indices, lengths, active, candidates: tl.constexpr, block: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    live = col < candidates
    valid = live & (tl.load(selected + row * candidates + col, live, -1) >= 0)
    rank = tl.cumsum(valid.to(tl.int32), 0) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    # The two store masks address disjoint output ranges.
    tl.store(order + row * candidates + rank, col.to(tl.int64), valid)
    tl.store(order + row * candidates + col, 0, live & (col >= count))
    native_index = tl.where(col < count, col, -1)
    # A fully padded row uses one safe physical KV row and is zeroed below.
    native_index = tl.where((count == 0) & (col == 0), 0, native_index)
    tl.store(indices + row * candidates + col, native_index, live)
    tl.store(lengths + row, tl.maximum(count, 1))
    tl.store(active + row, count > 0)


def compact_draft_c8_rows(rows, selected):
    """Pack valid candidate rows before calling the native sparse attention API."""
    batch, candidates, row_bytes = rows.shape
    queries = selected.shape[1] if selected.ndim == 3 else 1
    count = batch * queries
    order = torch.empty((count, candidates), device=rows.device, dtype=torch.int64)
    indices = torch.empty((count, candidates), device=rows.device, dtype=torch.int32)
    lengths = torch.empty(count, device=rows.device, dtype=torch.int32)
    active = torch.empty(count, device=rows.device, dtype=torch.bool)
    _compact_c8_indices_kernel[(count,)](
        selected.contiguous(), order, indices, lengths, active, candidates, triton.next_power_of_2(candidates)
    )
    source = rows[:, None].expand(batch, queries, candidates, row_bytes)
    gather = order.view(batch, queries, candidates, 1).expand(-1, -1, -1, row_bytes)
    packed = source.gather(2, gather).reshape(count, candidates, row_bytes)
    return packed, indices[:, None, None, :], lengths, active


@triton.jit
def _dual_rms_concat_kernel(
    a,
    b,
    wa,
    wb,
    out,
    a_stride: tl.constexpr,
    b_stride: tl.constexpr,
    a_col: tl.constexpr,
    b_col: tl.constexpr,
    width: tl.constexpr,
    eps_a: tl.constexpr,
    eps_b: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    side = tl.program_id(1)
    col = tl.arange(0, block)
    if side == 0:
        x = tl.load(a + row * a_stride + col * a_col, col < width, 0).to(tl.float32)
        weight = tl.load(wa + col, col < width, 0).to(tl.float32)
        inverse = tl.rsqrt(tl.sum(x * x, 0) / width + eps_a)
    else:
        x = tl.load(b + row * b_stride + col * b_col, col < width, 0).to(tl.float32)
        weight = tl.load(wb + col, col < width, 0).to(tl.float32)
        inverse = tl.rsqrt(tl.sum(x * x, 0) / width + eps_b)
    value = (x * inverse) * weight
    tl.store(out + row * (2 * width) + side * width + col, value, col < width)


@triton.jit
def _gated_residual_kernel(anchor, branch, gate, out, count: tl.constexpr, width: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    live = index < count
    x = tl.load(anchor + index, live, 0).to(tl.float32)
    y = tl.load(branch + index, live, 0).to(tl.float32)
    gain = tl.load(gate + index % width, live, 0).to(tl.float32)
    update = (gain * y).to(branch.dtype.element_ty).to(tl.float32)
    tl.store(out + index, x + update, live)


def dual_rms_concat(a, b, norm_a, norm_b):
    rows, width = a.shape
    output = torch.empty((rows, 2 * width), device=a.device, dtype=a.dtype)
    _dual_rms_concat_kernel[(rows, 2)](
        a,
        b,
        norm_a.weight,
        norm_b.weight,
        output,
        a.stride(0),
        b.stride(0),
        a.stride(1),
        b.stride(1),
        width,
        norm_a.variance_epsilon,
        norm_b.variance_epsilon,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return output


def gated_residual(anchor, branch, gate):
    output = torch.empty_like(anchor)
    _gated_residual_kernel[(triton.cdiv(anchor.numel(), RESIDUAL_BLOCK),)](
        anchor,
        branch,
        gate,
        output,
        anchor.numel(),
        anchor.shape[-1],
        RESIDUAL_BLOCK,
        enable_fp_fusion=False,
    )
    return output
