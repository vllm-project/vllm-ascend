# SPDX-License-Identifier: Apache-2.0
"""Decode-only Engram pointwise fusions with explicit BF16 boundaries."""

import torch
from vllm.triton_utils import tl, triton

_RESIDUAL_BLOCK = 1024


@triton.jit
def _residual_kernel(
    hidden,
    value,
    gamma,
    inverse,
    gate,
    scales,
    hashes,
    output,
    rows: tl.constexpr,
    channels: tl.constexpr,
    width: tl.constexpr,
    hash_width: tl.constexpr,
    hash_block: tl.constexpr,
    block: tl.constexpr,
):
    row_channel = tl.program_id(0)
    row = row_channel // channels
    channel = row_channel % channels
    column = tl.program_id(1) * block + tl.arange(0, block)
    live = column < width
    source = tl.load(value + row * width + column, live, 0).to(tl.float32)
    gain = tl.load(gamma + column, live, 0).to(tl.float32)
    rstd = tl.load(inverse + row).to(tl.float32)
    # Preserve gamma * value * inverse ordering, with contraction disabled.
    signal = (gain * source) * rstd
    probability = tl.load(gate + row_channel).to(tl.float32)
    offset = tl.load(scales + channel * width + column, live, 0).to(tl.float32)
    residual = ((probability + offset) * signal).to(tl.bfloat16).to(tl.float32)
    hash_index = tl.arange(0, hash_block)
    index = tl.load(hashes + row * hash_width + hash_index, hash_index < hash_width, -1)
    any_valid = tl.sum((index >= 0).to(tl.int32), 0) > 0
    residual = tl.where(any_valid, residual, 0.0)
    previous = tl.load(hidden + row_channel * width + column, live, 0).to(tl.float32)
    result = (previous + residual).to(tl.bfloat16)
    tl.store(output + row_channel * width + column, result, live)


def fused_residual(hidden, value, gamma, inverse, gate, scales, hashes):
    rows, channels, width = hidden.shape
    output = torch.empty_like(hidden)
    _residual_kernel[(rows * channels, triton.cdiv(width, _RESIDUAL_BLOCK))](
        hidden,
        value,
        gamma,
        inverse,
        gate,
        scales,
        hashes,
        output,
        rows,
        channels,
        width,
        hashes.shape[1],
        triton.next_power_of_2(hashes.shape[1]),
        _RESIDUAL_BLOCK,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _lookup_norm_kernel(
    source,
    hashes,
    gamma,
    output,
    row_stride: tl.constexpr,
    head_stride: tl.constexpr,
    col_stride: tl.constexpr,
    hash_row_stride: tl.constexpr,
    hash_head_stride: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, block)
    valid = tl.load(hashes + row * hash_row_stride + head * hash_head_stride) >= 0
    value = tl.load(
        source + row * row_stride + head * head_stride + col * col_stride,
        col < width,
        0,
    ).to(tl.float32)
    value = tl.where(valid, value, 0.0)
    inverse = tl.rsqrt(tl.sum(value * value, 0) / width + epsilon)
    weight = tl.load(gamma + head * width + col, col < width, 0).to(tl.float32)
    normalized = (value * inverse) * weight
    tl.store(output + (row * heads + head) * width + col, normalized, col < width)


def fused_lookup_norm(source, hashes, gamma, epsilon, output_dtype):
    rows, heads, width = source.shape
    output = torch.empty(source.shape, device=source.device, dtype=output_dtype)
    _lookup_norm_kernel[(rows, heads)](
        source,
        hashes,
        gamma,
        output,
        source.stride(0),
        source.stride(1),
        source.stride(2),
        hashes.stride(0),
        hashes.stride(1),
        heads,
        width,
        epsilon,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
        enable_graph_optimize=False,
    )
    return output


@triton.jit
def _norm_dot_gate_kernel(
    key,
    query,
    gamma_key,
    gamma_query,
    output,
    key_row_stride: tl.constexpr,
    key_col_stride: tl.constexpr,
    query_row_stride: tl.constexpr,
    query_col_stride: tl.constexpr,
    width: tl.constexpr,
    eps_key: tl.constexpr,
    eps_query: tl.constexpr,
    scale: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    live = col < width
    k = tl.load(key + row * key_row_stride + col * key_col_stride, live, 0).to(tl.float32)
    q = tl.load(query + row * query_row_stride + col * query_col_stride, live, 0).to(tl.float32)
    wk = tl.load(gamma_key + col, live, 0).to(tl.float32)
    wq = tl.load(gamma_query + col, live, 0).to(tl.float32)
    ik = tl.rsqrt(tl.sum(k * k, 0) / width + eps_key)
    iq = tl.rsqrt(tl.sum(q * q, 0) / width + eps_query)
    nk = ((k * ik) * wk).to(key.dtype.element_ty).to(tl.float32)
    nq = ((q * iq) * wq).to(query.dtype.element_ty).to(tl.float32)
    product = (nk * nq).to(output.dtype.element_ty).to(tl.float32)
    dot = tl.sum(product, 0).to(output.dtype.element_ty).to(tl.float32)
    scaled = (dot * scale).to(output.dtype.element_ty).to(tl.float32)
    floor = tl.full((), 1.0e-6, tl.float32).to(output.dtype.element_ty).to(tl.float32)
    magnitude = tl.maximum(tl.abs(scaled), floor, propagate_nan=tl.PropagateNan.ALL)
    magnitude = magnitude.to(output.dtype.element_ty).to(tl.float32)
    root = tl.sqrt(magnitude).to(output.dtype.element_ty).to(tl.float32)
    sign = tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))
    logit = (root * sign).to(output.dtype.element_ty).to(tl.float32)
    tl.store(output + row, tl.sigmoid(logit))


def fused_norm_dot_gate(key, query, key_norm, query_norm):
    rows, width = key.shape
    output = torch.empty((rows,), device=key.device, dtype=torch.promote_types(key.dtype, query.dtype))
    eps_key = key_norm.eps if key_norm.eps is not None else torch.finfo(torch.float32).eps
    eps_query = query_norm.eps if query_norm.eps is not None else torch.finfo(torch.float32).eps
    _norm_dot_gate_kernel[(rows,)](
        key,
        query,
        key_norm.weight,
        query_norm.weight,
        output,
        key.stride(0),
        key.stride(1),
        query.stride(0),
        query.stride(1),
        width,
        eps_key,
        eps_query,
        width**-0.5,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return output
