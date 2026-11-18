# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Five-kernel diagnostic PLE gate with four BF16 memory boundaries."""

import torch
from vllm.triton_utils import tl, triton


def ple_gate_materialized(key, value, hidden, norm_key, norm_query, norm_conv, *, hc=4, eps=1e-6):
    """Return gated and normed tensors; preserve all inputs and four scalars.

    Flattened hidden/key are [T, HC*H], value is [T, H]. Key and value
    may be strided row slices of the same KV projection. No cache is touched.
    The four returned BF16 scratch tensors expose dot, scaled dot, signed
    square root, and sigmoid, respectively, for same-input diagnosis.
    """
    if hidden.ndim != 2 or value.ndim != 2 or hc <= 0 or eps <= 0:
        raise ValueError("Expected 2D hidden/value, positive hc and eps")
    tokens, width = value.shape
    if width == 0 or hidden.shape != (tokens, hc * width) or key.shape != hidden.shape:
        raise ValueError("Expected key/hidden [T, HC*H] and value [T, H]")
    inputs = (key, value, hidden, norm_key, norm_query, norm_conv)
    if any(t.dtype != torch.bfloat16 or t.device != hidden.device for t in inputs):
        raise ValueError("All inputs must be BF16 on the same device")
    if hidden.device.type != "npu":
        raise ValueError("This diagnostic wrapper targets Ascend NPU")
    if not hidden.is_contiguous() or key.stride(1) != 1 or value.stride(1) != 1:
        raise ValueError("hidden must be contiguous; key/value require unit inner stride")
    if any(w.shape != (hc * width,) or not w.is_contiguous() for w in inputs[3:]):
        raise ValueError("Norm weights must be contiguous [HC*H]")
    gated, normed = torch.empty_like(hidden), torch.empty_like(hidden)
    scalars = [hidden.new_empty((tokens, hc)) for _ in range(4)]
    if tokens:
        kernels = (gate_0_1, gate_1_2, gate_2_3, gate_3_4, gate_4_5)
        for index, kernel in enumerate(kernels):
            previous = scalars[max(0, index - 1)]
            current = scalars[min(index, 3)]
            kernel[(tokens, hc)](
                previous,
                current,
                key,
                value,
                hidden,
                norm_key,
                norm_query,
                norm_conv,
                gated,
                normed,
                key.stride(0),
                value.stride(0),
                eps,
                H=width,
                HC=hc,
                BLOCK_H=triton.next_power_of_2(width),
                launch_pdl=False,
                num_warps=4,
            )
    return gated, normed, tuple(scalars)


@triton.jit
def gate_0_1(
    scratch_in,
    scratch_out,
    key_ptr,  # [T, HC*H] with row stride key_rs
    value_ptr,  # [T, H] with row stride value_rs
    hidden_ptr,  # [T, HC*H]
    nk_ptr,  # [HC*H] norm_key weight
    nq_ptr,  # [HC*H] norm_query weight
    ncw_ptr,  # [HC*H] norm_conv weight
    gated_ptr,  # [T, HC*H] out
    normed_ptr,  # [T, HC*H] out
    key_rs,
    value_rs,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_H)
    mask = lanes < H
    offs = s * H + lanes
    dtype: tl.constexpr = key_ptr.dtype.element_ty

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    k = tl.load(key_ptr + t * key_rs + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(hidden_ptr + t * HC * H + offs, mask=mask, other=0.0).to(tl.float32)
    nk = tl.load(nk_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    nq = tl.load(nq_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Match eager materialization at each intermediate tensor boundary.
    k_n = (k * tl.rsqrt(tl.sum(k * k) / H + eps) * (1.0 + nk)).to(dtype)
    q_n = (q * tl.rsqrt(tl.sum(q * q) / H + eps) * (1.0 + nq)).to(dtype)
    products = (k_n.to(tl.float32) * q_n.to(tl.float32)).to(dtype)
    dot = tl.sum(products.to(tl.float32)).to(dtype).to(tl.float32)
    tl.store(scratch_out + t * HC + s, dot)


@triton.jit
def gate_1_2(
    scratch_in,
    scratch_out,
    key_ptr,  # [T, HC*H] with row stride key_rs
    value_ptr,  # [T, H] with row stride value_rs
    hidden_ptr,  # [T, HC*H]
    nk_ptr,  # [HC*H] norm_key weight
    nq_ptr,  # [HC*H] norm_query weight
    ncw_ptr,  # [HC*H] norm_conv weight
    gated_ptr,  # [T, HC*H] out
    normed_ptr,  # [T, HC*H] out
    key_rs,
    value_rs,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_H)
    s * H + lanes
    dtype: tl.constexpr = key_ptr.dtype.element_ty

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    dot = tl.load(scratch_in + t * HC + s).to(tl.float32)
    d = dot / tl.sqrt(float(H))
    d = d.to(dtype).to(tl.float32)
    tl.store(scratch_out + t * HC + s, d)


@triton.jit
def gate_2_3(
    scratch_in,
    scratch_out,
    key_ptr,  # [T, HC*H] with row stride key_rs
    value_ptr,  # [T, H] with row stride value_rs
    hidden_ptr,  # [T, HC*H]
    nk_ptr,  # [HC*H] norm_key weight
    nq_ptr,  # [HC*H] norm_query weight
    ncw_ptr,  # [HC*H] norm_conv weight
    gated_ptr,  # [T, HC*H] out
    normed_ptr,  # [T, HC*H] out
    key_rs,
    value_rs,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_H)
    s * H + lanes
    dtype: tl.constexpr = key_ptr.dtype.element_ty

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    d = tl.load(scratch_in + t * HC + s).to(tl.float32)
    sign = tl.where(d < 0, -1.0, 0.0)
    sign = tl.where(d > 0, 1.0, sign)
    magnitude = tl.sqrt(tl.maximum(tl.abs(d), 1e-6)).to(dtype)
    tl.store(scratch_out + t * HC + s, sign * magnitude)


@triton.jit
def gate_3_4(
    scratch_in,
    scratch_out,
    key_ptr,  # [T, HC*H] with row stride key_rs
    value_ptr,  # [T, H] with row stride value_rs
    hidden_ptr,  # [T, HC*H]
    nk_ptr,  # [HC*H] norm_key weight
    nq_ptr,  # [HC*H] norm_query weight
    ncw_ptr,  # [HC*H] norm_conv weight
    gated_ptr,  # [T, HC*H] out
    normed_ptr,  # [T, HC*H] out
    key_rs,
    value_rs,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_H)
    s * H + lanes
    dtype: tl.constexpr = key_ptr.dtype.element_ty

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    signed_magnitude = tl.load(scratch_in + t * HC + s).to(tl.float32)
    g = tl.sigmoid(signed_magnitude)
    g = g.to(dtype).to(tl.float32)

    tl.store(scratch_out + t * HC + s, g)


@triton.jit
def gate_4_5(
    scratch_in,
    scratch_out,
    key_ptr,  # [T, HC*H] with row stride key_rs
    value_ptr,  # [T, H] with row stride value_rs
    hidden_ptr,  # [T, HC*H]
    nk_ptr,  # [HC*H] norm_key weight
    nq_ptr,  # [HC*H] norm_query weight
    ncw_ptr,  # [HC*H] norm_conv weight
    gated_ptr,  # [T, HC*H] out
    normed_ptr,  # [T, HC*H] out
    key_rs,
    value_rs,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_H)
    mask = lanes < H
    offs = s * H + lanes
    dtype: tl.constexpr = key_ptr.dtype.element_ty

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    g = tl.load(scratch_in + t * HC + s).to(tl.float32)
    v = tl.load(value_ptr + t * value_rs + lanes, mask=mask, other=0.0).to(tl.float32)
    gated = (g * v).to(dtype)
    gf = gated.to(tl.float32)
    ncw = tl.load(ncw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = gf * tl.rsqrt(tl.sum(gf * gf) / H + eps) * (1.0 + ncw)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(gated_ptr + t * HC * H + offs, gated, mask=mask)
    tl.store(normed_ptr + t * HC * H + offs, normed, mask=mask)
