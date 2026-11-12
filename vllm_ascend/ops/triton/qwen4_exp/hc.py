# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Triton-Ascend kernels for Qwen4Exp HyperConnection glue operations.

The math and public interfaces mirror
``vllm/models/qwen4_exp/nvidia/ops/hc.py``. CUDA PDL hooks are deliberately
omitted because triton-ascend launches on an NPU stream instead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    stride_x_row,
    stride_output_row,
    GROUP_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    WEIGHT_SHARED: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    row = program // NUM_GROUPS
    group = program % NUM_GROUPS
    group_offsets = tl.arange(0, BLOCK_SIZE)
    mask = group_offsets < GROUP_DIM
    offsets = group * GROUP_DIM + group_offsets
    weight_offsets = group_offsets if WEIGHT_SHARED else offsets

    values = tl.load(
        x_ptr + row * stride_x_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    reciprocal_rms = tl.rsqrt(tl.sum(values * values) / GROUP_DIM + EPS)
    normalized = values * reciprocal_rms
    # Gemma RMSNorm uses (1 + weight), expressed as an FMA-friendly update.
    normalized += normalized * weights.to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + offsets,
        normalized,
        mask=mask,
    )


def _grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    num_groups = int(num_groups)
    eps = float(eps)
    if x.ndim != 2:
        raise ValueError("grouped Gemma RMSNorm expects a two-dimensional input")
    rows, dim = x.shape
    dim = int(dim)
    if num_groups <= 0 or dim % num_groups:
        raise ValueError(f"hidden dimension {dim} is not divisible by {num_groups}")
    if x.stride(1) != 1:
        raise ValueError("grouped Gemma RMSNorm requires unit inner stride")
    if not weight.is_contiguous():
        raise ValueError("grouped Gemma RMSNorm weight must be contiguous")
    group_dim = dim // num_groups
    weight_numel = int(weight.numel())
    if weight_numel not in (group_dim, dim):
        raise ValueError(f"expected {group_dim} or {dim} norm weights, got {weight_numel}")
    block_size = triton.next_power_of_2(group_dim)
    output = x.new_empty(x.shape)
    if rows:
        _grouped_gemma_rmsnorm_kernel[(rows * num_groups,)](
            x,
            weight,
            output,
            x.stride(0),
            output.stride(0),
            GROUP_DIM=group_dim,
            BLOCK_SIZE=block_size,
            NUM_GROUPS=num_groups,
            WEIGHT_SHARED=weight_numel == group_dim,
            EPS=eps,
        )
    return output


@triton.jit
def _hc_silu_kernel(
    x_ptr,
    output_ptr,
    stride_x_row,
    stride_output_row,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HC_COUNT: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < DIM
    values = tl.load(
        x_ptr + row * stride_x_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values /= HC_COUNT
    output = values * tl.sigmoid(values)
    tl.store(
        output_ptr + row * stride_output_row + offsets,
        output,
        mask=mask,
    )


def _hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    hc_count = int(hc_count)
    if x.ndim != 2 or x.stride(1) != 1:
        raise ValueError("HC SiLU expects a two-dimensional input with unit inner stride")
    if hc_count <= 0:
        raise ValueError("HC stream count must be positive")
    rows, dim = x.shape
    dim = int(dim)
    block_size = triton.next_power_of_2(dim)
    output = x.new_empty(x.shape)
    if rows:
        _hc_silu_kernel[(rows,)](
            x,
            output,
            x.stride(0),
            output.stride(0),
            DIM=dim,
            BLOCK_SIZE=block_size,
            HC_COUNT=hc_count,
        )
    return output


@triton.jit
def _hc_gate_mix_kernel(
    x_ptr,
    gate_ptr,
    output_ptr,
    stride_x_row,
    stride_gate_row,
    stride_output_row,
    GROUP_DIM: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    inner_offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = inner_offsets < GROUP_DIM
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for stream in tl.static_range(HC_COUNT):
        offsets = stream * GROUP_DIM + inner_offsets
        gates = tl.load(
            gate_ptr + row * stride_gate_row + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        values = tl.load(
            x_ptr + row * stride_x_row + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sigmoid(gates) * values
    accumulator /= HC_COUNT
    tl.store(
        output_ptr + row * stride_output_row + inner_offsets,
        accumulator,
        mask=mask,
    )


def _hc_gate_mix(
    x: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    hc_count = int(hc_count)
    if x.ndim != 2 or x.shape != gate.shape:
        raise ValueError("HyperConnection input and gate must be matching 2D tensors")
    if hc_count <= 0 or x.shape[1] % hc_count:
        raise ValueError("hidden dimension must be divisible by the HC stream count")
    if x.stride(1) != 1 or gate.stride(1) != 1:
        raise ValueError("HC gate mix requires unit inner strides")
    rows, dim = x.shape
    dim = int(dim)
    group_dim = dim // hc_count
    output = x.new_empty((rows, group_dim))
    block_size = 512
    if rows:
        _hc_gate_mix_kernel[(rows, triton.cdiv(group_dim, block_size))](
            x,
            gate,
            output,
            x.stride(0),
            gate.stride(0),
            output.stride(0),
            GROUP_DIM=group_dim,
            HC_COUNT=hc_count,
            BLOCK_SIZE=block_size,
        )
    return output


@triton.jit
def _hc_combine_kernel(
    residual_ptr,
    block_output_ptr,
    injection_ptr,
    output_ptr,
    stride_residual_row,
    stride_block_row,
    stride_injection_row,
    stride_output_row,
    GROUP_DIM: tl.constexpr,
    HC_COUNT: tl.constexpr,
    HAS_INJECTION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    stream = tl.program_id(1)
    tile = tl.program_id(2)
    inner_offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = inner_offsets < GROUP_DIM
    offsets = stream * GROUP_DIM + inner_offsets
    residual = tl.load(
        residual_ptr + row * stride_residual_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    block_output = tl.load(
        block_output_ptr + row * stride_block_row + inner_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_INJECTION:
        injection = tl.load(injection_ptr + row * stride_injection_row + stream).to(tl.float32)
        block_output *= 2.0 * tl.sigmoid(injection / HC_COUNT)
    tl.store(
        output_ptr + row * stride_output_row + offsets,
        residual + block_output,
        mask=mask,
    )


def _validate_combine_inputs(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> tuple[int, int]:
    hc_count = int(hc_count)
    if residual.ndim != 2:
        raise ValueError("residual must be [rows, HC * hidden_size]")
    rows, dim = residual.shape
    dim = int(dim)
    if hc_count <= 0 or dim % hc_count:
        raise ValueError("residual must be [rows, HC * hidden_size]")
    group_dim = dim // hc_count
    if block_output.shape != (rows, group_dim):
        raise ValueError(f"block output must have shape {(rows, group_dim)}")
    if residual.stride(1) != 1 or block_output.stride(1) != 1:
        raise ValueError("HC combine requires unit inner strides")
    if injection_logits is not None:
        if injection_logits.shape != (rows, hc_count):
            raise ValueError(f"injection logits must have shape {(rows, hc_count)}")
        if injection_logits.stride(1) != 1:
            raise ValueError("HC injection logits require unit inner stride")
    return rows, group_dim


def _hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    hc_count = int(hc_count)
    rows, group_dim = _validate_combine_inputs(residual, block_output, injection_logits, hc_count)
    output = residual.new_empty(residual.shape)
    block_size = 512
    if rows:
        injection_ptr = residual if injection_logits is None else injection_logits
        injection_stride = 0 if injection_logits is None else injection_logits.stride(0)
        _hc_combine_kernel[(rows, hc_count, triton.cdiv(group_dim, block_size))](
            residual,
            block_output,
            injection_ptr,
            output,
            residual.stride(0),
            block_output.stride(0),
            injection_stride,
            output.stride(0),
            GROUP_DIM=group_dim,
            HC_COUNT=hc_count,
            HAS_INJECTION=injection_logits is not None,
            BLOCK_SIZE=block_size,
        )
    return output


@triton.jit
def _hc_combine_norm_kernel(
    residual_ptr,
    block_output_ptr,
    injection_ptr,
    weight_ptr,
    combined_ptr,
    normalized_ptr,
    stride_residual_row,
    stride_block_row,
    stride_injection_row,
    stride_combined_row,
    stride_normalized_row,
    GROUP_DIM: tl.constexpr,
    HC_COUNT: tl.constexpr,
    HAS_INJECTION: tl.constexpr,
    WEIGHT_SHARED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    row = program // HC_COUNT
    stream = program % HC_COUNT
    inner_offsets = tl.arange(0, BLOCK_SIZE)
    mask = inner_offsets < GROUP_DIM
    offsets = stream * GROUP_DIM + inner_offsets
    weight_offsets = inner_offsets if WEIGHT_SHARED else offsets

    residual = tl.load(
        residual_ptr + row * stride_residual_row + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    block_output = tl.load(
        block_output_ptr + row * stride_block_row + inner_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_INJECTION:
        injection = tl.load(injection_ptr + row * stride_injection_row + stream).to(tl.float32)
        block_output *= 2.0 * tl.sigmoid(injection / HC_COUNT)

    # Preserve the unfused combine -> BF16 materialization -> RMSNorm boundary.
    combined = (residual + block_output).to(combined_ptr.dtype.element_ty)
    tl.store(
        combined_ptr + row * stride_combined_row + offsets,
        combined,
        mask=mask,
    )
    combined_fp32 = combined.to(tl.float32)
    reciprocal_rms = tl.rsqrt(tl.sum(combined_fp32 * combined_fp32) / GROUP_DIM + EPS)
    weights = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0)
    normalized = combined_fp32 * reciprocal_rms
    normalized += normalized * weights.to(tl.float32)
    tl.store(
        normalized_ptr + row * stride_normalized_row + offsets,
        normalized,
        mask=mask,
    )


def _hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hc_count = int(hc_count)
    eps = float(eps)
    rows, group_dim = _validate_combine_inputs(residual, block_output, injection_logits, hc_count)
    if not norm_weight.is_contiguous():
        raise ValueError("HC norm weight must be contiguous")
    residual_dim = int(residual.shape[1])
    norm_weight_numel = int(norm_weight.numel())
    if norm_weight_numel not in (group_dim, residual_dim):
        raise ValueError(f"expected {group_dim} or {residual_dim} norm weights, got {norm_weight_numel}")
    block_size = triton.next_power_of_2(group_dim)
    combined = residual.new_empty(residual.shape)
    normalized = residual.new_empty(residual.shape)
    if rows:
        injection_ptr = residual if injection_logits is None else injection_logits
        injection_stride = 0 if injection_logits is None else injection_logits.stride(0)
        _hc_combine_norm_kernel[(rows * hc_count,)](
            residual,
            block_output,
            injection_ptr,
            norm_weight,
            combined,
            normalized,
            residual.stride(0),
            block_output.stride(0),
            injection_stride,
            combined.stride(0),
            normalized.stride(0),
            GROUP_DIM=group_dim,
            HC_COUNT=hc_count,
            HAS_INJECTION=injection_logits is not None,
            WEIGHT_SHARED=norm_weight_numel == group_dim,
            BLOCK_SIZE=block_size,
            EPS=eps,
        )
    return combined, normalized


def _same_shape_fake(x: torch.Tensor, *args) -> torch.Tensor:
    return x.new_empty(x.shape)


def _hc_gate_mix_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del gate
    return x.new_empty((x.shape[0], x.shape[1] // hc_count))


def _hc_combine_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    del block_output, injection_logits, hc_count
    return residual.new_empty(residual.shape)


def _hc_combine_norm_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, norm_weight, eps, hc_count
    return residual.new_empty(residual.shape), residual.new_empty(residual.shape)


direct_register_custom_op(
    op_name="qwen4_exp_ascend_grouped_gemma_rmsnorm",
    op_func=_grouped_gemma_rmsnorm,
    fake_impl=_same_shape_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="qwen4_exp_ascend_hc_silu",
    op_func=_hc_silu,
    fake_impl=_same_shape_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="qwen4_exp_ascend_hc_gate_mix",
    op_func=_hc_gate_mix,
    fake_impl=_hc_gate_mix_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="qwen4_exp_ascend_hc_combine",
    op_func=_hc_combine,
    fake_impl=_hc_combine_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="qwen4_exp_ascend_hc_combine_norm",
    op_func=_hc_combine_norm,
    fake_impl=_hc_combine_norm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def _fallback_grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    if x.shape[-1] % num_groups:
        raise ValueError("hidden size must be divisible by the HC stream count")
    group_dim = x.shape[-1] // num_groups
    grouped = x.unflatten(-1, (num_groups, group_dim)).float()
    reciprocal_rms = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
    if weight.numel() == group_dim:
        affine = weight.reshape(1, 1, group_dim)
    elif weight.numel() == x.shape[-1]:
        affine = weight.reshape(1, num_groups, group_dim)
    else:
        raise ValueError("HC norm weight has an incompatible shape")
    return (grouped * reciprocal_rms * (1.0 + affine.float())).flatten(-2).to(x.dtype)


def _fallback_hc_gate_mix(
    x: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if x.shape != gate.shape:
        raise ValueError("HC input and gate shapes must match")
    group_dim = x.shape[-1] // hc_count
    grouped = x.unflatten(-1, (hc_count, group_dim)).float()
    gates = torch.sigmoid(gate.float()).unflatten(-1, (hc_count, group_dim))
    return (grouped * gates).mean(dim=-2).to(x.dtype)


def _fallback_hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    group_dim = residual.shape[-1] // hc_count
    residual_grouped = residual.unflatten(-1, (hc_count, group_dim)).float()
    block = block_output.float().unsqueeze(-2)
    if injection_logits is not None:
        injection = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
        block = block * injection.unsqueeze(-1)
    return (residual_grouped + block).flatten(-2).to(residual.dtype)


def grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    if x.device.type != "npu":
        return _fallback_grouped_gemma_rmsnorm(x, weight, eps, num_groups)
    return torch.ops.vllm.qwen4_exp_ascend_grouped_gemma_rmsnorm(x, weight, eps, num_groups)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    if x.device.type != "npu":
        return F.silu(x.float() / hc_count).to(x.dtype)
    return torch.ops.vllm.qwen4_exp_ascend_hc_silu(x, hc_count)


def hc_gate_mix(
    x: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if x.device.type != "npu":
        return _fallback_hc_gate_mix(x, gate, hc_count)
    return torch.ops.vllm.qwen4_exp_ascend_hc_gate_mix(x, gate, hc_count)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    if residual.device.type != "npu":
        return _fallback_hc_combine(residual, block_output, injection_logits, hc_count)
    return torch.ops.vllm.qwen4_exp_ascend_hc_combine(residual, block_output, injection_logits, hc_count)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.device.type != "npu":
        combined = _fallback_hc_combine(residual, block_output, injection_logits, hc_count)
        normalized = _fallback_grouped_gemma_rmsnorm(combined, norm_weight, eps, hc_count)
        return combined, normalized
    return torch.ops.vllm.qwen4_exp_ascend_hc_combine_norm(
        residual,
        block_output,
        injection_logits,
        norm_weight,
        eps,
        hc_count,
    )


__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_gate_mix",
    "hc_silu",
]
