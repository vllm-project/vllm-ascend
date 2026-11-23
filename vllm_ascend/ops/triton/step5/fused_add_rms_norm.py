# SPDX-License-Identifier: Apache-2.0
"""Fused Step5 residual add and RMSNorm with BF16 output on Ascend A5."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["total_rows"])
def _fused_add_rms_norm_cast_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    residual_out_ptr,
    weight_ptr,
    eps,
    total_rows,
    hidden_size,
    HAS_RESIDUAL: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    core_id = tl.program_id(0)
    core_num = tl.num_programs(0)
    rows_per_core = tl.cdiv(total_rows, core_num)
    start_row = core_id * rows_per_core
    end_row = tl.minimum(start_row + rows_per_core, total_rows)

    offset_d = tl.arange(0, BLOCK_N)
    mask_d = offset_d < hidden_size
    weight = tl.load(weight_ptr + offset_d, mask=mask_d, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        weight = weight + 1.0

    # Retain the singleton row dimension so the RMS reduction returns a [1] tensor.
    # A 1D reduction lowered to scalar extraction and rebroadcast on Ascend 950DT,
    # increasing device latency, especially when a core processed multiple rows.
    mask = mask_d[None, :]
    for row in tl.range(start_row, end_row):
        offset = row * hidden_size + offset_d[None, :]
        values = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + offset, mask=mask, other=0.0)
            values = values + residual
            tl.store(residual_out_ptr + offset, values, mask=mask)

        variance = tl.sum(values * values, axis=1) / hidden_size
        rstd = tl.rsqrt(variance + eps)
        out = values * rstd[:, None] * weight[None, :]
        tl.store(out_ptr + offset, out, mask=mask)


def can_use_fused_add_rms_norm_cast(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    fp32_residual: bool,
) -> bool:
    """Check whether the caller can use the contiguous FP32-residual path."""
    # The unfused A5 norm can round both its input and its zero-centered
    # scale in the weight dtype. The kernel keeps both in FP32.
    if not (
        fp32_residual
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and weight.dtype == torch.float32
        and x.ndim > 0
        and x.shape[-1] > 0
        and weight.shape == (x.shape[-1],)
        and x.is_contiguous()
        and weight.is_contiguous()
    ):
        return False
    if residual is None:
        return x.dtype == torch.float32
    return residual.dtype == torch.float32 and residual.shape == x.shape and residual.is_contiguous()


def fused_add_rms_norm_cast(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    zero_centered: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BF16 norm output and FP32 residual, reusing x when residual is None."""
    out = x.new_empty(x.shape, dtype=torch.bfloat16)
    residual_out = x if residual is None else x.new_empty(x.shape, dtype=torch.float32)
    if x.numel() == 0:
        return out, residual_out

    hidden_size = x.shape[-1]
    total_rows = x.numel() // hidden_size
    grid = (min(get_vectorcore_num(), total_rows),)
    block_n = triton.next_power_of_2(hidden_size)
    _fused_add_rms_norm_cast_kernel[grid](
        x,
        residual,
        out,
        residual_out if residual is not None else None,
        weight,
        eps,
        total_rows,
        hidden_size,
        HAS_RESIDUAL=residual is not None,
        ZERO_CENTERED=zero_centered,
        BLOCK_N=block_n,
    )
    return out, residual_out
