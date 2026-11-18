# SPDX-License-Identifier: Apache-2.0
"""Fused Step5 residual add and RMSNorm with BF16 output on Ascend A5."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

# Use several rows per tile for narrow inputs. As the width of a row grows,
# reduce the number of rows to limit vector-core local memory use.
_SMALL_TILE_WIDTH = 1024
_SMALL_TILE_ROWS = 16
_MEDIUM_TILE_WIDTH = 2048
_MEDIUM_TILE_ROWS = 4


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
    BLOCK_M: tl.constexpr,
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

    for row_start in tl.range(start_row, end_row, BLOCK_M):
        offset_row = row_start + tl.arange(0, BLOCK_M)
        mask_r = offset_row < end_row
        mask = mask_r[:, None] & mask_d[None, :]

        offset = offset_row[:, None] * hidden_size + offset_d[None, :]
        values = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + offset, mask=mask, other=0.0)
            values = values + residual
            tl.store(residual_out_ptr + offset, values, mask=mask)

        variance = tl.sum(values * values, axis=-1) / hidden_size
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
        and HAS_TRITON
        and x.device.type == "npu"
        and is_950()
        and x.device == weight.device
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
    return (
        residual.device == x.device
        and residual.dtype == torch.float32
        and residual.shape == x.shape
        and residual.is_contiguous()
    )


def _fused_add_rms_norm_cast_impl(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    zero_centered: bool,
) -> list[torch.Tensor]:
    outputs = _fused_add_rms_norm_cast_fake(x, residual, weight, eps, zero_centered)
    if x.numel() == 0:
        return outputs

    hidden_size = x.shape[-1]
    total_rows = x.numel() // hidden_size
    grid = (min(get_vectorcore_num(), total_rows),)
    block_n = triton.next_power_of_2(hidden_size)
    if block_n <= _SMALL_TILE_WIDTH:
        block_m = _SMALL_TILE_ROWS
    elif block_n <= _MEDIUM_TILE_WIDTH:
        block_m = _MEDIUM_TILE_ROWS
    else:
        block_m = 1
    _fused_add_rms_norm_cast_kernel[grid](
        x,
        residual,
        outputs[0],
        outputs[1] if residual is not None else None,
        weight,
        eps,
        total_rows,
        hidden_size,
        HAS_RESIDUAL=residual is not None,
        ZERO_CENTERED=zero_centered,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return outputs


def _fused_add_rms_norm_cast_fake(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    zero_centered: bool,
) -> list[torch.Tensor]:
    outputs = [x.new_empty(x.shape, dtype=torch.bfloat16)]
    if residual is not None:
        outputs.append(x.new_empty(x.shape, dtype=torch.float32))
    return outputs


direct_register_custom_op(
    op_name="ascend_step5_fused_add_rms_norm_cast",
    op_func=_fused_add_rms_norm_cast_impl,
    mutates_args=[],
    fake_impl=_fused_add_rms_norm_cast_fake,
)


def fused_add_rms_norm_cast(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    zero_centered: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BF16 norm output and FP32 residual, reusing x when residual is None."""
    outputs = torch.ops.vllm.ascend_step5_fused_add_rms_norm_cast(x, residual, weight, eps, zero_centered)
    # Keep the alias outside the custom op, whose outputs use fresh storage.
    return outputs[0], x if residual is None else outputs[1]
