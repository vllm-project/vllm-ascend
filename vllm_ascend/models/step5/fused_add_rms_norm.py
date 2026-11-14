# SPDX-License-Identifier: Apache-2.0
"""Step5 fused residual-add + RMSNorm + param-dtype cast for Ascend.

Fuses the post-attention sequence in ``Step5DecoderLayer.forward``:

    hidden_states = self._cast_for_residual(hidden_states)
    hidden_states += residual
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self._cast_for_param_op(hidden_states)

into a single Triton kernel that emits the RMSNorm output in the param-op
dtype and the updated residual in the residual dtype (fp32 when
``fp32_residual_connection`` is enabled). The norm math matches
``Step5RMSNorm``: fp32 internal compute, scale = weight (+1 when the
checkpoint stores zero-centered weights). Row partitioning across vector
cores follows ``vllm_ascend.ops.triton.rms_norm``.
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["total_rows"])
def _fused_add_rms_norm_cast_kernel(
    x_ptr,
    x_row_stride,
    residual_ptr,
    residual_row_stride,
    out_ptr,
    out_row_stride,
    residual_out_ptr,
    residual_out_row_stride,
    weight_ptr,
    eps,
    total_rows,
    hidden_size,
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

        x_offset = offset_row[:, None] * x_row_stride + offset_d[None, :]
        residual_offset = offset_row[:, None] * residual_row_stride + offset_d[None, :]
        x = tl.load(x_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + residual_offset, mask=mask, other=0.0).to(
            tl.float32
        )
        summed = x + residual
        # Round the sum to the residual storage dtype before normalizing so
        # the result matches the unfused low-precision add.
        summed = summed.to(residual_out_ptr.dtype.element_ty)
        residual_out_offset = (
            offset_row[:, None] * residual_out_row_stride + offset_d[None, :]
        )
        tl.store(residual_out_ptr + residual_out_offset, summed, mask=mask)

        summed_f32 = summed.to(tl.float32)
        variance = tl.sum(summed_f32 * summed_f32, axis=-1) / hidden_size
        rstd = tl.rsqrt(variance + eps)
        out = summed_f32 * rstd[:, None] * weight[None, :]
        out_offset = offset_row[:, None] * out_row_stride + offset_d[None, :]
        tl.store(out_ptr + out_offset, out, mask=mask)


def _fused_add_rms_norm_cast_eager(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    zero_centered: bool,
    fp32_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = x.float() + residual if fp32_residual else x + residual
    compute = residual_out.float()
    variance = compute.pow(2).mean(dim=-1, keepdim=True)
    compute = compute * torch.rsqrt(variance + variance_epsilon)
    scale = weight.float() + 1.0 if zero_centered else weight.float()
    out_dtype = torch.bfloat16 if fp32_residual else x.dtype
    return (compute * scale).to(out_dtype), residual_out


def fused_add_rms_norm_cast(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    zero_centered: bool = False,
    fp32_residual: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (normed hidden states in param-op dtype, updated residual)."""
    if not HAS_TRITON or x.device.type != "npu":
        return _fused_add_rms_norm_cast_eager(
            x, residual, weight, variance_epsilon, zero_centered, fp32_residual
        )

    hidden_size = x.shape[-1]
    x_2d = x.reshape(-1, hidden_size).contiguous()
    residual_2d = residual.reshape(-1, hidden_size).contiguous()
    weight = weight.contiguous()
    total_rows = x_2d.shape[0]

    out_dtype = torch.bfloat16 if fp32_residual else x.dtype
    residual_dtype = torch.float32 if fp32_residual else x.dtype
    out = torch.empty_like(x_2d, dtype=out_dtype)
    residual_out = torch.empty_like(x_2d, dtype=residual_dtype)
    if total_rows == 0:
        return out.view(x.shape), residual_out.view(x.shape)

    num_vectorcore = get_vectorcore_num()
    grid = (max(1, min(num_vectorcore, total_rows)),)

    block_n = triton.next_power_of_2(hidden_size)
    # Keep the (BLOCK_M, BLOCK_N) tile within vector-core local memory.
    block_m = 16 if block_n <= 1024 else (4 if block_n <= 2048 else 1)
    _fused_add_rms_norm_cast_kernel[grid](
        x_2d,
        x_2d.stride(0),
        residual_2d,
        residual_2d.stride(0),
        out,
        out.stride(0),
        residual_out,
        residual_out.stride(0),
        weight,
        variance_epsilon,
        total_rows,
        hidden_size,
        ZERO_CENTERED=zero_centered,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out.view(x.shape), residual_out.view(x.shape)
