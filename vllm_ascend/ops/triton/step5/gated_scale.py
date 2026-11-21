# SPDX-License-Identifier: Apache-2.0
"""gated_scale -- Ascend release build (self-contained).

Fused head-wise attention gate: sigmoid + broadcast multiply in one Triton
kernel. Replaces the eager post-attention chain on Ascend --
``attn_output.view(t, H, D) * gate.unsqueeze(-1).sigmoid()`` -- where the
sigmoid and the multiply run in fp32, at least as accurate as the bf16 eager
path.

The gate is addressed through an offset-0 base tensor plus a scalar column
offset: ``out[t] = attn[t] * sigmoid(base[t, off:off+H])``. The base must
share the gate's storage and have ``storage_offset == 0`` -- for the DSA
layers the front row-slice (the ``qkv`` view) of the fused qkv+indexer GEMM
output, for the sliding-window layers either the packed ``qkvg`` GEMM
output (when ``fuse_qkv_gate`` packs q/k/v/gate into one GEMM) or the
standalone ``g_proj`` output, both with a scalar column offset. Handing a
mid-storage gate view to the op boundary instead makes the dispatch layer
materialize it first (TensorMove), so callers declare the layout explicitly
and no materializing copy is issued. Gate columns may lie beyond the
base's logical width as long as they stay inside its storage row span; the
DSA layout reads the gate tail of the shared qkv+indexer row this way.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _gated_scale_kernel(
    attn_ptr,
    gate_base_ptr,
    out_ptr,
    stride_attn_token,
    gate_row_stride,
    gate_col_offset,
    stride_out_token,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_ok = heads < NUM_HEADS
    dim_ok = dims < HEAD_DIM
    mask = head_ok[:, None] & dim_ok[None, :]
    g = tl.sigmoid(
        tl.load(
            gate_base_ptr + token * gate_row_stride + gate_col_offset + heads,
            mask=head_ok,
            other=0.0,
        ).to(tl.float32)
    )
    src = token * stride_attn_token + heads[:, None] * HEAD_DIM + dims[None, :]
    x = tl.load(attn_ptr + src, mask=mask, other=0.0).to(tl.float32)
    dst = token * stride_out_token + heads[:, None] * HEAD_DIM + dims[None, :]
    tl.store(
        out_ptr + dst,
        (x * g[:, None]).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


_LANE_BUDGET = 4096


def _pick_head_block(num_heads: int, head_dim: int) -> int:
    """Heads per program, sized to keep one token row's lanes within budget."""
    if num_heads * head_dim <= _LANE_BUDGET:
        return triton.next_power_of_2(num_heads)
    return triton.next_power_of_2(max(1, min(num_heads, _LANE_BUDGET // head_dim)))


def _gated_scale_impl(
    attn_output: torch.Tensor,
    gate_base: torch.Tensor,
    gate_col_offset: int,
    num_heads: int,
) -> torch.Tensor:
    gate_row_stride = gate_base.stride(0)
    assert gate_col_offset >= 0 and num_heads > 0 and gate_col_offset + num_heads <= gate_row_stride, (
        "gated_scale gate columns must stay inside the base's row span: "
        f"offset={gate_col_offset}, num_heads={num_heads}, "
        f"row_stride={gate_row_stride}"
    )
    if attn_output.shape[0] == 0:
        return torch.empty_like(attn_output)
    head_dim = attn_output.shape[-1] // num_heads
    block_h = _pick_head_block(num_heads, head_dim)
    out = torch.empty_like(attn_output)
    _gated_scale_kernel[(attn_output.shape[0], triton.cdiv(num_heads, block_h))](
        attn_output,
        gate_base,
        out,
        attn_output.stride(0),
        gate_row_stride,
        gate_col_offset,
        out.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return out


def _gated_scale_fake(
    attn_output: torch.Tensor,
    gate_base: torch.Tensor,
    gate_col_offset: int,
    num_heads: int,
) -> torch.Tensor:
    return torch.empty_like(attn_output)


direct_register_custom_op(
    op_name="ascend_step5_gated_scale",
    op_func=_gated_scale_impl,
    mutates_args=[],
    fake_impl=_gated_scale_fake,
)


def gated_scale(
    attn_output: torch.Tensor,
    gate_base: torch.Tensor,
    gate_col_offset: int,
    num_heads: int,
) -> torch.Tensor:
    """Apply the head-wise sigmoid gate: ``out[t] = attn[t] * sigmoid(gate[t])``.

    ``attn_output`` is ``(tokens, num_heads * head_dim)``. The gate is
    addressed through ``gate_base`` -- an offset-0 tensor sharing the gate's
    storage -- as ``gate_base[t, gate_col_offset:gate_col_offset+num_heads]``.
    The row stride is the base's own ``stride(0)``, so a front row-slice of a
    wider packed GEMM row (the DSA ``qkv`` view) also qualifies and the kernel
    reads the gate tail in place.
    """
    return torch.ops.vllm.ascend_step5_gated_scale(attn_output, gate_base, gate_col_offset, num_heads)
