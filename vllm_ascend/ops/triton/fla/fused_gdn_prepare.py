# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

_BLOCK_SIZE = 1024


@triton.jit
def fused_gdn_prepare_kernel(
    mixed_qkv,
    b,
    a,
    core_attn_out,
    mixed_qkvz,
    mixed_ba,
    QKVZ_ROW_STRIDE: tl.constexpr,
    BA_ROW_STRIDE: tl.constexpr,
    QKV_WIDTH: tl.constexpr,
    BA_WIDTH: tl.constexpr,
    CORE_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Materialize packed GDN projections and initialize the result buffer."""
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    qkv_mask = offsets < QKV_WIDTH
    qkv = tl.load(mixed_qkvz + row * QKVZ_ROW_STRIDE + offsets, mask=qkv_mask)
    tl.store(mixed_qkv + row * QKV_WIDTH + offsets, qkv, mask=qkv_mask)

    core_mask = offsets < CORE_WIDTH
    tl.store(core_attn_out + row * CORE_WIDTH + offsets, 0.0, mask=core_mask)

    ba_offsets = tl.arange(0, BLOCK_SIZE)
    ba_mask = (block == 0) & (ba_offsets < BA_WIDTH)
    b_data = tl.load(mixed_ba + row * BA_ROW_STRIDE + ba_offsets, mask=ba_mask)
    a_data = tl.load(mixed_ba + row * BA_ROW_STRIDE + BA_WIDTH + ba_offsets, mask=ba_mask)
    tl.store(b + row * BA_WIDTH + ba_offsets, b_data, mask=ba_mask)
    tl.store(a + row * BA_WIDTH + ba_offsets, a_data, mask=ba_mask)


def fused_gdn_prepare(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    qkv_width: int,
    num_v_heads: int,
    head_v_dim: int,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copy dense QKV/B/A and zero all output rows in one Triton launch.

    Inputs are non-interleaved, TP-local projections with unit column stride.
    Row padding and token slices are supported. Z aliases QKVZ storage. BA may
    have a different dtype from QKVZ; each output preserves its source dtype.
    Every output row is initialized, including graph-padding rows that the
    GDN core does not write. Empty batches do not launch a kernel.
    """
    if qkv_width <= 0 or head_v_dim <= 0 or not 0 < num_v_heads <= _BLOCK_SIZE:
        raise ValueError("GDN dimensions must be positive and local V heads must fit one block")
    if mixed_qkvz.ndim != 2 or mixed_ba.ndim != 2:
        raise ValueError("GDN projections must be rank-2 tensors")
    if mixed_qkvz.stride(1) != 1 or mixed_ba.stride(1) != 1:
        raise ValueError("GDN projections must have unit column stride")
    if mixed_qkvz.device != mixed_ba.device:
        raise ValueError("GDN projections must be on the same device")
    if mixed_qkvz.shape[0] != mixed_ba.shape[0]:
        raise ValueError("QKVZ and BA projections must have the same token count")
    if mixed_ba.shape[1] != 2 * num_v_heads:
        raise ValueError("BA projection width must be twice the local V-head count")
    if mixed_qkvz.shape[1] != qkv_width + num_v_heads * head_v_dim:
        raise ValueError("QKVZ projection width is inconsistent with the GDN dimensions")

    num_tokens = mixed_qkvz.shape[0]
    expected_core_shape = (num_tokens, num_v_heads, head_v_dim)
    if core_attn_out is None:
        core_attn_out = torch.empty(expected_core_shape, dtype=mixed_qkvz.dtype, device=mixed_qkvz.device)
    elif (
        core_attn_out.shape != expected_core_shape
        or core_attn_out.dtype != mixed_qkvz.dtype
        or core_attn_out.device != mixed_qkvz.device
        or not core_attn_out.is_contiguous()
    ):
        raise ValueError("core_attn_out must be contiguous and match the expected shape, dtype, and device")

    mixed_qkv = torch.empty((num_tokens, qkv_width), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device)
    b = torch.empty((num_tokens, num_v_heads), dtype=mixed_ba.dtype, device=mixed_ba.device)
    a = torch.empty_like(b)
    if num_tokens > 0:
        core_width = num_v_heads * head_v_dim
        grid = (num_tokens, triton.cdiv(max(qkv_width, core_width), _BLOCK_SIZE))
        fused_gdn_prepare_kernel[grid](
            mixed_qkv,
            b,
            a,
            core_attn_out,
            mixed_qkvz,
            mixed_ba,
            mixed_qkvz.stride(0),
            mixed_ba.stride(0),
            qkv_width,
            num_v_heads,
            core_width,
            BLOCK_SIZE=_BLOCK_SIZE,
        )

    z = mixed_qkvz[:, qkv_width:].reshape(num_tokens, num_v_heads, head_v_dim)
    return mixed_qkv, z, b, a, core_attn_out
