# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def fused_gdn_prepare_kernel(
    mixed_qkv,
    b,
    a,
    core_attn_out,
    mixed_qkvz,
    mixed_ba,
    N_ROWS: tl.constexpr,
    QKVZ_ROW_STRIDE: tl.constexpr,
    BA_ROW_STRIDE: tl.constexpr,
    QKV_WIDTH: tl.constexpr,
    BA_WIDTH: tl.constexpr,
    CORE_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BA_BLOCK_SIZE: tl.constexpr,
):
    """Materialize packed GDN projections and initialize the result buffer."""
    block = tl.program_id(1)
    if BLOCK_M == 1:
        row = tl.program_id(0)
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
    else:
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        rows = row_offsets[:, None]
        cols = col_offsets[None, :]
        row_mask = rows < N_ROWS

        qkv_mask = row_mask & (cols < QKV_WIDTH)
        qkv = tl.load(mixed_qkvz + rows * QKVZ_ROW_STRIDE + cols, mask=qkv_mask)
        tl.store(mixed_qkv + rows * QKV_WIDTH + cols, qkv, mask=qkv_mask)

        core_mask = row_mask & (cols < CORE_WIDTH)
        tl.store(core_attn_out + rows * CORE_WIDTH + cols, 0.0, mask=core_mask)

        ba_offsets = tl.arange(0, BA_BLOCK_SIZE)[None, :]
        ba_mask = (block == 0) & row_mask & (ba_offsets < BA_WIDTH)
        b_data = tl.load(mixed_ba + rows * BA_ROW_STRIDE + ba_offsets, mask=ba_mask)
        a_data = tl.load(
            mixed_ba + rows * BA_ROW_STRIDE + BA_WIDTH + ba_offsets,
            mask=ba_mask,
        )
        tl.store(b + rows * BA_WIDTH + ba_offsets, b_data, mask=ba_mask)
        tl.store(a + rows * BA_WIDTH + ba_offsets, a_data, mask=ba_mask)


def fused_gdn_prepare(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    qkv_width: int,
    num_v_heads: int,
    head_v_dim: int,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed QKVZ/BA projections and zero the padded GDN output.

    QKV, B, and A are materialized because their consumers require dense rows.
    Z remains a view over the packed projection. The output buffer is fully
    initialized so graph-padding rows remain deterministic when the GDN core
    only writes ``num_actual_tokens`` rows.
    """
    if mixed_qkvz.ndim != 2 or mixed_ba.ndim != 2:
        raise ValueError("GDN projections must be rank-2 tensors")
    if mixed_qkvz.shape[0] != mixed_ba.shape[0]:
        raise ValueError("QKVZ and BA projections must have the same token count")
    if mixed_ba.shape[1] != 2 * num_v_heads:
        raise ValueError("BA projection width must be twice the local V-head count")
    if mixed_qkvz.shape[1] != qkv_width + num_v_heads * head_v_dim:
        raise ValueError("QKVZ projection width is inconsistent with the GDN dimensions")

    num_tokens = mixed_qkvz.shape[0]
    mixed_qkv = torch.empty(
        (num_tokens, qkv_width),
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        (num_tokens, num_v_heads),
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)
    expected_core_shape = (num_tokens, num_v_heads, head_v_dim)
    if core_attn_out is None:
        core_attn_out = torch.empty(
            expected_core_shape,
            dtype=mixed_qkvz.dtype,
            device=mixed_qkvz.device,
        )
    elif (
        core_attn_out.shape != expected_core_shape
        or core_attn_out.dtype != mixed_qkvz.dtype
        or core_attn_out.device != mixed_qkvz.device
        or not core_attn_out.is_contiguous()
    ):
        raise ValueError("core_attn_out must be contiguous and match the expected shape, dtype, and device")

    if num_tokens > 0:
        block_size = 1024
        # Amortize scalar address and mask setup across rows. A smaller tile
        # retains enough programs for short prefills, while long prefills use
        # the wider tile to reduce scheduling overhead further.
        if num_tokens >= 2048:
            block_m = 16
        elif num_tokens >= 64:
            block_m = 8
        else:
            block_m = 1
        core_width = num_v_heads * head_v_dim
        grid = (
            triton.cdiv(num_tokens, block_m),
            triton.cdiv(max(qkv_width, core_width), block_size),
        )
        fused_gdn_prepare_kernel[grid](
            mixed_qkv,
            b,
            a,
            core_attn_out,
            mixed_qkvz,
            mixed_ba,
            num_tokens,
            mixed_qkvz.stride(0),
            mixed_ba.stride(0),
            qkv_width,
            num_v_heads,
            core_width,
            BLOCK_SIZE=block_size,
            BLOCK_M=block_m,
            BA_BLOCK_SIZE=32,
        )

    z = mixed_qkvz[:, qkv_width:].reshape(num_tokens, num_v_heads, head_v_dim)
    return mixed_qkv, z, b, a, core_attn_out
