# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK_SIZE = 1024
_PREFILL_MIN_TOKENS = 1024
_PREFILL_TILE_ROWS = 4
_PREFILL_MAX_WIDTH = 8192


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
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offsets = block.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

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


@triton.jit(do_not_specialize=["num_tokens"])
def fused_gdn_prepare_prefill_kernel(
    mixed_qkv,
    b,
    a,
    core_attn_out,
    mixed_qkvz,
    mixed_ba,
    num_tokens,
    QKVZ_ROW_STRIDE: tl.constexpr,
    BA_ROW_STRIDE: tl.constexpr,
    QKV_WIDTH: tl.constexpr,
    BA_WIDTH: tl.constexpr,
    CORE_WIDTH: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    # Group full rows so large prefill batches do not schedule eight programs
    # per token or repeat the small BA operation for each QKV column tile.
    rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.arange(0, BLOCK_WIDTH)
    valid_rows = rows[:, None] < num_tokens
    qkv_mask = valid_rows & (columns[None, :] < QKV_WIDTH)
    qkv = tl.load(mixed_qkvz + rows[:, None] * QKVZ_ROW_STRIDE + columns[None, :], qkv_mask, other=0)
    tl.store(mixed_qkv + rows[:, None] * QKV_WIDTH + columns[None, :], qkv, qkv_mask)
    core_mask = valid_rows & (columns[None, :] < CORE_WIDTH)
    tl.store(core_attn_out + rows[:, None] * CORE_WIDTH + columns[None, :], 0.0, core_mask)

    heads = tl.arange(0, BLOCK_HEADS)
    ba_mask = valid_rows & (heads[None, :] < BA_WIDTH)
    b_data = tl.load(mixed_ba + rows[:, None] * BA_ROW_STRIDE + heads[None, :], ba_mask, other=0)
    a_data = tl.load(mixed_ba + rows[:, None] * BA_ROW_STRIDE + BA_WIDTH + heads[None, :], ba_mask, other=0)
    tl.store(b + rows[:, None] * BA_WIDTH + heads[None, :], b_data, ba_mask)
    tl.store(a + rows[:, None] * BA_WIDTH + heads[None, :], a_data, ba_mask)


def supports_gdn_prepare(mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim):
    """Accept packed, TP-local projections with unit column strides."""
    return (
        mixed_qkvz.ndim == 2
        and mixed_ba.ndim == 2
        and mixed_qkvz.shape[0] == mixed_ba.shape[0]
        and mixed_qkvz.device == mixed_ba.device
        and mixed_qkvz.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and mixed_ba.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and mixed_qkvz.stride(1) == mixed_ba.stride(1) == 1
        and qkv_width > 0
        and head_v_dim > 0
        and 0 < num_v_heads <= _BLOCK_SIZE
        and mixed_qkvz.shape[1] == qkv_width + num_v_heads * head_v_dim
        and mixed_ba.shape[1] == 2 * num_v_heads
    )


def _prepare_fake(mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim):
    tokens = mixed_qkvz.shape[0]
    return (
        mixed_qkvz.new_empty((tokens, qkv_width)),
        mixed_ba.new_empty((tokens, num_v_heads)),
        mixed_ba.new_empty((tokens, num_v_heads)),
        mixed_qkvz.new_empty((tokens, num_v_heads, head_v_dim)),
    )


def _prepare(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    qkv_width: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not supports_gdn_prepare(mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim):
        raise ValueError("Unsupported packed GDN projection shape, dtype or stride")
    # Keep fresh allocation and writes inside one functional custom op, avoiding
    # AOT clones of empty output buffers before the Triton launch.
    mixed_qkv, b, a, core_attn_out = _prepare_fake(mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim)
    tokens = mixed_qkvz.shape[0]
    core_width = num_v_heads * head_v_dim
    if tokens > 0:
        if (
            tokens >= _PREFILL_MIN_TOKENS
            and max(qkv_width, core_width) <= _PREFILL_MAX_WIDTH
            and mixed_qkvz.dtype in (torch.float16, torch.bfloat16)
        ):
            # Four full rows fit the A3 UB budget for this copy/zero kernel.
            fused_gdn_prepare_prefill_kernel[(triton.cdiv(tokens, _PREFILL_TILE_ROWS),)](
                mixed_qkv,
                b,
                a,
                core_attn_out,
                mixed_qkvz,
                mixed_ba,
                tokens,
                mixed_qkvz.stride(0),
                mixed_ba.stride(0),
                qkv_width,
                num_v_heads,
                core_width,
                BLOCK_ROWS=_PREFILL_TILE_ROWS,
                BLOCK_WIDTH=triton.next_power_of_2(max(qkv_width, core_width)),
                BLOCK_HEADS=triton.next_power_of_2(num_v_heads),
            )
        else:
            grid = (tokens, triton.cdiv(max(qkv_width, core_width), _BLOCK_SIZE))
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
    return mixed_qkv, b, a, core_attn_out


direct_register_custom_op(
    op_name="ascend_fused_gdn_prepare",
    op_func=_prepare,
    fake_impl=_prepare_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def fused_gdn_prepare(mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim):
    """Copy QKV/B/A and zero the full output in one Triton launch.

    Z remains a view. No normalization, gate activation or state update occurs.
    Callers must keep their existing path when the support predicate is false.
    """
    mixed_qkv, b, a, core = torch.ops.vllm.ascend_fused_gdn_prepare(
        mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim
    )
    z = mixed_qkvz[:, qkv_width:].reshape(mixed_qkvz.shape[0], num_v_heads, head_v_dim)
    return mixed_qkv, z, b, a, core
