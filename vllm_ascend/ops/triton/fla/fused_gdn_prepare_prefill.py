# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_PREFILL_MIN_TOKENS = 1024
_PREFILL_TILE_ROWS = 4
_PREFILL_MAX_WIDTH = 8192


@triton.jit
def fused_gdn_prepare_prefill_kernel(
    mixed_qkv,
    b,
    a,
    core_attn_out,
    mixed_qkvz,
    mixed_ba,
    NUM_TOKENS: tl.constexpr,
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
    rows = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    columns = tl.arange(0, BLOCK_WIDTH)
    valid_rows = rows[:, None] < NUM_TOKENS
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


def supports_gdn_prepare_prefill(mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim):
    """Restrict the full-row tile to packed, TP-local projections fitting UB."""
    return (
        mixed_qkvz.ndim == 2
        and mixed_ba.ndim == 2
        and mixed_qkvz.shape[0] >= _PREFILL_MIN_TOKENS
        and mixed_qkvz.shape[0] == mixed_ba.shape[0]
        and mixed_qkvz.device == mixed_ba.device
        and mixed_qkvz.dtype in (torch.float16, torch.bfloat16)
        and mixed_ba.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and mixed_qkvz.stride(1) == mixed_ba.stride(1) == 1
        and qkv_width > 0
        and head_v_dim > 0
        and 0 < num_v_heads <= 1024
        and max(qkv_width, num_v_heads * head_v_dim) <= _PREFILL_MAX_WIDTH
        and mixed_qkvz.shape[1] == qkv_width + num_v_heads * head_v_dim
        and mixed_ba.shape[1] == 2 * num_v_heads
    )


def _prepare_prefill_fake(mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim):
    tokens = mixed_qkvz.shape[0]
    return (
        mixed_qkvz.new_empty((tokens, qkv_width)),
        mixed_ba.new_empty((tokens, num_v_heads)),
        mixed_ba.new_empty((tokens, num_v_heads)),
        mixed_qkvz.new_empty((tokens, num_v_heads, head_v_dim)),
    )


def _prepare_prefill(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    qkv_width: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not supports_gdn_prepare_prefill(mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim):
        raise ValueError("Unsupported packed GDN prefill projection shape, dtype or stride")
    # Keep fresh allocation and writes inside one functional custom op, avoiding
    # AOT clones of empty output buffers before the Triton launch.
    mixed_qkv, b, a, core_attn_out = _prepare_prefill_fake(mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim)
    tokens = mixed_qkvz.shape[0]
    core_width = num_v_heads * head_v_dim
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
    return mixed_qkv, b, a, core_attn_out


direct_register_custom_op(
    op_name="ascend_fused_gdn_prepare_prefill",
    op_func=_prepare_prefill,
    fake_impl=_prepare_prefill_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def fused_gdn_prepare_prefill(mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim):
    """Copy QKV/B/A and zero the full output in one large-batch launch.

    Z remains a view. No normalization, gate activation or state update occurs.
    Callers must keep their existing path when the support predicate is false.
    """
    mixed_qkv, b, a, core = torch.ops.vllm.ascend_fused_gdn_prepare_prefill(
        mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim
    )
    z = mixed_qkvz[:, qkv_width:].reshape(mixed_qkvz.shape[0], num_v_heads, head_v_dim)
    return mixed_qkv, z, b, a, core
