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


def _prepare_outputs(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    qkv_width: int,
    num_v_heads: int,
    head_v_dim: int,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = mixed_qkvz.shape[0]
    if core_attn_out is None:
        core_attn_out = torch.empty(
            (num_tokens, num_v_heads, head_v_dim), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
        )
    mixed_qkv = torch.empty((num_tokens, qkv_width), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device)
    b = torch.empty((num_tokens, num_v_heads), dtype=mixed_ba.dtype, device=mixed_ba.device)
    a = torch.empty_like(b)
    if num_tokens > 0:
        core_width = num_v_heads * head_v_dim
        if (
            num_tokens >= _PREFILL_MIN_TOKENS
            and max(qkv_width, core_width) <= _PREFILL_MAX_WIDTH
            and mixed_qkvz.dtype in (torch.float16, torch.bfloat16)
        ):
            # Bound the row tile to A3 UB capacity. Eight full 8192-wide rows
            # overflow UB with this fused copy/zero implementation.
            fused_gdn_prepare_prefill_kernel[(triton.cdiv(num_tokens, _PREFILL_TILE_ROWS),)](
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
                BLOCK_ROWS=_PREFILL_TILE_ROWS,
                BLOCK_WIDTH=triton.next_power_of_2(max(qkv_width, core_width)),
                BLOCK_HEADS=triton.next_power_of_2(num_v_heads),
            )
        else:
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
    return mixed_qkv, b, a, core_attn_out


def _prepare_fresh_outputs(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    qkv_width: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep allocations and writes inside one functional operator. Exposing the
    # four empty tensors as Triton mutation inputs makes AOT clone each of them.
    return _prepare_outputs(mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim)


def _prepare_fresh_outputs_fake(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    qkv_width: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = mixed_qkvz.shape[0]
    return (
        mixed_qkvz.new_empty((num_tokens, qkv_width)),
        mixed_ba.new_empty((num_tokens, num_v_heads)),
        mixed_ba.new_empty((num_tokens, num_v_heads)),
        mixed_qkvz.new_empty((num_tokens, num_v_heads, head_v_dim)),
    )


direct_register_custom_op(
    op_name="ascend_fused_gdn_prepare",
    op_func=_prepare_fresh_outputs,
    fake_impl=_prepare_fresh_outputs_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


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
    if core_attn_out is not None and (
        core_attn_out.shape != expected_core_shape
        or core_attn_out.dtype != mixed_qkvz.dtype
        or core_attn_out.device != mixed_qkvz.device
        or not core_attn_out.is_contiguous()
    ):
        raise ValueError("core_attn_out must be contiguous and match the expected shape, dtype, and device")

    if core_attn_out is None and num_tokens > 0:
        mixed_qkv, b, a, core_attn_out = torch.ops.vllm.ascend_fused_gdn_prepare(
            mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim
        )
    else:
        # The supplied-buffer API must preserve the caller's storage and writes.
        # Empty CPU tensors also keep the existing no-kernel behavior.
        mixed_qkv, b, a, core_attn_out = _prepare_outputs(
            mixed_qkvz, mixed_ba, qkv_width, num_v_heads, head_v_dim, core_attn_out
        )

    z = mixed_qkvz[:, qkv_width:].reshape(num_tokens, num_v_heads, head_v_dim)
    return mixed_qkv, z, b, a, core_attn_out
