# SPDX-License-Identifier: Apache-2.0
"""Layout and normalization operators for the batch-one Qwen vision encoder and packed GDN projection."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _split_rows(src, left, right, LEFT: tl.constexpr, RIGHT: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    values = tl.load(src + row * (LEFT + RIGHT) + col, col < LEFT + RIGHT, other=0)
    tl.store(left + row * LEFT + col, values, col < LEFT)
    tl.store(right + row * RIGHT + col - LEFT, values, (col >= LEFT) & (col < LEFT + RIGHT))


def split_packed_qkvz(src, qkv_width):
    rows, width = src.shape
    qkv = src.new_empty((rows, qkv_width))
    z = src.new_empty((rows, width - qkv_width))
    _split_rows[(rows, triton.cdiv(width, 1024))](src, qkv, z, qkv_width, width - qkv_width, 1024)
    return qkv, z


@triton.jit(do_not_specialize=["tokens"])
def _pack_vision_qk(src, dst, tokens, WIDTH: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    plane = tl.program_id(1)
    cols = tl.arange(0, COLS)
    mask = (rows[:, None] < tokens) & (cols[None, :] < WIDTH)
    # Inside each token the source has Q, K, V; the destination has separate Q and K planes.
    src_rows = (rows * 3 + plane) * WIDTH
    dst_rows = (plane * tokens + rows) * WIDTH
    values = tl.load(src + src_rows[:, None] + cols[None, :], mask, other=0)
    tl.store(dst + dst_rows[:, None] + cols[None, :], values, mask)


def pack_vision_qk(qkv):
    """Pack [T, 3, H, D] into the contiguous [2, T, H, D] input of native RoPE."""
    tokens, _, heads, dim = qkv.shape
    width = heads * dim
    if not qkv.is_contiguous() or tokens == 0 or tokens * 3 * width >= 2**31:
        return qkv[:, :2].permute(1, 0, 2, 3).contiguous()
    out = qkv.new_empty((2, tokens, heads, dim))
    rows = 8
    _pack_vision_qk[(triton.cdiv(tokens, rows), 2)](qkv, out, tokens, width, rows, triton.next_power_of_2(width))
    return out


def vision_add_norm(x, attn, norm):
    normalized, _, _, residual = torch.ops.npu.npu_add_layer_norm(
        x, attn, norm.weight, norm.bias, epsilon=norm.eps, additional_output=True
    )
    return residual, normalized
