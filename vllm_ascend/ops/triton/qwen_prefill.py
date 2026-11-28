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


@triton.jit
def _vision_rope(src, cos, sin, dst, heads: tl.constexpr, dim: tl.constexpr, tokens, TILE: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    plane = tl.program_id(1)
    i = tl.arange(0, TILE)
    half = dim // 2
    mask = i < heads * half
    lane = i % half
    offset = (i // half) * dim + lane
    x = src + ((token * 3 + plane) * heads * dim) + offset
    y = dst + ((plane * tokens + token) * heads * dim) + offset
    a = tl.load(x, mask, other=0).to(tl.float32)
    b = tl.load(x + half, mask, other=0).to(tl.float32)
    c = tl.load(cos + token * half + lane).to(tl.float32)
    s = tl.load(sin + token * half + lane).to(tl.float32)
    tl.store(y, a * c - b * s, mask)
    tl.store(y + half, b * c + a * s, mask)


def vision_qk_rope(qkv, cos, sin):
    tokens, _, heads, dim = qkv.shape
    out = torch.empty((2, 1, tokens, heads, dim), device=qkv.device, dtype=qkv.dtype)
    _vision_rope[(tokens, 2)](qkv, cos, sin, out, heads, dim, tokens, triton.next_power_of_2(heads * dim // 2))
    return out.unbind(0)


def vision_add_norm(x, attn, norm):
    normalized, _, _, residual = torch.ops.npu.npu_add_layer_norm(
        x, attn, norm.weight, norm.bias, epsilon=norm.eps, additional_output=True
    )
    return residual, normalized
