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


def vision_add_norm(x, attn, norm):
    normalized, _, _, residual = torch.ops.npu.npu_add_layer_norm(
        x, attn, norm.weight, norm.bias, epsilon=norm.eps, additional_output=True
    )
    return residual, normalized
