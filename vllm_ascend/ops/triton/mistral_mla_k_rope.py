# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Rotate the positional part of a Mistral Large 4 MLA KV projection."""

import torch
import triton
import triton.language as tl

from .mistral_mla_q import _round_bf16


@triton.jit
def _slice_k_rope(kv, cos, sin, output, COS_STRIDE: tl.constexpr, SIN_STRIDE: tl.constexpr):
    token = tl.program_id(0)
    pair = tl.arange(0, 32)
    rope = kv + token * 576 + 512
    even = tl.load(rope + 2 * pair).to(tl.float32)
    odd = tl.load(rope + 2 * pair + 1).to(tl.float32)
    c = tl.load(cos + token * COS_STRIDE + pair).to(tl.float32)
    s = tl.load(sin + token * SIN_STRIDE + pair).to(tl.float32)
    # The packed cache expects the split-half output of npu_interleave_rope.
    row = output + token * 64
    tl.store(row + pair, _round_bf16(even * c - odd * s))
    tl.store(row + 32 + pair, _round_bf16(odd * c + even * s))


def try_slice_k_rope(kv, cos, sin):
    """Return rotated BF16 K PE for a supported decode shape."""
    if kv.ndim != 4:
        return None
    tokens = kv.shape[0]
    if (
        kv.device.type != "npu"
        or kv.dtype != torch.bfloat16
        or kv.shape != (tokens, 1, 1, 576)
        or not 0 < tokens <= 8
        or not kv.is_contiguous()
        or not all(
            x is not None
            and x.shape == (tokens, 1, 1, 64)
            and x.device == kv.device
            and x.dtype == kv.dtype
            and x.stride(-1) == 1
            for x in (cos, sin)
        )
    ):
        return None

    output = kv.new_empty((tokens, 1, 1, 64))
    _slice_k_rope[(tokens,)](kv, cos, sin, output, cos.stride(0), sin.stride(0), enable_fp_fusion=False)
    return output
