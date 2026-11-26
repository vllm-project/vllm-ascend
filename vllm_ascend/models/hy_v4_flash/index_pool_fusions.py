# SPDX-License-Identifier: Apache-2.0
"""Fused four-token IndexPool pooling, softmax and RMS normalization."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _pool_norm_kernel(projected, ape, gamma, output, ratio: tl.constexpr, width: tl.constexpr, epsilon: tl.constexpr):
    group = tl.program_id(0)
    token = tl.arange(0, ratio)
    col = tl.arange(0, width)
    offset = (group * ratio + token[:, None]) * (2 * width) + col[None, :]
    value = tl.load(projected + offset)
    gate = tl.load(projected + offset + width) + tl.load(ape + token[:, None] * width + col[None, :])
    probability = tl.exp(gate - tl.max(gate, 0)[None, :])
    probability = probability / tl.sum(probability, 0)[None, :]
    pooled = tl.sum(value * probability, 0)
    inverse = tl.rsqrt(tl.sum(pooled * pooled, 0) / width + epsilon)
    gain = tl.load(gamma + col)
    tl.store(output + group * width + col, (pooled * inverse) * gain)


def pool_norm(raw, projection, ape, gamma, epsilon):
    groups, ratio, width = raw.shape
    projected = torch.nn.functional.linear(raw.flatten(0, 1).float(), projection)
    output = torch.empty((groups, width), device=raw.device, dtype=raw.dtype)
    _pool_norm_kernel[(groups,)](
        projected,
        ape,
        gamma,
        output,
        ratio,
        width,
        epsilon,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
