# SPDX-License-Identifier: Apache-2.0
"""Combine weighted IndexPool reduction and RMS statistics for assembled groups."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["rows", "epsilon"])
def _index_pool_weighted_stats(values, probabilities, output, stats, rows, epsilon, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    channel = tl.arange(0, 128)
    source = row[:, None] * 512 + channel[None, :]
    valid = row < rows
    live = valid[:, None]
    p0 = tl.load(values + source, live, other=0) * tl.load(probabilities + source, live, other=0)
    p1 = tl.load(values + source + 128, live, other=0) * tl.load(probabilities + source + 128, live, other=0)
    p2 = tl.load(values + source + 256, live, other=0) * tl.load(probabilities + source + 256, live, other=0)
    p3 = tl.load(values + source + 384, live, other=0) * tl.load(probabilities + source + 384, live, other=0)
    # The four-lane CANN reduction pairs opposite lanes before the final add.
    pooled = (p0 + p2) + (p1 + p3)
    mean = tl.sum(pooled * pooled, axis=1) * (1.0 / 128.0)
    tl.store(stats + row, mean + epsilon, valid)
    tl.store(output + row[:, None] * 128 + channel[None, :], pooled, live)


def pool_with_inverse_norm(values, probabilities, epsilon):
    rows = values.shape[0]
    pooled = torch.empty((rows, 128), dtype=torch.float32, device=values.device)
    stats = torch.empty(rows, dtype=torch.float32, device=values.device)
    tile = 16
    _index_pool_weighted_stats[(triton.cdiv(rows, tile),)](
        values, probabilities, pooled, stats, rows, epsilon, tile, enable_fp_fusion=False
    )
    return pooled, stats.rsqrt()
