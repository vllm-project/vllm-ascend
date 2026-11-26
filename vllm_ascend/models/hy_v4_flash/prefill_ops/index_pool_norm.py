# SPDX-License-Identifier: Apache-2.0
"""Preserve CANN's rsqrt rounding while fusing the surrounding IndexPool norm."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["rows", "epsilon"])
def _index_pool_mean_square(keys, stats, rows, epsilon, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    channel = tl.arange(0, 128)
    valid = row < rows
    key = tl.load(keys + row[:, None] * 128 + channel[None, :], valid[:, None], other=0).to(tl.float32)
    mean = tl.sum(key * key, axis=1) * (1.0 / 128.0)
    tl.store(stats + row, mean + epsilon, valid)


def token_key_norm_inverse(keys, epsilon):
    rows = keys.shape[0]
    stats = torch.empty(rows, dtype=torch.float32, device=keys.device)
    tile = 16
    _index_pool_mean_square[(triton.cdiv(rows, tile),)](keys, stats, rows, epsilon, tile, enable_fp_fusion=False)
    return stats.rsqrt()
