# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.attention.utils import SFA_QSFA_TILE_SIZE

_LATENT = 512
_ROPE = 64
_ROWS_PER_PROGRAM = 8


@triton.jit(do_not_specialize=["count"])
def _read_c8_rows(
    cache,
    rope_cache,
    scale_cache,
    block_table,
    positions,
    output,
    count,
    ROW_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE_TILE: tl.constexpr,
    ROWS: tl.constexpr,
):
    i = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    in_bounds = i < count
    position = tl.load(positions + i, in_bounds, -1)
    live = in_bounds & (position >= 0)
    page = tl.load(block_table + position // PAGE_SIZE, live, 0).to(tl.int64)
    slot = page * PAGE_SIZE + position % PAGE_SIZE
    lane = tl.arange(0, LATENT)
    bits = tl.load(cache + slot[:, None] * ROW_BYTES + lane[None, :], live[:, None], 0).to(tl.int32)
    exponent = (bits >> 3) & 15
    fraction = bits & 7
    # E4M3FN normal numbers embed exactly in FP32. The smallest
    # E4M3FN subnormal is 2**-9.
    normal = (((exponent + 120) << 23) | (fraction << 20)).to(tl.float32, bitcast=True)
    magnitude = tl.where(exponent == 0, fraction.to(tl.float32) * (2.0**-9), normal)
    value = tl.where(bits & 128 != 0, -magnitude, magnitude)
    value = tl.where(bits & 127 == 127, float("nan"), value)
    scale = tl.load(
        scale_cache + slot[:, None] * (ROW_BYTES // 4) + (LATENT + 2 * ROPE) // 4 + lane[None, :] // SCALE_TILE,
        live[:, None],
        0.0,
    )
    dst = i[:, None] * (LATENT + ROPE)
    tl.store(output + dst + lane[None, :], value * scale, in_bounds[:, None])
    r = tl.arange(0, ROPE)
    rope = tl.load(rope_cache + slot[:, None] * (ROW_BYTES // 2) + LATENT // 2 + r[None, :], live[:, None], 0.0).to(
        tl.float32
    )
    tl.store(output + dst + LATENT + r[None, :], rope, in_bounds[:, None])


def gather_c8_rows(cache, block_table, block_size, positions):
    output = torch.empty((*positions.shape, _LATENT + _ROPE), dtype=torch.float32, device=cache.device)
    count = positions.numel()
    if count:
        _read_c8_rows[(triton.cdiv(count, _ROWS_PER_PROGRAM),)](
            cache,
            cache.view(torch.bfloat16),
            cache.view(torch.float32),
            block_table,
            positions.contiguous(),
            output,
            count,
            cache.shape[-1],
            block_size,
            _LATENT,
            _ROPE,
            SFA_QSFA_TILE_SIZE,
            _ROWS_PER_PROGRAM,
            num_warps=1,
        )
    return output
