from functools import lru_cache

import torch
from vllm.triton_utils import tl, triton


@lru_cache(maxsize=32)
def rope_constants(device, stream, theta, rotary_dim, sections, interleaved):
    frequency = theta ** (-torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
    if sections and interleaved:
        channels = torch.arange(rotary_dim // 2, device=device)
        height = (channels % 3 == 1) & (channels < sections[1] * 3)
        width = (channels % 3 == 2) & (channels < sections[2] * 3)
        return frequency, height, width
    return frequency, None, None


def make_factors(positions, theta, rotary_dim, sections, interleaved, stream):
    frequency, height, width = rope_constants(positions.device, stream, theta, rotary_dim, sections, interleaved)
    phase = positions.float().unsqueeze(-1) * frequency
    if positions.ndim == 2:
        if interleaved:
            phase = torch.where(width, phase[2], torch.where(height, phase[1], phase[0]))
        else:
            phase = torch.cat([part[axis] for axis, part in enumerate(phase.split(sections, dim=-1))], dim=-1)
    return phase.cos(), phase.sin()


@triton.jit(do_not_specialize=["Rows", "Start"])
def _index_pool_rotate_rows(
    Vectors,
    Cos,
    Sin,
    Out,
    Rows,
    Heads: tl.constexpr,
    Start,
    Step: tl.constexpr,
    Block: tl.constexpr,
    GroupMap,
    HistoryCos,
    HistorySin,
    MAPPED: tl.constexpr,
    WITH_HISTORY: tl.constexpr,
):
    row = tl.program_id(0) * Block + tl.arange(0, Block)
    col = tl.arange(0, 64)
    angle = tl.arange(0, 32)
    base = row[:, None] * 128
    mask = row[:, None] < Rows
    rope = tl.load(Vectors + base + 64 + col[None, :], mask, other=0).to(tl.float32)
    even, odd = tl.split(tl.reshape(rope, (Block, 32, 2)))
    if MAPPED:
        pos = tl.load(GroupMap + row, row < Rows, other=0)
    else:
        pos = Start + row // Heads * Step
    offset = pos[:, None] * 32 + angle[None, :]
    if WITH_HISTORY:
        is_history = pos[:, None] < 0
        history_offset = (-1 - pos[:, None]) * 32 + angle[None, :]
        cos = tl.load(Cos + offset, mask & ~is_history, other=0)
        sin = tl.load(Sin + offset, mask & ~is_history, other=0)
        old_cos = tl.load(HistoryCos + history_offset, mask & is_history, other=0)
        old_sin = tl.load(HistorySin + history_offset, mask & is_history, other=0)
        cos = tl.where(is_history, old_cos, cos)
        sin = tl.where(is_history, old_sin, sin)
    else:
        cos = tl.load(Cos + offset, mask, other=0)
        sin = tl.load(Sin + offset, mask, other=0)
    even_rotated = even * cos - odd * sin
    odd_rotated = odd * cos + even * sin
    nope = tl.load(Vectors + base + col[None, :], mask, other=0)
    tl.store(Out + base + col[None, :], nope, mask)
    tl.store(Out + base + 64 + angle[None, :], even_rotated, mask)
    tl.store(Out + base + 96 + angle[None, :], odd_rotated, mask)


def apply_factors(vectors, factors, start=0, step=1, *, group_map=None, history_factors=None):
    out = torch.empty_like(vectors)
    heads = vectors.shape[1] if vectors.ndim == 3 else 1
    rows = vectors.shape[0] * heads
    block = 128 if rows >= 32768 else 32
    _index_pool_rotate_rows[(triton.cdiv(rows, block),)](
        vectors,
        *factors,
        out,
        rows,
        heads,
        start,
        step,
        block,
        vectors if group_map is None else group_map,
        *(factors if history_factors is None else history_factors),
        group_map is not None,
        history_factors is not None,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out
