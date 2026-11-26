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


def _make_factors(positions, theta, rotary_dim, sections, interleaved, stream):
    frequency, height, width = rope_constants(positions.device, stream, theta, rotary_dim, sections, interleaved)
    phase = positions.float().unsqueeze(-1) * frequency
    if positions.ndim == 2:
        if interleaved:
            phase = torch.where(width, phase[2], torch.where(height, phase[1], phase[0]))
        else:
            phase = torch.cat([part[axis] for axis, part in enumerate(phase.split(sections, dim=-1))], dim=-1)
    return phase.cos(), phase.sin()


@triton.jit(do_not_specialize=["rows"])
def _index_pool_rope_phase(
    positions, frequency, phase, rows, ROW_STRIDE: tl.constexpr, AXIS_STRIDE: tl.constexpr,
    HEIGHT_LIMIT: tl.constexpr, WIDTH_LIMIT: tl.constexpr, ROW_TILE: tl.constexpr,
):
    row = tl.program_id(0) * ROW_TILE + tl.arange(0, ROW_TILE)
    col = tl.arange(0, 32)
    axis = tl.where((col % 3 == 2) & (col < WIDTH_LIMIT), 2,
                    tl.where((col % 3 == 1) & (col < HEIGHT_LIMIT), 1, 0))
    position = tl.load(
        positions + row[:, None] * ROW_STRIDE + axis[None, :] * AXIS_STRIDE,
        row[:, None] < rows,
        other=0,
    ).to(tl.float32)
    inverse = tl.load(frequency + col)
    tl.store(phase + row[:, None] * 32 + col[None, :],
             position * inverse[None, :], row[:, None] < rows)


def make_factors(positions, theta, rotary_dim, sections, interleaved, stream):
    if (
        positions.device.type != "npu"
        or positions.dtype not in (torch.int32, torch.int64)
        or rotary_dim != 64
        or not 0 < positions.shape[-1] <= 4096
        or not (positions.ndim == 2 and positions.shape[0] == 3 and interleaved and sections)
    ):
        return _make_factors(positions, theta, rotary_dim, sections, interleaved, stream)
    frequency = rope_constants(positions.device, stream, theta, rotary_dim, sections, interleaved)[0]
    rows = positions.shape[-1]
    phase = torch.empty((rows, 32), dtype=torch.float32, device=positions.device)
    tile = 32
    _index_pool_rope_phase[(triton.cdiv(rows, tile),)](
        positions, frequency, phase, rows, positions.stride(-1),
        positions.stride(0), sections[1] * 3, sections[2] * 3,
        tile,
    )
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
    NormWeight,
    NormInverse,
    APPLY_NORM: tl.constexpr,
):
    row = tl.program_id(0) * Block + tl.arange(0, Block)
    col = tl.arange(0, 64)
    angle = tl.arange(0, 32)
    base = row[:, None] * 128
    mask = row[:, None] < Rows
    rope = tl.load(Vectors + base + 64 + col[None, :], mask, other=0).to(tl.float32)
    if APPLY_NORM:
        inverse = tl.load(NormInverse + row, row < Rows, other=0)
        rotary_weight = tl.load(NormWeight + 64 + col)
        rope = (rope * inverse[:, None]) * rotary_weight[None, :]
        rope = tl.cast(rope, tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
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
    if APPLY_NORM:
        nope_weight = tl.load(NormWeight + col)
        nope = (nope * inverse[:, None]) * nope_weight[None, :]
    tl.store(Out + base + col[None, :], nope, mask)
    tl.store(Out + base + 64 + angle[None, :], even_rotated, mask)
    tl.store(Out + base + 96 + angle[None, :], odd_rotated, mask)


def apply_factors(vectors, factors, start=0, step=1, *, group_map=None, history_factors=None, normalization=None):
    out = torch.empty_like(vectors) if normalization is None else torch.empty_like(vectors, dtype=torch.bfloat16)
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
        *(normalization if normalization is not None else (vectors, vectors)),
        normalization is not None,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out
