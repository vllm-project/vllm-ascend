# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.attention.utils import SFA_QSFA_TILE_SIZE

_LATENT = 512
_ROPE = 64
_ROWS_PER_PROGRAM = 8
_MAX_FIA_COMPACT_CANDIDATES = 8192
_QK_HEAD_TILE = 16
_QK_CANDIDATE_TILE = 32


@triton.jit
def _decode_e4m3fn(bits):
    bits = bits.to(tl.int32)
    exponent = (bits >> 3) & 15
    fraction = bits & 7
    # E4M3FN normal numbers embed exactly in FP32. The smallest
    # E4M3FN subnormal is 2**-9.
    normal = (((exponent + 120) << 23) | (fraction << 20)).to(tl.float32, bitcast=True)
    magnitude = tl.where(exponent == 0, fraction.to(tl.float32) * (2.0**-9), normal)
    value = tl.where(bits & 128 != 0, -magnitude, magnitude)
    return tl.where(bits & 127 == 127, float("nan"), value)


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
    bits = tl.load(cache + slot[:, None] * ROW_BYTES + lane[None, :], live[:, None], 0)
    groups = tl.arange(0, LATENT // SCALE_TILE)
    scale = tl.load(
        scale_cache + slot[:, None] * (ROW_BYTES // 4) + (LATENT + 2 * ROPE) // 4 + groups[None, :],
        live[:, None],
        0.0,
    )
    # Read each scale once and broadcast it in the vector buffer to its 128 values.
    value = _decode_e4m3fn(bits).reshape((ROWS, LATENT // SCALE_TILE, SCALE_TILE))
    latent = (value * scale[:, :, None]).reshape((ROWS, LATENT))
    dst = i[:, None] * (LATENT + ROPE)
    tl.store(output + dst + lane[None, :], latent, in_bounds[:, None])
    r = tl.arange(0, ROPE)
    rope = tl.load(rope_cache + slot[:, None] * (ROW_BYTES // 2) + LATENT // 2 + r[None, :], live[:, None], 0.0).to(
        tl.float32
    )
    tl.store(output + dst + LATENT + r[None, :], rope, in_bounds[:, None])


def gather_c8_rows(cache, block_table, block_size, positions, *, output=None):
    shape = (*positions.shape, _LATENT + _ROPE)
    if output is None:
        output = torch.empty(shape, dtype=torch.float32, device=cache.device)
    elif (
        output.shape != shape
        or output.dtype != torch.float32
        or output.device != cache.device
        or not output.is_contiguous()
    ):
        raise ValueError("C8 output must be contiguous FP32 with the requested shape and device")
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


@triton.jit(do_not_specialize=["count"])
def _read_c8_banded(
    cache,
    rope_cache,
    scale_cache,
    block_table,
    key,
    value,
    count,
    ROW_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE_TILE: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    live = row < count
    page = tl.load(block_table + row // PAGE_SIZE, live, 0).to(tl.int64)
    slot = page * PAGE_SIZE + row % PAGE_SIZE
    lane = tl.arange(0, LATENT)
    group = tl.arange(0, LATENT // SCALE_TILE)
    bits = tl.load(cache + slot[:, None] * ROW_BYTES + lane[None, :], live[:, None], 0)
    scales = tl.load(
        scale_cache + slot[:, None] * (ROW_BYTES // 4) + (LATENT + 2 * ROPE) // 4 + group[None, :],
        live[:, None],
        0.0,
    )
    decoded = _decode_e4m3fn(bits).reshape((ROWS, LATENT // SCALE_TILE, SCALE_TILE))
    latent = (decoded * scales[:, :, None]).reshape((ROWS, LATENT)).to(tl.bfloat16)
    key_row = key + row[:, None] * (LATENT + ROPE)
    tl.store(key_row + lane[None, :], latent, live[:, None])
    tl.store(value + row[:, None] * LATENT + lane[None, :], latent, live[:, None])
    rope_lane = tl.arange(0, ROPE)
    rope = tl.load(
        rope_cache + slot[:, None] * (ROW_BYTES // 2) + LATENT // 2 + rope_lane[None, :],
        live[:, None],
        0.0,
    )
    tl.store(key_row + LATENT + rope_lane[None, :], rope, live[:, None])


def read_c8_banded(cache, block_table, block_size, rows):
    """Build contiguous BF16 K and V for the CANN banded attention path."""
    key = torch.empty((rows, _LATENT + _ROPE), device=cache.device, dtype=torch.bfloat16)
    value = torch.empty((rows, _LATENT), device=cache.device, dtype=torch.bfloat16)
    _read_c8_banded[(triton.cdiv(rows, _ROWS_PER_PROGRAM),)](
        cache,
        cache.view(torch.bfloat16),
        cache.view(torch.float32),
        block_table,
        key,
        value,
        rows,
        cache.shape[-1],
        block_size,
        _ROWS_PER_PROGRAM,
        _LATENT,
        _ROPE,
        SFA_QSFA_TILE_SIZE,
        num_warps=1,
    )
    return key, value


@triton.jit(do_not_specialize=["count"])
def _read_c8_fia_rows(
    cache,
    rope_cache,
    scale_cache,
    block_table,
    positions,
    latent_output,
    rope_output,
    mask_output,
    count,
    ROW_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE_TILE: tl.constexpr,
    ROWS: tl.constexpr,
):
    index = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    in_bounds = index < count
    position = tl.load(positions + index, in_bounds, -1)
    live = in_bounds & (position >= 0)
    page = tl.load(block_table + position // PAGE_SIZE, live, 0).to(tl.int64)
    slot = page * PAGE_SIZE + position % PAGE_SIZE
    lane = tl.arange(0, LATENT)
    bits = tl.load(cache + slot[:, None] * ROW_BYTES + lane[None, :], live[:, None], 0)
    group = tl.arange(0, LATENT // SCALE_TILE)
    scales = tl.load(
        scale_cache + slot[:, None] * (ROW_BYTES // 4) + (LATENT + 2 * ROPE) // 4 + group[None, :],
        live[:, None],
        0.0,
    )
    decoded = _decode_e4m3fn(bits).reshape((ROWS, LATENT // SCALE_TILE, SCALE_TILE))
    latent = (decoded * scales[:, :, None]).reshape((ROWS, LATENT))
    tl.store(latent_output + index[:, None] * LATENT + lane[None, :], latent, in_bounds[:, None])
    rope_lane = tl.arange(0, ROPE)
    rope = tl.load(
        rope_cache + slot[:, None] * (ROW_BYTES // 2) + LATENT // 2 + rope_lane[None, :],
        live[:, None],
        0.0,
    )
    tl.store(rope_output + index[:, None] * ROPE + rope_lane[None, :], rope, in_bounds[:, None])
    tl.store(mask_output + index, position < 0, in_bounds)


@triton.jit
def _compact_fia_positions(
    positions, output, candidates: tl.constexpr, input_stride: tl.constexpr, block: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    value = tl.load(positions + row * input_stride + col, col < candidates, -1)
    valid = (col < candidates) & (value >= 0)
    destination = tl.cumsum(valid.to(tl.int32), 0) - 1
    count = tl.sum(valid.to(tl.int32), 0)
    # The two stores have disjoint destinations: [count, K) and [0, count).
    # Every output is written, including all-invalid rows and graph padding.
    tl.store(output + row * candidates + col, -1, (col < candidates) & (col >= count))
    tl.store(output + row * candidates + destination, value, valid)


def compact_fia_positions(positions):
    """Stable-pack integer indices before decoding the much larger C8 rows."""
    if positions.ndim != 2 or positions.stride(1) != 1 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("FIA index compaction requires 2D integer indices with unit column stride")
    queries, candidates = positions.shape
    output = torch.empty((queries, candidates), dtype=positions.dtype, device=positions.device)
    if queries and candidates:
        _compact_fia_positions[(queries,)](
            positions, output, candidates, positions.stride(0), triton.next_power_of_2(candidates), num_warps=1
        )
    return output


def empty_c8_fia_rows(queries, candidates, device, *, dtype=torch.bfloat16):
    latent = torch.empty((queries, candidates, 1, _LATENT), dtype=dtype, device=device)
    rope = torch.empty((queries, candidates, 1, _ROPE), dtype=dtype, device=device)
    mask = torch.empty((queries, 1, 1, candidates), dtype=torch.bool, device=device)
    return latent, rope, mask


def empty_compact_c8_fia_rows(queries, candidates, device):
    from .fia_lengths import CompactFIAKV

    return CompactFIAKV(*empty_c8_fia_rows(queries, candidates, device, dtype=torch.float32))


def gather_compact_c8_fia_rows(cache, block_table, block_size, positions, *, output=None):
    """Read stable-packed C8 in FP32, retaining the original dequantization."""
    from .fia_lengths import CompactFIAKV, compact_fia_rows

    if (
        positions.device.type != "npu"
        or positions.ndim != 2
        or positions.stride(1) != 1
        or positions.dtype not in (torch.int32, torch.int64)
        or not 0 < positions.shape[1] <= _MAX_FIA_COMPACT_CANDIDATES
    ):
        rows = gather_c8_fia_rows(cache, block_table, block_size, positions, output=output, dtype=torch.float32)
        nope, rope = compact_fia_rows(positions, rows[0], rows[1])
        rows[0].copy_(nope)
        rows[1].copy_(rope)
        count = (positions >= 0).sum(dim=-1)
        rows[2].copy_((torch.arange(positions.shape[1], device=positions.device) >= count[:, None])[:, None, None])
        return CompactFIAKV(*rows)
    packed = compact_fia_positions(positions)
    rows = gather_c8_fia_rows(cache, block_table, block_size, packed, output=output, dtype=torch.float32)
    return CompactFIAKV(*rows)


def gather_c8_fia_rows(cache, block_table, block_size, positions, *, output=None, dtype=torch.bfloat16):
    """Gather sparse rows directly into contiguous BF16 FIA K/rope and mask."""
    queries, candidates = positions.shape
    if output is None:
        output = empty_c8_fia_rows(queries, candidates, cache.device, dtype=dtype)
    count = positions.numel()
    if count:
        _read_c8_fia_rows[(triton.cdiv(count, _ROWS_PER_PROGRAM),)](
            cache,
            cache.view(torch.bfloat16),
            cache.view(torch.float32),
            block_table,
            positions.contiguous(),
            *output,
            count,
            cache.shape[-1],
            block_size,
            _LATENT,
            _ROPE,
            SFA_QSFA_TILE_SIZE,
            _ROWS_PER_PROGRAM,
            num_warps=1,
            enable_fp_fusion=False,
            enable_graph_optimize=False,
        )
    return output


@triton.jit
def _read_c8_qk(
    query,
    cache,
    rope_cache,
    scale_cache,
    block_table,
    positions,
    scores,
    values,
    ROW_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LATENT: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE_TILE: tl.constexpr,
    HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    CANDIDATE_TILE: tl.constexpr,
):
    qi = tl.program_id(0)
    row = tl.program_id(1) * CANDIDATE_TILE + tl.arange(0, CANDIDATE_TILE)
    head = tl.arange(0, HEAD_TILE)
    position = tl.load(positions + qi * TOPK + row, row < TOPK, -1)
    live = position >= 0
    page = tl.load(block_table + position // PAGE_SIZE, live, 0).to(tl.int64)
    slot = page * PAGE_SIZE + position % PAGE_SIZE
    q_base = (qi * HEADS + head[:, None]) * (LATENT + ROPE)
    value_base = (qi * TOPK + row[:, None]) * LATENT
    acc = tl.zeros((HEAD_TILE, CANDIDATE_TILE), tl.float32)

    lane = tl.arange(0, SCALE_TILE)
    for group in tl.static_range(LATENT // SCALE_TILE):
        col = group * SCALE_TILE + lane
        q = tl.load(query + q_base + col[None, :], head[:, None] < HEADS, 0).to(tl.float32)
        bits = tl.load(cache + slot[:, None] * ROW_BYTES + col[None, :], live[:, None], 0)
        scale = tl.load(scale_cache + slot * (ROW_BYTES // 4) + (LATENT + 2 * ROPE) // 4 + group, live, 0)
        k = _decode_e4m3fn(bits) * scale[:, None]
        tl.store(values + value_base + col[None, :], k, row[:, None] < TOPK)
        acc = tl.dot(q, tl.trans(k), acc, input_precision="ieee")

    rope = tl.arange(0, ROPE)
    q_rope = tl.load(query + q_base + LATENT + rope[None, :], head[:, None] < HEADS, 0).to(tl.float32)
    k_rope = tl.load(rope_cache + slot[:, None] * (ROW_BYTES // 2) + LATENT // 2 + rope[None, :], live[:, None], 0).to(
        tl.float32
    )
    acc = tl.dot(q_rope, tl.trans(k_rope), acc, input_precision="ieee")
    tl.store(
        scores + (qi * HEADS + head[:, None]) * TOPK + row[None, :],
        acc,
        (head[:, None] < HEADS) & (row[None, :] < TOPK),
    )


def supports_c8_qk_heads(heads: int) -> bool:
    return 0 < heads <= _QK_HEAD_TILE


def gather_c8_qk(query, cache, block_table, block_size, positions):
    """Return FP32 QK and the 512 latent values needed by PV for each candidate."""
    queries, heads, _ = query.shape
    topk = positions.shape[1]
    scores = torch.empty((queries, heads, topk), dtype=torch.float32, device=query.device)
    values = torch.empty((queries, topk, _LATENT), dtype=torch.float32, device=query.device)
    _read_c8_qk[(queries, triton.cdiv(topk, _QK_CANDIDATE_TILE))](
        query,
        cache,
        cache.view(torch.bfloat16),
        cache.view(torch.float32),
        block_table,
        positions.contiguous(),
        scores,
        values,
        cache.shape[-1],
        block_size,
        _LATENT,
        _ROPE,
        SFA_QSFA_TILE_SIZE,
        heads,
        topk,
        _QK_HEAD_TILE,
        _QK_CANDIDATE_TILE,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return scores, values
