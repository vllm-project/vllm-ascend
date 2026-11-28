# SPDX-License-Identifier: Apache-2.0
"""Materialize only QSA pages needed by this step's BF16 Union attention."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _mark_pages(
    indices,
    positions,
    lengths,
    requests,
    table,
    used,
    index_stride,
    table_stride,
    column_stride,
    num_requests,
    GROUPS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    group_col = tl.arange(0, TILE)
    req = tl.load(requests + row)
    pos = tl.load(positions + row)
    valid_req = (req >= 0) & (req < num_requests) & (pos >= 0)
    length = tl.load(lengths + req, valid_req, other=0)
    visible = tl.minimum(pos + 1, length)
    group = tl.load(indices + row * index_stride + group_col, group_col < GROUPS, other=-1)
    valid = valid_req & (group >= 0) & (group < visible // 4)
    page = tl.load(table + req * table_stride + (group // 32) * column_stride, valid, other=0)
    tl.store(used + page, 1, valid)
    if valid_req and visible % 4 != 0:
        tail_page = tl.load(table + req * table_stride + (visible // 128) * column_stride)
        tl.store(used + tail_page, 1)


@triton.jit
def _mark_request_pages(table, lengths, used, row_stride, col_stride, cols, TILE: tl.constexpr):
    req = tl.program_id(0)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    length = tl.load(lengths + req)
    valid = (col < cols) & (col * 128 < length)
    page = tl.load(table + req * row_stride + col * col_stride, valid, other=0)
    tl.store(used + page, 1, valid)


@triton.jit
def _decode_e4m3(bits):
    bits = bits.to(tl.uint32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    normal = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
    magnitude = tl.where(exponent == 0, mantissa.to(tl.float32) * 0.001953125, normal)
    magnitude = tl.where((bits & 127) == 127, float("nan"), magnitude)
    return tl.where(bits >= 128, -magnitude, magnitude)


@triton.jit
def _scale(byte):
    exponent = byte.to(tl.uint32)
    value = (exponent << 23).to(tl.float32, bitcast=True)
    return tl.where(exponent == 255, float("nan"), tl.where(exponent == 0, 5.877471754111438e-39, value))


@triton.jit
def _materialize(
    pages,
    current_rows,
    k,
    v,
    ks,
    vs,
    current_k,
    current_v,
    out_k,
    out_v,
    k_row_stride,
    v_row_stride,
    k_col_stride,
    v_col_stride,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
):
    shard = tl.program_id(0)
    dest_page = shard // (128 // ROWS)
    src_page = tl.load(pages + dest_page)
    row = (shard % (128 // ROWS)) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, WIDTH)
    current = tl.load(current_rows + dest_page * 128 + row)
    write_mask = col[None, :] < DIM
    hist = (current[:, None] < 0) & write_mask
    fresh = (current[:, None] >= 0) & write_mask
    src = ((src_page * 128 + row[:, None]) * DIM + col[None, :]).to(tl.int64)
    if tl.sum((current < 0).to(tl.int32), 0) > 0:
        kb = tl.load(k + src, hist, other=0)
        vb = tl.load(v + src, hist, other=0)
        kscale = tl.load(ks + src // 32, hist, other=127)
        # The two V-scale lanes and both scale tiles contain the layer's static channel scale.
        vscale = tl.load(vs + src_page * 4 * DIM + col * 2, col < DIM, other=127)
        kh = _decode_e4m3(kb) * _scale(kscale)
        vh = _decode_e4m3(vb) * _scale(vscale)[None, :]
    else:
        kh = tl.full((ROWS, WIDTH), 0, tl.float32)
        vh = tl.full((ROWS, WIDTH), 0, tl.float32)
    kc = tl.load(current_k + current[:, None] * k_row_stride + col[None, :] * k_col_stride, fresh, other=0)
    vc = tl.load(current_v + current[:, None] * v_row_stride + col[None, :] * v_col_stride, fresh, other=0)
    dest = (dest_page * 128 + row[:, None]) * DIM + col[None, :]
    tl.store(out_k + dest, tl.where(fresh, kc, kh), write_mask)
    tl.store(out_v + dest, tl.where(fresh, vc, vh), write_mask)


def prepare_union_pages(num_pages, slots, indices, positions, lengths, requests, table, request_page_limit):
    groups = indices.shape[1] - 3
    used = torch.zeros(num_pages, dtype=torch.int32, device=table.device)
    if request_page_limit:
        cols = min(request_page_limit, table.shape[1])
        _mark_request_pages[(table.shape[0], triton.cdiv(cols, 64))](
            table,
            lengths.contiguous(),
            used,
            table.stride(0),
            table.stride(1),
            cols,
            TILE=64,
        )
    else:
        _mark_pages[(indices.shape[0],)](
            indices,
            positions.contiguous(),
            lengths.contiguous(),
            requests.contiguous(),
            table,
            used,
            indices.stride(0),
            table.stride(0),
            table.stride(1),
            table.shape[0],
            GROUPS=groups,
            TILE=triton.next_power_of_2(groups),
        )
    pages = torch.nonzero(used).flatten()
    page_map = torch.where(used.bool(), used.cumsum(0, dtype=torch.int32) - 1, -1)
    n = pages.numel()
    # The extra row receives writes for unused and padded slots.
    current = torch.full((n * 128 + 1,), -1, dtype=torch.int32, device=table.device)
    slot = slots.reshape(-1).long()
    valid_slot = (slot >= 0) & (slot < num_pages * 128)
    page = page_map[(slot // 128).clamp(0, num_pages - 1)].long()
    dest = torch.where(valid_slot & (page >= 0), page * 128 + slot.remainder(128), n * 128)
    current.scatter_(0, dest, torch.arange(slot.numel(), device=table.device, dtype=torch.int32))
    return pages, current, page_map


def materialize_prepared_pages(cache, current_kv, pages, current):
    k, v, ks, vs = cache
    current_k, current_v = current_kv
    n = pages.numel()
    shape = (max(n, 1), 128, 1, k.shape[-1])
    out_k = current_k.new_empty(shape)
    out_v = current_v.new_empty(shape)
    if n:
        _materialize[(n * 16,)](
            pages,
            current,
            k.view(torch.uint8),
            v.view(torch.uint8),
            ks,
            vs,
            current_k,
            current_v,
            out_k,
            out_v,
            current_k.stride(0),
            current_v.stride(0),
            current_k.stride(-1),
            current_v.stride(-1),
            DIM=k.shape[-1],
            WIDTH=triton.next_power_of_2(k.shape[-1]),
            ROWS=8,
        )
    return out_k, out_v
