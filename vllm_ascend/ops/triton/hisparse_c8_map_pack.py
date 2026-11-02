# SPDX-License-Identifier: Apache-2.0
"""Staged C8 mapping/query packing; no changes to installed model runtime."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _map32(
    source,
    output,
    ROW: tl.constexpr,
    COL: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    token = tl.load(source + row.to(tl.int64) * ROW + column * COL, mask=column < WIDTH, other=-1)
    valid = (column < WIDTH) & (token >= 0) & (token < CAPACITY)
    count = tl.sum(valid.to(tl.int32), 0)
    safe = tl.maximum(token, 0)
    mapped = safe // BLOCK * STRIDE + safe % BLOCK
    base = output + row.to(tl.int64) * WIDTH
    prefix = tl.sum(((column < count) & ~valid).to(tl.int32), 0) == 0
    if prefix:
        tl.store(base + column, tl.where(valid, mapped, -1), mask=column < WIDTH)
    else:
        destination = tl.cumsum(valid.to(tl.int32), 0) - 1
        tl.store(base + destination, mapped, mask=valid)
        tl.store(base + column, -1, mask=(column < WIDTH) & (column >= count))


@triton.jit
def _map_pack(
    source,
    output,
    query,
    rope,
    packed,
    qend,
    klen,
    ROW: tl.constexpr,
    COL: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    TILE: tl.constexpr,
    HEADS: tl.constexpr,
    QROW: tl.constexpr,
    QHEAD: tl.constexpr,
    QCOL: tl.constexpr,
    RROW: tl.constexpr,
    RHEAD: tl.constexpr,
    RCOL: tl.constexpr,
    ROWS: tl.constexpr,
    KEY_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    if head == 0:
        column = tl.arange(0, TILE)
        token = tl.load(source + row.to(tl.int64) * ROW + column * COL, mask=column < WIDTH, other=-1)
        valid = (column < WIDTH) & (token >= 0) & (token < CAPACITY)
        count = tl.sum(valid.to(tl.int32), 0)
        safe = tl.maximum(token, 0)
        mapped = safe // BLOCK * STRIDE + safe % BLOCK
        base = output + row.to(tl.int64) * WIDTH
        prefix = tl.sum(((column < count) & ~valid).to(tl.int32), 0) == 0
        if prefix:
            tl.store(base + column, tl.where(valid, mapped, -1), mask=column < WIDTH)
        else:
            destination = tl.cumsum(valid.to(tl.int32), 0) - 1
            tl.store(base + destination, mapped, mask=valid)
            tl.store(base + column, -1, mask=(column < WIDTH) & (column >= count))
    column = tl.arange(0, 1024)
    a = tl.load(query + row.to(tl.int64) * QROW + head * QHEAD + column * QCOL, mask=column < 512, other=0)
    b = tl.load(
        rope + row.to(tl.int64) * RROW + head * RHEAD + (column - 512) * RCOL,
        mask=(column >= 512) & (column < 576),
        other=0,
    )
    tl.store(packed + (row.to(tl.int64) * HEADS + head) * 576 + column, tl.where(column < 512, a, b), mask=column < 576)
    if (row == 0) & (head == 0):
        tl.store(qend, ROWS)
        tl.store(klen, KEY_ROWS)


def geometry(indices, block, stride, capacity):
    assert indices.ndim == 2 and indices.dtype == torch.int32 and indices.device.type == "npu"
    assert 0 < indices.shape[1] <= 4096 and 0 < block <= stride and capacity > 0 and capacity % block == 0
    assert (capacity // block - 1) * stride + block <= 2**31 - 1
    return (
        indices.stride(0),
        indices.stride(1),
        indices.shape[1],
        block,
        stride,
        capacity,
        triton.next_power_of_2(indices.shape[1]),
    )


def map32(indices, block, stride, capacity):
    args = geometry(indices, block, stride, capacity)
    output = torch.empty(indices.shape, dtype=torch.int32, device=indices.device)
    if indices.shape[0]:
        _map32[(indices.shape[0],)](indices, output, *args)
    return output


def map_pack(indices, block, stride, capacity, query, rope, qend, klen, key_rows):
    args = geometry(indices, block, stride, capacity)
    rows, heads, width = query.shape
    assert rows == indices.shape[0] and rows > 0 and heads > 0 and width == 512 and query.dtype == torch.bfloat16
    assert rope.shape == (rows, heads, 64) and rope.dtype == query.dtype
    assert qend.shape == klen.shape == (1,) and qend.dtype == klen.dtype == torch.int32
    assert all(v.device == indices.device for v in [query, rope, qend, klen]) and 0 < key_rows <= 2**31 - 1
    output = torch.empty(indices.shape, dtype=torch.int32, device=indices.device)
    packed = torch.empty((rows, heads, 576), dtype=query.dtype, device=query.device)
    _map_pack[(rows, heads)](
        indices,
        output,
        query.view(torch.uint16),
        rope.view(torch.uint16),
        packed.view(torch.uint16),
        qend,
        klen,
        *args,
        heads,
        *query.stride(),
        *rope.stride(),
        rows,
        key_rows,
    )
    return output, packed
