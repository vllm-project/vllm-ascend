# SPDX-License-Identifier: Apache-2.0
"""NPU copies for request-owned SWA rings and packed temporary PA pages."""

from vllm.triton_utils import tl, triton


@triton.jit
def _assemble(
    RING, CURRENT, IDS, QSTART, SEQLEN, BASE, OUT, WIDTH: tl.constexpr, WINDOW: tl.constexpr, ROWS: tl.constexpr
):
    req = tl.program_id(0)
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    dim = tl.arange(0, WIDTH)
    first = tl.load(QSTART + req)
    end = tl.load(QSTART + req + 1)
    length = end - first
    computed = tl.load(SEQLEN + req) - length
    state = tl.load(IDS + req)
    # FIA graph padding can have a positive query length with seq_len zero.
    # Keep valid null-slot graph capture, but never read invalid padding KV.
    valid = (state >= 0) & (length > 0) & (computed >= 0)
    old = tl.where(valid, tl.minimum(computed, WINDOW - 1), 0)
    length = tl.where(valid, length, 0)
    base = tl.load(BASE + req)
    old_slot = (computed - old + row) % WINDOW
    previous = tl.load(
        RING + (state * WINDOW + old_slot[:, None]) * WIDTH + dim[None, :], mask=(row < old)[:, None], other=0
    )
    new = tl.load(
        CURRENT + (first + row[:, None] - old) * WIDTH + dim[None, :],
        mask=((row >= old) & (row < old + length))[:, None],
        other=0,
    )
    value = tl.where((row < old)[:, None], previous, new)
    tl.store(OUT + (base + row[:, None]) * WIDTH + dim[None, :], value, mask=(row < old + length)[:, None])


@triton.jit
def _commit(RING, CURRENT, IDS, QSTART, SEQLEN, WIDTH: tl.constexpr, WINDOW: tl.constexpr, ROWS: tl.constexpr):
    req = tl.program_id(0)
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    dim = tl.arange(0, WIDTH)
    first = tl.load(QSTART + req)
    end = tl.load(QSTART + req + 1)
    length = end - first
    computed = tl.load(SEQLEN + req) - length
    state = tl.load(IDS + req)
    # Capture dummy rows can share reserved slot 0. They participate in
    # staging, but only real request slots may receive persistent writes.
    valid = (state > 0) & (length > 0) & (computed >= 0)
    count = tl.where(valid, tl.minimum(length, WINDOW), 0)
    absolute = computed + length - count + row
    value = tl.load(CURRENT + (end - count + row[:, None]) * WIDTH + dim[None, :], mask=(row < count)[:, None], other=0)
    tl.store(
        RING + (state * WINDOW + absolute[:, None] % WINDOW) * WIDTH + dim[None, :], value, mask=(row < count)[:, None]
    )


def assemble(ring, current, state_ids, query_start, seq_lens, row_starts, output, max_query_len):
    assert ring.is_contiguous() and current.is_contiguous() and output.is_contiguous()
    nreq = state_ids.numel()
    width = ring.shape[-1]
    _assemble[(nreq, triton.cdiv(max_query_len + 127, 16))](
        ring, current, state_ids, query_start, seq_lens, row_starts, output, WIDTH=width, WINDOW=128, ROWS=16
    )


def commit(ring, current, state_ids, query_start, seq_lens):
    assert ring.is_contiguous() and current.is_contiguous()
    _commit[(state_ids.numel(), 8)](
        ring, current, state_ids, query_start, seq_lens, WIDTH=ring.shape[-1], WINDOW=128, ROWS=16
    )
