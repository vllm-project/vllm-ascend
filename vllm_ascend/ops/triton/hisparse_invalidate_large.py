# SPDX-License-Identifier: Apache-2.0
"""Exact large-write membership without unsupported CAS lowering.

Sort one invocation-owned vector of (persistent state, Host slot) keys;
probe every group's cached key by bounded binary search. Decode unchanged.
"""

import torch
from vllm.triton_utils import tl, triton

_SLOT_RANGE = 1 << 31
_SENTINEL = (1 << 63) - 1


@triton.jit
def _make_keys(
    output,
    request_states,
    requests,
    written,
    TOKENS: tl.constexpr,
    REQUESTS: tl.constexpr,
    STATES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(requests + offset, mask=offset < TOKENS, other=-1)
    state = tl.load(request_states + request, mask=(offset < TOKENS) & (request >= 0) & (request < REQUESTS), other=-1)
    slot = tl.load(written + offset, mask=offset < TOKENS, other=-1).to(tl.int64)
    valid = (state >= 0) & (state < STATES) & (slot >= 0) & (slot < 2147483648)
    key = state.to(tl.int64) * 2147483648 + slot
    tl.store(output + offset, tl.where(valid, key, 9223372036854775807), mask=offset < TOKENS)


@triton.jit
def _invalidate_sorted(keys, tables, TOKENS: tl.constexpr, HOT: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    state = tl.program_id(0)
    group = tl.program_id(1)
    offset = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    table = tl.load(tables + group).to(tl.pointer_type(tl.int32))
    row = table + state * HOT + offset
    cached = tl.load(row, mask=offset < HOT, other=-1)
    valid = (offset < HOT) & (cached >= 0)
    wanted = state.to(tl.int64) * 2147483648 + cached.to(tl.int64)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), TOKENS, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        value = tl.load(keys + mid, mask=valid & (mid < TOKENS), other=9223372036854775807)
        before = value < wanted
        lo = tl.where(valid & before, mid + 1, lo)
        hi = tl.where(valid & ~before, mid, hi)
    found = tl.load(keys + lo, mask=valid & (lo < TOKENS), other=-1)
    tl.store(row, -1, mask=valid & (found == wanted))


def invalidate_large_written_groups(tables, states, requests, written, num_states, hot):
    tokens = written.numel()
    if tokens < 256 or not 0 < num_states <= 32:
        raise ValueError("Large invalidation requires >=256 writes and 1..32 states")
    unsorted = torch.empty(tokens, dtype=torch.int64, device=written.device)
    _make_keys[(triton.cdiv(tokens, 256),)](
        unsorted,
        states,
        requests,
        written,
        TOKENS=tokens,
        REQUESTS=states.numel(),
        STATES=num_states,
        BLOCK=256,
        multibuffer=False,
    )
    keys = torch.sort(unsorted).values
    _invalidate_sorted[(num_states, tables.numel(), triton.cdiv(hot, 256))](
        keys,
        tables,
        TOKENS=tokens,
        HOT=hot,
        STEPS=tokens.bit_length(),
        BLOCK=256,
        multibuffer=False,
    )
