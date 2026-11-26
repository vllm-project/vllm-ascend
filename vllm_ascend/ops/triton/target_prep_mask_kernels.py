# SPDX-License-Identifier: Apache-2.0
"""Imported lazily only after default-off graph-external host eligibility."""

from vllm.triton_utils import tl, triton


@triton.jit
def _prefix_mask_write_kernel(
    mask,
    MASK_STRIDE: tl.constexpr,
    SIZE: tl.constexpr,
    ACTUAL: tl.constexpr,
    PREFIX_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.where(lane < ACTUAL, PREFIX_VALUE, not PREFIX_VALUE)
    tl.store(mask + lane * MASK_STRIDE, value, lane < SIZE)


@triton.jit
def _prepare_pos_seq_lens_padding_kernel(
    pos,
    seq_lens,
    idx_mapping,
    query_start_loc,
    num_computed_tokens,
    padding,
    MAX_REQS: tl.constexpr,
    PADDED_TOKENS: tl.constexpr,
    ACTUAL_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Clone of the original graph-external preparation kernel, plus one output.
    # Its final program owns seq_lens padding and now also writes is_padding.
    req = tl.program_id(0)
    nreq = tl.num_programs(0) - 1
    if req == nreq:
        for begin in tl.range(nreq, MAX_REQS, BLOCK):
            lane = begin + tl.arange(0, BLOCK)
            tl.store(seq_lens + lane, 0, lane < MAX_REQS)
        for begin in tl.range(0, PADDED_TOKENS, BLOCK):
            lane = begin + tl.arange(0, BLOCK)
            tl.store(padding + lane, lane >= ACTUAL_TOKENS, lane < PADDED_TOKENS)
    else:
        state = tl.load(idx_mapping + req)
        computed = tl.load(num_computed_tokens + state)
        first = tl.load(query_start_loc + req)
        stop = tl.load(query_start_loc + req + 1)
        length = stop - first
        tl.store(seq_lens + req, computed + length)
        for begin in tl.range(0, length, BLOCK):
            lane = begin + tl.arange(0, BLOCK)
            tl.store(pos + first + lane, computed + lane, lane < length)
