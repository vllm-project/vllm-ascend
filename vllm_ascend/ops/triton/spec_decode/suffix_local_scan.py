# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local history scan kernels and launch wrapper for MRV2 suffix decoding.

Adapted from vLLM #60339, merge 190df0b90007eead441d97d3abf6270f84f2c63e.
"""

import torch
from vllm.triton_utils import tl, triton

# Distinct negative fill tokens prevent masked loads from matching each other.
_MASKED_TAIL_TOKEN = tl.constexpr(-2)
_MASKED_CANDIDATE_TOKEN = tl.constexpr(-1)


@triton.jit
def _suffix_local_scan_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    idx_mapping_ptr,  # *int32  [B]  batch_idx -> req_state_idx
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    scratch_ptr,  # *int64  [B, scratch_stride]  (output)
    scratch_stride,
    Lp1,  # int64 packing radix (> any position)
    MAX_DEPTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pack the longest, rightmost local match per request and block."""
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    seq_len = tl.load(total_len_ptr + req).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b)
    row = token_ids_ptr + req * token_ids_stride

    scratch_off = b * scratch_stride + blk
    # Blocks past the last candidate (an occurrence must leave a token).
    if blk * BLOCK >= seq_len:
        tl.store(scratch_ptr + scratch_off, tl.zeros((), tl.int64))
        return

    pos = blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    active = (pos >= 1) & (pos < seq_len) & (num_sampled > 0)
    # Keep indirect-load masks independent of loop-carried match state so
    # Triton-Ascend can lower the loop. Matching still stops at the first miss.
    candidate_mask = active
    match_len = tl.zeros([BLOCK], dtype=tl.int64)
    for t in range(MAX_DEPTH):
        # Request token t back from its end (newest first).
        tail_pos = seq_len - 1 - t
        tail_tok = tl.load(row + tl.maximum(tail_pos, 0), mask=tail_pos >= 0, other=_MASKED_TAIL_TOKEN)
        src_pos = pos - 1 - t
        load_mask = candidate_mask & (src_pos >= 0) & (tail_pos >= 0)
        tok = tl.load(
            row + tl.minimum(tl.maximum(src_pos, 0), token_ids_stride - 1),
            mask=load_mask,
            other=_MASKED_CANDIDATE_TOKEN,
        )
        active = active & load_mask & (tok == tail_tok)
        match_len += active.to(tl.int64)

    score = tl.where(match_len > 0, match_len * Lp1 + pos, 0)
    tl.store(scratch_ptr + scratch_off, tl.max(score, axis=0))


def launch_suffix_local_scan(
    token_ids_ptr: torch.Tensor,
    token_ids_stride: int,
    idx_mapping_ptr: torch.Tensor,
    total_len_ptr: torch.Tensor,
    num_sampled_ptr: torch.Tensor,
    scratch_ptr: torch.Tensor,
    scratch_stride: int,
    Lp1: int,
    MAX_DEPTH: int,
    BLOCK: int,
    **launch_options,
) -> None:
    """Scan active request rows using the actual history and scratch strides."""
    launch_options.update(
        MAX_DEPTH=MAX_DEPTH,
        BLOCK=BLOCK,
    )
    num_reqs = idx_mapping_ptr.shape[0]
    _suffix_local_scan_kernel[(num_reqs, scratch_ptr.shape[1])](
        token_ids_ptr,
        token_ids_ptr.stride(0),
        idx_mapping_ptr,
        total_len_ptr,
        num_sampled_ptr,
        scratch_ptr,
        scratch_ptr.stride(0),
        Lp1,
        **launch_options,
    )
